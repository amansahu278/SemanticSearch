import argparse
from asr.loader import load_asr
from preprocessing.sentence_segmentation import segment_sentences
from preprocessing.chunking import chunk_sentences
from retrieval.retrievers import (
    build_bm25_retriever,
    build_deep_retriever,
    _custom_bm25_preprocessing,
)
from utils.utils import extract_keywords, datetime_to_str
from llm.summarizer import summarize

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

from transformers import AutoTokenizer
import phonetics
from config import CHUNK_SIZE, CHUNK_OVERLAP

from thefuzz import fuzz
from thefuzz import process as fuzz_process

import logging
from logging import getLogger

logger = getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser(description="Semantic Search Script")
    parser.add_argument(
        "--filepath", type=str, required=True, help="Path to the ASR file"
    )
    parser.add_argument("--query", type=str, help="Query for semantic search")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument(
        "--phonetic_enrichment",
        action="store_true",
        help="Enable phonetic enrichment for BM25",
    )
    parser.add_argument(
        "--query_expansion",
        type=str,
        choices=["phonetic", "fuzzy"],
        help="Enable query expansion for BM25. Options: 'phonetic' or 'fuzzy'.",
    )
    return parser.parse_args()


def load_and_process_asr(filepath, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    # Load ASR data
    asr_data = load_asr(filepath, word=True)
    sentences = segment_sentences(asr_data)
    chunks = chunk_sentences(sentences, chunk_size, chunk_overlap)
    logger.debug(f"Number of chunks: {len(chunks)}")

    return chunks


def initialize_retrievers(
    chunks,
    idx_name="faiss_index",
    top_bm25=5,
    top_faiss=10,
    use_phonetic_enrichment=False,
):
    # BM25 retriever initialization
    bm25_retriever = build_bm25_retriever(
        chunks, k=top_bm25, use_phonetic_enrichment=use_phonetic_enrichment
    )

    # FAISS retriever initialization
    embedding_model = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large-instruct"
    )
    faiss_retriever = build_deep_retriever(
        chunks, FAISS, embedding_model, k=top_faiss, index_name=idx_name
    )

    return bm25_retriever, faiss_retriever


def retrieve_and_summarize_phonetic_enrichment(
    query, bm25_retriever, faiss_retriever, top_keywords=1
):
    logger.info(f"Retrieving with phonetic enrichment")
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-instruct")
    # tokenizer = tiktoken.encoding_for_model("gpt-4")

    # Use keyword extraction for better input to the BM25 model. Look into Yake.
    keywords_list = extract_keywords(query, top=top_keywords)
    logger.debug(f"Keywords extracted: {keywords_list}")

    # Find phonetic keys for the keywords
    query_terms_for_bm25 = set()
    for keyword in keywords_list:
        key = phonetics.dmetaphone(keyword[0].strip().lower())
        primary_key = key[0]
        query_terms_for_bm25.add(primary_key)

    # Create the query string for BM25
    final_query = " ".join(query_terms_for_bm25)
    logging.debug(f"Final query for BM25: {final_query}")

    bm_25_relevant_docs = []
    retrieved_docs = bm25_retriever.invoke(final_query)
    for doc in retrieved_docs:
        # Retain original text
        doc.page_content = doc.metadata["original_text"]
        bm_25_relevant_docs.append(doc)
    logger.info(f"Number of BM25 extracted documents: {len(bm_25_relevant_docs)}")

    # FAISS retrieval
    faiss_relevant_docs = faiss_retriever.invoke(query)
    logger.info(f"Number of FAISS extracted documents: {len(faiss_relevant_docs)}")

    # Combine and sort documents
    combined_docs = bm_25_relevant_docs + faiss_relevant_docs
    combined_docs = sorted(
        combined_docs, key=lambda x: x.metadata.get("start_time", float("inf"))
    )

    # Create context
    combined_text = "\n\n".join(doc.page_content for doc in combined_docs)
    logger.debug("Context is: %s", combined_text)

    num_tokens = len(tokenizer.encode(combined_text))
    logger.info(
        f"Number of documents: {len(bm_25_relevant_docs) + len(faiss_relevant_docs)}"
    )
    logger.info(f"Number of tokens in context: {num_tokens}")

    summary = summarize(combined_text, query)

    logger.info(f"\n📝 Answer:\n{summary}")
    for doc in retrieved_docs:
        logger.info(
            f"{datetime_to_str(doc.metadata['start_time'])}: {datetime_to_str(doc.metadata['end_time'])}"
        )


def _query_expansion_preprocessing(chunks, create_phonetic_map=True):
    tokenized_corpus_full = list(
        map(lambda x: _custom_bm25_preprocessing(x["text"]), chunks)
    )
    corpus_vocab = list(
        set(token for doc_tokens in tokenized_corpus_full for token in doc_tokens)
    )

    phonetic_map = None
    if create_phonetic_map:
        phonetic_map = {}
        for word in corpus_vocab:
            try:
                keys = phonetics.dmetaphone(word)
                primary_key = keys[0]
                if primary_key not in phonetic_map:
                    phonetic_map[primary_key] = set()
                phonetic_map[primary_key].add(word)
            except IndexError:
                pass  # Ignoring words that don't produce phonetic keys

    return corpus_vocab, phonetic_map


def retrieve_and_summarize_query_expansion(
    query,
    bm25_retriever,
    faiss_retriever,
    corpus_vocab=None,
    phonetic_map=None,
    fuzzy_score_cutoff=95,
    fuzzy_limit=5,
):
    logger.info(
        f"Retrieving with query expansion, mode: {'phonetic' if phonetic_map else 'fuzzy'}"
    )
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-instruct")
    # tokenizer = tiktoken.encoding_for_model("gpt-4")

    # Use keyword extraction for better input to the BM25 model. Look into Yake.
    keywords_list = extract_keywords(query, top=1)
    logger.debug(f"Keywords extracted: {keywords_list}")

    phrase_tokens = _custom_bm25_preprocessing(keywords_list[0][0])  # call, of, duty

    # TODO: Maybe add stop word removal for terms to expand
    terms_to_expand = set(phrase_tokens)
    all_query_terms = set(phrase_tokens)

    for term in terms_to_expand:
        # Phonetic
        if phonetic_map:
            try:
                key = phonetics.dmetaphone(term.lower())
                primary_key = key[0]
                if primary_key and primary_key in phonetic_map:
                    variations = phonetic_map[primary_key]  # call, caal, calll...
                    all_query_terms.update(variations)
            except IndexError:
                pass
        else:
            # FUZZY
            # --- ---
            fuzzy_matches = fuzz_process.extract(
                term, corpus_vocab, scorer=fuzz.WRatio, limit=fuzzy_limit
            )
            for match, score in fuzzy_matches:
                logger.info(f"Term:{term}, Fuzzy match: {match}, Score: {score}")
                if score >= fuzzy_score_cutoff:
                    all_query_terms.add(match)
            # --- ---

    final_query = " ".join(all_query_terms)
    logger.debug(f"Final query for BM25: {final_query}")

    bm_25_relevant_docs = bm25_retriever.invoke(final_query)
    logger.info(f"Number of BM25 extracted documents: {len(bm_25_relevant_docs)}")

    # FAISS retrieval
    faiss_relevant_docs = faiss_retriever.invoke(query)
    logger.info(f"Number of FAISS extracted documents: {len(faiss_relevant_docs)}")

    # Combine and sort documents
    combined_docs = bm_25_relevant_docs + faiss_relevant_docs
    combined_docs = sorted(
        combined_docs, key=lambda x: x.metadata.get("start_time", float("inf"))
    )

    # Create context
    combined_text = "\n\n".join(doc.page_content for doc in combined_docs)
    logger.debug("Context is: %s", combined_text)

    num_tokens = len(tokenizer.encode(combined_text))
    logger.info(f"Number of documents: {combined_docs}")
    logger.info(f"Number of tokens in context: {num_tokens}")

    summary = summarize(combined_text, query)

    logger.info(f"\n📝 Answer:\n{summary}")
    for doc in combined_docs:
        logger.info(
            f"{datetime_to_str(doc.metadata['start_time'])}: {datetime_to_str(doc.metadata['end_time'])}"
        )


def retrieve_and_summarize_standard(
    query, bm25_retriever, faiss_retriever, top_keywords=1
):
    logger.info(f"Retrieving with standard BM25 and FAISS")
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-instruct")
    # tokenizer = tiktoken.encoding_for_model("gpt-4")

    # BM25 retrieval
    # Use keyword extraction for better input to the BM25 model.
    keywords_list = extract_keywords(query, top=top_keywords)
    logger.debug(f"Keywords extracted: {keywords_list}")

    final_query = " ".join(map(lambda x: x[0].lower(), keywords_list))

    bm_25_relevant_docs = bm25_retriever.invoke(final_query)
    logger.info(f"Number of BM25 extracted documents: {len(bm_25_relevant_docs)}")

    # FAISS retrieval
    faiss_relevant_docs = faiss_retriever.invoke(query)
    logger.info(f"Number of FAISS extracted documents: {len(faiss_relevant_docs)}")

    # Combine and sort documents
    combined_docs = bm_25_relevant_docs + faiss_relevant_docs
    combined_docs = sorted(
        combined_docs, key=lambda x: x.metadata.get("start_time", float("inf"))
    )

    # Create context
    combined_text = "\n\n".join(doc.page_content for doc in combined_docs)
    logger.debug("Context is: %s", combined_text)

    num_tokens = len(tokenizer.encode(combined_text))
    logger.info(f"Number of documents: {combined_docs}")
    logger.info(f"Number of tokens in context: {num_tokens}")

    summary = summarize(combined_text, query)

    logger.info(f"\n📝 Answer:\n{summary}")
    for doc in combined_docs:
        logger.info(
            f"{datetime_to_str(doc.metadata['start_time'])}: {datetime_to_str(doc.metadata['end_time'])}"
        )


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level="INFO",
        handlers=[logging.StreamHandler(), logging.FileHandler("semantic_search.log")],
    )

    args = get_args()
    filepath = args.filepath
    query = args.query

    if args.verbose:
        logger.setLevel("DEBUG")
    else:
        logger.setLevel("INFO")

    # Load and process ASR data
    chunks = load_and_process_asr(filepath)

    # Initialize retrievers
    bm25_retriever, faiss_retriever = initialize_retrievers(
        chunks,
        idx_name=filepath.split("/")[-1].split(".")[0],
        use_phonetic_enrichment=args.phonetic_enrichment,
    )

    # With the keyword extraction, better not to use the ensemble retriever
    # ensemble_retriever = build_ensemble_retriever(bm25_retriever, faiss_retriever)

    if query:
        if args.phonetic_enrichment:
            retrieve_and_summarize_phonetic_enrichment(
                query, bm25_retriever, faiss_retriever
            )
        elif args.query_expansion:
            expansion_type = args.query_expansion
            corpus_vocab, phonetic_map = _query_expansion_preprocessing(
                chunks, create_phonetic_map=(expansion_type == "phonetic")
            )
            retrieve_and_summarize_query_expansion(
                query,
                bm25_retriever,
                faiss_retriever,
                corpus_vocab=corpus_vocab,
                phonetic_map=phonetic_map,
            )
        else:
            retrieve_and_summarize_standard(query, bm25_retriever, faiss_retriever)
    else:
        while True:
            query = input("Enter your query (or type 'exit' to quit): ").strip()
            if query.lower() == "exit":
                print("Exiting the program.")
                break
            if args.phonetic_enrichment:
                retrieve_and_summarize_phonetic_enrichment(
                    query, bm25_retriever, faiss_retriever
                )
            elif args.query_expansion:
                expansion_type = args.query_expansion
                corpus_vocab, phonetic_map = _query_expansion_preprocessing(
                    chunks, create_phonetic_map=(expansion_type == "phonetic")
                )
                retrieve_and_summarize_query_expansion(
                    query,
                    bm25_retriever,
                    faiss_retriever,
                    corpus_vocab=corpus_vocab,
                    phonetic_map=phonetic_map,
                )
            else:
                retrieve_and_summarize_standard(query, bm25_retriever, faiss_retriever)
