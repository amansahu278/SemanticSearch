import argparse
from asr.loader import load_asr
from preprocessing.sentence_segmentation import segment_sentences
from preprocessing.chunking import chunk_sentences
from retrieval.retrievers import build_bm25_retriever, build_ensemble_retriever, build_deep_retriever
from utils.utils import extract_keywords, datetime_to_str
from llm.summarizer import summarize

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

# import sys, os
# sys.path.append(os.path.abspath("/home/user/code/pwesuite"))

# from models.metric_learning.model import RNNMetricLearner
# from models.metric_learning.preprocessor import preprocess_dataset_foreign
# from main.utils import load_multi_data

import logging
from logging import getLogger
logger = getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser(description="Semantic Search Script")
    parser.add_argument("--filepath", type=str, required=True, help="Path to the ASR file")
    parser.add_argument("--query", type=str, required=True, help="Query for semantic search")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    return parser.parse_args()

def load_and_process_asr(filepath):
    # Load ASR data
    asr_data = load_asr(filepath, word=True)
    sentences = segment_sentences(asr_data)
    chunks = chunk_sentences(sentences)
    return chunks

def initialize_retrievers(chunks):
    # BM25 retriever initialization
    bm25_retriever = build_bm25_retriever(chunks, k=20)

    # FAISS retriever initialization
    embedding_model = HuggingFaceEmbeddings(model_name='intfloat/multilingual-e5-large-instruct')
    faiss_retriever = build_deep_retriever(chunks, FAISS, embedding_model, k=30)

    return bm25_retriever, faiss_retriever

def retrieve_and_summarize(query, bm25_retriever, faiss_retriever):
    # Use keyword extraction for better input to the BM25 model. Look into Yake.
    keywords_list = extract_keywords(query)
    logger.debug(f"Keywords extracted: {keywords_list}")

    # BM25 extraction
    relevant_docs = []
    for keywords in keywords_list:
        retrieved_docs = bm25_retriever.invoke(keywords[0])
        relevant_docs.extend(doc for doc in retrieved_docs)

    # FAISS retrieval
    retrieved_docs = faiss_retriever.invoke(query)
    relevant_docs.extend(doc for doc in retrieved_docs)

    combined_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    logger.debug("Context is: %s", combined_text)
    summary = summarize(combined_text, query)

    logger.info(f"\n📝 Answer:\n{summary}")
    retrieved_docs.sort(key=lambda x: x.metadata['start_time'])
    for doc in retrieved_docs:
        logger.info(f"{datetime_to_str(doc.metadata['start_time'])}: {datetime_to_str(doc.metadata['end_time'])}")

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level="INFO",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("semantic_search.log")
        ]
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
    bm25_retriever, faiss_retriever = initialize_retrievers(chunks)
    
    # With the keyword extraction, better not to use the ensemble retriever
    # ensemble_retriever = build_ensemble_retriever(bm25_retriever, faiss_retriever)

    while True:
        query = input("Enter your query (or type 'exit' to quit): ").strip()
        if query.lower() == 'exit':
            print("Exiting the program.")
            break
        retrieve_and_summarize(query, bm25_retriever, faiss_retriever)