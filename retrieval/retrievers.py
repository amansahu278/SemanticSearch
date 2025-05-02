import re
from typing import List
from langchain_community.retrievers import BM25Retriever
from langchain.schema.document import Document
from langchain.retrievers import EnsembleRetriever
from langchain_community.vectorstores import FAISS

from preprocessing.chunking import chunks_to_docs
import phonetics

from logging import getLogger
import os

logger = getLogger(__name__)


def _custom_bm25_preprocessing(text: str) -> List[str]:
    """
    Custom preprocessing function for BM25 tokenization.
    """
    bm25_tokenization_regex = r"(?u)\b\w+\b"
    text = text.lower()
    tokenizer = re.compile(bm25_tokenization_regex).findall
    # tokens = re.findall(bm25_tokenization_regex, text)
    return tokenizer(text)


def _enrich_text_with_phonetics(text):
    tokens = _custom_bm25_preprocessing(text)
    enriched_tokens = set(tokens)
    for token in tokens:
        try:
            keys = phonetics.dmetaphone(token)
            primary_key = keys[0]
            if primary_key:
                enriched_tokens.add(primary_key)
        except IndexError:
            pass
    return " ".join(list(enriched_tokens))


def build_bm25_retriever(chunks, k=5, use_phonetic_enrichment=False):
    logger.info("Initializing BM25 retriever")
    docs = chunks_to_docs(chunks)

    if use_phonetic_enrichment:
        logger.info("Using phonetics for BM25 tokenization")
        enriched_docs = []
        for doc in docs:
            metadata = doc.metadata
            metadata["original_text"] = doc.page_content
            enriched_doc = Document(
                page_content=_enrich_text_with_phonetics(doc.page_content),
                metadata=doc.metadata,
            )
            enriched_docs.append(enriched_doc)
        docs = enriched_docs

    retriever = BM25Retriever.from_documents(docs)
    retriever.k = k
    return retriever


def build_deep_retriever(chunks, vector_store, embeddings, k=5, index_name=None):
    logger.info("Initializing FAISS retriever")
    docs = chunks_to_docs(chunks)

    if index_name:
        index_path = os.path.join("faiss_index", index_name)
        if os.path.exists(index_path):
            logger.info(f"Loading FAISS index from {index_path}")
            vectordb = vector_store.load_local(
                index_path, embeddings, allow_dangerous_deserialization=True
            )
        else:
            logger.info(f"Creating and saving FAISS index to {index_path}")
            vectordb = vector_store.from_documents(docs, embeddings)
            vectordb.save_local(index_path)
    else:
        logger.info("Creating FAISS index without persistence")
        vectordb = vector_store.from_documents(docs, embeddings)

    retriever = vectordb.as_retriever(
        search_type="mmr", search_kwargs={"k": k, "score_threshold": 0.7}
    )
    return retriever


def build_hyde_retriever(chunks, vector_store, llm, k=5):
    docs = [
        Document(
            page_content=chunk["text"],
            metadata={"start_time": chunk["start_time"], "end_time": chunk["end_time"]},
        )
        for chunk in chunks
    ]
    # TODO: hyde embeddings
    vectordb = vector_store.from_documents(docs, embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": k})
    return retriever


def build_ensemble_retriever(bm25_retriever, faiss_retriever):
    ensemble = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5]
    )
    return ensemble
