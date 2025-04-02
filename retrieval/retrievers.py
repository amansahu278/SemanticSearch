from langchain_community.retrievers import BM25Retriever
from langchain.schema.document import Document
from langchain.retrievers import EnsembleRetriever

from langchain_community.vectorstores import FAISS

from preprocessing.chunking import chunks_to_docs

def build_bm25_retriever(chunks, k=5):
  docs = chunks_to_docs(chunks)
  retriever = BM25Retriever.from_documents(docs)
  retriever.k = k
  return retriever

def build_deep_retriever(chunks, vector_store, embeddings, k=5):
  docs = chunks_to_docs(chunks)
  vectordb = vector_store.from_documents(docs, embeddings)
  retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={'k': k, 'score_threshold': 0.7})
  return retriever

def build_hyde_retriever(chunks, vector_store, llm, k=5):
  docs = [
      Document(
          page_content=chunk['text'],
          metadata={'start_time': chunk['start_time'], 'end_time': chunk['end_time']}
      ) for chunk in chunks
  ]
  # TODO: hyde embeddings
  vectordb = vector_store.from_documents(docs, embeddings)
  retriever = vectordb.as_retriever(search_kwargs={'k': k})
  return retriever

def build_ensemble_retriever(bm25_retriever, faiss_retriever):
    ensemble = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5])
    return ensemble