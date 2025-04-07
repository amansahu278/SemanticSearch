from asr.loader import load_asr
from preprocessing.sentence_segmentation import segment_sentences
from preprocessing.chunking import chunk_sentences
from retrieval.retrievers import build_bm25_retriever, build_ensemble_retriever, build_deep_retriever
from utils.utils import extract_keywords, datetime_to_str
from llm.summarizer import summarize

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

if __name__ == "__main__":    
    filepath = "/Users/amansahu/Downloads/s3_vtt/1449380350_asr.vtt"

    asr_data = load_asr(filepath, word=True)
    sentences = segment_sentences(asr_data)
    chunks = chunk_sentences(sentences)

    # # BM25 retriever initialization
    bm25_retriever = build_bm25_retriever(chunks, k=20)

    # # FAISS retriever initialization
    embedding_model = HuggingFaceEmbeddings(model_name='intfloat/multilingual-e5-large-instruct')
    faiss_retriever = build_deep_retriever(chunks, FAISS, embedding_model, k=30)
    
    # With the keyword extraction, better not to use the ensemble retriever
    # ensemble_retriever = build_ensemble_retriever(bm25_retriever, faiss_retriever)

    query = "What does the speaker think about bret of frie 2?"
    # Use keyword extraction for better input to the BM25 model. Look into Yake.
    keywords_list = extract_keywords(query)
    # keywords_list = [query]
    # print(keywords_list)
    # BM25 extraction
    relevant_docs = []
    for keywords in keywords_list:
        retrieved_docs = bm25_retriever.invoke(keywords[0])
        # print(retrieved_docs, type(retrieved_docs), len(retrieved_docs))
        relevant_docs.append(doc for doc in retrieved_docs)

    # # FAISS retrieval
    retrieved_docs = faiss_retriever.invoke(query)
    relevant_docs.extend(doc for doc in retrieved_docs)
    # # print(relevant_docs)

    combined_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    print("Context is: ", combined_text)
    summary = summarize(combined_text, query)


    print(f"\n📝 Answer:\n{summary}")
    for doc in retrieved_docs:
        print((datetime_to_str(doc.metadata['start_time']), datetime_to_str(doc.metadata['end_time'])))