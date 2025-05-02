from config import CHUNK_OVERLAP, CHUNK_SIZE
from langchain.schema.document import Document
from logging import getLogger

logger = getLogger(__name__)


def chunk_sentences(sentences, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Creating chunks from sentences
    """
    logger.debug(
        f"Creating chunks of {chunk_size} sentences with overlap of {overlap} sentences"
    )
    chunks = []
    for i in range(0, len(sentences), chunk_size):
        segment = sentences[
            max(0, i - overlap) : min(len(sentences) - 1, i + chunk_size + overlap)
        ]
        text = " ".join(sentence["text"] for sentence in segment)
        start = segment[0]["start_time"]
        end = segment[-1]["end_time"]
        chunks.append({"text": text, "start_time": start, "end_time": end})
    return chunks


def chunks_to_docs(chunks):
    """
    Convert chunks to Langchain Document
    """
    docs = [
        Document(
            id=idx,  # NOTE: This is unique only to given chunks, not universally
            page_content=chunk["text"],
            metadata={"start_time": chunk["start_time"], "end_time": chunk["end_time"]},
        )
        for idx, chunk in enumerate(chunks)
    ]
    return docs
