from config import SENTENCE_SIZE


def segment_sentences(words, n=SENTENCE_SIZE):
    """
    Words -> Sentences
    """
    sentences = []
    for i in range(0, len(words), n):
        segment = words[i : i + n]
        text = " ".join(w["text"] for w in segment)
        start = segment[0]["start_time"]
        end = segment[-1]["end_time"]
        sentences.append({"text": text, "start_time": start, "end_time": end})
    return sentences
