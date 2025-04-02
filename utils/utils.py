from yake import KeywordExtractor
from datetime import datetime

def extract_keywords(query, top=2):
    # Note: the extractor works better if the topic is in bold.
    keyword_extractor = KeywordExtractor().extract_keywords(query)
    return keyword_extractor[:top]

def datetime_to_str(datetime):
    return datetime.strftime("%H:%M:%S.%f")

def str_to_datetime(datetime_str):
    return datetime.strptime(datetime_str, "%H:%M:%S.%f")