from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

def summarize(context, query, model="llama-3.1-8b-instant"):
    chain = _load_summarizer(model)
    return chain.invoke({"context": context, "query": query})

def _load_summarizer(model="llama-3.1-8b-instant"):
    llm = ChatGroq(model=model, temperature=0)
    prompt = PromptTemplate(
    input_variables=["context", "query"],
    template="""
    You are a summarizion assistant. 
    You are given a query and a context. 
    The context is a collection of ASR transcripts. Your task is to summarize the context based on the query.
    You should focus on the most relevant parts of the context that answer the query.
    You should not include any information that is not relevant to the query.
    You should not include any information that is not in the context.

    Given the query: '{query}', answer appropriately based on the following text: {context}

    Respond clearly and briefly. Be detailed in your approach, maybe citing what the speaker had said."""
    )
    chain = prompt | llm | StrOutputParser()
    return chain
