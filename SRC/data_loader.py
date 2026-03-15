import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_data(path: str):
    loader = PyPDFLoader(path)
    return loader.load()

def filter_data(documents):
    clean_data = ""
    for doc in documents:
        doc.page_content = doc.page_content.replace("CLASSIFICATION | CONFIDENTIAL", " ")
        clean_data += doc.page_content + "\n"
    return clean_data

def create_chunks(clean_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_text(clean_data)