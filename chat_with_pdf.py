import os
import streamlit as st

from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import faiss
from langchain_google_genai import GoogleGenerativeAIEmbeddings

import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()
genai.configure(api_key = os.getenv('GOOGLE_API_KEY'))

def get_pdf_content(pdf_docs):
    text = ""
    for doc in pdf_docs:
        pdf_reader = PdfReader(doc)
        for content in pdf_reader.pages:
            text += content.extract_text()
    return text

def get_text_chunks(content):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(content)
    return chunks

def get_vector_store(text_chunk):
    embed_model = GoogleGenerativeAIEmbeddings(model='embedding-001')
    vector_store = faiss.FAISS.from_texts(text_chunk, embedding=embed_model)
    vector_store.save_local("faiss-index")

def get_response():
    pass

def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Gemini💁")

    user_query = st.text_input("Ask a Question from the PDF Files")
    button = st.button("Get Response")

    if button:
        get_response(user_query)

    with st.sidebar:
        st.title("Menu:")
        pdf_documents = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_content = get_pdf_content(pdf_documents)
                chunks_text = get_text_chunks(raw_content)
                get_vector_store(chunks_text)
                st.success("Done")