import os
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()

genai.configure(api_key = os.getenv('GOOGLE_API_KEY'))

def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Gemini💁")

    user_query = st.text_input("Ask a Question from the PDF Files")
    button = st.button("Get Response")

    if button:
        get_reponse(user_query)

    with st.sidebar:
        st.title("Menu:")
        pdf_documents = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_content = get_pdf_content(pdf_documents)
                chunks_text = get_text_chunks(raw_content)
                get_vector_store(chunks_text)
                st.success("Done")