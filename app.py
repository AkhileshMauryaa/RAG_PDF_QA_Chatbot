import streamlit as st
import os
from qa_chatbot import process_pdf_and_answer

st.set_page_config(page_title="PDF Q&A Chatbot", page_icon="📄")

st.title("📄 AI PDF Q&A RAG Chatbot")
st.write("Upload a PDF and ask questions from its content.")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
query = st.text_input("Enter your question")

if uploaded_file and query:
    pdf_path = f"temp_{uploaded_file.name}"
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.write("Processing your request...")
    with st.spinner("Fetching answer..."):
        answer = process_pdf_and_answer(pdf_path, query)
    
    st.subheader("Answer:")
    st.write(answer)
    
    os.remove(pdf_path)
