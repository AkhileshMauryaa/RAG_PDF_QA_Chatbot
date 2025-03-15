import os
import faiss
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# Load API Key
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Configure Gemini API
genai.configure(api_key=GOOGLE_API_KEY)

# Function to process PDF and generate answer
def process_pdf_and_answer(pdf_path, query):
    # Step 1: Extract Text from PDF using LangChain
    print("Extracting text from PDF using LangChain...")
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()
    text = "\n".join([doc.page_content for doc in docs])

    # Step 2: Split Text into Chunks
    print("Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100, length_function=len)
    chunks = text_splitter.split_text(text)

    # Step 3: Generate Embeddings
    print("Generating embeddings...")
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", api_key=GOOGLE_API_KEY)
    embeddings = embedding_model.embed_documents(chunks)

    # Step 4: Store in FAISS
    print("Storing embeddings in FAISS...")
    vector_store = FAISS.from_texts(chunks, embedding_model)

    # Step 5: Retrieve Relevant Chunks
    print(f"Retrieving relevant text for query: {query}")
    docs = vector_store.similarity_search(query, k=3)
    relevant_text = "\n".join([doc.page_content for doc in docs])

    # Step 6: Generate Answer from Gemini API
    print("Generating answer from Gemini API...")
    prompt = f"""
    You are a helpful AI assistant. Answer the question based on the given context:
    Context: {relevant_text}
    Question: {query}
    Answer:
    """
    response = genai.GenerativeModel("gemini-1.5-flash").generate_content(prompt)
    return response.text.strip()
