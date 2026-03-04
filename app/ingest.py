import os
import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

PDF_DIR = "pdf"
VECTOR_STORE_PATH = "faiss_index"

def ingest_documents():
    pdf_files = glob.glob(os.path.join(PDF_DIR, "*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {PDF_DIR}/ directory.")
        return

    # Use the first PDF found
    file_path = pdf_files[0]
    print(f"Loading document: {file_path}")

    loader = PyPDFLoader(file_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages.")

    # Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")

    # Embeddings
    print("Generating embeddings and creating FAISS index...")
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is not set in environment variables.")

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    
    # Create vector store
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # Save local
    vectorstore.save_local(VECTOR_STORE_PATH)
    print(f"FAISS index saved successfully to {VECTOR_STORE_PATH}/")

if __name__ == "__main__":
    ingest_documents()
