import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

VECTOR_STORE_PATH = "faiss_index"

# Global references to avoid reloading continuously
_embeddings = None
_vectorstore = None
_retriever = None
_llm = None

def get_rag_chain():
    global _embeddings, _vectorstore, _retriever, _llm
    
    if _llm is None:
        if not os.path.exists(VECTOR_STORE_PATH):
            raise RuntimeError(f"FAISS index not found at {VECTOR_STORE_PATH}. Please run ingest.py first.")

        # Initialize Embeddings & Vectorstore
        _embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        _vectorstore = FAISS.load_local(VECTOR_STORE_PATH, _embeddings, allow_dangerous_deserialization=True)
        
        # We retrieve top 4 contexts
        _retriever = _vectorstore.as_retriever(search_kwargs={"k": 4})

        # Initialize LLM
        _llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)

    template = """You are an AI assistant built to answer questions based ONLY on the Swiggy Annual Report.
Use the following pieces of retrieved context to answer the question.
If the answer is not contained in the provided context, loudly and clearly state: "I don't know the answer to this question based on the Swiggy Annual Report." Do not try to make up an answer or hallucinate.

Context:
{context}

Question: {question}

Helpful Answer:"""

    prompt = PromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(f"Source Page: {doc.metadata.get('page', 'Unknown')}\nContent: {doc.page_content}" for doc in docs)

    # We want to return both the answer and the context documents, so we'll build a custom runnable
    
    def run_rag(query: str):
        docs = _retriever.invoke(query)
        context_str = format_docs(docs)
        
        chain = prompt | _llm | StrOutputParser()
        answer = chain.invoke({"context": context_str, "question": query})
        
        return {
            "answer": answer,
            "context_docs": [{"content": d.page_content, "page": d.metadata.get("page", 0)} for d in docs]
        }

    return run_rag

def query_rag(query: str):
    chain_func = get_rag_chain()
    return chain_func(query)
