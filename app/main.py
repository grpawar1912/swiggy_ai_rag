import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.rag import query_rag
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Swiggy RAG App")

class QueryRequest(BaseModel):
    query: str

class SourceDoc(BaseModel):
    content: str
    page: int

class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceDoc]

# Get absolute path for static directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
INDEX_PATH = os.path.join(STATIC_DIR, "index.html")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
async def read_index():
    if os.path.exists(INDEX_PATH):
        return FileResponse(INDEX_PATH)
    return {"message": f"Index file not found at {INDEX_PATH}"}

@app.post("/api/chat", response_model=QueryResponse)
async def chat_endpoint(request: QueryRequest):
    try:
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty.")
            
        result = query_rag(request.query)
        return QueryResponse(answer=result["answer"], sources=result["context_docs"])
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        print(f"Error during RAG: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while processing your request.")
