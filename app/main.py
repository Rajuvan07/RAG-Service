from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from app.rag import RAGService

app = FastAPI(
    title="Simple RAG Service",
    description="A minimal Retrieval-Augmented Generation API.",
    version="1.0.0",
)

rag = RAGService()
DOCS_DIR = Path("data/documents")


class IngestRequest(BaseModel):
    path: str = Field(default="data/documents", description="Folder containing .txt and .md files")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int = Field(default=3, ge=1, le=10)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest")
def ingest(request: IngestRequest):
    folder = Path(request.path)
    if not folder.exists() or not folder.is_dir():
        raise HTTPException(status_code=400, detail=f"Folder not found: {folder}")
    try:
        result = rag.ingest_folder(folder)
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/ask")
def ask(request: AskRequest):
    try:
        return rag.ask(request.question, request.top_k)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
