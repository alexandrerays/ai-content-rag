"""FastAPI application for the RAG system."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import DEFAULT_TOP_K
from src.rag.pipeline import RAGPipeline

logger = logging.getLogger(__name__)

pipeline: RAGPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    try:
        pipeline = RAGPipeline()
        logger.info("RAG pipeline loaded successfully")
    except FileNotFoundError:
        logger.error("Vector index not found. Run: python -m src.indexing.build_index")
        pipeline = None
    yield


app = FastAPI(
    title="RAG System API",
    description="Retrieval-Augmented Generation API for domain-specific Q&A",
    version="1.0.0",
    lifespan=lifespan,
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The question to answer")
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=20, description="Number of chunks to retrieve")


class Citation(BaseModel):
    title: str
    source_url: str
    section: str
    score: float


class AskResponse(BaseModel):
    answer: str
    retrieved_contexts: list[dict]
    citations: list[Citation]
    query: str


class RetrieveRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=20)


class RetrieveResponse(BaseModel):
    chunks: list[dict]
    query: str
    total: int


@app.get("/health")
async def health():
    return {
        "status": "healthy" if pipeline else "degraded",
        "index_loaded": pipeline is not None,
    }


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Index not loaded. Build the index first.")

    response = pipeline.ask(request.question, top_k=request.top_k)
    return AskResponse(
        answer=response.answer,
        retrieved_contexts=response.retrieved_contexts,
        citations=[Citation(**c) for c in response.citations],
        query=response.query,
    )


@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(request: RetrieveRequest):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Index not loaded. Build the index first.")

    chunks = pipeline.retrieve_only(request.question, top_k=request.top_k)
    return RetrieveResponse(
        chunks=chunks,
        query=request.question,
        total=len(chunks),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
