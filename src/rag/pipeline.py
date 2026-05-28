"""RAG pipeline combining retrieval and generation."""

import logging
from dataclasses import dataclass

from src.config import DEFAULT_TOP_K
from src.indexing.vector_store import VectorStore
from src.rag.generator import generate_answer
from src.rag.prompts import (
    BASE_LLM_PROMPT,
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    format_context,
)
from src.rag.retriever import Retriever

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    answer: str
    retrieved_contexts: list[dict]
    citations: list[dict]
    query: str


class RAGPipeline:
    def __init__(self, vector_store: VectorStore | None = None):
        if vector_store is None:
            vector_store = VectorStore()
            vector_store.load()
        self.vector_store = vector_store
        self.retriever = Retriever(vector_store)

    def ask(self, question: str, top_k: int = DEFAULT_TOP_K) -> RAGResponse:
        """Answer a question using RAG."""
        retrieved_chunks = self.retriever.retrieve(question, top_k=top_k)
        context = format_context(retrieved_chunks)
        user_prompt = USER_PROMPT_TEMPLATE.format(
            question=question, retrieved_context=context
        )
        answer = generate_answer(SYSTEM_PROMPT, user_prompt)

        citations = [
            {
                "title": chunk.get("title", ""),
                "source_url": chunk.get("source_url", ""),
                "section": chunk.get("section", ""),
                "score": chunk.get("score", 0.0),
            }
            for chunk in retrieved_chunks
        ]

        return RAGResponse(
            answer=answer,
            retrieved_contexts=retrieved_chunks,
            citations=citations,
            query=question,
        )

    def ask_base_llm(self, question: str) -> str:
        """Answer using only the base LLM without retrieval."""
        prompt = BASE_LLM_PROMPT.format(question=question)
        return generate_answer("You are a helpful AI assistant.", prompt)

    def retrieve_only(self, question: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
        """Retrieve relevant chunks without generation."""
        return self.retriever.retrieve(question, top_k=top_k)
