"""Retrieval module with vector and optional BM25 hybrid search."""

import logging

import numpy as np
from rank_bm25 import BM25Okapi

from src.config import DEFAULT_TOP_K
from src.indexing.embeddings import generate_query_embedding
from src.indexing.vector_store import VectorStore

logger = logging.getLogger(__name__)


class Retriever:
    def __init__(self, vector_store: VectorStore, use_hybrid: bool = True):
        self.vector_store = vector_store
        self.use_hybrid = use_hybrid
        self.bm25: BM25Okapi | None = None

        if use_hybrid and vector_store.is_loaded:
            self._build_bm25_index()

    def _build_bm25_index(self) -> None:
        """Build BM25 index from stored chunks."""
        tokenized_corpus = [
            chunk["text"].lower().split() for chunk in self.vector_store.chunks
        ]
        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info("Built BM25 index for hybrid retrieval")

    def retrieve_vector(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
        """Retrieve chunks using vector similarity."""
        query_embedding = generate_query_embedding(query)
        return self.vector_store.search(query_embedding, top_k=top_k)

    def retrieve_bm25(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
        """Retrieve chunks using BM25 keyword matching."""
        if self.bm25 is None:
            return []

        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                chunk = self.vector_store.chunks[idx].copy()
                chunk["bm25_score"] = float(scores[idx])
                results.append(chunk)

        return results

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        hybrid_weight: float = 0.7,
    ) -> list[dict]:
        """Retrieve chunks using hybrid search (vector + BM25)."""
        vector_results = self.retrieve_vector(query, top_k=top_k * 2)

        if not self.use_hybrid or self.bm25 is None:
            return vector_results[:top_k]

        bm25_results = self.retrieve_bm25(query, top_k=top_k * 2)

        combined_scores: dict[int, dict] = {}

        for rank, chunk in enumerate(vector_results):
            chunk_idx = chunk.get("chunk_index", rank)
            key = (chunk.get("document_id", ""), chunk_idx)
            vector_score = 1.0 / (rank + 1)
            combined_scores[key] = {
                "chunk": chunk,
                "score": hybrid_weight * vector_score,
            }

        for rank, chunk in enumerate(bm25_results):
            chunk_idx = chunk.get("chunk_index", rank)
            key = (chunk.get("document_id", ""), chunk_idx)
            bm25_score = 1.0 / (rank + 1)
            if key in combined_scores:
                combined_scores[key]["score"] += (1 - hybrid_weight) * bm25_score
            else:
                combined_scores[key] = {
                    "chunk": chunk,
                    "score": (1 - hybrid_weight) * bm25_score,
                }

        sorted_results = sorted(
            combined_scores.values(), key=lambda x: x["score"], reverse=True
        )

        final_results = []
        for item in sorted_results[:top_k]:
            chunk = item["chunk"]
            chunk["score"] = item["score"]
            final_results.append(chunk)

        return final_results
