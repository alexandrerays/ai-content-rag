"""Retrieval module with hybrid search and cross-encoder reranking."""

import logging
import re
import string

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from src.config import DEFAULT_TOP_K, RERANKER_MODEL
from src.indexing.embeddings import generate_query_embedding
from src.indexing.vector_store import VectorStore

logger = logging.getLogger(__name__)

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "above",
    "below", "between", "out", "off", "over", "under", "again", "further",
    "then", "once", "here", "there", "when", "where", "why", "how", "all",
    "each", "every", "both", "few", "more", "most", "other", "some",
    "such", "no", "nor", "not", "only", "own", "same", "so", "than",
    "too", "very", "just", "because", "but", "and", "or", "if", "while",
    "about", "this", "that", "these", "those", "it", "its", "what",
    "which", "who", "whom", "whose",
}


def tokenize_for_bm25(text: str) -> list[str]:
    """Tokenize text for BM25 with stopword removal and normalization."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


class Retriever:
    def __init__(self, vector_store: VectorStore, use_hybrid: bool = True, use_reranker: bool = True):
        self.vector_store = vector_store
        self.use_hybrid = use_hybrid
        self.use_reranker = use_reranker
        self.bm25: BM25Okapi | None = None
        self.reranker: CrossEncoder | None = None

        if use_hybrid and vector_store.is_loaded:
            self._build_bm25_index()

        if use_reranker:
            self._load_reranker()

    def _build_bm25_index(self) -> None:
        """Build BM25 index from stored chunks."""
        tokenized_corpus = [
            tokenize_for_bm25(chunk["text"]) for chunk in self.vector_store.chunks
        ]
        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info("Built BM25 index for hybrid retrieval")

    def _load_reranker(self) -> None:
        """Load cross-encoder reranker model."""
        try:
            self.reranker = CrossEncoder(RERANKER_MODEL)
            logger.info(f"Loaded reranker: {RERANKER_MODEL}")
        except Exception as e:
            logger.warning(f"Failed to load reranker: {e}. Continuing without reranking.")
            self.reranker = None

    def rerank(self, query: str, chunks: list[dict], top_k: int) -> list[dict]:
        """Rerank chunks using cross-encoder."""
        if not self.reranker or not chunks:
            return chunks[:top_k]

        pairs = [(query, chunk["text"]) for chunk in chunks]
        scores = self.reranker.predict(pairs)

        for i, chunk in enumerate(chunks):
            chunk["rerank_score"] = float(scores[i])

        reranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]

    def retrieve_vector(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
        """Retrieve chunks using vector similarity."""
        query_embedding = generate_query_embedding(query)
        return self.vector_store.search(query_embedding, top_k=top_k)

    def retrieve_bm25(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
        """Retrieve chunks using BM25 keyword matching."""
        if self.bm25 is None:
            return []

        tokenized_query = tokenize_for_bm25(query)
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
        """Retrieve chunks using hybrid search with optional reranking."""
        candidate_k = top_k * 4

        vector_results = self.retrieve_vector(query, top_k=candidate_k)

        if not self.use_hybrid or self.bm25 is None:
            if self.use_reranker and self.reranker:
                return self.rerank(query, vector_results, top_k)
            return vector_results[:top_k]

        bm25_results = self.retrieve_bm25(query, top_k=candidate_k)

        combined_scores: dict[tuple, dict] = {}

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

        candidates = []
        for item in sorted_results[:candidate_k]:
            chunk = item["chunk"]
            chunk["score"] = item["score"]
            candidates.append(chunk)

        if self.use_reranker and self.reranker:
            return self.rerank(query, candidates, top_k)

        return candidates[:top_k]
