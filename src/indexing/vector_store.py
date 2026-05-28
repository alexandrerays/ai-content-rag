"""FAISS vector store for chunk storage and retrieval."""

import json
import logging
from dataclasses import asdict
from pathlib import Path

import faiss
import numpy as np

from src.config import VECTOR_STORE_PATH
from src.ingestion.chunker import Chunk

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self, dimension: int = 384, store_path: str = VECTOR_STORE_PATH):
        self.dimension = dimension
        self.store_path = Path(store_path)
        self.index: faiss.IndexFlatIP | None = None
        self.chunks: list[dict] = []

    def build_index(self, embeddings: np.ndarray, chunks: list[Chunk]) -> None:
        """Build FAISS index from embeddings and chunks."""
        self.dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.dimension)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.chunks = [asdict(c) for c in chunks]
        logger.info(f"Built index with {self.index.ntotal} vectors of dimension {self.dimension}")

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[dict]:
        """Search for similar chunks."""
        if self.index is None:
            raise RuntimeError("Index not built. Call build_index() or load() first.")

        query = query_embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query)
        scores, indices = self.index.search(query, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = self.chunks[idx].copy()
            chunk["score"] = float(score)
            results.append(chunk)

        return results

    def save(self) -> None:
        """Save index and metadata to disk."""
        self.store_path.mkdir(parents=True, exist_ok=True)
        index_path = self.store_path / "index.faiss"
        metadata_path = self.store_path / "chunks.json"

        faiss.write_index(self.index, str(index_path))
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(
                {"dimension": self.dimension, "chunks": self.chunks},
                f,
                ensure_ascii=False,
            )

        logger.info(f"Saved index to {self.store_path}")

    def load(self) -> None:
        """Load index and metadata from disk."""
        index_path = self.store_path / "index.faiss"
        metadata_path = self.store_path / "chunks.json"

        if not index_path.exists():
            raise FileNotFoundError(f"No index found at {index_path}")

        self.index = faiss.read_index(str(index_path))

        with open(metadata_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.dimension = data["dimension"]
            self.chunks = data["chunks"]

        logger.info(f"Loaded index with {self.index.ntotal} vectors from {self.store_path}")

    @property
    def is_loaded(self) -> bool:
        return self.index is not None and len(self.chunks) > 0
