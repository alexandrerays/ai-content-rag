"""Embedding generation for document chunks."""

import logging

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)

_model_cache: dict[str, SentenceTransformer] = {}


def get_embedding_model(model_name: str = EMBEDDING_MODEL) -> SentenceTransformer:
    """Get or create a cached embedding model."""
    if model_name not in _model_cache:
        logger.info(f"Loading embedding model: {model_name}")
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


def generate_embeddings(
    texts: list[str],
    model_name: str = EMBEDDING_MODEL,
    batch_size: int = 32,
) -> np.ndarray:
    """Generate embeddings for a list of texts."""
    model = get_embedding_model(model_name)
    logger.info(f"Generating embeddings for {len(texts)} texts")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return np.array(embeddings, dtype=np.float32)


def generate_query_embedding(
    query: str,
    model_name: str = EMBEDDING_MODEL,
) -> np.ndarray:
    """Generate embedding for a single query."""
    model = get_embedding_model(model_name)
    embedding = model.encode([query], normalize_embeddings=True)
    return np.array(embedding, dtype=np.float32)[0]
