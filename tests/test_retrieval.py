"""Tests for retrieval functionality."""

import numpy as np
import pytest

from src.indexing.vector_store import VectorStore
from src.ingestion.chunker import Chunk


@pytest.fixture
def sample_chunks():
    return [
        Chunk(
            chunk_id="c1",
            document_id="d1",
            source_url="https://example.com/1",
            title="AI Safety",
            section="Introduction",
            chunk_index=0,
            text="AI safety research focuses on ensuring AI systems behave as intended.",
            token_count=12,
            metadata={},
        ),
        Chunk(
            chunk_id="c2",
            document_id="d1",
            source_url="https://example.com/1",
            title="AI Safety",
            section="Methods",
            chunk_index=1,
            text="Alignment techniques include RLHF and constitutional AI methods.",
            token_count=10,
            metadata={},
        ),
        Chunk(
            chunk_id="c3",
            document_id="d2",
            source_url="https://example.com/2",
            title="Machine Learning",
            section="Overview",
            chunk_index=0,
            text="Machine learning models learn patterns from data to make predictions.",
            token_count=11,
            metadata={},
        ),
    ]


@pytest.fixture
def sample_embeddings():
    np.random.seed(42)
    return np.random.randn(3, 384).astype(np.float32)


def test_vector_store_build(sample_chunks, sample_embeddings):
    store = VectorStore(dimension=384, store_path="/tmp/test_faiss")
    store.build_index(sample_embeddings, sample_chunks)
    assert store.index is not None
    assert store.index.ntotal == 3
    assert len(store.chunks) == 3


def test_vector_store_search(sample_chunks, sample_embeddings):
    store = VectorStore(dimension=384, store_path="/tmp/test_faiss")
    store.build_index(sample_embeddings, sample_chunks)

    query = np.random.randn(384).astype(np.float32)
    results = store.search(query, top_k=2)
    assert len(results) == 2
    assert "score" in results[0]
    assert "text" in results[0]


def test_vector_store_search_top_k(sample_chunks, sample_embeddings):
    store = VectorStore(dimension=384, store_path="/tmp/test_faiss")
    store.build_index(sample_embeddings, sample_chunks)

    query = np.random.randn(384).astype(np.float32)
    results = store.search(query, top_k=1)
    assert len(results) == 1


def test_vector_store_save_load(sample_chunks, sample_embeddings, tmp_path):
    store_path = str(tmp_path / "test_index")
    store = VectorStore(dimension=384, store_path=store_path)
    store.build_index(sample_embeddings, sample_chunks)
    store.save()

    loaded_store = VectorStore(dimension=384, store_path=store_path)
    loaded_store.load()
    assert loaded_store.index.ntotal == 3
    assert len(loaded_store.chunks) == 3


def test_vector_store_not_loaded():
    store = VectorStore(dimension=384, store_path="/tmp/nonexistent")
    assert not store.is_loaded

    query = np.random.randn(384).astype(np.float32)
    with pytest.raises(RuntimeError):
        store.search(query, top_k=5)
