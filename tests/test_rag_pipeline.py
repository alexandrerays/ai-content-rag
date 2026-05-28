"""Tests for RAG pipeline."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.ingestion.chunker import Chunk
from src.ingestion.cleaner import clean_text, remove_boilerplate
from src.rag.prompts import format_context


def test_clean_text_removes_html():
    text = "<p>Hello <b>world</b></p>"
    cleaned = clean_text(text)
    assert "<" not in cleaned
    assert "Hello" in cleaned
    assert "world" in cleaned


def test_clean_text_normalizes_whitespace():
    text = "Hello    world\n\n\n\nmultiple   spaces"
    cleaned = clean_text(text)
    assert "    " not in cleaned
    assert "\n\n\n" not in cleaned


def test_clean_text_removes_html_entities():
    text = "Hello&nbsp;world&amp;test"
    cleaned = clean_text(text)
    assert "&nbsp;" not in cleaned
    assert "&amp;" not in cleaned


def test_remove_boilerplate():
    text = "Main content here.\nCookie policy: we use cookies.\nMore content."
    cleaned = remove_boilerplate(text)
    assert "Main content" in cleaned


def test_format_context():
    chunks = [
        {
            "source_url": "https://example.com/1",
            "title": "Test Doc",
            "section": "Intro",
            "text": "This is the content.",
            "score": 0.95,
        }
    ]
    context = format_context(chunks)
    assert "Test Doc" in context
    assert "https://example.com/1" in context
    assert "This is the content." in context
    assert "0.950" in context


def test_format_context_multiple_chunks():
    chunks = [
        {"source_url": "https://a.com", "title": "A", "section": "", "text": "AAA", "score": 0.9},
        {"source_url": "https://b.com", "title": "B", "section": "S", "text": "BBB", "score": 0.8},
    ]
    context = format_context(chunks)
    assert "[Source 1]" in context
    assert "[Source 2]" in context
    assert "---" in context


@patch("src.rag.pipeline.RAGPipeline.__init__", return_value=None)
def test_pipeline_ask_returns_response(mock_init):
    from src.rag.pipeline import RAGPipeline, RAGResponse

    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline.vector_store = MagicMock()
    pipeline.retriever = MagicMock()

    mock_chunks = [
        {
            "text": "AI safety is important.",
            "source_url": "https://example.com",
            "title": "AI Doc",
            "section": "Safety",
            "score": 0.9,
            "chunk_id": "c1",
            "document_id": "d1",
            "chunk_index": 0,
            "token_count": 5,
            "metadata": {},
        }
    ]
    pipeline.retriever.retrieve.return_value = mock_chunks

    with patch("src.rag.pipeline.generate_answer", return_value="AI safety is crucial."):
        response = pipeline.ask("What is AI safety?")

    assert isinstance(response, RAGResponse)
    assert response.answer == "AI safety is crucial."
    assert len(response.citations) == 1
    assert response.citations[0]["source_url"] == "https://example.com"
