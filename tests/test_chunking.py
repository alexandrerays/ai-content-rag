"""Tests for text chunking."""

import pytest

from src.ingestion.chunker import (
    Chunk,
    chunk_document,
    chunk_text,
    count_tokens,
    extract_sections,
    generate_chunk_id,
    generate_document_id,
)


def test_count_tokens():
    text = "Hello, this is a test sentence."
    tokens = count_tokens(text)
    assert tokens > 0
    assert isinstance(tokens, int)


def test_count_tokens_empty():
    assert count_tokens("") == 0


def test_generate_document_id():
    doc_id = generate_document_id("https://example.com/page")
    assert len(doc_id) == 12
    assert generate_document_id("https://example.com/page") == doc_id


def test_generate_chunk_id():
    chunk_id = generate_chunk_id("doc123", 0)
    assert len(chunk_id) == 16
    assert generate_chunk_id("doc123", 0) == chunk_id
    assert generate_chunk_id("doc123", 1) != chunk_id


def test_chunk_text_short():
    text = "This is a short text."
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=20)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_text_long():
    text = "Word " * 500
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=20)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) > 0


def test_chunk_text_overlap():
    text = "Sentence one. Sentence two. Sentence three. " * 50
    chunks = chunk_text(text, chunk_size=50, chunk_overlap=10)
    assert len(chunks) > 1


def test_extract_sections_with_headings():
    text = "# Introduction\nSome intro text.\n# Methods\nSome methods text."
    sections = extract_sections(text)
    assert len(sections) >= 2


def test_extract_sections_no_headings():
    text = "Just a paragraph of text without any headings or structure."
    sections = extract_sections(text)
    assert len(sections) == 1


def test_chunk_document():
    doc = {
        "source_url": "https://example.com/test",
        "title": "Test Document",
        "content_text": "This is a test document. " * 100,
        "author": "Test Author",
        "published_date": "2024-01-01",
        "ingestion_timestamp": "2024-01-01T00:00:00Z",
    }
    chunks = chunk_document(doc, chunk_size=50, chunk_overlap=10)
    assert len(chunks) > 0
    assert all(isinstance(c, Chunk) for c in chunks)

    first = chunks[0]
    assert first.source_url == "https://example.com/test"
    assert first.title == "Test Document"
    assert first.document_id == generate_document_id("https://example.com/test")
    assert first.chunk_index == 0
    assert first.token_count > 0


def test_chunk_document_preserves_metadata():
    doc = {
        "source_url": "https://example.com",
        "title": "Doc",
        "content_text": "Content " * 200,
        "author": "Author",
        "published_date": "2024-06-01",
        "ingestion_timestamp": "2024-06-01T12:00:00Z",
    }
    chunks = chunk_document(doc, chunk_size=50, chunk_overlap=10)
    for chunk in chunks:
        assert chunk.metadata["author"] == "Author"
        assert chunk.metadata["published_date"] == "2024-06-01"
