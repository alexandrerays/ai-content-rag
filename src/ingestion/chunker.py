"""Document chunking with metadata preservation."""

import hashlib
import logging
import re
from dataclasses import dataclass, field

import tiktoken

from src.config import CHUNK_OVERLAP, CHUNK_SIZE

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    source_url: str
    title: str
    section: str
    chunk_index: int
    text: str
    token_count: int
    metadata: dict = field(default_factory=dict)


def count_tokens(text: str, model: str = "cl100k_base") -> int:
    """Count tokens using tiktoken."""
    enc = tiktoken.get_encoding(model)
    return len(enc.encode(text))


def generate_chunk_id(document_id: str, chunk_index: int) -> str:
    """Generate a unique chunk ID."""
    content = f"{document_id}_{chunk_index}"
    return hashlib.md5(content.encode()).hexdigest()[:16]


def generate_document_id(source_url: str) -> str:
    """Generate a document ID from source URL."""
    return hashlib.md5(source_url.encode()).hexdigest()[:12]


def extract_sections(text: str) -> list[tuple[str, str]]:
    """Split text into sections based on headings and structural cues."""
    lines = text.split("\n")
    sections = []
    current_section = ""
    current_text = []

    for line in lines:
        is_heading = False
        if line.startswith("#"):
            is_heading = True
        elif re.match(r"^[A-Z][A-Za-z0-9 :,\-]{5,80}$", line.strip()):
            stripped = line.strip()
            if not stripped.endswith(".") and not stripped.endswith(","):
                is_heading = True

        if is_heading:
            if current_text:
                sections.append((current_section, "\n".join(current_text)))
            current_section = line.lstrip("#").strip()
            current_text = []
        else:
            current_text.append(line)

    if current_text:
        sections.append((current_section, "\n".join(current_text)))

    return sections if sections else [("", text)]


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences for better chunk boundaries."""
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split text into overlapping chunks with sentence-aware boundaries."""
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)

    if len(tokens) <= chunk_size:
        return [text]

    sentences = split_into_sentences(text)
    if len(sentences) <= 1:
        return _chunk_by_tokens(text, chunk_size, chunk_overlap, enc)

    chunks = []
    current_chunk_sentences = []
    current_token_count = 0

    for sentence in sentences:
        sentence_tokens = len(enc.encode(sentence))

        if sentence_tokens > chunk_size:
            if current_chunk_sentences:
                chunks.append(" ".join(current_chunk_sentences))
                current_chunk_sentences = []
                current_token_count = 0
            sub_chunks = _chunk_by_tokens(sentence, chunk_size, chunk_overlap, enc)
            chunks.extend(sub_chunks)
            continue

        if current_token_count + sentence_tokens > chunk_size and current_chunk_sentences:
            chunks.append(" ".join(current_chunk_sentences))

            overlap_sentences = []
            overlap_tokens = 0
            for s in reversed(current_chunk_sentences):
                s_tokens = len(enc.encode(s))
                if overlap_tokens + s_tokens > chunk_overlap:
                    break
                overlap_sentences.insert(0, s)
                overlap_tokens += s_tokens

            current_chunk_sentences = overlap_sentences
            current_token_count = overlap_tokens

        current_chunk_sentences.append(sentence)
        current_token_count += sentence_tokens

    if current_chunk_sentences:
        chunks.append(" ".join(current_chunk_sentences))

    return [c.strip() for c in chunks if c.strip()]


def _chunk_by_tokens(
    text: str, chunk_size: int, chunk_overlap: int, enc: tiktoken.Encoding
) -> list[str]:
    """Fallback: split by raw token count."""
    tokens = enc.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_str = enc.decode(tokens[start:end])
        chunks.append(chunk_str.strip())
        start = end - chunk_overlap
        if start >= len(tokens):
            break
    return chunks


def chunk_document(
    doc: dict,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Chunk]:
    """Chunk a single document into smaller pieces."""
    source_url = doc.get("source_url", "")
    document_id = generate_document_id(source_url)
    title = doc.get("title", "")
    text = doc.get("content_text", "")

    sections = extract_sections(text)
    chunks = []
    chunk_index = 0

    for section_title, section_text in sections:
        if not section_text.strip():
            continue

        text_chunks = chunk_text(section_text, chunk_size, chunk_overlap)

        for chunk_str in text_chunks:
            if not chunk_str.strip():
                continue

            chunk_id = generate_chunk_id(document_id, chunk_index)
            token_count = count_tokens(chunk_str)

            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    source_url=source_url,
                    title=title,
                    section=section_title,
                    chunk_index=chunk_index,
                    text=chunk_str,
                    token_count=token_count,
                    metadata={
                        "author": doc.get("author"),
                        "published_date": doc.get("published_date"),
                        "ingestion_timestamp": doc.get("ingestion_timestamp"),
                    },
                )
            )
            chunk_index += 1

    logger.info(f"Document '{title}' chunked into {len(chunks)} chunks")
    return chunks


def chunk_documents(
    documents: list[dict],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Chunk]:
    """Chunk all documents."""
    all_chunks = []
    for doc in documents:
        chunks = chunk_document(doc, chunk_size, chunk_overlap)
        all_chunks.extend(chunks)

    logger.info(f"Total chunks created: {len(all_chunks)}")
    return all_chunks
