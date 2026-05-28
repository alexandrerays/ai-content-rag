"""Text cleaning and preprocessing."""

import logging
import re

logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """Clean and normalize document text."""
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"&[a-zA-Z]+;", " ", text)
    text = re.sub(r"&#\d+;", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if line:
            cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def remove_boilerplate(text: str) -> str:
    """Remove common boilerplate patterns."""
    boilerplate_patterns = [
        r"(?i)cookie\s*(policy|consent|notice).*?\n",
        r"(?i)subscribe\s*to\s*our\s*newsletter.*?\n",
        r"(?i)follow\s*us\s*on.*?\n",
        r"(?i)share\s*this\s*(article|post|page).*?\n",
        r"(?i)all\s*rights\s*reserved.*?\n",
        r"(?i)terms\s*(of\s*service|and\s*conditions).*?\n",
        r"(?i)privacy\s*policy.*?\n",
    ]
    for pattern in boilerplate_patterns:
        text = re.sub(pattern, "", text)
    return text


def clean_document(doc: dict) -> dict:
    """Clean a single document."""
    cleaned_text = doc.get("content_text", "")
    cleaned_text = clean_text(cleaned_text)
    cleaned_text = remove_boilerplate(cleaned_text)

    return {
        **doc,
        "content_text": cleaned_text,
        "cleaned": True,
    }


def clean_documents(documents: list[dict]) -> list[dict]:
    """Clean all documents."""
    cleaned = []
    for doc in documents:
        cleaned_doc = clean_document(doc)
        if len(cleaned_doc["content_text"]) > 50:
            cleaned.append(cleaned_doc)
        else:
            logger.debug(f"Skipping short document: {doc.get('title', 'unknown')}")

    logger.info(f"Cleaned {len(cleaned)}/{len(documents)} documents")
    return cleaned
