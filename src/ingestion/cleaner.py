"""Text cleaning and preprocessing."""

import logging
import re

logger = logging.getLogger(__name__)

BOILERPLATE_PATTERNS = [
    r"(?i)cookie\s*(policy|consent|notice).*?\n",
    r"(?i)subscribe\s*to\s*our\s*newsletter.*?\n",
    r"(?i)follow\s*us\s*on.*?\n",
    r"(?i)share\s*this\s*(article|post|page).*?\n",
    r"(?i)all\s*rights\s*reserved.*?\n",
    r"(?i)terms\s*(of\s*service|and\s*conditions).*?\n",
    r"(?i)privacy\s*policy.*?\n",
    r"(?i)sign\s*(up|in)\s*(for|to).*?\n",
    r"(?i)log\s*in\s*to.*?\n",
    r"(?i)click\s*here\s*to.*?\n",
    r"(?i)read\s*more\s*›.*?\n",
    r"(?i)skip\s*to\s*(main\s*)?content.*?\n",
    r"(?i)table\s*of\s*contents.*?\n",
    r"(?i)back\s*to\s*top.*?\n",
    r"(?i)©\s*\d{4}.*?\n",
    r"(?i)powered\s*by.*?\n",
    r"(?i)contact\s*us.*?\n",
    r"(?i)loading\.{0,3}\s*$",
]

NAV_LINE_PATTERNS = [
    r"^(Home|About|Contact|Blog|FAQ|Menu|Search|Login|Sign Up)\s*$",
    r"^(Previous|Next|Back|Forward)\s*(page|article|post)?\s*$",
    r"^\s*[|•·›»←→]\s*$",
    r"^(Share|Tweet|Pin|Email)\s*$",
]


def clean_text(text: str) -> str:
    """Clean and normalize document text."""
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"&[a-zA-Z]+;", " ", text)
    text = re.sub(r"&#\d+;", " ", text)
    text = re.sub(r"\u00a0", " ", text)
    text = re.sub(r"[ \t]+", " ", text)

    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if len(line) < 3:
            continue
        is_nav = False
        for pattern in NAV_LINE_PATTERNS:
            if re.match(pattern, line, re.IGNORECASE):
                is_nav = True
                break
        if not is_nav:
            cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def remove_boilerplate(text: str) -> str:
    """Remove common boilerplate patterns."""
    for pattern in BOILERPLATE_PATTERNS:
        text = re.sub(pattern, "", text)
    return text


def remove_repeated_headers(text: str) -> str:
    """Remove lines that appear multiple times (likely repeated headers/footers)."""
    lines = text.split("\n")
    line_counts: dict[str, int] = {}
    for line in lines:
        stripped = line.strip().lower()
        if len(stripped) > 5:
            line_counts[stripped] = line_counts.get(stripped, 0) + 1

    repeated = {line for line, count in line_counts.items() if count > 2}

    filtered = []
    for line in lines:
        if line.strip().lower() not in repeated:
            filtered.append(line)

    return "\n".join(filtered)


def clean_document(doc: dict) -> dict:
    """Clean a single document."""
    cleaned_text = doc.get("content_text", "")
    cleaned_text = clean_text(cleaned_text)
    cleaned_text = remove_boilerplate(cleaned_text)
    cleaned_text = remove_repeated_headers(cleaned_text)

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
        if len(cleaned_doc["content_text"]) > 100:
            cleaned.append(cleaned_doc)
        else:
            logger.debug(f"Skipping short document: {doc.get('title', 'unknown')}")

    logger.info(f"Cleaned {len(cleaned)}/{len(documents)} documents")
    return cleaned
