"""Load raw documents from disk."""

import json
import logging
from pathlib import Path

from src.config import RAW_DIR

logger = logging.getLogger(__name__)


def load_raw_documents(input_dir: Path = RAW_DIR) -> list[dict]:
    """Load all raw documents from the data directory."""
    documents = []
    json_files = sorted(input_dir.glob("doc_*.json"))

    if not json_files:
        logger.warning(f"No documents found in {input_dir}")
        return documents

    for filepath in json_files:
        with open(filepath, "r", encoding="utf-8") as f:
            doc = json.load(f)
            doc["file_path"] = str(filepath)
            documents.append(doc)

    logger.info(f"Loaded {len(documents)} documents from {input_dir}")
    return documents
