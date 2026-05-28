"""QA dataset management for evaluation."""

import json
import logging
from pathlib import Path

from src.config import QA_DIR

logger = logging.getLogger(__name__)


def load_qa_dataset(path: Path | None = None) -> list[dict]:
    """Load the QA evaluation dataset."""
    if path is None:
        path = QA_DIR / "qa_dataset.json"

    if not path.exists():
        raise FileNotFoundError(f"QA dataset not found at {path}")

    with open(path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    logger.info(f"Loaded {len(dataset)} QA examples from {path}")
    return dataset


def save_qa_dataset(dataset: list[dict], path: Path | None = None) -> None:
    """Save the QA evaluation dataset."""
    if path is None:
        path = QA_DIR / "qa_dataset.json"

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved {len(dataset)} QA examples to {path}")
