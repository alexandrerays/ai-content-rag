"""Retrieval evaluation metrics."""

import json
import logging
from pathlib import Path

from src.config import DEFAULT_TOP_K, PROCESSED_DIR
from src.evaluation.qa_dataset import load_qa_dataset
from src.rag.pipeline import RAGPipeline

logger = logging.getLogger(__name__)


def hit_rate_at_k(
    pipeline: RAGPipeline, dataset: list[dict], k: int = DEFAULT_TOP_K
) -> float:
    """Calculate hit rate@k - proportion of queries where gold source appears in top-k."""
    hits = 0
    for item in dataset:
        question = item["question"]
        gold_url = item.get("gold_source_url", "")

        results = pipeline.retrieve_only(question, top_k=k)
        retrieved_urls = [r.get("source_url", "") for r in results]

        if gold_url in retrieved_urls:
            hits += 1

    return hits / len(dataset) if dataset else 0.0


def recall_at_k(
    pipeline: RAGPipeline, dataset: list[dict], k: int = DEFAULT_TOP_K
) -> float:
    """Calculate recall@k - whether gold context snippet appears in retrieved chunks."""
    recalls = 0
    for item in dataset:
        question = item["question"]
        gold_snippet = item.get("gold_context_snippet", "").lower()

        results = pipeline.retrieve_only(question, top_k=k)
        retrieved_text = " ".join(r.get("text", "").lower() for r in results)

        key_phrases = [p.strip() for p in gold_snippet.split(".") if len(p.strip()) > 10]
        if not key_phrases:
            key_phrases = [gold_snippet]

        matched = sum(1 for phrase in key_phrases if phrase in retrieved_text)
        if key_phrases and matched / len(key_phrases) > 0.5:
            recalls += 1

    return recalls / len(dataset) if dataset else 0.0


def evaluate_retrieval(
    pipeline: RAGPipeline | None = None,
    dataset: list[dict] | None = None,
    k_values: list[int] | None = None,
) -> dict:
    """Run full retrieval evaluation."""
    if pipeline is None:
        pipeline = RAGPipeline()
    if dataset is None:
        dataset = load_qa_dataset()
    if k_values is None:
        k_values = [3, 5, 10]

    results = {}
    for k in k_values:
        hr = hit_rate_at_k(pipeline, dataset, k)
        rc = recall_at_k(pipeline, dataset, k)
        results[f"hit_rate@{k}"] = hr
        results[f"recall@{k}"] = rc
        logger.info(f"k={k}: hit_rate={hr:.3f}, recall={rc:.3f}")

    return results


def run_and_save(output_path: Path | None = None) -> dict:
    """Run retrieval evaluation and save results."""
    if output_path is None:
        output_path = PROCESSED_DIR / "retrieval_metrics.json"

    results = evaluate_retrieval()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Retrieval metrics saved to {output_path}")
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_and_save()
