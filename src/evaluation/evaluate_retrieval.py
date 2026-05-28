"""Retrieval evaluation metrics."""

import json
import logging
import re
from pathlib import Path

from src.config import DEFAULT_TOP_K, PROCESSED_DIR
from src.evaluation.qa_dataset import load_qa_dataset
from src.rag.pipeline import RAGPipeline

logger = logging.getLogger(__name__)


def _urls_match(retrieved_url: str, gold_url: str) -> bool:
    """Check if URLs match, allowing prefix and path variations."""
    if not retrieved_url or not gold_url:
        return False
    r = retrieved_url.rstrip("/")
    g = gold_url.rstrip("/")
    if r == g:
        return True
    if r.startswith(g) or g.startswith(r):
        return True
    return False


def _ngram_overlap(text_a: str, text_b: str, n: int = 4) -> float:
    """Compute overlap of character n-grams (proportion of a found in b)."""
    text_a = re.sub(r"\s+", " ", text_a.lower().strip())
    text_b = re.sub(r"\s+", " ", text_b.lower().strip())
    if len(text_a) < n or len(text_b) < n:
        return 0.0
    ngrams_a = {text_a[i : i + n] for i in range(len(text_a) - n + 1)}
    ngrams_b = {text_b[i : i + n] for i in range(len(text_b) - n + 1)}
    intersection = ngrams_a & ngrams_b
    return len(intersection) / len(ngrams_a) if ngrams_a else 0.0


def hit_rate_at_k(
    pipeline: RAGPipeline, dataset: list[dict], k: int = DEFAULT_TOP_K
) -> float:
    """Calculate hit rate@k - proportion of queries where gold source appears in top-k."""
    hits = 0
    for item in dataset:
        question = item["question"]
        gold_url = item.get("gold_source_url", "")
        gold_snippet = item.get("gold_context_snippet", "")

        results = pipeline.retrieve_only(question, top_k=k)

        hit = False
        for r in results:
            if _urls_match(r.get("source_url", ""), gold_url):
                hit = True
                break
            if gold_snippet and _ngram_overlap(gold_snippet, r.get("text", ""), n=5) > 0.3:
                hit = True
                break
        if hit:
            hits += 1

    return hits / len(dataset) if dataset else 0.0


def recall_at_k(
    pipeline: RAGPipeline, dataset: list[dict], k: int = DEFAULT_TOP_K
) -> float:
    """Calculate recall@k - whether key phrases from gold context appear in retrieved chunks."""
    recalls = 0
    for item in dataset:
        question = item["question"]
        gold_snippet = item.get("gold_context_snippet", "").lower()

        results = pipeline.retrieve_only(question, top_k=k)
        retrieved_text = " ".join(r.get("text", "").lower() for r in results)

        key_phrases = [p.strip() for p in gold_snippet.split(".") if len(p.strip()) > 10]
        if not key_phrases:
            key_phrases = [gold_snippet]

        matched = 0
        for phrase in key_phrases:
            overlap = _ngram_overlap(phrase, retrieved_text, n=4)
            if overlap > 0.3:
                matched += 1
                continue
            words = [w for w in phrase.split() if len(w) > 3]
            if words:
                word_matches = sum(1 for w in words if w in retrieved_text)
                if word_matches / len(words) > 0.5:
                    matched += 1

        if key_phrases and matched / len(key_phrases) > 0.3:
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
