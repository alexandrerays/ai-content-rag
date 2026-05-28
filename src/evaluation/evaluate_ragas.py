"""RAGAS-style evaluation metrics."""

import json
import logging
import re
from pathlib import Path
from urllib.parse import urlparse

from src.config import PROCESSED_DIR
from src.evaluation.qa_dataset import load_qa_dataset
from src.indexing.embeddings import generate_embeddings
from src.rag.generator import generate_answer
from src.rag.pipeline import RAGPipeline

logger = logging.getLogger(__name__)


def _extract_ngrams(text: str, n: int) -> set[str]:
    """Extract character n-grams from text."""
    text = re.sub(r"\s+", " ", text.lower().strip())
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def _ngram_overlap(text_a: str, text_b: str, n: int = 4) -> float:
    """Compute Jaccard-like overlap of character n-grams."""
    ngrams_a = _extract_ngrams(text_a, n)
    ngrams_b = _extract_ngrams(text_b, n)
    if not ngrams_a or not ngrams_b:
        return 0.0
    intersection = ngrams_a & ngrams_b
    return len(intersection) / len(ngrams_a)


def _cosine_similarity(vec_a, vec_b) -> float:
    """Compute cosine similarity between two vectors."""
    import numpy as np

    dot = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def _urls_match(retrieved_url: str, gold_url: str) -> bool:
    """Check if URLs match, allowing prefix and path variations."""
    if retrieved_url == gold_url:
        return True
    r = retrieved_url.rstrip("/")
    g = gold_url.rstrip("/")
    if r == g:
        return True
    if r.startswith(g) or g.startswith(r):
        return True
    return False


def evaluate_faithfulness(answer: str, contexts: list[dict]) -> float:
    """Evaluate if the answer is grounded in the retrieved context using n-gram overlap."""
    context_text = " ".join(c.get("text", "") for c in contexts)
    answer_sentences = [s.strip() for s in answer.split(".") if len(s.strip()) > 10]

    if not answer_sentences:
        return 1.0

    grounded = 0
    for sentence in answer_sentences:
        overlap = _ngram_overlap(sentence, context_text, n=4)
        if overlap > 0.25:
            grounded += 1
            continue
        words = [w for w in sentence.lower().split() if len(w) > 3]
        if not words:
            grounded += 1
            continue
        context_lower = context_text.lower()
        matched = sum(1 for w in words if w in context_lower)
        if matched / len(words) > 0.4:
            grounded += 1

    return grounded / len(answer_sentences)


def evaluate_answer_relevance(answer: str, question: str) -> float:
    """Evaluate answer relevance using embedding similarity."""
    embeddings = generate_embeddings([question, answer])
    return max(0.0, _cosine_similarity(embeddings[0], embeddings[1]))


def evaluate_context_precision(contexts: list[dict], gold_url: str, gold_snippet: str = "") -> float:
    """Evaluate if relevant contexts are ranked higher using URL and content matching."""
    if not contexts:
        return 0.0

    relevant_positions = []
    for i, ctx in enumerate(contexts):
        if _urls_match(ctx.get("source_url", ""), gold_url):
            relevant_positions.append(i + 1)
            continue
        if gold_snippet:
            chunk_text = ctx.get("text", "")
            overlap = _ngram_overlap(gold_snippet, chunk_text, n=4)
            if overlap > 0.15:
                relevant_positions.append(i + 1)

    if not relevant_positions:
        return 0.0

    precision_sum = 0.0
    for i, pos in enumerate(relevant_positions, 1):
        precision_sum += i / pos

    return precision_sum / len(relevant_positions)


def evaluate_context_recall(
    contexts: list[dict], gold_snippet: str
) -> float:
    """Evaluate how much of the gold context is captured using n-gram matching."""
    retrieved_text = " ".join(c.get("text", "") for c in contexts)
    gold_sentences = [s.strip() for s in gold_snippet.split(".") if len(s.strip()) > 10]

    if not gold_sentences:
        return 1.0

    recalled = 0
    for sentence in gold_sentences:
        overlap = _ngram_overlap(sentence, retrieved_text, n=4)
        if overlap > 0.3:
            recalled += 1
            continue
        key_words = [w for w in sentence.lower().split() if len(w) > 3]
        if not key_words:
            recalled += 1
            continue
        retrieved_lower = retrieved_text.lower()
        matched = sum(1 for w in key_words if w in retrieved_lower)
        if matched / len(key_words) > 0.5:
            recalled += 1

    return recalled / len(gold_sentences)


def evaluate_ragas(
    pipeline: RAGPipeline | None = None,
    dataset: list[dict] | None = None,
) -> dict:
    """Run full RAGAS-style evaluation."""
    if pipeline is None:
        pipeline = RAGPipeline()
    if dataset is None:
        dataset = load_qa_dataset()

    metrics = {
        "faithfulness": [],
        "answer_relevance": [],
        "context_precision": [],
        "context_recall": [],
    }
    details = []

    for item in dataset:
        question = item["question"]
        gold_url = item.get("gold_source_url", "")
        gold_snippet = item.get("gold_context_snippet", "")

        response = pipeline.ask(question)
        contexts = response.retrieved_contexts

        faith = evaluate_faithfulness(response.answer, contexts)
        relevance = evaluate_answer_relevance(response.answer, question)
        precision = evaluate_context_precision(contexts, gold_url, gold_snippet)
        recall = evaluate_context_recall(contexts, gold_snippet)

        metrics["faithfulness"].append(faith)
        metrics["answer_relevance"].append(relevance)
        metrics["context_precision"].append(precision)
        metrics["context_recall"].append(recall)

        details.append({
            "question": question,
            "answer": response.answer,
            "faithfulness": faith,
            "answer_relevance": relevance,
            "context_precision": precision,
            "context_recall": recall,
        })

    averages = {k: sum(v) / len(v) if v else 0.0 for k, v in metrics.items()}
    return {"averages": averages, "details": details}


def run_and_save(output_path: Path | None = None) -> dict:
    """Run RAGAS evaluation and save results."""
    if output_path is None:
        output_path = PROCESSED_DIR / "evaluation_results.json"

    results = evaluate_ragas()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"RAGAS evaluation results saved to {output_path}")
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_and_save()
