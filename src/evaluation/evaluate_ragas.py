"""RAGAS-style evaluation metrics."""

import json
import logging
from pathlib import Path

from src.config import PROCESSED_DIR
from src.evaluation.qa_dataset import load_qa_dataset
from src.rag.generator import generate_answer
from src.rag.pipeline import RAGPipeline

logger = logging.getLogger(__name__)


def evaluate_faithfulness(answer: str, contexts: list[dict]) -> float:
    """Evaluate if the answer is faithful to the retrieved context."""
    context_text = " ".join(c.get("text", "") for c in contexts).lower()
    answer_sentences = [s.strip() for s in answer.split(".") if len(s.strip()) > 10]

    if not answer_sentences:
        return 1.0

    grounded = 0
    for sentence in answer_sentences:
        words = sentence.lower().split()
        key_words = [w for w in words if len(w) > 4]
        if not key_words:
            grounded += 1
            continue
        matched = sum(1 for w in key_words if w in context_text)
        if matched / len(key_words) > 0.3:
            grounded += 1

    return grounded / len(answer_sentences)


def evaluate_answer_relevance(answer: str, question: str) -> float:
    """Evaluate if the answer is relevant to the question."""
    question_words = set(question.lower().split())
    answer_words = set(answer.lower().split())

    question_keywords = {w for w in question_words if len(w) > 3}
    if not question_keywords:
        return 1.0

    overlap = question_keywords & answer_words
    return len(overlap) / len(question_keywords)


def evaluate_context_precision(contexts: list[dict], gold_url: str) -> float:
    """Evaluate if relevant contexts are ranked higher."""
    if not contexts:
        return 0.0

    relevant_positions = []
    for i, ctx in enumerate(contexts):
        if ctx.get("source_url", "") == gold_url:
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
    """Evaluate how much of the gold context is captured."""
    retrieved_text = " ".join(c.get("text", "").lower() for c in contexts)
    gold_sentences = [s.strip() for s in gold_snippet.split(".") if len(s.strip()) > 10]

    if not gold_sentences:
        return 1.0

    recalled = 0
    for sentence in gold_sentences:
        key_words = [w for w in sentence.lower().split() if len(w) > 4]
        if not key_words:
            recalled += 1
            continue
        matched = sum(1 for w in key_words if w in retrieved_text)
        if matched / len(key_words) > 0.4:
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
        precision = evaluate_context_precision(contexts, gold_url)
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
