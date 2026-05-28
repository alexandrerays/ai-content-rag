"""Compare base LLM vs RAG-enhanced answers."""

import json
import logging
from pathlib import Path

from src.config import PROCESSED_DIR
from src.evaluation.qa_dataset import load_qa_dataset
from src.rag.generator import generate_answer
from src.rag.pipeline import RAGPipeline

logger = logging.getLogger(__name__)


def compute_answer_similarity(answer: str, expected: str) -> float:
    """Compute simple word overlap similarity between answer and expected."""
    answer_words = set(answer.lower().split())
    expected_words = set(expected.lower().split())
    key_expected = {w for w in expected_words if len(w) > 3}

    if not key_expected:
        return 1.0

    overlap = answer_words & key_expected
    return len(overlap) / len(key_expected)


def compare_base_vs_rag(
    pipeline: RAGPipeline | None = None,
    dataset: list[dict] | None = None,
) -> dict:
    """Generate answers from both base LLM and RAG, then compare."""
    if pipeline is None:
        pipeline = RAGPipeline()
    if dataset is None:
        dataset = load_qa_dataset()

    comparisons = []
    base_scores = []
    rag_scores = []

    for item in dataset:
        question = item["question"]
        expected = item["expected_answer"]

        logger.info(f"Evaluating: {question[:50]}...")

        base_answer = pipeline.ask_base_llm(question)
        rag_response = pipeline.ask(question)

        base_score = compute_answer_similarity(base_answer, expected)
        rag_score = compute_answer_similarity(rag_response.answer, expected)

        base_scores.append(base_score)
        rag_scores.append(rag_score)

        comparisons.append({
            "question": question,
            "expected_answer": expected,
            "base_answer": base_answer,
            "rag_answer": rag_response.answer,
            "base_score": base_score,
            "rag_score": rag_score,
            "rag_better": rag_score > base_score,
            "citations": [c["source_url"] for c in rag_response.citations],
        })

    avg_base = sum(base_scores) / len(base_scores) if base_scores else 0
    avg_rag = sum(rag_scores) / len(rag_scores) if rag_scores else 0
    rag_wins = sum(1 for c in comparisons if c["rag_better"])

    summary = {
        "total_questions": len(comparisons),
        "avg_base_score": avg_base,
        "avg_rag_score": avg_rag,
        "rag_improvement": avg_rag - avg_base,
        "rag_wins": rag_wins,
        "base_wins": len(comparisons) - rag_wins,
        "rag_win_rate": rag_wins / len(comparisons) if comparisons else 0,
    }

    return {"summary": summary, "comparisons": comparisons}


def generate_report(results: dict) -> str:
    """Generate a markdown comparison report."""
    summary = results["summary"]
    comparisons = results["comparisons"]

    report = "# Base LLM vs RAG Comparison Report\n\n"
    report += "## Summary\n\n"
    report += f"| Metric | Value |\n|--------|-------|\n"
    report += f"| Total Questions | {summary['total_questions']} |\n"
    report += f"| Avg Base LLM Score | {summary['avg_base_score']:.3f} |\n"
    report += f"| Avg RAG Score | {summary['avg_rag_score']:.3f} |\n"
    report += f"| RAG Improvement | {summary['rag_improvement']:+.3f} |\n"
    report += f"| RAG Wins | {summary['rag_wins']}/{summary['total_questions']} ({summary['rag_win_rate']:.0%}) |\n\n"

    report += "## Detailed Comparisons\n\n"
    for i, comp in enumerate(comparisons, 1):
        winner = "RAG" if comp["rag_better"] else "Base LLM"
        report += f"### Question {i}\n\n"
        report += f"**Q:** {comp['question']}\n\n"
        report += f"**Expected:** {comp['expected_answer'][:200]}...\n\n"
        report += f"**Base LLM** (score: {comp['base_score']:.3f}):\n> {comp['base_answer'][:200]}...\n\n"
        report += f"**RAG** (score: {comp['rag_score']:.3f}):\n> {comp['rag_answer'][:200]}...\n\n"
        report += f"**Winner:** {winner}\n\n---\n\n"

    return report


def run_and_save(output_dir: Path | None = None) -> dict:
    """Run comparison and save results."""
    if output_dir is None:
        output_dir = PROCESSED_DIR

    results = compare_base_vs_rag()
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "base_vs_rag_comparison.json", "w") as f:
        json.dump(results, f, indent=2)

    report = generate_report(results)
    with open(output_dir / "base_vs_rag_comparison.md", "w") as f:
        f.write(report)

    logger.info(f"Comparison results saved to {output_dir}")
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_and_save()
