"""Evaluation using the RAGAS framework.

Uses RAGAS metrics (faithfulness, answer_relevancy, context_precision,
context_recall) evaluated by an LLM judge (OpenAI gpt-3.5-turbo by default).

Run as: python -m src.evaluation.evaluate_ragas
"""

import asyncio
import json
import logging
import math
import sys
from pathlib import Path

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from ragas.run_config import RunConfig

from src.config import PROCESSED_DIR
from src.evaluation.qa_dataset import load_qa_dataset
from src.rag.pipeline import RAGPipeline

logger = logging.getLogger(__name__)

# --- Python 3.13+ compatibility patches for RAGAS ---
# RAGAS 0.1.x uses asyncio patterns incompatible with Python 3.13+:
# 1. Metric.ascore uses asyncio.wait_for which requires being inside a task
# 2. Executor.results calls asyncio.as_completed without a running loop

import ragas.metrics.base as _ragas_base  # noqa: E402
import ragas.executor as _ragas_executor  # noqa: E402


async def _patched_ascore(self, row, callbacks=None, timeout=None):
    return await self._ascore(row, callbacks)


_ragas_base.Metric.ascore = _patched_ascore

_original_results = _ragas_executor.Executor.results


def _patched_results(self):
    """Patch Executor.results to run within asyncio.run() for Python 3.13+ compat."""
    from tqdm import tqdm
    from ragas.executor import as_completed

    run_config = self.run_config or RunConfig()

    async def _aresults():
        futures_as_they_finish = as_completed(
            coros=[afunc(*args, **kwargs) for afunc, args, kwargs, _ in self.jobs],
            max_workers=run_config.max_workers,
        )
        results = []
        for future in tqdm(
            futures_as_they_finish,
            desc=self.desc,
            total=len(self.jobs),
            leave=self.keep_progress_bar,
        ):
            r = await future
            results.append(r)
        return results

    results = asyncio.run(_aresults())
    sorted_results = sorted(results, key=lambda x: x[0])
    return [r[1] for r in sorted_results]


_ragas_executor.Executor.results = _patched_results
# --- End patches ---


def _build_ragas_dataset(
    pipeline: RAGPipeline, qa_dataset: list[dict]
) -> Dataset:
    """Run RAG pipeline on QA dataset and format for RAGAS evaluation."""
    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for i, item in enumerate(qa_dataset):
        question = item["question"]
        expected_answer = item.get("expected_answer", "")

        logger.info(f"Processing question {i + 1}/{len(qa_dataset)}: {question[:50]}...")
        response = pipeline.ask(question)

        questions.append(question)
        answers.append(response.answer)
        contexts.append([c.get("text", "") for c in response.retrieved_contexts])
        ground_truths.append(expected_answer)

    return Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })


def _safe_float(val) -> float:
    """Convert to float, handling NaN."""
    try:
        f = float(val)
        return 0.0 if math.isnan(f) else f
    except (TypeError, ValueError):
        return 0.0


def evaluate_ragas(
    pipeline: RAGPipeline | None = None,
    dataset: list[dict] | None = None,
) -> dict:
    """Run RAGAS evaluation on the RAG pipeline.

    RAGAS uses LLM-as-judge (OpenAI by default) to evaluate:
    - faithfulness: Is the answer grounded in the retrieved context?
    - answer_relevancy: Is the answer relevant to the question?
    - context_precision: Are relevant documents ranked higher?
    - context_recall: Does the context cover the ground truth?
    """
    if pipeline is None:
        pipeline = RAGPipeline()
    if dataset is None:
        dataset = load_qa_dataset()

    logger.info("Running RAG pipeline on QA dataset...")
    ragas_dataset = _build_ragas_dataset(pipeline, dataset)

    logger.info("Running RAGAS evaluation (LLM-judged metrics)...")
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    run_config = RunConfig(timeout=600, max_retries=10, max_workers=4)

    result = evaluate(
        dataset=ragas_dataset,
        metrics=metrics,
        run_config=run_config,
        raise_exceptions=False,
    )

    scores = {}
    for k, v in result.items():
        if isinstance(v, (int, float)):
            scores[k] = _safe_float(v)

    details = []
    result_df = result.to_pandas()
    for _, row in result_df.iterrows():
        details.append({
            "question": row.get("question", ""),
            "answer": row.get("answer", ""),
            "faithfulness": _safe_float(row.get("faithfulness")),
            "answer_relevancy": _safe_float(row.get("answer_relevancy")),
            "context_precision": _safe_float(row.get("context_precision")),
            "context_recall": _safe_float(row.get("context_recall")),
        })

    return {"averages": scores, "details": details}


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
    results = run_and_save()
    print(json.dumps(results["averages"], indent=2))
