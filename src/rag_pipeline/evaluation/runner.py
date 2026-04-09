"""Evaluation runner that executes the pipeline against the eval dataset."""

from __future__ import annotations

import json
from typing import Any

import structlog

from rag_pipeline.config import Settings
from rag_pipeline.evaluation.dataset import EVAL_DATASET
from rag_pipeline.evaluation.metrics import (
    compute_answer_relevancy,
    compute_context_precision,
    compute_faithfulness,
)
from rag_pipeline.pipeline import RAGPipeline

logger = structlog.get_logger()


def run_evaluation(
    settings: Settings,
    output_base: str = "evaluation_report",
) -> dict[str, Any]:
    """Run the full evaluation suite against the eval dataset.

    Args:
        settings: Pipeline settings.
        output_base: Base filename for output reports (without extension).

    Returns:
        Evaluation report as a dictionary.
    """
    pipeline = RAGPipeline(settings=settings)
    results: list[dict[str, Any]] = []

    for i, sample in enumerate(EVAL_DATASET):
        logger.info("evaluating", sample=i + 1, total=len(EVAL_DATASET))

        try:
            query_result = pipeline.query(sample.question)
            context = query_result.context
            answer = query_result.answer
        except Exception:
            logger.exception("query_failed_during_eval", question=sample.question)
            results.append({
                "question": sample.question,
                "ground_truth": sample.ground_truth,
                "answer": "ERROR: Query failed",
                "context": "",
                "faithfulness": 0.0,
                "answer_relevancy": 0.0,
                "context_precision": 0.0,
            })
            continue

        faithfulness = compute_faithfulness(
            sample.question, answer, context, settings.generation
        )
        answer_relevancy = compute_answer_relevancy(
            sample.question, answer, settings.generation
        )
        context_precision = compute_context_precision(
            sample.question, context, sample.ground_truth, settings.generation
        )

        results.append({
            "question": sample.question,
            "ground_truth": sample.ground_truth,
            "answer": answer,
            "context": context[:500],
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_precision": context_precision,
        })

        logger.info(
            "sample_evaluated",
            sample=i + 1,
            faithfulness=faithfulness,
            relevancy=answer_relevancy,
            precision=context_precision,
        )

    report = _build_report(results)

    # Write JSON report
    json_path = f"{output_base}.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("json_report_written", path=json_path)

    # Write Markdown summary
    md_path = f"{output_base}.md"
    with open(md_path, "w") as f:
        f.write(_format_markdown(report))
    logger.info("markdown_report_written", path=md_path)

    return report


def _build_report(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the evaluation report from individual results."""
    n = len(results)
    if n == 0:
        return {"summary": {}, "results": []}

    avg_faith = sum(r["faithfulness"] for r in results) / n
    avg_relevancy = sum(r["answer_relevancy"] for r in results) / n
    avg_precision = sum(r["context_precision"] for r in results) / n

    return {
        "summary": {
            "num_samples": n,
            "avg_faithfulness": round(avg_faith, 4),
            "avg_answer_relevancy": round(avg_relevancy, 4),
            "avg_context_precision": round(avg_precision, 4),
        },
        "results": results,
    }


def _format_markdown(report: dict[str, Any]) -> str:
    """Format the evaluation report as a Markdown table."""
    summary = report["summary"]
    lines = [
        "# RAG Pipeline Evaluation Report\n",
        "## Summary\n",
        "| Metric | Score |",
        "|--------|-------|",
        f"| Samples | {summary['num_samples']} |",
        f"| Avg Faithfulness | {summary['avg_faithfulness']:.4f} |",
        f"| Avg Answer Relevancy | {summary['avg_answer_relevancy']:.4f} |",
        f"| Avg Context Precision | {summary['avg_context_precision']:.4f} |",
        "",
        "## Detailed Results\n",
        "| # | Question | Faithfulness | Relevancy | Precision |",
        "|---|----------|-------------|-----------|-----------|",
    ]

    for i, r in enumerate(report["results"], 1):
        q = r["question"][:50] + "..." if len(r["question"]) > 50 else r["question"]
        lines.append(
            f"| {i} | {q} | {r['faithfulness']:.2f} | "
            f"{r['answer_relevancy']:.2f} | {r['context_precision']:.2f} |"
        )

    lines.append("")
    return "\n".join(lines)
