"""RAGAS-style evaluation metrics: faithfulness, answer relevancy, context precision."""

from __future__ import annotations

import re

import structlog
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from rag_pipeline.config import GenerationConfig
from rag_pipeline.generation import get_llm

logger = structlog.get_logger()


def _get_eval_llm(config: GenerationConfig):
    """Create an LLM instance for evaluation (uses same backend as generation)."""
    eval_config = config.model_copy()
    eval_config.temperature = 0.01  # near-deterministic for eval
    eval_config.max_tokens = 256
    return get_llm(eval_config)


def compute_faithfulness(
    question: str,
    answer: str,
    context: str,
    config: GenerationConfig,
) -> float:
    """Measure how faithful the answer is to the provided context.

    Scores how well the answer is supported by the retrieved context,
    penalizing claims not found in the context.

    Args:
        question: The original question.
        answer: The generated answer.
        context: The retrieved context used for generation.
        config: Generation config for the evaluator LLM.

    Returns:
        Score between 0.0 and 1.0.
    """
    prompt = ChatPromptTemplate.from_template(
        "Rate the faithfulness of the answer to the context. "
        "1.0 means fully supported, 0.0 means unsupported. "
        "Respond with ONLY a number.\n\n"
        "Context: {context}\n\nQuestion: {question}\n\nAnswer: {answer}\n\nScore:"
    )
    chain = prompt | _get_eval_llm(config) | StrOutputParser()

    try:
        result = chain.invoke({"context": context, "question": question, "answer": answer})
        return _parse_score(result)
    except Exception:
        logger.exception("faithfulness_eval_failed")
        return 0.0


def compute_answer_relevancy(
    question: str,
    answer: str,
    config: GenerationConfig,
) -> float:
    """Measure how relevant the answer is to the question.

    Evaluates whether the answer addresses the question directly
    and provides useful information.

    Args:
        question: The original question.
        answer: The generated answer.
        config: Generation config for the evaluator LLM.

    Returns:
        Score between 0.0 and 1.0.
    """
    prompt = ChatPromptTemplate.from_template(
        "Rate how relevant the answer is to the question. "
        "1.0 means perfectly relevant, 0.0 means irrelevant. "
        "Respond with ONLY a number.\n\n"
        "Question: {question}\n\nAnswer: {answer}\n\nScore:"
    )
    chain = prompt | _get_eval_llm(config) | StrOutputParser()

    try:
        result = chain.invoke({"question": question, "answer": answer})
        return _parse_score(result)
    except Exception:
        logger.exception("answer_relevancy_eval_failed")
        return 0.0


def compute_context_precision(
    question: str,
    context: str,
    ground_truth: str,
    config: GenerationConfig,
) -> float:
    """Measure how precise the retrieved context is for answering the question.

    Evaluates whether the retrieved context contains the information needed
    to answer the question, relative to the ground truth answer.

    Args:
        question: The original question.
        context: The retrieved context.
        ground_truth: The expected correct answer.
        config: Generation config for the evaluator LLM.

    Returns:
        Score between 0.0 and 1.0.
    """
    prompt = ChatPromptTemplate.from_template(
        "Rate if the context contains info needed to answer the question. "
        "1.0 means all info present, 0.0 means irrelevant. "
        "Respond with ONLY a number.\n\n"
        "Question: {question}\n\nContext: {context}\n\n"
        "Expected answer: {ground_truth}\n\nScore:"
    )
    chain = prompt | _get_eval_llm(config) | StrOutputParser()

    try:
        result = chain.invoke({
            "question": question,
            "context": context,
            "ground_truth": ground_truth,
        })
        return _parse_score(result)
    except Exception:
        logger.exception("context_precision_eval_failed")
        return 0.0


def _parse_score(text: str) -> float:
    """Extract a numeric score from LLM output."""
    text = text.strip()
    match = re.search(r"(\d+\.?\d*)", text)
    if match:
        score = float(match.group(1))
        return max(0.0, min(1.0, score))
    return 0.0
