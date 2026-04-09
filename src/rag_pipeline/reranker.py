"""Cross-encoder re-ranking for improved retrieval precision."""

from __future__ import annotations

import structlog
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from rag_pipeline.config import RerankerConfig

logger = structlog.get_logger()

_reranker_cache: dict[str, CrossEncoder] = {}


def get_reranker(config: RerankerConfig) -> CrossEncoder:
    """Get or create a cached cross-encoder re-ranker model.

    Args:
        config: Re-ranker configuration.

    Returns:
        A CrossEncoder instance.
    """
    if config.model_name not in _reranker_cache:
        logger.info("loading_reranker", model=config.model_name)
        _reranker_cache[config.model_name] = CrossEncoder(config.model_name)
        logger.info("reranker_loaded", model=config.model_name)
    return _reranker_cache[config.model_name]


def rerank(
    query: str,
    documents: list[Document],
    config: RerankerConfig,
) -> list[Document]:
    """Re-rank documents using a cross-encoder model.

    The cross-encoder scores each (query, document) pair jointly, producing
    more accurate relevance scores than the bi-encoder used for initial retrieval.

    Args:
        query: The user's query.
        documents: Documents from initial retrieval to re-rank.
        config: Re-ranker configuration.

    Returns:
        Top-k documents re-ordered by cross-encoder scores.
    """
    if not config.enabled or not documents:
        return documents[: config.top_k]

    model = get_reranker(config)
    pairs = [[query, doc.page_content] for doc in documents]

    logger.info("reranking", num_documents=len(documents))
    scores = model.predict(pairs)

    scored_docs = sorted(
        zip(documents, scores, strict=True), key=lambda x: x[1], reverse=True
    )
    reranked = [doc for doc, _ in scored_docs[: config.top_k]]

    logger.info("reranked", top_k=config.top_k, top_score=float(scored_docs[0][1]))
    return reranked
