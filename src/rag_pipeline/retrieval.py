"""Retrieval strategies: similarity search, MMR, and hybrid (dense + BM25)."""

from __future__ import annotations

import structlog
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from rag_pipeline.config import RetrievalConfig

logger = structlog.get_logger()


def retrieve(
    query: str,
    vectorstore: FAISS,
    config: RetrievalConfig,
    all_chunks: list[Document] | None = None,
) -> list[Document]:
    """Retrieve relevant documents for a query.

    Args:
        query: The user's question or search query.
        vectorstore: The FAISS vector store to search.
        config: Retrieval configuration (strategy, top_k, etc.).
        all_chunks: All document chunks (required for hybrid strategy).

    Returns:
        List of relevant documents ranked by relevance.
    """
    logger.info("retrieving", strategy=config.strategy, top_k=config.top_k)

    if config.strategy == "mmr":
        docs = vectorstore.max_marginal_relevance_search(
            query,
            k=config.top_k,
            lambda_mult=config.mmr_lambda,
        )
    elif config.strategy == "similarity":
        docs = vectorstore.similarity_search(
            query,
            k=config.top_k,
        )
    elif config.strategy == "hybrid":
        from rag_pipeline.hybrid_retrieval import hybrid_retrieve

        if all_chunks is None:
            logger.warning("hybrid_no_chunks_fallback", fallback="similarity")
            docs = vectorstore.similarity_search(query, k=config.top_k)
        else:
            docs = hybrid_retrieve(
                query, vectorstore, all_chunks, config, rrf_k=config.rrf_k
            )
    else:
        raise ValueError(f"Unknown retrieval strategy: {config.strategy}")

    logger.info("retrieved", num_results=len(docs), strategy=config.strategy)
    return docs


def retrieve_with_scores(
    query: str,
    vectorstore: FAISS,
    config: RetrievalConfig,
) -> list[tuple[Document, float]]:
    """Retrieve documents with relevance scores.

    Args:
        query: The user's question or search query.
        vectorstore: The FAISS vector store to search.
        config: Retrieval configuration.

    Returns:
        List of (document, score) tuples sorted by relevance.
    """
    results = vectorstore.similarity_search_with_score(query, k=config.top_k)

    if config.score_threshold > 0:
        results = [(doc, score) for doc, score in results if score >= config.score_threshold]

    logger.info("retrieved_with_scores", num_results=len(results))
    return results
