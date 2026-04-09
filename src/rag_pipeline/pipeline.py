"""End-to-end RAG pipeline orchestrating all components."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import structlog
from langchain_core.documents import Document

from rag_pipeline.chunking import chunk_documents
from rag_pipeline.config import Settings, load_settings
from rag_pipeline.embeddings import get_embedding_model
from rag_pipeline.generation import format_context, generate
from rag_pipeline.ingestion import load_directory, load_document
from rag_pipeline.metrics import Timer, collector, estimate_tokens
from rag_pipeline.reranker import rerank
from rag_pipeline.retrieval import retrieve
from rag_pipeline.vectorstore import (
    add_documents,
    create_vectorstore,
    load_vectorstore,
)

logger = structlog.get_logger()


@dataclass
class QueryResult:
    """Result from a RAG query."""

    question: str
    answer: str
    source_documents: list[Document]
    context: str
    num_chunks_retrieved: int = 0
    num_chunks_after_rerank: int = 0
    latency_ms: float = 0.0


class RAGPipeline:
    """Main RAG pipeline coordinating ingestion, retrieval, and generation."""

    def __init__(self, settings: Settings | None = None, config_path: str | None = None):
        """Initialize the pipeline.

        Args:
            settings: Pre-built Settings object. If None, loads from config_path.
            config_path: Path to config YAML. Used only if settings is None.
        """
        self.settings = settings or load_settings(config_path)
        self._embedding_model = get_embedding_model(self.settings.embedding)
        self._vectorstore = None
        self._all_chunks: list[Document] = []
        logger.info("pipeline_initialized")

    @property
    def vectorstore(self):
        """Lazy-load the vector store from disk."""
        if self._vectorstore is None:
            self._vectorstore = load_vectorstore(
                self._embedding_model, self.settings.vectorstore
            )
        return self._vectorstore

    def ingest(self, path: str | Path) -> int:
        """Ingest documents from a file or directory.

        Args:
            path: Path to a file or directory to ingest.

        Returns:
            Number of chunks created and stored.
        """
        with Timer("ingestion"):
            path = Path(path)
            documents = load_directory(path) if path.is_dir() else load_document(path)

            if not documents:
                logger.warning("no_documents_found", path=str(path))
                return 0

            chunks = chunk_documents(documents, self.settings.chunking)
            self._all_chunks.extend(chunks)

            if self._vectorstore is None:
                try:
                    self._vectorstore = load_vectorstore(
                        self._embedding_model, self.settings.vectorstore
                    )
                    add_documents(self._vectorstore, chunks, self.settings.vectorstore)
                except FileNotFoundError:
                    self._vectorstore = create_vectorstore(
                        chunks, self._embedding_model, self.settings.vectorstore
                    )
            else:
                add_documents(self._vectorstore, chunks, self.settings.vectorstore)

        collector.record_ingestion(len(chunks))
        logger.info("ingestion_complete", chunks=len(chunks))
        return len(chunks)

    def query(self, question: str) -> QueryResult:
        """Run the full RAG pipeline on a question.

        Args:
            question: The user's question.

        Returns:
            QueryResult with the answer and supporting context.
        """
        logger.info("query_start", question=question[:100])
        total_timer = Timer("query_total")
        total_timer.__enter__()

        # Retrieve
        with Timer("retrieval"):
            docs = retrieve(
                question,
                self.vectorstore,
                self.settings.retrieval,
                all_chunks=self._all_chunks or None,
            )
        num_retrieved = len(docs)

        # Re-rank
        with Timer("reranking"):
            docs = rerank(question, docs, self.settings.reranker)
        num_reranked = len(docs)

        # Generate
        with Timer("generation"):
            context = format_context(docs)
            answer = generate(question, docs, self.settings.generation)

        total_timer.__exit__(None, None, None)

        # Track tokens
        token_estimate = estimate_tokens(context) + estimate_tokens(answer)
        collector.record_tokens(token_estimate)
        collector.record_query()

        result = QueryResult(
            question=question,
            answer=answer,
            source_documents=docs,
            context=context,
            num_chunks_retrieved=num_retrieved,
            num_chunks_after_rerank=num_reranked,
            latency_ms=total_timer.elapsed_ms,
        )

        logger.info(
            "query_complete",
            retrieved=num_retrieved,
            reranked=num_reranked,
            answer_length=len(answer),
            latency_ms=round(total_timer.elapsed_ms, 1),
            est_tokens=token_estimate,
        )
        return result

    def retrieve_only(self, question: str) -> list[Document]:
        """Retrieve documents without generation (useful for debugging).

        Args:
            question: The search query.

        Returns:
            List of relevant documents after retrieval and re-ranking.
        """
        docs = retrieve(
            question,
            self.vectorstore,
            self.settings.retrieval,
            all_chunks=self._all_chunks or None,
        )
        return rerank(question, docs, self.settings.reranker)
