"""FastAPI route handlers for the RAG pipeline."""

from __future__ import annotations

import tempfile
from pathlib import Path

import structlog
from fastapi import APIRouter, HTTPException, UploadFile, status

from rag_pipeline import __version__
from rag_pipeline.api.schemas import (
    ErrorResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    MetricsResponse,
    QueryRequest,
    QueryResponse,
    SourceDocument,
)
from rag_pipeline.pipeline import RAGPipeline

logger = structlog.get_logger()

router = APIRouter()

_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    """Get the global pipeline instance."""
    if _pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pipeline not initialized",
        )
    return _pipeline


def set_pipeline(pipeline: RAGPipeline) -> None:
    """Set the global pipeline instance."""
    global _pipeline
    _pipeline = pipeline


@router.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    pipeline = _pipeline
    index_loaded = pipeline is not None and pipeline._vectorstore is not None
    return HealthResponse(
        status="ok",
        version=__version__,
        index_loaded=index_loaded,
    )


@router.get("/metrics", response_model=MetricsResponse)
async def metrics():
    """Return pipeline performance metrics."""
    from rag_pipeline.metrics import collector

    return collector.snapshot()


@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)
async def ingest_path(request: IngestRequest):
    """Ingest documents from a local file or directory path."""
    pipeline = get_pipeline()
    path = Path(request.path)

    if not path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Path not found: {request.path}",
        )

    try:
        chunks = pipeline.ingest(path)
        return IngestResponse(
            status="success",
            chunks_created=chunks,
            message=f"Ingested {chunks} chunks from {request.path}",
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)
        ) from e


@router.post(
    "/ingest/upload",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    responses={400: {"model": ErrorResponse}},
)
async def ingest_upload(file: UploadFile):
    """Ingest an uploaded document file."""
    pipeline = get_pipeline()

    suffix = Path(file.filename or "upload.txt").suffix.lower()
    if suffix not in {".pdf", ".md", ".txt"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: {suffix}. Supported: .pdf, .md, .txt",
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        chunks = pipeline.ingest(tmp_path)
        return IngestResponse(
            status="success",
            chunks_created=chunks,
            message=f"Ingested {chunks} chunks from uploaded file '{file.filename}'",
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)
        ) from e
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@router.post(
    "/query",
    response_model=QueryResponse,
    responses={
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def query(request: QueryRequest):
    """Query the RAG pipeline with a question."""
    pipeline = get_pipeline()

    # Apply per-request overrides
    if request.top_k is not None:
        pipeline.settings.retrieval.top_k = request.top_k
    if request.strategy is not None:
        pipeline.settings.retrieval.strategy = request.strategy
    if request.rerank is not None:
        pipeline.settings.reranker.enabled = request.rerank

    try:
        result = pipeline.query(request.question)
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No vector index found. Ingest documents first.",
        ) from e
    except Exception as e:
        logger.exception("query_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query failed: {str(e)}",
        ) from e

    sources = [
        SourceDocument(
            content=doc.page_content[:500],
            source=doc.metadata.get("source", "unknown"),
            chunk_index=doc.metadata.get("chunk_index"),
        )
        for doc in result.source_documents
    ]

    return QueryResponse(
        question=result.question,
        answer=result.answer,
        sources=sources,
        num_chunks_retrieved=result.num_chunks_retrieved,
        num_chunks_after_rerank=result.num_chunks_after_rerank,
        latency_ms=round(result.latency_ms, 1),
    )
