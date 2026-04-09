"""Pydantic models for API request/response schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"
    version: str
    index_loaded: bool


class IngestRequest(BaseModel):
    """Document ingestion request (for JSON-based path ingestion)."""

    path: str = Field(..., description="Path to file or directory to ingest")


class IngestResponse(BaseModel):
    """Document ingestion response."""

    status: str
    chunks_created: int
    message: str


class QueryRequest(BaseModel):
    """Question-answering query request."""

    question: str = Field(..., min_length=1, max_length=2000, description="The question to answer")
    top_k: int | None = Field(default=None, ge=1, le=100, description="Override retrieval top_k")
    strategy: str | None = Field(
        default=None,
        pattern=r"^(similarity|mmr|hybrid)$",
        description="Override retrieval strategy",
    )
    rerank: bool | None = Field(default=None, description="Override re-ranking setting")


class SourceDocument(BaseModel):
    """A source document returned with query results."""

    content: str
    source: str
    chunk_index: int | None = None


class QueryResponse(BaseModel):
    """Question-answering query response."""

    question: str
    answer: str
    sources: list[SourceDocument]
    num_chunks_retrieved: int
    num_chunks_after_rerank: int
    latency_ms: float | None = None


class StageMetrics(BaseModel):
    """Metrics for a single pipeline stage."""

    total_calls: int
    avg_latency_ms: float
    min_latency_ms: float | None
    max_latency_ms: float | None


class MetricsResponse(BaseModel):
    """Pipeline metrics response."""

    total_queries: int
    total_ingestions: int
    total_chunks_ingested: int
    estimated_tokens_used: int
    stages: dict[str, StageMetrics]


class ErrorResponse(BaseModel):
    """Standard error response."""

    detail: str
    error_type: str = "error"
