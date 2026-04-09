"""Tests for FastAPI endpoints."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from rag_pipeline.api.app import create_app
from rag_pipeline.api.routes import set_pipeline
from rag_pipeline.config import Settings
from rag_pipeline.pipeline import QueryResult


@pytest.fixture
def mock_pipeline():
    """Create a mocked RAGPipeline."""
    pipeline = MagicMock()
    pipeline._vectorstore = MagicMock()
    pipeline.settings = Settings()
    return pipeline


@pytest.fixture
def client(mock_pipeline):
    """Create a test client with mocked pipeline."""
    with patch("rag_pipeline.api.app.RAGPipeline", return_value=mock_pipeline):
        app = create_app()
        set_pipeline(mock_pipeline)
        yield TestClient(app)


class TestHealthEndpoint:
    def test_health_ok(self, client, mock_pipeline):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data

    def test_health_no_index(self, client, mock_pipeline):
        mock_pipeline._vectorstore = None
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["index_loaded"] is False


class TestIngestEndpoint:
    def test_ingest_path_not_found(self, client, mock_pipeline):
        response = client.post("/ingest", json={"path": "/nonexistent/path"})
        assert response.status_code == 404

    def test_ingest_success(self, client, mock_pipeline, tmp_path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content for ingestion")
        mock_pipeline.ingest.return_value = 3

        response = client.post("/ingest", json={"path": str(test_file)})
        assert response.status_code == 201
        data = response.json()
        assert data["status"] == "success"
        assert data["chunks_created"] == 3

    def test_ingest_invalid_request(self, client):
        response = client.post("/ingest", json={})
        assert response.status_code == 422  # Validation error


class TestQueryEndpoint:
    def test_query_success(self, client, mock_pipeline):
        mock_pipeline.query.return_value = QueryResult(
            question="What is RAG?",
            answer="RAG is Retrieval-Augmented Generation.",
            source_documents=[
                Document(
                    page_content="RAG combines retrieval with generation.",
                    metadata={"source": "test.md", "chunk_index": 0},
                )
            ],
            context="Context text",
            num_chunks_retrieved=5,
            num_chunks_after_rerank=3,
        )

        response = client.post("/query", json={"question": "What is RAG?"})
        assert response.status_code == 200
        data = response.json()
        assert data["question"] == "What is RAG?"
        assert data["answer"] == "RAG is Retrieval-Augmented Generation."
        assert len(data["sources"]) == 1
        assert data["num_chunks_retrieved"] == 5

    def test_query_no_index(self, client, mock_pipeline):
        mock_pipeline.query.side_effect = FileNotFoundError("No index")

        response = client.post("/query", json={"question": "test question"})
        assert response.status_code == 404
        assert "Ingest documents first" in response.json()["detail"]

    def test_query_empty_question(self, client):
        response = client.post("/query", json={"question": ""})
        assert response.status_code == 422

    def test_query_with_overrides(self, client, mock_pipeline):
        mock_pipeline.query.return_value = QueryResult(
            question="test",
            answer="answer",
            source_documents=[],
            context="",
            num_chunks_retrieved=3,
            num_chunks_after_rerank=2,
        )

        response = client.post(
            "/query",
            json={
                "question": "test",
                "top_k": 10,
                "strategy": "mmr",
                "rerank": False,
            },
        )
        assert response.status_code == 200

    def test_query_server_error(self, client, mock_pipeline):
        mock_pipeline.query.side_effect = RuntimeError("LLM connection failed")

        response = client.post("/query", json={"question": "test"})
        assert response.status_code == 500
        assert "Query failed" in response.json()["detail"]


class TestUploadEndpoint:
    def test_upload_unsupported_type(self, client, mock_pipeline):
        response = client.post(
            "/ingest/upload",
            files={"file": ("test.jpg", b"fake image data", "image/jpeg")},
        )
        assert response.status_code == 400
        assert "Unsupported file type" in response.json()["detail"]

    def test_upload_text_file(self, client, mock_pipeline):
        mock_pipeline.ingest.return_value = 2

        response = client.post(
            "/ingest/upload",
            files={"file": ("doc.txt", b"Test document content", "text/plain")},
        )
        assert response.status_code == 201
        assert response.json()["chunks_created"] == 2
