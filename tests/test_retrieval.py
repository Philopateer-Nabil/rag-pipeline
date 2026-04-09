"""Tests for retrieval and vector store logic."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from rag_pipeline.config import RerankerConfig, RetrievalConfig


class TestRetrieval:
    def test_similarity_search(self):
        config = RetrievalConfig(strategy="similarity", top_k=2)
        mock_vs = MagicMock()
        mock_vs.similarity_search.return_value = [
            Document(page_content="Result 1", metadata={}),
            Document(page_content="Result 2", metadata={}),
        ]

        from rag_pipeline.retrieval import retrieve

        results = retrieve("test query", mock_vs, config)
        assert len(results) == 2
        mock_vs.similarity_search.assert_called_once_with("test query", k=2)

    def test_mmr_search(self):
        config = RetrievalConfig(strategy="mmr", top_k=3, mmr_lambda=0.7)
        mock_vs = MagicMock()
        mock_vs.max_marginal_relevance_search.return_value = [
            Document(page_content="Result 1", metadata={}),
            Document(page_content="Result 2", metadata={}),
            Document(page_content="Result 3", metadata={}),
        ]

        from rag_pipeline.retrieval import retrieve

        results = retrieve("test query", mock_vs, config)
        assert len(results) == 3
        mock_vs.max_marginal_relevance_search.assert_called_once_with(
            "test query", k=3, lambda_mult=0.7
        )

    def test_invalid_strategy_raises(self):
        config = RetrievalConfig(strategy="similarity")
        object.__setattr__(config, "strategy", "invalid")
        mock_vs = MagicMock()

        from rag_pipeline.retrieval import retrieve

        with pytest.raises(ValueError, match="Unknown retrieval strategy"):
            retrieve("test", mock_vs, config)

    def test_retrieve_with_scores(self):
        config = RetrievalConfig(strategy="similarity", top_k=2, score_threshold=0.0)
        mock_vs = MagicMock()
        mock_vs.similarity_search_with_score.return_value = [
            (Document(page_content="High relevance", metadata={}), 0.95),
            (Document(page_content="Low relevance", metadata={}), 0.3),
        ]

        from rag_pipeline.retrieval import retrieve_with_scores

        results = retrieve_with_scores("test", mock_vs, config)
        assert len(results) == 2
        assert results[0][1] == 0.95

    def test_retrieve_with_score_threshold(self):
        config = RetrievalConfig(strategy="similarity", top_k=5, score_threshold=0.5)
        mock_vs = MagicMock()
        mock_vs.similarity_search_with_score.return_value = [
            (Document(page_content="High", metadata={}), 0.9),
            (Document(page_content="Medium", metadata={}), 0.6),
            (Document(page_content="Low", metadata={}), 0.2),
        ]

        from rag_pipeline.retrieval import retrieve_with_scores

        results = retrieve_with_scores("test", mock_vs, config)
        assert len(results) == 2  # Only scores >= 0.5


class TestReranker:
    def test_rerank_disabled(self):
        config = RerankerConfig(enabled=False, top_k=2)
        docs = [
            Document(page_content="Doc 1", metadata={}),
            Document(page_content="Doc 2", metadata={}),
            Document(page_content="Doc 3", metadata={}),
        ]

        from rag_pipeline.reranker import rerank

        results = rerank("query", docs, config)
        assert len(results) == 2
        assert results[0].page_content == "Doc 1"

    def test_rerank_empty_docs(self):
        config = RerankerConfig(enabled=True, top_k=3)

        from rag_pipeline.reranker import rerank

        results = rerank("query", [], config)
        assert results == []

    @patch("rag_pipeline.reranker.get_reranker")
    def test_rerank_reorders_by_score(self, mock_get_reranker):
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.1, 0.9, 0.5]
        mock_get_reranker.return_value = mock_model

        config = RerankerConfig(enabled=True, top_k=2)
        docs = [
            Document(page_content="Low score", metadata={}),
            Document(page_content="High score", metadata={}),
            Document(page_content="Mid score", metadata={}),
        ]

        from rag_pipeline.reranker import rerank

        results = rerank("query", docs, config)
        assert len(results) == 2
        assert results[0].page_content == "High score"
        assert results[1].page_content == "Mid score"
