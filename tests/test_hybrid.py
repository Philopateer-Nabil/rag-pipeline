"""Tests for hybrid retrieval and BM25."""

from __future__ import annotations

from unittest.mock import MagicMock

from langchain_core.documents import Document

from rag_pipeline.config import RetrievalConfig
from rag_pipeline.hybrid_retrieval import BM25, hybrid_retrieve, reciprocal_rank_fusion


class TestBM25:
    def test_scores_relevant_doc_higher(self):
        docs = [
            Document(page_content="The transformer architecture uses attention", metadata={}),
            Document(page_content="Cooking recipes for pasta and salad", metadata={}),
            Document(page_content="Attention mechanism in deep learning", metadata={}),
        ]
        bm25 = BM25(docs)
        scores = bm25.score("transformer attention")
        # First and third docs should score higher than the cooking doc
        assert scores[0] > scores[1]
        assert scores[2] > scores[1]

    def test_empty_query(self):
        docs = [Document(page_content="Some content", metadata={})]
        bm25 = BM25(docs)
        scores = bm25.score("")
        assert scores == [0.0]

    def test_single_document(self):
        docs = [Document(page_content="FAISS vector database", metadata={})]
        bm25 = BM25(docs)
        scores = bm25.score("FAISS")
        assert len(scores) == 1
        assert scores[0] > 0


class TestReciprocalRankFusion:
    def test_fuses_two_lists(self):
        doc_a = Document(page_content="Doc A", metadata={})
        doc_b = Document(page_content="Doc B", metadata={})
        doc_c = Document(page_content="Doc C", metadata={})

        list1 = [(doc_a, 0.9), (doc_b, 0.8)]
        list2 = [(doc_b, 0.95), (doc_c, 0.7)]

        fused = reciprocal_rank_fusion([list1, list2], k=60)
        # doc_b appears in both lists, should be ranked first
        assert fused[0].page_content == "Doc B"
        assert len(fused) == 3

    def test_single_list(self):
        doc = Document(page_content="Only doc", metadata={})
        fused = reciprocal_rank_fusion([[(doc, 1.0)]])
        assert len(fused) == 1

    def test_empty_lists(self):
        fused = reciprocal_rank_fusion([[], []])
        assert fused == []


class TestHybridRetrieve:
    def test_hybrid_returns_top_k(self):
        config = RetrievalConfig(strategy="hybrid", top_k=2, rrf_k=60)
        chunks = [
            Document(page_content="Transformers use attention mechanisms", metadata={}),
            Document(page_content="FAISS is a vector search library", metadata={}),
            Document(page_content="RAG combines retrieval with generation", metadata={}),
        ]

        mock_vs = MagicMock()
        mock_vs.similarity_search_with_score.return_value = [
            (chunks[0], 0.9),
            (chunks[2], 0.8),
            (chunks[1], 0.5),
        ]

        results = hybrid_retrieve("attention", mock_vs, chunks, config)
        assert len(results) <= 2
