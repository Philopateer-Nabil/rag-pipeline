"""Hybrid retrieval combining dense (FAISS) and sparse (BM25) search with RRF."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

import structlog
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from rag_pipeline.config import RetrievalConfig

logger = structlog.get_logger()


class BM25:
    """Lightweight BM25 implementation over a set of documents.

    This avoids pulling in rank_bm25 as a dependency and gives us full control.
    """

    def __init__(self, documents: list[Document], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents = documents
        self._corpus: list[list[str]] = []
        self._doc_freqs: Counter[str] = Counter()
        self._doc_lens: list[int] = []
        self._avg_dl: float = 0.0
        self._n = len(documents)

        self._index(documents)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())

    def _index(self, documents: list[Document]) -> None:
        for doc in documents:
            tokens = self._tokenize(doc.page_content)
            self._corpus.append(tokens)
            self._doc_lens.append(len(tokens))
            unique = set(tokens)
            for token in unique:
                self._doc_freqs[token] += 1

        self._avg_dl = sum(self._doc_lens) / self._n if self._n else 1.0

    def _idf(self, term: str) -> float:
        df = self._doc_freqs.get(term, 0)
        return math.log((self._n - df + 0.5) / (df + 0.5) + 1.0)

    def score(self, query: str) -> list[float]:
        """Score all documents against a query. Returns list of BM25 scores."""
        query_tokens = self._tokenize(query)
        scores = []
        for i, doc_tokens in enumerate(self._corpus):
            tf_map = Counter(doc_tokens)
            dl = self._doc_lens[i]
            doc_score = 0.0
            for qt in query_tokens:
                tf = tf_map.get(qt, 0)
                idf = self._idf(qt)
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * dl / self._avg_dl)
                doc_score += idf * numerator / denominator
            scores.append(doc_score)
        return scores


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[Document, float]]],
    k: int = 60,
) -> list[Document]:
    """Combine multiple ranked lists using Reciprocal Rank Fusion (RRF).

    RRF score = sum(1 / (k + rank_i)) across all lists.
    This is robust to score scale differences between retrieval methods.

    Args:
        ranked_lists: List of ranked result lists, each containing (doc, score) tuples.
        k: RRF constant (default 60, as in the original paper).

    Returns:
        Documents re-ordered by fused RRF score.
    """
    doc_scores: dict[str, float] = defaultdict(float)
    doc_map: dict[str, Document] = {}

    for ranked_list in ranked_lists:
        for rank, (doc, _score) in enumerate(ranked_list, start=1):
            doc_id = _doc_key(doc)
            doc_scores[doc_id] += 1.0 / (k + rank)
            doc_map[doc_id] = doc

    sorted_ids = sorted(doc_scores, key=lambda d: doc_scores[d], reverse=True)
    return [doc_map[doc_id] for doc_id in sorted_ids]


def _doc_key(doc: Document) -> str:
    """Create a unique key for deduplication based on content hash."""
    return str(hash(doc.page_content))


def hybrid_retrieve(
    query: str,
    vectorstore: FAISS,
    all_chunks: list[Document],
    config: RetrievalConfig,
    rrf_k: int = 60,
) -> list[Document]:
    """Retrieve documents using hybrid dense + sparse search with RRF fusion.

    Args:
        query: The user's query.
        vectorstore: FAISS vector store for dense retrieval.
        all_chunks: All document chunks for BM25 sparse retrieval.
        config: Retrieval configuration.
        rrf_k: RRF constant.

    Returns:
        Top-k documents after fusing dense and sparse rankings.
    """
    fetch_k = config.top_k * 3  # over-fetch to give RRF enough candidates

    # Dense retrieval via FAISS
    dense_results = vectorstore.similarity_search_with_score(query, k=fetch_k)
    dense_ranked = [(doc, float(score)) for doc, score in dense_results]

    # Sparse retrieval via BM25
    bm25 = BM25(all_chunks)
    bm25_scores = bm25.score(query)
    sparse_ranked = sorted(
        zip(all_chunks, bm25_scores, strict=True),
        key=lambda x: x[1],
        reverse=True,
    )[:fetch_k]

    # Fuse with RRF
    fused = reciprocal_rank_fusion([dense_ranked, sparse_ranked], k=rrf_k)

    logger.info(
        "hybrid_retrieved",
        dense_candidates=len(dense_ranked),
        sparse_candidates=len(sparse_ranked),
        fused_total=len(fused),
        returning=min(config.top_k, len(fused)),
    )
    return fused[: config.top_k]
