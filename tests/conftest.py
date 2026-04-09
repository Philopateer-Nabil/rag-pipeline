"""Shared test fixtures."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from rag_pipeline.config import (
    ChunkingConfig,
    EmbeddingConfig,
    GenerationConfig,
    RerankerConfig,
    RetrievalConfig,
    Settings,
    VectorStoreConfig,
)


@pytest.fixture
def sample_documents() -> list[Document]:
    """Create sample documents for testing."""
    return [
        Document(
            page_content=(
                "The Transformer architecture was introduced by Vaswani et al. in 2017. "
                "It relies on self-attention mechanisms to capture dependencies between "
                "input and output positions. The key innovation is multi-head attention, "
                "which allows the model to attend to different representation subspaces."
            ),
            metadata={"source": "test_transformers.md", "file_type": ".md"},
        ),
        Document(
            page_content=(
                "FAISS is an open-source library developed by Meta for efficient similarity "
                "search and clustering of dense vectors. It supports multiple index types "
                "including flat, IVF, HNSW, and product quantization. FAISS can run on "
                "both CPU and GPU hardware."
            ),
            metadata={"source": "test_faiss.md", "file_type": ".md"},
        ),
        Document(
            page_content=(
                "Retrieval-Augmented Generation combines the power of large language models "
                "with external knowledge retrieval. RAG reduces hallucinations by grounding "
                "generation in retrieved documents. It was first formalized by Lewis et al."
            ),
            metadata={"source": "test_rag.md", "file_type": ".md"},
        ),
    ]


@pytest.fixture
def long_document() -> Document:
    """A longer document for testing chunking."""
    paragraphs = [
        "Paragraph about transformers. " * 20,
        "Paragraph about attention. " * 20,
        "Paragraph about embeddings. " * 20,
        "Paragraph about vector databases. " * 20,
    ]
    return Document(
        page_content="\n\n".join(paragraphs),
        metadata={"source": "long_doc.md"},
    )


@pytest.fixture
def chunking_config() -> ChunkingConfig:
    return ChunkingConfig(strategy="recursive", chunk_size=200, chunk_overlap=20)


@pytest.fixture
def fixed_chunking_config() -> ChunkingConfig:
    return ChunkingConfig(strategy="fixed", chunk_size=200, chunk_overlap=20)


@pytest.fixture
def test_settings(tmp_path: Path) -> Settings:
    """Settings configured for testing with a temp index path."""
    return Settings(
        chunking=ChunkingConfig(strategy="recursive", chunk_size=200, chunk_overlap=20),
        embedding=EmbeddingConfig(model_name="sentence-transformers/all-MiniLM-L6-v2"),
        vectorstore=VectorStoreConfig(index_path=str(tmp_path / "test_index")),
        retrieval=RetrievalConfig(strategy="similarity", top_k=2),
        reranker=RerankerConfig(enabled=False),
        generation=GenerationConfig(model_name="llama3"),
    )


@pytest.fixture
def mock_llm_response():
    """Mock LLM that returns a fixed response."""
    with patch("rag_pipeline.generation.build_rag_chain") as mock_chain:
        mock_runnable = MagicMock()
        mock_runnable.invoke.return_value = "This is a test answer from the mocked LLM."
        mock_chain.return_value = mock_runnable
        yield mock_runnable
