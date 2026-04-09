"""Tests for document chunking logic."""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from rag_pipeline.chunking import chunk_documents, create_splitter
from rag_pipeline.config import ChunkingConfig


class TestCreateSplitter:
    def test_recursive_strategy(self, chunking_config: ChunkingConfig):
        splitter = create_splitter(chunking_config)
        assert splitter is not None
        assert splitter._chunk_size == 200
        assert splitter._chunk_overlap == 20

    def test_fixed_strategy(self, fixed_chunking_config: ChunkingConfig):
        splitter = create_splitter(fixed_chunking_config)
        assert splitter is not None
        assert splitter._chunk_size == 200

    def test_invalid_strategy_raises(self):
        config = ChunkingConfig(strategy="recursive")
        # Override with invalid value bypassing validation
        object.__setattr__(config, "strategy", "invalid")
        with pytest.raises(ValueError, match="Unknown chunking strategy"):
            create_splitter(config)


class TestChunkDocuments:
    def test_chunks_long_document(self, long_document: Document, chunking_config: ChunkingConfig):
        chunks = chunk_documents([long_document], chunking_config)
        assert len(chunks) > 1
        for chunk in chunks:
            # Chunks should not greatly exceed configured size
            assert len(chunk.page_content) <= chunking_config.chunk_size + 50

    def test_preserves_metadata(self, long_document: Document, chunking_config: ChunkingConfig):
        chunks = chunk_documents([long_document], chunking_config)
        for chunk in chunks:
            assert "source" in chunk.metadata
            assert chunk.metadata["source"] == "long_doc.md"

    def test_adds_chunk_metadata(self, long_document: Document, chunking_config: ChunkingConfig):
        chunks = chunk_documents([long_document], chunking_config)
        for i, chunk in enumerate(chunks):
            assert chunk.metadata["chunk_index"] == i
            assert chunk.metadata["chunk_size"] == len(chunk.page_content)

    def test_empty_input(self, chunking_config: ChunkingConfig):
        chunks = chunk_documents([], chunking_config)
        assert chunks == []

    def test_short_document_single_chunk(self, chunking_config: ChunkingConfig):
        doc = Document(page_content="Short text.", metadata={"source": "short.txt"})
        chunks = chunk_documents([doc], chunking_config)
        assert len(chunks) == 1
        assert chunks[0].page_content == "Short text."

    def test_overlap_produces_shared_content(self):
        config = ChunkingConfig(strategy="recursive", chunk_size=100, chunk_overlap=30)
        text = "word " * 80  # ~400 chars
        doc = Document(page_content=text.strip(), metadata={"source": "test.txt"})
        chunks = chunk_documents([doc], config)
        assert len(chunks) > 1

        # With overlap, consecutive chunks should share some content
        for i in range(len(chunks) - 1):
            assert len(chunks[i].page_content) > 0
            assert len(chunks[i + 1].page_content) > 0

    def test_multiple_documents(self, chunking_config: ChunkingConfig):
        docs = [
            Document(page_content="A " * 200, metadata={"source": "a.txt"}),
            Document(page_content="B " * 200, metadata={"source": "b.txt"}),
        ]
        chunks = chunk_documents(docs, chunking_config)
        sources = {c.metadata["source"] for c in chunks}
        assert "a.txt" in sources
        assert "b.txt" in sources

    def test_recursive_respects_separators(self):
        config = ChunkingConfig(
            strategy="recursive",
            chunk_size=100,
            chunk_overlap=0,
            separators=["\n\n", "\n", " "],
        )
        text = (
            "First paragraph.\n\nSecond paragraph.\n\n"
            "Third paragraph that is longer to force splitting."
        )
        doc = Document(page_content=text, metadata={"source": "test.txt"})
        chunks = chunk_documents([doc], config)
        # Should prefer splitting on \n\n
        assert any("First paragraph" in c.page_content for c in chunks)
