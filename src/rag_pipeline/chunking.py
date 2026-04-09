"""Document chunking with fixed-size and recursive character splitting strategies."""

from __future__ import annotations

import structlog
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

from rag_pipeline.config import ChunkingConfig

logger = structlog.get_logger()


def create_splitter(
    config: ChunkingConfig,
) -> CharacterTextSplitter | RecursiveCharacterTextSplitter:
    """Create a text splitter based on configuration.

    Args:
        config: Chunking configuration specifying strategy and parameters.

    Returns:
        A configured text splitter instance.

    Raises:
        ValueError: If the strategy is not recognized.
    """
    if config.strategy == "fixed":
        return CharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separator="\n",
        )
    elif config.strategy == "recursive":
        return RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=config.separators,
        )
    else:
        raise ValueError(f"Unknown chunking strategy: {config.strategy}")


def chunk_documents(
    documents: list[Document],
    config: ChunkingConfig,
) -> list[Document]:
    """Split documents into chunks using the configured strategy.

    Args:
        documents: List of documents to chunk.
        config: Chunking configuration.

    Returns:
        List of chunked documents with preserved and enriched metadata.
    """
    splitter = create_splitter(config)
    chunks = splitter.split_documents(documents)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        chunk.metadata["chunk_size"] = len(chunk.page_content)

    logger.info(
        "documents_chunked",
        input_docs=len(documents),
        output_chunks=len(chunks),
        strategy=config.strategy,
        chunk_size=config.chunk_size,
        overlap=config.chunk_overlap,
    )
    return chunks
