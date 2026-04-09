"""FAISS vector store management with disk persistence."""

from __future__ import annotations

from pathlib import Path

import structlog
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from rag_pipeline.config import VectorStoreConfig

logger = structlog.get_logger()


def create_vectorstore(
    documents: list[Document],
    embedding_model: HuggingFaceEmbeddings,
    config: VectorStoreConfig,
) -> FAISS:
    """Create a FAISS vector store from documents and save to disk.

    Args:
        documents: Chunked documents to embed and store.
        embedding_model: The embedding model to use.
        config: Vector store configuration.

    Returns:
        The created FAISS vector store.
    """
    logger.info("creating_vectorstore", num_documents=len(documents))
    vectorstore = FAISS.from_documents(documents, embedding_model)

    save_path = Path(config.index_path)
    save_path.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(save_path))
    logger.info("vectorstore_saved", path=str(save_path))

    return vectorstore


def load_vectorstore(
    embedding_model: HuggingFaceEmbeddings,
    config: VectorStoreConfig,
) -> FAISS:
    """Load a FAISS vector store from disk.

    Args:
        embedding_model: The embedding model (must match the one used to create the store).
        config: Vector store configuration with index path.

    Returns:
        The loaded FAISS vector store.

    Raises:
        FileNotFoundError: If the index does not exist at the configured path.
    """
    save_path = Path(config.index_path)
    if not save_path.exists():
        raise FileNotFoundError(
            f"No FAISS index found at {save_path}. Run ingestion first."
        )

    logger.info("loading_vectorstore", path=str(save_path))
    vectorstore = FAISS.load_local(
        str(save_path),
        embedding_model,
        allow_dangerous_deserialization=True,
    )
    logger.info("vectorstore_loaded", path=str(save_path))
    return vectorstore


def add_documents(
    vectorstore: FAISS,
    documents: list[Document],
    config: VectorStoreConfig,
) -> FAISS:
    """Add documents to an existing vector store and persist.

    Args:
        vectorstore: Existing FAISS vector store.
        documents: New documents to add.
        config: Vector store configuration.

    Returns:
        The updated vector store.
    """
    logger.info("adding_documents", num_documents=len(documents))
    vectorstore.add_documents(documents)

    save_path = Path(config.index_path)
    save_path.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(save_path))
    logger.info("vectorstore_updated", path=str(save_path))

    return vectorstore
