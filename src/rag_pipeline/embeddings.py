"""Embedding generation using HuggingFace sentence-transformers."""

from __future__ import annotations

import structlog
from langchain_huggingface import HuggingFaceEmbeddings

from rag_pipeline.config import EmbeddingConfig

logger = structlog.get_logger()

_embeddings_cache: dict[str, HuggingFaceEmbeddings] = {}


def get_embedding_model(config: EmbeddingConfig) -> HuggingFaceEmbeddings:
    """Get or create a cached HuggingFace embedding model.

    Args:
        config: Embedding configuration with model name and device.

    Returns:
        A HuggingFaceEmbeddings instance ready for encoding.
    """
    cache_key = f"{config.model_name}:{config.device}"
    if cache_key not in _embeddings_cache:
        logger.info(
            "loading_embedding_model",
            model=config.model_name,
            device=config.device,
        )
        _embeddings_cache[cache_key] = HuggingFaceEmbeddings(
            model_name=config.model_name,
            model_kwargs={"device": config.device},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info("embedding_model_loaded", model=config.model_name)
    return _embeddings_cache[cache_key]
