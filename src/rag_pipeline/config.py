"""Configuration management with YAML loading and environment variable overrides."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import structlog
import yaml
from pydantic import BaseModel, Field

logger = structlog.get_logger()

_DEFAULT_CONFIG_PATH = Path("config.yaml")


class ChunkingConfig(BaseModel):
    """Chunking strategy configuration."""

    strategy: str = Field(default="recursive", pattern=r"^(fixed|recursive)$")
    chunk_size: int = Field(default=512, ge=64, le=8192)
    chunk_overlap: int = Field(default=64, ge=0)
    separators: list[str] = Field(default=["\n\n", "\n", ". ", " ", ""])


class EmbeddingConfig(BaseModel):
    """Embedding model configuration."""

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cpu"


class VectorStoreConfig(BaseModel):
    """Vector store configuration."""

    index_path: str = "./data/faiss_index"
    distance_metric: str = Field(default="cosine", pattern=r"^(cosine|l2|ip)$")


class RetrievalConfig(BaseModel):
    """Retrieval configuration."""

    strategy: str = Field(default="mmr", pattern=r"^(similarity|mmr|hybrid)$")
    top_k: int = Field(default=5, ge=1, le=100)
    mmr_lambda: float = Field(default=0.5, ge=0.0, le=1.0)
    score_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    rrf_k: int = Field(default=60, ge=1, description="RRF constant for hybrid search")


class RerankerConfig(BaseModel):
    """Re-ranker configuration."""

    enabled: bool = True
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_k: int = Field(default=3, ge=1, le=50)


class GenerationConfig(BaseModel):
    """LLM generation configuration."""

    llm_backend: str = Field(
        default="ollama", pattern=r"^(ollama|huggingface)$"
    )
    model_name: str = "llama3"
    base_url: str = "http://localhost:11434"
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, ge=64, le=32768)


class ApiConfig(BaseModel):
    """API server configuration."""

    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = "INFO"
    format: str = Field(default="console", pattern=r"^(console|json)$")


class Settings(BaseModel):
    """Root settings combining all configuration sections."""

    chunking: ChunkingConfig = ChunkingConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    vectorstore: VectorStoreConfig = VectorStoreConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    reranker: RerankerConfig = RerankerConfig()
    generation: GenerationConfig = GenerationConfig()
    api: ApiConfig = ApiConfig()
    logging: LoggingConfig = LoggingConfig()


def _apply_env_overrides(data: dict[str, Any]) -> dict[str, Any]:
    """Apply RAG_ prefixed environment variable overrides.

    Supports flat keys like RAG_CHUNK_SIZE mapping to chunking.chunk_size,
    and nested keys like RAG_CHUNKING__STRATEGY mapping to chunking.strategy.
    """
    env_map = {
        "RAG_CHUNK_SIZE": ("chunking", "chunk_size"),
        "RAG_CHUNK_OVERLAP": ("chunking", "chunk_overlap"),
        "RAG_CHUNK_STRATEGY": ("chunking", "strategy"),
        "RAG_EMBEDDING_MODEL": ("embedding", "model_name"),
        "RAG_EMBEDDING_DEVICE": ("embedding", "device"),
        "RAG_INDEX_PATH": ("vectorstore", "index_path"),
        "RAG_RETRIEVAL_STRATEGY": ("retrieval", "strategy"),
        "RAG_RETRIEVAL_TOP_K": ("retrieval", "top_k"),
        "RAG_MMR_LAMBDA": ("retrieval", "mmr_lambda"),
        "RAG_RERANKER_ENABLED": ("reranker", "enabled"),
        "RAG_RERANKER_TOP_K": ("reranker", "top_k"),
        "RAG_LLM_MODEL": ("generation", "model_name"),
        "RAG_LLM_BASE_URL": ("generation", "base_url"),
        "RAG_LLM_TEMPERATURE": ("generation", "temperature"),
        "RAG_LLM_MAX_TOKENS": ("generation", "max_tokens"),
        "RAG_API_HOST": ("api", "host"),
        "RAG_API_PORT": ("api", "port"),
        "RAG_LOG_LEVEL": ("logging", "level"),
    }

    for env_key, (section, key) in env_map.items():
        value = os.environ.get(env_key)
        if value is not None:
            data.setdefault(section, {})[key] = value
            logger.debug("env_override_applied", env_key=env_key, section=section, key=key)

    return data


def load_settings(config_path: str | Path | None = None) -> Settings:
    """Load settings from YAML config file with environment variable overrides.

    Args:
        config_path: Path to config YAML file. Defaults to ./config.yaml.

    Returns:
        Validated Settings instance.
    """
    path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
    data: dict[str, Any] = {}

    if path.exists():
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        logger.info("config_loaded", path=str(path))
    else:
        logger.warning("config_not_found", path=str(path), using="defaults")

    data = _apply_env_overrides(data)
    return Settings(**data)
