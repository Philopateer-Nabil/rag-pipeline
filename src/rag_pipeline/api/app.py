"""FastAPI application factory."""

from __future__ import annotations

from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI

from rag_pipeline import __version__
from rag_pipeline.api.routes import router, set_pipeline
from rag_pipeline.config import Settings, load_settings
from rag_pipeline.pipeline import RAGPipeline

logger = structlog.get_logger()


def create_app(settings: Settings | None = None, config_path: str | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        settings: Pre-built Settings. If None, loads from config_path.
        config_path: Path to config YAML.

    Returns:
        Configured FastAPI application.
    """
    settings = settings or load_settings(config_path)
    pipeline = RAGPipeline(settings=settings)
    set_pipeline(pipeline)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("api_startup", version=__version__)
        try:
            _ = pipeline.vectorstore
            logger.info("vectorstore_preloaded")
        except FileNotFoundError:
            logger.warning("no_existing_index", hint="Ingest documents to create the index")
        yield

    app = FastAPI(
        title="RAG Pipeline API",
        description="Retrieval-Augmented Generation pipeline with FAISS and Ollama",
        version=__version__,
        lifespan=lifespan,
    )

    app.include_router(router)
    return app
