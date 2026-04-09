"""Tests for configuration loading."""

from __future__ import annotations

from pathlib import Path

from rag_pipeline.config import Settings, load_settings


class TestSettings:
    def test_default_settings(self):
        s = Settings()
        assert s.chunking.strategy == "recursive"
        assert s.chunking.chunk_size == 512
        assert s.embedding.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert s.retrieval.top_k == 5

    def test_load_from_yaml(self, tmp_path: Path):
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(
            "chunking:\n  chunk_size: 256\n  strategy: fixed\n"
            "retrieval:\n  top_k: 10\n"
        )
        settings = load_settings(config_file)
        assert settings.chunking.chunk_size == 256
        assert settings.chunking.strategy == "fixed"
        assert settings.retrieval.top_k == 10

    def test_load_missing_file_uses_defaults(self, tmp_path: Path):
        settings = load_settings(tmp_path / "nonexistent.yaml")
        assert settings.chunking.chunk_size == 512

    def test_env_override(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("RAG_CHUNK_SIZE", "1024")
        settings = load_settings(tmp_path / "nonexistent.yaml")
        assert settings.chunking.chunk_size == 1024
        monkeypatch.delenv("RAG_CHUNK_SIZE")
