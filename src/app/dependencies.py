"""Application configuration and dependency container."""

import os
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    app_env: str = "development"
    log_level: str = "INFO"

    # LLM & Embeddings
    openai_api_key: str = "sk-proj-xxxxxxxxxxxxxxxxxxxxxxxx"
    openai_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    # Vector Storage
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "production_kb"

    # Reranker
    reranker_provider: str = "flashrank"
    reranker_model: str = "ms-marco-TinyBERT-L-2-v2"
    cohere_api_key: str = ""

    # Pipeline Tuning
    chunk_size: int = 600
    chunk_overlap: int = 100
    top_k_retrieval: int = 10
    top_n_rerank: int = 4
    hybrid_alpha: float = 0.5

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


@lru_cache()
def get_settings() -> AppSettings:
    """Thread-safe cached settings singleton."""
    return AppSettings()
