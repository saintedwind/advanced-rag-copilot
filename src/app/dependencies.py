"""Explicit local defaults; no paid API calls or credentials required."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class AppSettings(BaseSettings):
    app_env: str = "local"
    qdrant_url: str = ""
    qdrant_path: str = str(ROOT / ".runtime/qdrant")
    qdrant_api_key: str = ""
    qdrant_collection: str = "knowledge_v2"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_cache: str = str(ROOT / ".runtime/models")
    reranker_provider: str = "none"
    reranker_model: str = "ms-marco-TinyBERT-L-2-v2"
    llm_provider: str = "extractive"
    local_llm_url: str = "http://127.0.0.1:8081/v1/chat/completions"
    top_k_retrieval: int = 10
    top_n_rerank: int = 4
    hybrid_alpha: float = Field(default=0.5, ge=0, le=1)
    evidence_threshold: float = Field(default=0.55, ge=0, le=1)
    max_context_chars: int = 3000
    source_root: str = str(ROOT / "data/sample_docs")
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @model_validator(mode="after")
    def validate_providers(self):
        if self.llm_provider not in {"extractive", "local"}:
            raise ValueError("Use extractive or local LLM provider")
        if self.reranker_provider not in {"none", "flashrank"}:
            raise ValueError("Unknown reranker")
        if self.local_llm_url != "http://127.0.0.1:8081/v1/chat/completions":
            raise ValueError("Local inference must use fixed loopback endpoint")
        return self


@lru_cache()
def get_settings():
    return AppSettings()
