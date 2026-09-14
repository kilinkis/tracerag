"""Application configuration loaded from environment variables.

Pattern source:
https://docs.pydantic.dev/latest/concepts/pydantic_settings/
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Validated runtime configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="TRACERAG_",
        extra="ignore",
    )

    app_name: str = "TraceRAG API"
    database_url: str = "postgresql://tracerag:tracerag@localhost:5432/tracerag"
    corpus_path: Path = Path("data/nfl")
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dimensions: int = 384
    embedding_cache_path: Path = Path(".cache/fastembed")
    corpus_season: int = 2026
    retrieval_top_k: int = 5
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    """Return one immutable-by-convention settings instance per process."""

    return Settings()
