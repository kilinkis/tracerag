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
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    """Return one immutable-by-convention settings instance per process."""

    return Settings()
