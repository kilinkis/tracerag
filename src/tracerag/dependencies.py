"""Construction of replaceable application services."""

from functools import lru_cache

from tracerag.config import get_settings
from tracerag.embeddings import FastEmbedProvider
from tracerag.retrieval.service import Retriever
from tracerag.retrieval.store import PgVectorStore


@lru_cache
def get_embedder() -> FastEmbedProvider:
    settings = get_settings()
    return FastEmbedProvider(
        model_name=settings.embedding_model,
        dimensions=settings.embedding_dimensions,
        cache_dir=settings.embedding_cache_path,
    )


@lru_cache
def get_vector_store() -> PgVectorStore:
    settings = get_settings()
    return PgVectorStore(
        settings.database_url,
        dimensions=settings.embedding_dimensions,
    )


@lru_cache
def get_retriever() -> Retriever:
    return Retriever(get_embedder(), get_vector_store())
