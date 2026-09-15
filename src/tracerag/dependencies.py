"""Construction of replaceable application services."""

from functools import lru_cache

from tracerag.answering.generator import GroqAnswerGenerator
from tracerag.answering.service import AnswerService
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


@lru_cache
def get_answer_generator() -> GroqAnswerGenerator:
    settings = get_settings()
    return GroqAnswerGenerator(
        api_key=(
            settings.groq_api_key.get_secret_value() if settings.groq_api_key is not None else None
        ),
        model_name=settings.generation_model,
        timeout_seconds=settings.generation_timeout_seconds,
        max_completion_tokens=settings.generation_max_completion_tokens,
        reasoning_effort=settings.generation_reasoning_effort,
        structured_output_retries=settings.generation_structured_output_retries,
    )


@lru_cache
def get_answer_service() -> AnswerService:
    return AnswerService(get_retriever(), get_answer_generator())
