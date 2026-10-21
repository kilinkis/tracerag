"""Construction of replaceable application services."""

from functools import lru_cache

from groq import Groq

from tracerag.answering.generator import SYSTEM_PROMPT, GroqAnswerGenerator
from tracerag.answering.prompts import CONCISE_EXPLANATION
from tracerag.answering.service import AnswerService
from tracerag.answering.verification import GroqAnswerVerifier, VerifiedAnswerGenerator
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
        system_prompt=SYSTEM_PROMPT
        + (CONCISE_EXPLANATION if settings.generation_prompt_profile == "concise" else ""),
    )


@lru_cache
def get_answer_service() -> AnswerService:
    settings = get_settings()
    generator = get_answer_generator()
    if settings.verification_enabled:
        key = settings.groq_api_key
        if key is None:
            raise ValueError("Verification requires TRACERAG_GROQ_API_KEY")
        generator = VerifiedAnswerGenerator(
            generator,
            GroqAnswerVerifier(
                Groq(
                    api_key=key.get_secret_value(),
                    timeout=settings.generation_timeout_seconds,
                    max_retries=0,
                ),
                settings.verification_model,
            ),
        )
    return AnswerService(get_retriever(), generator)
