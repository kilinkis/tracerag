"""API smoke tests.

HTTPX2 ASGI transport pattern source:
https://pydantic.dev/docs/httpx2/advanced/transports/#asgi-transport
"""

import asyncio

import httpx2
import pytest

from tracerag.answering.generator import GenerationError, GenerationUnavailableError
from tracerag.answering.models import AnswerResult, AnswerTrace
from tracerag.api import app
from tracerag.dependencies import get_answer_service, get_retriever
from tracerag.retrieval.models import RetrievalResult


async def request(method: str, path: str, **kwargs: object) -> httpx2.Response:
    transport = httpx2.ASGITransport(app=app)
    async with httpx2.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        return await client.request(method, path, **kwargs)


async def get(path: str) -> httpx2.Response:
    return await request("GET", path)


def test_health_reports_service_version() -> None:
    response = asyncio.run(get("/health"))

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "tracerag",
        "version": "0.1.0",
    }


def test_corpus_status_reports_loaded_snapshot() -> None:
    response = asyncio.run(get("/corpus/status"))

    assert response.status_code == 200
    assert response.json() == {
        "season": 2026,
        "documents": 6,
        "chunks": 21,
        "rule_references": 23,
    }


class StubRetriever:
    def search(self, question: str, *, season: int, limit: int) -> RetrievalResult:
        assert season == 2026
        assert limit == 3
        return RetrievalResult(
            question=question,
            season=season,
            embedding_model="test-embedding-model",
            matches=(),
        )


def test_retrieval_endpoint_returns_ranked_evidence_contract() -> None:
    app.dependency_overrides[get_retriever] = StubRetriever
    try:
        response = asyncio.run(
            request(
                "POST",
                "/retrieval/search",
                json={"question": "What makes a catch complete?", "top_k": 3},
            )
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {
        "question": "What makes a catch complete?",
        "season": 2026,
        "embedding_model": "test-embedding-model",
        "matches": [],
    }


def test_retrieval_endpoint_rejects_blank_question() -> None:
    response = asyncio.run(
        request(
            "POST",
            "/retrieval/search",
            json={"question": "   "},
        )
    )

    assert response.status_code == 422


class StubAnswerService:
    def answer(self, question: str, *, season: int, limit: int) -> AnswerResult:
        assert season == 2026
        assert limit == 3
        return AnswerResult(
            question=question,
            season=season,
            ruling=None,
            explanation="The available evidence does not resolve the scenario.",
            abstained=True,
            abstention_reason="The available evidence does not resolve the scenario.",
            citations=(),
            evidence=(),
            trace=AnswerTrace(
                embedding_model="test-embedding-model",
                generation_model="test-generation-model",
                retrieved_chunks=0,
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                generation_attempts=1,
                latency_ms=1.5,
            ),
        )


def test_answer_endpoint_returns_grounded_contract() -> None:
    app.dependency_overrides[get_answer_service] = StubAnswerService
    try:
        response = asyncio.run(
            request(
                "POST",
                "/answers",
                json={"question": "Was the catch complete?", "top_k": 3},
            )
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {
        "question": "Was the catch complete?",
        "season": 2026,
        "ruling": None,
        "explanation": "The available evidence does not resolve the scenario.",
        "abstained": True,
        "abstention_reason": "The available evidence does not resolve the scenario.",
        "citations": [],
        "evidence": [],
        "trace": {
            "embedding_model": "test-embedding-model",
            "generation_model": "test-generation-model",
            "retrieved_chunks": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "generation_attempts": 1,
            "latency_ms": 1.5,
        },
    }


class FailingAnswerService:
    def __init__(self, error: GenerationError) -> None:
        self.error = error

    def answer(self, question: str, *, season: int, limit: int) -> AnswerResult:
        raise self.error


@pytest.mark.parametrize(
    ("error", "expected_status", "expected_detail"),
    (
        (
            GenerationUnavailableError("missing key"),
            503,
            "answer generation is not configured",
        ),
        (GenerationError("provider failed"), 502, "answer generation provider failed"),
    ),
)
def test_answer_endpoint_maps_generation_failures(
    error: GenerationError,
    expected_status: int,
    expected_detail: str,
) -> None:
    app.dependency_overrides[get_answer_service] = lambda: FailingAnswerService(error)
    try:
        response = asyncio.run(
            request(
                "POST",
                "/answers",
                json={"question": "Was the catch complete?"},
            )
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == expected_status
    assert response.json() == {"detail": expected_detail}
