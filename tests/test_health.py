"""API smoke tests.

HTTPX2 ASGI transport pattern source:
https://pydantic.dev/docs/httpx2/advanced/transports/#asgi-transport
"""

import asyncio

import httpx2

from tracerag.api import app
from tracerag.dependencies import get_retriever
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
