"""API smoke tests.

HTTPX2 ASGI transport pattern source:
https://pydantic.dev/docs/httpx2/advanced/transports/#asgi-transport
"""

import asyncio

import httpx2

from tracerag.api import app


async def get(path: str) -> httpx2.Response:
    transport = httpx2.ASGITransport(app=app)
    async with httpx2.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        return await client.get(path)


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
