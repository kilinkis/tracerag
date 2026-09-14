"""API smoke tests.

HTTPX2 ASGI transport pattern source:
https://pydantic.dev/docs/httpx2/advanced/transports/#asgi-transport
"""

import asyncio

import httpx2

from tracerag.api import app


async def get_health() -> httpx2.Response:
    transport = httpx2.ASGITransport(app=app)
    async with httpx2.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        return await client.get("/health")


def test_health_reports_service_version() -> None:
    response = asyncio.run(get_health())

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "tracerag",
        "version": "0.1.0",
    }
