"""HTTP API entrypoint.

FastAPI application and route pattern source:
https://fastapi.tiangolo.com/tutorial/first-steps/
"""

from fastapi import FastAPI
from pydantic import BaseModel

from tracerag import __version__
from tracerag.config import get_settings

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=__version__,
    description="Evidence-first retrieval augmented generation.",
)


class HealthResponse(BaseModel):
    """Health check payload."""

    status: str
    service: str
    version: str


@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health() -> HealthResponse:
    """Report whether the API process is available."""

    return HealthResponse(status="ok", service="tracerag", version=__version__)

