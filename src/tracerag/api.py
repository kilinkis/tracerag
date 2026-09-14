"""HTTP API entrypoint.

FastAPI application and route pattern source:
https://fastapi.tiangolo.com/tutorial/first-steps/
"""

from fastapi import FastAPI
from pydantic import BaseModel

from tracerag import __version__
from tracerag.config import get_settings
from tracerag.corpus.loader import chunk_corpus, load_corpus

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


class CorpusStatusResponse(BaseModel):
    """Summary of the version-controlled rule corpus."""

    season: int
    documents: int
    chunks: int
    rule_references: int


@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health() -> HealthResponse:
    """Report whether the API process is available."""

    return HealthResponse(status="ok", service="tracerag", version=__version__)


@app.get("/corpus/status", response_model=CorpusStatusResponse, tags=["corpus"])
def corpus_status() -> CorpusStatusResponse:
    """Report the corpus snapshot that will back retrieval."""

    # The standard-library file reads are blocking, so FastAPI recommends a normal def handler:
    # https://fastapi.tiangolo.com/async/#in-a-hurry
    documents = load_corpus(settings.corpus_path)
    chunks = chunk_corpus(documents)
    return CorpusStatusResponse(
        season=documents[0].season,
        documents=len(documents),
        chunks=len(chunks),
        rule_references=len(
            {reference for document in documents for reference in document.rule_references}
        ),
    )
