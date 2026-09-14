"""HTTP API entrypoint.

FastAPI application and route pattern source:
https://fastapi.tiangolo.com/tutorial/first-steps/
https://fastapi.tiangolo.com/tutorial/dependencies/#declare-the-dependency-in-the-dependant
"""

from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field, StringConstraints

from tracerag import __version__
from tracerag.answering.generator import GenerationError, GenerationUnavailableError
from tracerag.answering.models import AnswerResult
from tracerag.answering.service import AnswerService
from tracerag.config import get_settings
from tracerag.corpus.loader import chunk_corpus, load_corpus
from tracerag.dependencies import get_answer_service, get_retriever
from tracerag.retrieval.models import RetrievalResult
from tracerag.retrieval.service import Retriever

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


class RetrievalRequest(BaseModel):
    """Validated evidence search request."""

    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=3, max_length=500)]
    top_k: int | None = Field(default=None, ge=1, le=20)


class AnswerRequest(BaseModel):
    """Validated grounded-ruling request."""

    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=3, max_length=500)]
    top_k: int | None = Field(default=None, ge=1, le=20)


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


@app.post("/retrieval/search", response_model=RetrievalResult, tags=["retrieval"])
def search_evidence(
    request: RetrievalRequest,
    retriever: Annotated[Retriever, Depends(get_retriever)],
) -> RetrievalResult:
    """Rank corpus evidence for a rule question without generating an answer."""

    return retriever.search(
        request.question,
        season=settings.corpus_season,
        limit=request.top_k or settings.retrieval_top_k,
    )


@app.post("/answers", response_model=AnswerResult, tags=["answers"])
def answer_question(
    request: AnswerRequest,
    answer_service: Annotated[AnswerService, Depends(get_answer_service)],
) -> AnswerResult:
    """Generate a ruling grounded in retrieved corpus evidence."""

    try:
        return answer_service.answer(
            request.question,
            season=settings.corpus_season,
            limit=request.top_k or settings.retrieval_top_k,
        )
    except GenerationUnavailableError as exc:
        raise HTTPException(status_code=503, detail="answer generation is not configured") from exc
    except GenerationError as exc:
        raise HTTPException(status_code=502, detail="answer generation provider failed") from exc
