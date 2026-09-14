"""Validated retrieval results returned by the baseline search pipeline."""

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class RetrievalMatch(BaseModel):
    """One evidence passage ranked for a question."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    document_id: str
    document_title: str
    season: int
    heading_path: tuple[str, ...]
    text: str
    source_title: str
    source_url: HttpUrl
    rule_references: tuple[str, ...]
    relative_path: str
    score: float = Field(ge=-1.0, le=1.0)


class RetrievalResult(BaseModel):
    """Ranked evidence and retrieval metadata for one question."""

    model_config = ConfigDict(frozen=True)

    question: str
    season: int
    embedding_model: str
    matches: tuple[RetrievalMatch, ...]
