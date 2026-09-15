"""Typed contracts for grounded ruling generation and API responses.

Official validation and JSON Schema sources:
https://pydantic.dev/docs/validation/latest/concepts/validators/#model-validators
https://pydantic.dev/docs/validation/latest/concepts/json_schema/
"""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator

from tracerag.retrieval.models import RetrievalMatch

NonEmptyString = Annotated[str, Field(min_length=1)]


class AnswerDraft(BaseModel):
    """Provider output before citations are resolved against retrieved evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    ruling: NonEmptyString | None
    explanation: NonEmptyString
    cited_chunk_ids: tuple[str, ...]
    abstained: bool
    abstention_reason: NonEmptyString | None

    @model_validator(mode="after")
    def validate_draft_invariants(self) -> "AnswerDraft":
        if self.abstained:
            if self.ruling is not None or self.cited_chunk_ids or not self.abstention_reason:
                raise ValueError(
                    "an abstention draft requires a reason and cannot contain a ruling or citations"
                )
        elif self.ruling is None or not self.cited_chunk_ids or self.abstention_reason is not None:
            raise ValueError(
                "a supported draft requires a ruling and citations without an abstention reason"
            )
        return self


class GenerationUsage(BaseModel):
    """Token usage reported by the generation provider."""

    model_config = ConfigDict(frozen=True)

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)


class GenerationResult(BaseModel):
    """Validated output and metadata from one generation request."""

    model_config = ConfigDict(frozen=True)

    model: str
    draft: AnswerDraft
    attempts: int = Field(ge=1)
    usage: GenerationUsage


class AnswerCitation(BaseModel):
    """Trusted citation resolved from one retrieved corpus passage."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    document_id: str
    document_title: str
    heading_path: tuple[str, ...]
    source_title: str
    source_url: HttpUrl
    rule_references: tuple[str, ...]


class AnswerTrace(BaseModel):
    """Observable retrieval and generation metadata for one ruling."""

    model_config = ConfigDict(frozen=True)

    embedding_model: str
    generation_model: str
    retrieved_chunks: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    generation_attempts: int = Field(ge=0)
    latency_ms: float = Field(ge=0)


class AnswerResult(BaseModel):
    """A supported ruling or an explicit abstention with its evidence trace."""

    model_config = ConfigDict(frozen=True)

    question: str
    season: int
    ruling: str | None
    explanation: str
    abstained: bool
    abstention_reason: str | None
    citations: tuple[AnswerCitation, ...]
    evidence: tuple[RetrievalMatch, ...]
    trace: AnswerTrace

    @model_validator(mode="after")
    def validate_grounding_invariants(self) -> "AnswerResult":
        if self.abstained:
            if self.ruling is not None or self.citations or not self.abstention_reason:
                raise ValueError(
                    "an abstention requires a reason and cannot contain a ruling or citations"
                )
        elif self.ruling is None or not self.citations or self.abstention_reason is not None:
            raise ValueError(
                "a supported ruling requires citations and cannot contain an abstention reason"
            )
        return self
