"""Typed contracts for retrieval and answer evaluation."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator

NonEmptyString = Annotated[str, Field(min_length=1)]
Score = Annotated[float, Field(ge=0.0, le=1.0)]


class EvaluationCase(BaseModel):
    """One question with its expected corpus-level outcome."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: NonEmptyString
    question: NonEmptyString
    expected_document_ids: tuple[NonEmptyString, ...] = ()
    expected_abstained: bool

    @model_validator(mode="after")
    def validate_expectations(self) -> "EvaluationCase":
        if len(set(self.expected_document_ids)) != len(self.expected_document_ids):
            raise ValueError("expected document identifiers must be unique")
        if self.expected_abstained and self.expected_document_ids:
            raise ValueError("an abstention case cannot require supporting documents")
        if not self.expected_abstained and not self.expected_document_ids:
            raise ValueError("an answerable case requires at least one supporting document")
        return self


class EvaluationDataset(BaseModel):
    """A versioned set of questions evaluated against one corpus season."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = Field(ge=1)
    name: NonEmptyString
    season: int = Field(ge=2000)
    cases: tuple[EvaluationCase, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_case_ids(self) -> "EvaluationDataset":
        case_ids = [case.id for case in self.cases]
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("evaluation case identifiers must be unique")
        return self


class RetrievalCaseResult(BaseModel):
    """Retrieval measurements for one evaluation case."""

    model_config = ConfigDict(frozen=True)

    case_id: str
    question: str
    expected_document_ids: tuple[str, ...]
    retrieved_document_ids: tuple[str, ...]
    recall_at_k: Score | None
    reciprocal_rank: Score | None
    first_relevant_rank: int | None = Field(default=None, ge=1)


class RetrievalMetrics(BaseModel):
    """Aggregate retrieval quality over answerable cases."""

    model_config = ConfigDict(frozen=True)

    evaluated_cases: int = Field(ge=0)
    hit_rate_at_k: Score
    recall_at_k: Score
    mean_reciprocal_rank: Score


class AnswerCaseResult(BaseModel):
    """Abstention and citation measurements for one generated ruling."""

    model_config = ConfigDict(frozen=True)

    case_id: str
    expected_abstained: bool
    actual_abstained: bool | None
    abstention_correct: bool
    cited_document_ids: tuple[str, ...]
    citation_document_precision: Score | None
    grounded_ruling_correct: bool | None
    generation_error: str | None
    latency_ms: float | None = Field(default=None, ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)


class AnswerMetrics(BaseModel):
    """Aggregate answer behavior and provider usage."""

    model_config = ConfigDict(frozen=True)

    evaluated_cases: int = Field(ge=0)
    generation_failures: int = Field(ge=0)
    abstention_accuracy: Score
    grounded_ruling_accuracy: Score
    citation_document_precision: Score | None
    mean_latency_ms: float | None = Field(default=None, ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)


class EvaluationReport(BaseModel):
    """Machine-readable benchmark report."""

    model_config = ConfigDict(frozen=True)

    dataset: str
    season: int
    top_k: int = Field(ge=1)
    embedding_model: str
    retrieval_metrics: RetrievalMetrics
    retrieval_cases: tuple[RetrievalCaseResult, ...]
    answer_metrics: AnswerMetrics | None
    answer_cases: tuple[AnswerCaseResult, ...] | None
