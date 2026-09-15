"""Unit coverage for evaluation dataset validation and metric calculation."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from tracerag.answering.generator import GenerationError
from tracerag.answering.models import AnswerCitation, AnswerResult, AnswerTrace
from tracerag.evaluation.models import EvaluationCase, EvaluationDataset
from tracerag.evaluation.runner import EvaluationRunner
from tracerag.retrieval.models import RetrievalMatch, RetrievalResult


def match(document_id: str, *, score: float) -> RetrievalMatch:
    return RetrievalMatch(
        chunk_id=document_id.ljust(16, "0"),
        document_id=document_id,
        document_title=document_id.title(),
        season=2026,
        heading_path=(document_id.title(),),
        text="Evidence",
        source_title="Rulebook",
        source_url="https://example.com/rules",
        rule_references=("1-1",),
        relative_path=f"{document_id}.md",
        score=score,
    )


def answer(
    question: str,
    *,
    abstained: bool,
    cited_document_ids: tuple[str, ...] = (),
) -> AnswerResult:
    citations = tuple(
        AnswerCitation(
            chunk_id=document_id.ljust(16, "0"),
            document_id=document_id,
            document_title=document_id.title(),
            heading_path=(document_id.title(),),
            source_title="Rulebook",
            source_url="https://example.com/rules",
            rule_references=("1-1",),
        )
        for document_id in cited_document_ids
    )
    return AnswerResult(
        question=question,
        season=2026,
        ruling=None if abstained else "Supported ruling",
        explanation="Insufficient evidence" if abstained else "Grounded explanation",
        abstained=abstained,
        abstention_reason="Insufficient evidence" if abstained else None,
        citations=citations,
        evidence=(),
        trace=AnswerTrace(
            embedding_model="test-embedding-model",
            generation_model="test-generation-model",
            retrieved_chunks=2,
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
            latency_ms=100,
        ),
    )


class StubRetriever:
    def __init__(self, results: dict[str, tuple[RetrievalMatch, ...]]) -> None:
        self.results = results

    def search(self, question: str, *, season: int, limit: int) -> RetrievalResult:
        assert season == 2026
        return RetrievalResult(
            question=question,
            season=season,
            embedding_model="test-embedding-model",
            matches=self.results[question][:limit],
        )


class StubAnswerer:
    def __init__(self, results: dict[str, AnswerResult]) -> None:
        self.results = results

    def answer(self, question: str, *, season: int, limit: int) -> AnswerResult:
        assert season == 2026
        assert limit == 2
        return self.results[question]


class FailingAnswerer:
    def answer(self, question: str, *, season: int, limit: int) -> AnswerResult:
        raise GenerationError("provider failed")


def dataset() -> EvaluationDataset:
    return EvaluationDataset(
        schema_version=1,
        name="test-dataset",
        season=2026,
        cases=(
            EvaluationCase(
                id="supported-a",
                question="Question A?",
                expected_document_ids=("doc-a",),
                expected_abstained=False,
            ),
            EvaluationCase(
                id="supported-b",
                question="Question B?",
                expected_document_ids=("doc-b", "doc-c"),
                expected_abstained=False,
            ),
            EvaluationCase(
                id="unsupported",
                question="Question C?",
                expected_abstained=True,
            ),
        ),
    )


def test_dataset_rejects_conflicting_expectations_and_duplicate_ids() -> None:
    with pytest.raises(ValidationError, match="cannot require supporting documents"):
        EvaluationCase(
            id="invalid",
            question="Question?",
            expected_document_ids=("doc-a",),
            expected_abstained=True,
        )

    with pytest.raises(ValidationError, match="identifiers must be unique"):
        EvaluationDataset(
            schema_version=1,
            name="invalid",
            season=2026,
            cases=(
                EvaluationCase(
                    id="duplicate",
                    question="Question A?",
                    expected_document_ids=("doc-a",),
                    expected_abstained=False,
                ),
                EvaluationCase(
                    id="duplicate",
                    question="Question B?",
                    expected_document_ids=("doc-b",),
                    expected_abstained=False,
                ),
            ),
        )


def test_runner_measures_retrieval_abstention_citations_and_usage() -> None:
    benchmark = dataset()
    retriever = StubRetriever(
        {
            "Question A?": (match("noise", score=0.9), match("doc-a", score=0.8)),
            "Question B?": (match("doc-b", score=0.9), match("noise", score=0.8)),
            "Question C?": (match("noise", score=0.7),),
        }
    )
    answerer = StubAnswerer(
        {
            "Question A?": answer("Question A?", abstained=False, cited_document_ids=("doc-a",)),
            "Question B?": answer("Question B?", abstained=True),
            "Question C?": answer("Question C?", abstained=True),
        }
    )

    report = EvaluationRunner(retriever, answerer).run(
        benchmark,
        top_k=2,
        evaluate_answers=True,
    )

    assert report.embedding_model == "test-embedding-model"
    assert report.retrieval_metrics.model_dump() == {
        "evaluated_cases": 2,
        "hit_rate_at_k": 1.0,
        "recall_at_k": 0.75,
        "mean_reciprocal_rank": 0.75,
    }
    assert report.retrieval_cases[0].first_relevant_rank == 2
    assert report.retrieval_cases[2].recall_at_k is None
    assert report.answer_metrics is not None
    assert report.answer_metrics.model_dump() == {
            "evaluated_cases": 3,
            "generation_failures": 0,
            "abstention_accuracy": pytest.approx(2 / 3),
            "grounded_ruling_accuracy": 0.5,
            "citation_document_precision": 1.0,
            "mean_latency_ms": 100.0,
            "input_tokens": 30,
            "output_tokens": 15,
            "total_tokens": 45,
        }


def test_runner_requires_an_answerer_for_answer_evaluation() -> None:
    with pytest.raises(ValueError, match="requires an answerer"):
        EvaluationRunner(StubRetriever({})).run(
            dataset(),
            top_k=3,
            evaluate_answers=True,
        )


def test_runner_records_generation_failures_and_continues() -> None:
    benchmark = EvaluationDataset(
        schema_version=1,
        name="failure-dataset",
        season=2026,
        cases=(
            EvaluationCase(
                id="unsupported",
                question="Unsupported question?",
                expected_abstained=True,
            ),
        ),
    )
    retriever = StubRetriever(
        {"Unsupported question?": (match("noise", score=0.5),)}
    )

    report = EvaluationRunner(retriever, FailingAnswerer()).run(
        benchmark,
        top_k=2,
        evaluate_answers=True,
    )

    assert report.retrieval_metrics.evaluated_cases == 0
    assert report.retrieval_metrics.hit_rate_at_k == 0.0
    assert report.answer_metrics is not None
    assert report.answer_metrics.generation_failures == 1
    assert report.answer_metrics.abstention_accuracy == 0.0
    assert report.answer_metrics.mean_latency_ms is None
    assert report.answer_cases is not None
    assert report.answer_cases[0].generation_error == "provider failed"


def test_repository_evaluation_dataset_is_valid() -> None:
    from tracerag.evaluation.dataset import load_evaluation_dataset

    path = Path(__file__).parents[1] / "evals" / "nfl-rules.json"
    loaded = load_evaluation_dataset(path)

    assert loaded.name == "nfl-rules-v1"
    assert len(loaded.cases) >= 30
