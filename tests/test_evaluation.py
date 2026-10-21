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
            generation_attempts=1,
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
        raise GenerationError(
            "provider failed",
            attempts=2,
            provider_status_code=400,
            provider_error_code="json_validate_failed",
        )


def dataset() -> EvaluationDataset:
    return EvaluationDataset(
        schema_version=1,
        name="test-dataset",
        season=2026,
        cases=(
            EvaluationCase(
                id="supported-a",
                question="Question A?",
                tags=("baseline",),
                expected_document_ids=("doc-a",),
                expected_abstained=False,
            ),
            EvaluationCase(
                id="supported-b",
                question="Question B?",
                tags=("baseline", "multi_evidence"),
                expected_document_ids=("doc-b", "doc-c"),
                expected_abstained=False,
            ),
            EvaluationCase(
                id="unsupported",
                question="Question C?",
                tags=("near_domain",),
                expected_abstained=True,
            ),
        ),
    )


def test_dataset_rejects_conflicting_expectations_and_duplicate_ids() -> None:
    with pytest.raises(ValidationError, match="cannot require supporting documents"):
        EvaluationCase(
            id="invalid",
            question="Question?",
            tags=("baseline",),
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
                    tags=("baseline",),
                    expected_document_ids=("doc-a",),
                    expected_abstained=False,
                ),
                EvaluationCase(
                    id="duplicate",
                    question="Question B?",
                    tags=("baseline",),
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
    assert report.retrieval_metrics_by_tag["baseline"].model_dump() == {
        "evaluated_cases": 2,
        "hit_rate_at_k": 1.0,
        "recall_at_k": 0.75,
        "mean_reciprocal_rank": 0.75,
    }
    assert report.retrieval_metrics_by_tag["multi_evidence"].recall_at_k == 0.5
    assert report.retrieval_metrics_by_tag["near_domain"].evaluated_cases == 0
    assert report.answer_metrics is not None
    assert report.answer_metrics.model_dump() == {
        "evaluated_cases": 3,
        "generation_failures": 0,
        "retried_cases": 0,
        "generation_attempts": 3,
        "abstention_accuracy": pytest.approx(2 / 3),
        "expected_document_coverage": 0.5,
        "citation_document_precision": 1.0,
        "mean_latency_ms": 100.0,
        "input_tokens": 30,
        "output_tokens": 15,
        "total_tokens": 45,
    }
    assert report.answer_metrics_by_tag is not None
    assert report.answer_metrics_by_tag["baseline"].abstention_accuracy == 0.5
    assert report.answer_metrics_by_tag["near_domain"].abstention_accuracy == 1.0


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
                tags=("near_domain",),
                expected_abstained=True,
            ),
        ),
    )
    retriever = StubRetriever({"Unsupported question?": (match("noise", score=0.5),)})

    report = EvaluationRunner(retriever, FailingAnswerer()).run(
        benchmark,
        top_k=2,
        evaluate_answers=True,
    )

    assert report.retrieval_metrics.evaluated_cases == 0
    assert report.retrieval_metrics.hit_rate_at_k == 0.0
    assert report.answer_metrics is not None
    assert report.answer_metrics.generation_failures == 1
    assert report.answer_metrics.retried_cases == 1
    assert report.answer_metrics.generation_attempts == 2
    assert report.answer_metrics.abstention_accuracy == 0.0
    assert report.answer_metrics.mean_latency_ms is None
    assert report.answer_cases is not None
    assert report.answer_cases[0].generation_error == "provider failed"
    assert report.answer_cases[0].provider_status_code == 400
    assert report.answer_cases[0].provider_error_code == "json_validate_failed"


def test_evaluation_case_requires_unique_normalized_tags() -> None:
    with pytest.raises(ValidationError, match="evaluation tags must be unique"):
        EvaluationCase(
            id="duplicate-tags",
            question="Question?",
            tags=("baseline", "baseline"),
            expected_abstained=True,
        )

    with pytest.raises(ValidationError, match="string_pattern_mismatch"):
        EvaluationCase(
            id="invalid-tag",
            question="Question?",
            tags=("Multi Evidence",),
            expected_abstained=True,
        )


def test_repository_evaluation_dataset_is_valid() -> None:
    from tracerag.evaluation.dataset import load_evaluation_dataset

    path = Path(__file__).parents[1] / "evals" / "nfl-rules.json"
    loaded = load_evaluation_dataset(path)

    assert loaded.name == "nfl-rules-v3.1"
    assert len(loaded.cases) == 50
    assert {tag for case in loaded.cases for tag in case.tags} == {
        "adversarial",
        "ambiguous",
        "baseline",
        "confounder",
        "multi_evidence",
        "near_domain",
        "paraphrase",
    }


def review_fixture():
    """Two answerable outcomes: a wrong ruling with right citations, and an abstention."""
    from tracerag.evaluation.review import create_review

    benchmark = dataset()
    benchmark = benchmark.model_copy(
        update={
            "cases": tuple(
                case.model_copy(
                    update={
                        "answer_rubric": (
                            "A touchdown is worth six points.",
                            "No unsupported rule claims.",
                        ),
                    }
                )
                if not case.expected_abstained
                else case
                for case in benchmark.cases
            ),
        }
    )
    retriever = StubRetriever({case.question: () for case in benchmark.cases})
    wrong = answer("Question A?", abstained=False, cited_document_ids=("doc-a",))
    wrong = wrong.model_copy(update={"ruling": "A touchdown is worth ten points."})
    report = EvaluationRunner(
        retriever,
        StubAnswerer(
            {
                "Question A?": wrong,
                "Question B?": answer("Question B?", abstained=True),
                "Question C?": answer("Question C?", abstained=True),
            }
        ),
    ).run(benchmark, top_k=2, evaluate_answers=True)
    review = create_review(benchmark, report)
    review.reviewer = "Test reviewer"
    return benchmark, report, review


def test_correct_citations_do_not_imply_correct_ruling() -> None:
    from tracerag.evaluation.review import score_review

    benchmark, report, review = review_fixture()
    assert report.answer_cases[0].expected_documents_matched is True
    assert report.answer_cases[0].ruling == "A touchdown is worth ten points."
    assert review.cases[0].criteria[0].passed is None
    assert all(c.passed is False for c in review.cases[1].criteria)
    for criterion in review.cases[0].criteria:
        criterion.passed = False
        criterion.notes = "The answer says ten points; the corpus requires six."
    scored = score_review(benchmark, report, review)
    assert scored["answer_correctness"] == 0.0
    assert scored["evaluated_cases"] == 2
    assert scored["cases"][0]["failures"]


def test_review_scores_partial_credit_and_requires_all_criteria_for_correctness() -> None:
    from tracerag.evaluation.review import score_review

    benchmark, report, review = review_fixture()
    review.cases[0].criteria[0].passed = False
    review.cases[0].criteria[0].notes = "Wrong point value."
    review.cases[0].criteria[1].passed = True
    scored = score_review(benchmark, report, review)
    assert scored["answer_correctness"] == 0.0
    assert scored["mean_criterion_score"] == 0.25
    assert scored["cases"][0]["criterion_score"] == 0.5


@pytest.mark.parametrize(
    "change, message",
    [
        ("incomplete", "incomplete review"),
        ("reviewer", "reviewer must be identified"),
        ("missing_case", "every answerable case"),
        ("duplicate_case", "every answerable case"),
        ("rubric", "answer text or rubric changed"),
        ("answer", "answer text or rubric changed"),
        ("fingerprint", "fingerprints do not match"),
        ("abstention_pass", "cannot pass correctness"),
        ("no_reason", "failed criterion requires notes"),
    ],
)
def test_review_rejects_incomplete_or_mismatched_grades(change: str, message: str) -> None:
    from tracerag.evaluation.review import score_review

    benchmark, report, review = review_fixture()
    if change != "incomplete":
        for criterion in review.cases[0].criteria:
            criterion.passed = True
    if change == "reviewer":
        review.reviewer = " "
    elif change == "missing_case":
        review.cases = review.cases[:1]
    elif change == "duplicate_case":
        review.cases = (review.cases[0], review.cases[0])
    elif change == "rubric":
        review.cases[0].criteria[0].requirement = "A touchdown is worth ten points."
    elif change == "answer":
        review.cases[0].ruling = "Edited answer."
    elif change == "fingerprint":
        review.report_fingerprint = "changed"
    elif change == "abstention_pass":
        review.cases[1].criteria[0].passed = True
    elif change == "no_reason":
        review.cases[0].criteria[0].passed = False
    with pytest.raises(ValueError, match=message):
        score_review(benchmark, report, review)


def test_review_cli_roundtrip_is_offline(tmp_path, monkeypatch, capsys) -> None:
    import json
    import sys

    from tracerag import cli
    from tracerag.evaluation.review import AnswerReview

    benchmark, report, _ = review_fixture()
    dataset_path = tmp_path / "dataset.json"
    report_path = tmp_path / "report.json"
    review_path = tmp_path / "review.json"
    dataset_path.write_text(benchmark.model_dump_json())
    report_path.write_text(report.model_dump_json())
    arguments = ["tracerag-review", "--dataset", str(dataset_path), "--report", str(report_path)]
    monkeypatch.setattr(sys, "argv", arguments)
    cli.review_answers()
    review = AnswerReview.model_validate_json(capsys.readouterr().out)
    review.reviewer = "Test reviewer"
    for criterion in review.cases[0].criteria:
        criterion.passed = True
    review_path.write_text(review.model_dump_json())
    monkeypatch.setattr(sys, "argv", [*arguments, "--review", str(review_path)])
    cli.review_answers()
    assert json.loads(capsys.readouterr().out)["answer_correctness"] == 0.5


def test_review_rejects_retrieval_only_report_and_missing_text() -> None:
    from tracerag.evaluation.review import create_review

    benchmark, report, _ = review_fixture()
    with pytest.raises(ValueError, match="answer evaluation report"):
        create_review(benchmark, report.model_copy(update={"answer_cases": None}))
    cases = report.answer_cases
    old_case = cases[0].model_copy(update={"ruling": None, "explanation": None})
    with pytest.raises(ValueError, match="missing generated text"):
        create_review(benchmark, report.model_copy(update={"answer_cases": (old_case, *cases[1:])}))
    wrong_report = report.model_copy(update={"dataset": "another-version"})
    with pytest.raises(ValueError, match="does not match"):
        create_review(benchmark, wrong_report)


def test_generation_failure_also_fails_answer_correctness() -> None:
    from tracerag.evaluation.review import create_review, score_review

    benchmark, report, _ = review_fixture()
    cases = report.answer_cases
    failed = cases[0].model_copy(
        update={
            "generation_error": "rate limited",
            "actual_abstained": None,
            "ruling": None,
            "explanation": None,
        }
    )
    report = report.model_copy(update={"answer_cases": (failed, *cases[1:])})
    review = create_review(benchmark, report)
    review.reviewer = "Test reviewer"
    assert score_review(benchmark, report, review)["answer_correctness"] == 0.0


def test_v3_requires_rubrics_for_all_answerable_cases() -> None:
    from tracerag.evaluation.dataset import load_evaluation_dataset

    loaded = load_evaluation_dataset(Path(__file__).parents[1] / "evals" / "nfl-rules.json")
    assert sum(bool(case.answer_rubric) for case in loaded.cases) == 40
    assert all(len(case.answer_rubric) >= 2 for case in loaded.cases if not case.expected_abstained)
    with pytest.raises(ValidationError, match="require an answer rubric"):
        EvaluationDataset.model_validate({**dataset().model_dump(), "schema_version": 3})


@pytest.mark.parametrize(
    "field, value",
    [
        ("question", "A changed question under the same case ID?"),
        ("answer_rubric", ("Changed requirement.",)),
        ("expected_document_ids", ("a-different-document",)),
    ],
)
def test_review_rejects_changed_dataset_even_when_name_and_case_ids_match(field, value) -> None:
    from tracerag.evaluation.review import create_review

    benchmark, report, _ = review_fixture()
    changed = benchmark.cases[0].model_copy(update={field: value})
    benchmark = benchmark.model_copy(update={"cases": (changed, *benchmark.cases[1:])})
    with pytest.raises(ValueError, match="report dataset fingerprint differs"):
        create_review(benchmark, report)


def test_model_assisted_review_is_identified_in_score() -> None:
    from tracerag.evaluation.review import score_review

    benchmark, report, review = review_fixture()
    review.review_kind = "model_assisted"
    review.reviewer = "Codex (model-assisted review)"
    for criterion in review.cases[0].criteria:
        criterion.passed = True
    scored = score_review(benchmark, report, review)
    assert scored["scoring_method"] == "model_assisted_rubric_review"
    assert scored["reviewer"] == review.reviewer
