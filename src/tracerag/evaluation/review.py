"""Create and score attributed rubric reviews bound to saved benchmark outputs."""

from __future__ import annotations

import hashlib
from typing import Literal

from pydantic import BaseModel, ConfigDict, StrictBool

from tracerag.evaluation.dataset import fingerprint_dataset
from tracerag.evaluation.models import EvaluationDataset, EvaluationReport


class CriterionReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requirement: str
    passed: StrictBool | None = None
    notes: str = ""


class CaseReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    question: str
    ruling: str | None
    explanation: str | None
    criteria: tuple[CriterionReview, ...]


class AnswerReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    dataset_fingerprint: str
    report_fingerprint: str
    reviewer: str = ""
    review_kind: Literal["human", "model_assisted"] = "human"
    cases: tuple[CaseReview, ...]


def _fingerprint(value: BaseModel) -> str:
    return hashlib.sha256(value.model_dump_json().encode()).hexdigest()


def create_review(dataset: EvaluationDataset, report: EvaluationReport) -> AnswerReview:
    """Expose actual answers beside their rubrics; failures and abstentions score zero."""
    if report.dataset != dataset.name or report.season != dataset.season:
        raise ValueError("report does not match the dataset name and season")
    if report.dataset_fingerprint != fingerprint_dataset(dataset):
        raise ValueError(
            "report dataset fingerprint differs; generate a new report for this dataset"
        )
    if not report.answer_cases:
        raise ValueError("review requires an answer evaluation report")
    results = {case.case_id: case for case in report.answer_cases}
    if len(results) != len(report.answer_cases) or set(results) != {c.id for c in dataset.cases}:
        raise ValueError("report must contain exactly one answer result for every dataset case")

    reviews = []
    for case in dataset.cases:
        answer = results[case.id]
        if answer.expected_abstained != case.expected_abstained:
            raise ValueError(f"report expectations differ for {case.id}")
        if case.expected_abstained:
            continue
        if not case.answer_rubric:
            raise ValueError(f"missing answer rubric for {case.id}")
        failed = answer.generation_error is not None or answer.actual_abstained is True
        if not failed and (
            answer.actual_abstained is not False or not answer.ruling or not answer.explanation
        ):
            raise ValueError(f"missing generated text for {case.id}; generate a new answer report")
        reviews.append(
            CaseReview(
                case_id=case.id,
                question=case.question,
                ruling=answer.ruling,
                explanation=answer.explanation,
                criteria=tuple(
                    CriterionReview(
                        requirement=requirement,
                        passed=False if failed else None,
                        notes="No ruling: generation failed or system abstained." if failed else "",
                    )
                    for requirement in case.answer_rubric
                ),
            )
        )
    if not reviews:
        raise ValueError("dataset has no answerable cases to review")
    return AnswerReview(
        dataset_fingerprint=fingerprint_dataset(dataset),
        report_fingerprint=_fingerprint(report),
        cases=tuple(reviews),
    )


def score_review(
    dataset: EvaluationDataset,
    report: EvaluationReport,
    review: AnswerReview,
) -> dict[str, object]:
    """Require a complete review and exact source match before reporting correctness."""
    template = create_review(dataset, report)
    if (
        review.dataset_fingerprint != template.dataset_fingerprint
        or review.report_fingerprint != template.report_fingerprint
    ):
        raise ValueError("review fingerprints do not match this dataset and report")
    if not review.reviewer.strip():
        raise ValueError("reviewer must be identified")
    reviews = {case.case_id: case for case in review.cases}
    if len(reviews) != len(review.cases) or set(reviews) != {c.case_id for c in template.cases}:
        raise ValueError("review must include every answerable case exactly once")

    scores = []
    for expected in template.cases:
        actual = reviews[expected.case_id]
        if (actual.question, actual.ruling, actual.explanation) != (
            expected.question,
            expected.ruling,
            expected.explanation,
        ) or tuple(c.requirement for c in actual.criteria) != tuple(
            c.requirement for c in expected.criteria
        ):
            raise ValueError(f"answer text or rubric changed for {expected.case_id}")
        for criterion, original in zip(actual.criteria, expected.criteria, strict=True):
            if criterion.passed is None:
                raise ValueError(f"incomplete review for {expected.case_id}")
            if original.passed is False and criterion.passed:
                raise ValueError("an abstention or generation failure cannot pass correctness")
            if criterion.passed is False and not criterion.notes.strip():
                raise ValueError(f"failed criterion requires notes for {expected.case_id}")
        passed = sum(c.passed is True for c in actual.criteria)
        scores.append(
            {
                "case_id": actual.case_id,
                "criteria_passed": passed,
                "criteria_total": len(actual.criteria),
                "criterion_score": passed / len(actual.criteria),
                "answer_correct": passed == len(actual.criteria),
                "failures": [c.model_dump() for c in actual.criteria if c.passed is False],
            }
        )
    return {
        "dataset": dataset.name,
        "report_fingerprint": review.report_fingerprint,
        "reviewer": review.reviewer,
        "scoring_method": f"{review.review_kind}_rubric_review",
        "evaluated_cases": len(scores),
        "answer_correctness": sum(s["answer_correct"] for s in scores) / len(scores),
        "mean_criterion_score": sum(s["criterion_score"] for s in scores) / len(scores),
        "cases": scores,
    }
