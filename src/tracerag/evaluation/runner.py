"""Run one dataset through retrieval and optional answer evaluation."""

from __future__ import annotations

from typing import Protocol

from tracerag.answering.generator import GenerationError
from tracerag.answering.models import AnswerResult
from tracerag.evaluation.models import (
    AnswerCaseResult,
    AnswerMetrics,
    EvaluationCase,
    EvaluationDataset,
    EvaluationReport,
    RetrievalCaseResult,
    RetrievalMetrics,
)
from tracerag.retrieval.models import RetrievalResult


class RetrievalSearcher(Protocol):
    def search(self, question: str, *, season: int, limit: int) -> RetrievalResult: ...


class RulingAnswerer(Protocol):
    def answer(self, question: str, *, season: int, limit: int) -> AnswerResult: ...


class EvaluationRunner:
    """Measure pipeline quality behind one small, dependency-injected interface."""

    def __init__(
        self,
        retriever: RetrievalSearcher,
        answerer: RulingAnswerer | None = None,
    ) -> None:
        self._retriever = retriever
        self._answerer = answerer

    def run(
        self,
        dataset: EvaluationDataset,
        *,
        top_k: int,
        evaluate_answers: bool = False,
    ) -> EvaluationReport:
        if top_k < 1:
            raise ValueError("top_k must be at least one")
        if evaluate_answers and self._answerer is None:
            raise ValueError("answer evaluation requires an answerer")

        retrieval_cases: list[RetrievalCaseResult] = []
        answer_cases: list[AnswerCaseResult] = []
        embedding_model = ""

        for case in dataset.cases:
            retrieval = self._retriever.search(
                case.question,
                season=dataset.season,
                limit=top_k,
            )
            embedding_model = retrieval.embedding_model
            retrieval_cases.append(self._measure_retrieval(case, retrieval))

            if evaluate_answers:
                assert self._answerer is not None
                try:
                    answer = self._answerer.answer(
                        case.question,
                        season=dataset.season,
                        limit=top_k,
                    )
                except GenerationError as exc:
                    answer_cases.append(self._generation_failure(case, exc))
                else:
                    answer_cases.append(self._measure_answer(case, answer))

        return EvaluationReport(
            dataset=dataset.name,
            season=dataset.season,
            top_k=top_k,
            embedding_model=embedding_model,
            retrieval_metrics=self._retrieval_metrics(retrieval_cases),
            retrieval_cases=tuple(retrieval_cases),
            answer_metrics=self._answer_metrics(answer_cases) if evaluate_answers else None,
            answer_cases=tuple(answer_cases) if evaluate_answers else None,
        )

    @staticmethod
    def _measure_retrieval(
        case: EvaluationCase,
        retrieval: RetrievalResult,
    ) -> RetrievalCaseResult:
        retrieved_ids = tuple(match.document_id for match in retrieval.matches)
        if case.expected_abstained:
            return RetrievalCaseResult(
                case_id=case.id,
                question=case.question,
                expected_document_ids=case.expected_document_ids,
                retrieved_document_ids=retrieved_ids,
                recall_at_k=None,
                reciprocal_rank=None,
                first_relevant_rank=None,
            )

        expected_ids = set(case.expected_document_ids)
        retrieved_unique = set(retrieved_ids)
        recall = len(expected_ids & retrieved_unique) / len(expected_ids)
        first_rank = next(
            (
                rank
                for rank, document_id in enumerate(retrieved_ids, start=1)
                if document_id in expected_ids
            ),
            None,
        )
        return RetrievalCaseResult(
            case_id=case.id,
            question=case.question,
            expected_document_ids=case.expected_document_ids,
            retrieved_document_ids=retrieved_ids,
            recall_at_k=recall,
            reciprocal_rank=1 / first_rank if first_rank is not None else 0.0,
            first_relevant_rank=first_rank,
        )

    @staticmethod
    def _measure_answer(case: EvaluationCase, answer: AnswerResult) -> AnswerCaseResult:
        expected_ids = set(case.expected_document_ids)
        cited_ids = tuple(citation.document_id for citation in answer.citations)
        matching_citations = sum(document_id in expected_ids for document_id in cited_ids)
        citation_precision = matching_citations / len(cited_ids) if cited_ids else None
        grounded_ruling_correct = None
        if not case.expected_abstained:
            cited_unique = set(cited_ids)
            grounded_ruling_correct = (
                not answer.abstained
                and bool(cited_ids)
                and cited_unique == expected_ids
            )

        return AnswerCaseResult(
            case_id=case.id,
            expected_abstained=case.expected_abstained,
            actual_abstained=answer.abstained,
            abstention_correct=case.expected_abstained == answer.abstained,
            cited_document_ids=cited_ids,
            citation_document_precision=citation_precision,
            grounded_ruling_correct=grounded_ruling_correct,
            generation_error=None,
            latency_ms=answer.trace.latency_ms,
            input_tokens=answer.trace.input_tokens,
            output_tokens=answer.trace.output_tokens,
            total_tokens=answer.trace.total_tokens,
        )

    @staticmethod
    def _generation_failure(case: EvaluationCase, error: GenerationError) -> AnswerCaseResult:
        return AnswerCaseResult(
            case_id=case.id,
            expected_abstained=case.expected_abstained,
            actual_abstained=None,
            abstention_correct=False,
            cited_document_ids=(),
            citation_document_precision=None,
            grounded_ruling_correct=False if not case.expected_abstained else None,
            generation_error=str(error),
            latency_ms=None,
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
        )

    @staticmethod
    def _retrieval_metrics(cases: list[RetrievalCaseResult]) -> RetrievalMetrics:
        measured = [case for case in cases if case.recall_at_k is not None]
        if not measured:
            return RetrievalMetrics(
                evaluated_cases=0,
                hit_rate_at_k=0.0,
                recall_at_k=0.0,
                mean_reciprocal_rank=0.0,
            )
        return RetrievalMetrics(
            evaluated_cases=len(measured),
            hit_rate_at_k=(
                sum(case.first_relevant_rank is not None for case in measured) / len(measured)
            ),
            recall_at_k=sum(case.recall_at_k or 0.0 for case in measured) / len(measured),
            mean_reciprocal_rank=(
                sum(case.reciprocal_rank or 0.0 for case in measured) / len(measured)
            ),
        )

    @staticmethod
    def _answer_metrics(cases: list[AnswerCaseResult]) -> AnswerMetrics:
        answerable = [case for case in cases if case.grounded_ruling_correct is not None]
        citation_cases = [case for case in cases if case.citation_document_precision is not None]
        citation_count = sum(len(case.cited_document_ids) for case in citation_cases)
        matching_citations = sum(
            (case.citation_document_precision or 0.0) * len(case.cited_document_ids)
            for case in citation_cases
        )
        completed_latencies = [case.latency_ms for case in cases if case.latency_ms is not None]
        return AnswerMetrics(
            evaluated_cases=len(cases),
            generation_failures=sum(case.generation_error is not None for case in cases),
            abstention_accuracy=sum(case.abstention_correct for case in cases) / len(cases),
            grounded_ruling_accuracy=(
                sum(bool(case.grounded_ruling_correct) for case in answerable) / len(answerable)
                if answerable
                else 0.0
            ),
            citation_document_precision=(
                matching_citations / citation_count
                if citation_count
                else None
            ),
            mean_latency_ms=(
                sum(completed_latencies) / len(completed_latencies)
                if completed_latencies
                else None
            ),
            input_tokens=sum(case.input_tokens for case in cases),
            output_tokens=sum(case.output_tokens for case in cases),
            total_tokens=sum(case.total_tokens for case in cases),
        )
