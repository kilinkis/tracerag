"""Question-to-ruling orchestration with citation verification."""

from __future__ import annotations

from collections.abc import Callable
from time import perf_counter

from tracerag.answering.generator import AnswerGenerator
from tracerag.answering.models import (
    AnswerCitation,
    AnswerResult,
    AnswerTrace,
    GenerationUsage,
)
from tracerag.retrieval.models import RetrievalMatch
from tracerag.retrieval.service import Retriever

NO_EVIDENCE_REASON = "No evidence was retrieved from the configured NFL rules corpus."
UNVERIFIED_REASON = "The generated ruling could not be verified against the retrieved evidence."


class AnswerService:
    """Retrieve evidence, generate a ruling, and resolve only trusted citations."""

    def __init__(
        self,
        retriever: Retriever,
        generator: AnswerGenerator,
        *,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        self._retriever = retriever
        self._generator = generator
        self._clock = clock

    def answer(self, question: str, *, season: int, limit: int) -> AnswerResult:
        started_at = self._clock()
        retrieval = self._retriever.search(question, season=season, limit=limit)

        if not retrieval.matches:
            return self._abstention(
                retrieval.question,
                season=retrieval.season,
                reason=NO_EVIDENCE_REASON,
                evidence=retrieval.matches,
                embedding_model=retrieval.embedding_model,
                generation_model=self._generator.model_name,
                usage=GenerationUsage(input_tokens=0, output_tokens=0, total_tokens=0),
                generation_attempts=0,
                started_at=started_at,
            )

        generation = self._generator.generate(retrieval.question, retrieval.matches)
        if generation.draft.abstained:
            return self._abstention(
                retrieval.question,
                season=retrieval.season,
                reason=generation.draft.abstention_reason or UNVERIFIED_REASON,
                evidence=retrieval.matches,
                embedding_model=retrieval.embedding_model,
                generation_model=generation.model,
                usage=generation.usage,
                generation_attempts=generation.attempts,
                started_at=started_at,
            )

        matches_by_id = {match.chunk_id: match for match in retrieval.matches}
        cited_ids = tuple(dict.fromkeys(generation.draft.cited_chunk_ids))
        if not cited_ids or any(chunk_id not in matches_by_id for chunk_id in cited_ids):
            return self._abstention(
                retrieval.question,
                season=retrieval.season,
                reason=UNVERIFIED_REASON,
                evidence=retrieval.matches,
                embedding_model=retrieval.embedding_model,
                generation_model=generation.model,
                usage=generation.usage,
                generation_attempts=generation.attempts,
                started_at=started_at,
            )

        citations = tuple(self._citation(matches_by_id[chunk_id]) for chunk_id in cited_ids)
        return AnswerResult(
            question=retrieval.question,
            season=retrieval.season,
            ruling=generation.draft.ruling,
            explanation=generation.draft.explanation,
            abstained=False,
            abstention_reason=None,
            citations=citations,
            evidence=retrieval.matches,
            trace=self._trace(
                retrieval.embedding_model,
                generation.model,
                retrieved_chunks=len(retrieval.matches),
                usage=generation.usage,
                generation_attempts=generation.attempts,
                started_at=started_at,
            ),
        )

    def _abstention(
        self,
        question: str,
        *,
        season: int,
        reason: str,
        evidence: tuple[RetrievalMatch, ...],
        embedding_model: str,
        generation_model: str,
        usage: GenerationUsage,
        generation_attempts: int,
        started_at: float,
    ) -> AnswerResult:
        return AnswerResult(
            question=question,
            season=season,
            ruling=None,
            explanation=reason,
            abstained=True,
            abstention_reason=reason,
            citations=(),
            evidence=evidence,
            trace=self._trace(
                embedding_model,
                generation_model,
                retrieved_chunks=len(evidence),
                usage=usage,
                generation_attempts=generation_attempts,
                started_at=started_at,
            ),
        )

    def _trace(
        self,
        embedding_model: str,
        generation_model: str,
        *,
        retrieved_chunks: int,
        usage: GenerationUsage,
        generation_attempts: int,
        started_at: float,
    ) -> AnswerTrace:
        return AnswerTrace(
            embedding_model=embedding_model,
            generation_model=generation_model,
            retrieved_chunks=retrieved_chunks,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
            generation_attempts=generation_attempts,
            latency_ms=round((self._clock() - started_at) * 1000, 2),
        )

    @staticmethod
    def _citation(match: RetrievalMatch) -> AnswerCitation:
        return AnswerCitation(
            chunk_id=match.chunk_id,
            document_id=match.document_id,
            document_title=match.document_title,
            heading_path=match.heading_path,
            source_title=match.source_title,
            source_url=match.source_url,
            rule_references=match.rule_references,
        )
