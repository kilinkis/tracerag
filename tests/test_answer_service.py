"""Unit tests for grounded ruling orchestration and citation verification."""

from tracerag.answering.models import AnswerDraft, GenerationResult, GenerationUsage
from tracerag.answering.service import NO_EVIDENCE_REASON, UNVERIFIED_REASON, AnswerService
from tracerag.retrieval.models import RetrievalMatch, RetrievalResult


def retrieval_match(*, chunk_id: str = "a" * 16) -> RetrievalMatch:
    return RetrievalMatch(
        chunk_id=chunk_id,
        document_id="completing-a-catch",
        document_title="Completing a Catch",
        season=2026,
        heading_path=("Completing a Catch", "Sideline catches"),
        text="A receiver must control the ball and get both feet inbounds.",
        source_title="2026 NFL Rulebook",
        source_url="https://example.com/rules",
        rule_references=("8-1-3",),
        relative_path="completing-a-catch.md",
        score=0.91,
    )


class StubRetriever:
    def __init__(self, matches: tuple[RetrievalMatch, ...]) -> None:
        self.matches = matches

    def search(self, question: str, *, season: int, limit: int) -> RetrievalResult:
        assert season == 2026
        assert limit == 3
        return RetrievalResult(
            question=question.strip(),
            season=season,
            embedding_model="test-embedding-model",
            matches=self.matches,
        )


class StubGenerator:
    model_name = "test-generation-model"

    def __init__(self, draft: AnswerDraft) -> None:
        self.draft = draft
        self.calls: list[tuple[str, tuple[RetrievalMatch, ...]]] = []

    def generate(
        self,
        question: str,
        evidence: tuple[RetrievalMatch, ...],
    ) -> GenerationResult:
        self.calls.append((question, evidence))
        return GenerationResult(
            model=self.model_name,
            draft=self.draft,
            attempts=1,
            usage=GenerationUsage(input_tokens=100, output_tokens=40, total_tokens=140),
        )


def test_answer_service_resolves_citations_from_retrieved_metadata() -> None:
    match = retrieval_match()
    generator = StubGenerator(
        AnswerDraft(
            ruling="The catch is complete if both feet came down inbounds after control.",
            explanation="Control plus two feet inbounds satisfies the cited catch requirement.",
            cited_chunk_ids=(match.chunk_id,),
            abstained=False,
            abstention_reason=None,
        )
    )
    clock = iter((10.0, 10.125)).__next__
    service = AnswerService(StubRetriever((match,)), generator, clock=clock)

    result = service.answer("  Is the sideline catch complete?  ", season=2026, limit=3)

    assert result.abstained is False
    assert result.ruling is not None
    assert result.citations[0].model_dump(mode="json") == {
        "chunk_id": match.chunk_id,
        "document_id": "completing-a-catch",
        "document_title": "Completing a Catch",
        "heading_path": ["Completing a Catch", "Sideline catches"],
        "source_title": "2026 NFL Rulebook",
        "source_url": "https://example.com/rules",
        "rule_references": ["8-1-3"],
    }
    assert result.evidence == (match,)
    assert result.trace.model_dump() == {
        "embedding_model": "test-embedding-model",
        "generation_model": "test-generation-model",
        "retrieved_chunks": 1,
        "input_tokens": 100,
        "output_tokens": 40,
        "total_tokens": 140,
        "generation_attempts": 1,
        "latency_ms": 125.0,
    }
    assert generator.calls == [("Is the sideline catch complete?", (match,))]


def test_answer_service_abstains_when_generator_cites_unknown_evidence() -> None:
    match = retrieval_match()
    generator = StubGenerator(
        AnswerDraft(
            ruling="Touchdown.",
            explanation="The play scores.",
            cited_chunk_ids=("f" * 16,),
            abstained=False,
            abstention_reason=None,
        )
    )
    service = AnswerService(StubRetriever((match,)), generator, clock=iter((1.0, 1.0)).__next__)

    result = service.answer("Was it a touchdown?", season=2026, limit=3)

    assert result.abstained is True
    assert result.ruling is None
    assert result.citations == ()
    assert result.abstention_reason == UNVERIFIED_REASON


def test_answer_service_preserves_generator_abstention_reason() -> None:
    match = retrieval_match()
    generator = StubGenerator(
        AnswerDraft(
            ruling=None,
            explanation="The scenario does not say whether the receiver maintained control.",
            cited_chunk_ids=(),
            abstained=True,
            abstention_reason="The play scenario omits whether control was maintained.",
        )
    )
    service = AnswerService(StubRetriever((match,)), generator, clock=iter((1.0, 1.0)).__next__)

    result = service.answer("Was it a catch?", season=2026, limit=3)

    assert result.abstained is True
    assert result.abstention_reason == "The play scenario omits whether control was maintained."
    assert result.evidence == (match,)


def test_answer_service_skips_generation_when_retrieval_returns_nothing() -> None:
    generator = StubGenerator(
        AnswerDraft(
            ruling=None,
            explanation="Unused",
            cited_chunk_ids=(),
            abstained=True,
            abstention_reason="Unused",
        )
    )
    service = AnswerService(StubRetriever(()), generator, clock=iter((1.0, 1.0)).__next__)

    result = service.answer("What is the rule?", season=2026, limit=3)

    assert result.abstained is True
    assert result.abstention_reason == NO_EVIDENCE_REASON
    assert result.trace.total_tokens == 0
    assert result.trace.generation_attempts == 0
    assert generator.calls == []
