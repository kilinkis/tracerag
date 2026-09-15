"""Unit tests for schema-constrained answer generation."""

from types import SimpleNamespace
from typing import cast

import httpx
import pytest
from groq import BadRequestError, Groq
from pydantic import ValidationError

from tracerag.answering.generator import (
    GenerationError,
    GenerationUnavailableError,
    GroqAnswerGenerator,
)
from tracerag.answering.models import AnswerDraft
from tracerag.retrieval.models import RetrievalMatch


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


class FakeCompletions:
    def __init__(self, outcomes: str | list[str | Exception]) -> None:
        self.outcomes = [outcomes] if isinstance(outcomes, str) else list(outcomes)
        self.call: dict[str, object] = {}
        self.calls: list[dict[str, object]] = []
        self.usage: SimpleNamespace | None = SimpleNamespace(
            prompt_tokens=80,
            completion_tokens=30,
            total_tokens=110,
        )

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.call = kwargs
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=outcome))],
            usage=self.usage,
        )


def bad_request(code: str) -> BadRequestError:
    response = httpx.Response(
        400,
        request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"),
    )
    return BadRequestError(
        "provider rejected the request",
        response=response,
        body={"error": {"code": code}},
    )


def groq_generator(completions: FakeCompletions) -> GroqAnswerGenerator:
    fake_client = cast(
        Groq,
        SimpleNamespace(chat=SimpleNamespace(completions=completions)),
    )
    return GroqAnswerGenerator(
        api_key=None,
        model_name="openai/gpt-oss-20b",
        timeout_seconds=20,
        max_completion_tokens=700,
        client=fake_client,
    )


def test_groq_generator_uses_strict_schema_and_passage_identifiers() -> None:
    match = retrieval_match()
    draft = AnswerDraft(
        ruling="Complete catch.",
        explanation="The evidence states the receiver needs control and two feet inbounds.",
        cited_chunk_ids=(match.chunk_id,),
        abstained=False,
        abstention_reason=None,
    )
    completions = FakeCompletions(draft.model_dump_json())

    result = groq_generator(completions).generate("Is this a catch?", (match,))

    assert result.draft == draft
    assert result.attempts == 1
    assert result.usage.total_tokens == 110
    assert completions.call["model"] == "openai/gpt-oss-20b"
    assert completions.call["citation_options"] == "disabled"
    assert completions.call["reasoning_effort"] == "medium"
    response_format = cast(dict[str, object], completions.call["response_format"])
    schema_config = cast(dict[str, object], response_format["json_schema"])
    schema = cast(dict[str, object], schema_config["schema"])
    assert schema_config["strict"] is True
    assert schema["additionalProperties"] is False
    assert set(cast(list[str], schema["required"])) == {
        "ruling",
        "explanation",
        "cited_chunk_ids",
        "abstained",
        "abstention_reason",
    }
    messages = cast(list[dict[str, str]], completions.call["messages"])
    assert match.chunk_id in messages[1]["content"]
    assert "Is this a catch?" in messages[1]["content"]


def test_groq_generator_retries_json_validation_failure_once() -> None:
    match = retrieval_match()
    draft = AnswerDraft(
        ruling="Complete catch.",
        explanation="The evidence supports the catch.",
        cited_chunk_ids=(match.chunk_id,),
        abstained=False,
        abstention_reason=None,
    )
    completions = FakeCompletions(
        [bad_request("json_validate_failed"), draft.model_dump_json()]
    )

    result = groq_generator(completions).generate("Is this a catch?", (match,))

    assert result.draft == draft
    assert result.attempts == 2
    assert len(completions.calls) == 2


def test_groq_generator_stops_after_json_validation_retry_is_exhausted() -> None:
    completions = FakeCompletions(
        [bad_request("json_validate_failed"), bad_request("json_validate_failed")]
    )

    with pytest.raises(GenerationError, match="provider request failed") as error:
        groq_generator(completions).generate("Is this a catch?", (retrieval_match(),))

    assert len(completions.calls) == 2
    assert error.value.attempts == 2
    assert error.value.provider_status_code == 400
    assert error.value.provider_error_code == "json_validate_failed"


def test_groq_generator_does_not_retry_other_bad_requests() -> None:
    completions = FakeCompletions(
        [bad_request("invalid_request"), "unused successful response"]
    )

    with pytest.raises(GenerationError, match="provider request failed") as error:
        groq_generator(completions).generate("Is this a catch?", (retrieval_match(),))

    assert len(completions.calls) == 1
    assert error.value.attempts == 1
    assert error.value.provider_status_code == 400
    assert error.value.provider_error_code == "invalid_request"


def test_groq_generator_requires_api_key_only_when_called() -> None:
    generator = GroqAnswerGenerator(
        api_key=None,
        model_name="openai/gpt-oss-20b",
        timeout_seconds=20,
        max_completion_tokens=700,
    )

    with pytest.raises(GenerationUnavailableError, match="TRACERAG_GROQ_API_KEY"):
        generator.generate("Is this a catch?", (retrieval_match(),))


def test_groq_generator_rejects_missing_usage_metadata() -> None:
    match = retrieval_match()
    draft = AnswerDraft(
        ruling="Complete catch.",
        explanation="The evidence supports the catch.",
        cited_chunk_ids=(match.chunk_id,),
        abstained=False,
        abstention_reason=None,
    )
    completions = FakeCompletions(draft.model_dump_json())
    completions.usage = None

    with pytest.raises(GenerationError, match="usage metadata"):
        groq_generator(completions).generate("Is this a catch?", (match,))


def test_answer_draft_rejects_contradictory_abstention() -> None:
    with pytest.raises(ValidationError, match="abstention draft"):
        AnswerDraft(
            ruling="Touchdown.",
            explanation="The evidence is insufficient.",
            cited_chunk_ids=(),
            abstained=True,
            abstention_reason="The play scenario omits required facts.",
        )
