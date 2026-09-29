"""Unit tests for the optional LangChain answer-generator adapter."""

from typing import cast

import httpx
import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from openai import BadRequestError

from tracerag.answering.generator import GenerationError, GenerationUnavailableError
from tracerag.answering.langchain_generator import (
    STRUCTURED_OUTPUT_ERROR_CODE,
    LangChainGroqAnswerGenerator,
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


def answer_draft(chunk_id: str = "a" * 16) -> AnswerDraft:
    return AnswerDraft(
        ruling="Complete catch.",
        explanation="The evidence requires control and two feet inbounds.",
        cited_chunk_ids=(chunk_id,),
        abstained=False,
        abstention_reason=None,
    )


def answer_envelope(
    draft: AnswerDraft | None,
    *,
    parsing_error: BaseException | None = None,
    with_usage: bool = True,
) -> dict[str, object]:
    usage = {"input_tokens": 80, "output_tokens": 30, "total_tokens": 110} if with_usage else None
    return {
        "raw": AIMessage(content="", usage_metadata=usage),
        "parsed": draft,
        "parsing_error": parsing_error,
    }


class FakeRunnable:
    def __init__(self, outcomes: object | list[object]) -> None:
        self.outcomes = outcomes if isinstance(outcomes, list) else [outcomes]
        self.calls: list[object] = []

    def invoke(self, value: object) -> object:
        self.calls.append(value)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class FakeChatModel:
    def __init__(self, runnable: FakeRunnable) -> None:
        self.runnable = runnable
        self.structured_output_call: dict[str, object] = {}

    def with_structured_output(self, schema: object, **kwargs: object) -> FakeRunnable:
        self.structured_output_call = {"schema": schema, **kwargs}
        return self.runnable


def generator(runnable: FakeRunnable) -> tuple[LangChainGroqAnswerGenerator, FakeChatModel]:
    model = FakeChatModel(runnable)
    return (
        LangChainGroqAnswerGenerator(
            api_key=None,
            model_name="openai/gpt-oss-20b",
            timeout_seconds=20,
            max_completion_tokens=700,
            model=cast(BaseChatModel, model),
        ),
        model,
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


def test_langchain_generator_uses_native_schema_and_preserves_usage() -> None:
    match = retrieval_match()
    draft = answer_draft(match.chunk_id)
    runnable = FakeRunnable(answer_envelope(draft))
    answer_generator, model = generator(runnable)

    result = answer_generator.generate("Is this a catch?", (match,))

    assert result.draft == draft
    assert result.attempts == 1
    assert result.usage.total_tokens == 110
    assert model.structured_output_call == {
        "schema": AnswerDraft,
        "method": "json_schema",
        "include_raw": True,
        "strict": True,
    }
    messages = cast(list[tuple[str, str]], runnable.calls[0])
    assert messages[0][0] == "system"
    assert messages[1][0] == "human"
    assert match.chunk_id in messages[1][1]
    assert "Is this a catch?" in messages[1][1]


def test_langchain_generator_retries_captured_parsing_failure_once() -> None:
    runnable = FakeRunnable(
        [
            answer_envelope(None, parsing_error=ValueError("invalid JSON")),
            answer_envelope(answer_draft()),
        ]
    )
    answer_generator, _ = generator(runnable)

    result = answer_generator.generate("Is this a catch?", (retrieval_match(),))

    assert result.attempts == 2
    assert len(runnable.calls) == 2


def test_langchain_generator_reports_exhausted_parsing_failure() -> None:
    runnable = FakeRunnable(
        [
            answer_envelope(None, parsing_error=ValueError("invalid JSON")),
            answer_envelope(None, parsing_error=ValueError("invalid JSON")),
        ]
    )
    answer_generator, _ = generator(runnable)

    with pytest.raises(GenerationError, match="invalid structured answer") as error:
        answer_generator.generate("Is this a catch?", (retrieval_match(),))

    assert error.value.attempts == 2
    assert error.value.provider_error_code == STRUCTURED_OUTPUT_ERROR_CODE


def test_langchain_generator_retries_provider_json_validation_failure() -> None:
    runnable = FakeRunnable([bad_request("json_validate_failed"), answer_envelope(answer_draft())])
    answer_generator, _ = generator(runnable)

    result = answer_generator.generate("Is this a catch?", (retrieval_match(),))

    assert result.attempts == 2
    assert len(runnable.calls) == 2


def test_langchain_generator_wraps_other_provider_failures() -> None:
    runnable = FakeRunnable(bad_request("invalid_request"))
    answer_generator, _ = generator(runnable)

    with pytest.raises(GenerationError, match="LangChain generation provider") as error:
        answer_generator.generate("Is this a catch?", (retrieval_match(),))

    assert error.value.attempts == 1
    assert error.value.provider_status_code == 400
    assert error.value.provider_error_code == "invalid_request"


def test_langchain_generator_requires_api_key_only_when_called() -> None:
    answer_generator = LangChainGroqAnswerGenerator(
        api_key=None,
        model_name="openai/gpt-oss-20b",
        timeout_seconds=20,
        max_completion_tokens=700,
    )

    with pytest.raises(GenerationUnavailableError, match="TRACERAG_GROQ_API_KEY"):
        answer_generator.generate("Is this a catch?", (retrieval_match(),))


def test_langchain_generator_sends_groq_extensions_through_extra_body() -> None:
    answer_generator = LangChainGroqAnswerGenerator(
        api_key="test-key",
        model_name="openai/gpt-oss-20b",
        timeout_seconds=20,
        max_completion_tokens=700,
    )

    model = answer_generator._model

    assert isinstance(model, ChatOpenAI)
    assert model.extra_body == {"citation_options": "disabled"}
    assert "citation_options" not in model.model_kwargs


def test_langchain_generator_rejects_missing_usage_metadata() -> None:
    runnable = FakeRunnable(answer_envelope(answer_draft(), with_usage=False))
    answer_generator, _ = generator(runnable)

    with pytest.raises(GenerationError, match="usage metadata"):
        answer_generator.generate("Is this a catch?", (retrieval_match(),))
