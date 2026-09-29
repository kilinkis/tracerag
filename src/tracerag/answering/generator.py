"""Replaceable answer-generator interface and Groq implementation.

Official SDK and structured-output sources:
https://github.com/groq/groq-python#usage
https://console.groq.com/docs/structured-outputs
https://console.groq.com/docs/api-reference
https://console.groq.com/docs/reasoning
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import cached_property
from typing import Literal, Protocol

from groq import APIError, BadRequestError, Groq
from pydantic import ValidationError

from tracerag.answering.models import AnswerDraft, GenerationResult, GenerationUsage
from tracerag.answering.prompt import SYSTEM_PROMPT, build_user_message
from tracerag.retrieval.models import RetrievalMatch

JSON_VALIDATION_ERROR_CODE = "json_validate_failed"


class GenerationError(RuntimeError):
    """The generation provider failed or returned unusable output."""

    def __init__(
        self,
        message: str,
        *,
        attempts: int = 0,
        provider_status_code: int | None = None,
        provider_error_code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.provider_status_code = provider_status_code
        self.provider_error_code = provider_error_code


class GenerationUnavailableError(GenerationError):
    """Answer generation is not configured for this runtime."""


class AnswerGenerator(Protocol):
    """Produces a typed ruling draft from retrieved evidence."""

    @property
    def model_name(self) -> str: ...

    def generate(
        self,
        question: str,
        evidence: Sequence[RetrievalMatch],
    ) -> GenerationResult: ...


class GroqAnswerGenerator:
    """Generate schema-constrained ruling drafts through Groq."""

    def __init__(
        self,
        *,
        api_key: str | None,
        model_name: str,
        timeout_seconds: float,
        max_completion_tokens: int,
        reasoning_effort: Literal["low", "medium", "high"] = "medium",
        structured_output_retries: int = 1,
        client: Groq | None = None,
    ) -> None:
        if structured_output_retries < 0:
            raise ValueError("structured_output_retries cannot be negative")
        self._api_key = api_key
        self._model_name = model_name
        self._timeout_seconds = timeout_seconds
        self._max_completion_tokens = max_completion_tokens
        self._reasoning_effort = reasoning_effort
        self._structured_output_retries = structured_output_retries
        self._provided_client = client

    @property
    def model_name(self) -> str:
        return self._model_name

    @cached_property
    def _client(self) -> Groq:
        if self._provided_client is not None:
            return self._provided_client
        if not self._api_key:
            raise GenerationUnavailableError("answer generation requires TRACERAG_GROQ_API_KEY")
        return Groq(
            api_key=self._api_key,
            timeout=self._timeout_seconds,
            max_retries=2,
        )

    def generate(
        self,
        question: str,
        evidence: Sequence[RetrievalMatch],
    ) -> GenerationResult:
        if not evidence:
            raise ValueError("answer generation requires retrieved evidence")

        attempts = 0
        while True:
            attempts += 1
            try:
                completion = self._client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": build_user_message(question, evidence),
                        },
                    ],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "nfl_ruling",
                            "strict": True,
                            "schema": AnswerDraft.model_json_schema(),
                        },
                    },
                    citation_options="disabled",
                    temperature=0,
                    reasoning_effort=self._reasoning_effort,
                    max_completion_tokens=self._max_completion_tokens,
                )
                break
            except BadRequestError as exc:
                if (
                    not self._is_json_validation_failure(exc)
                    or attempts > self._structured_output_retries
                ):
                    raise self._provider_failure(exc, attempts=attempts) from exc
            except APIError as exc:
                raise self._provider_failure(exc, attempts=attempts) from exc

        content = completion.choices[0].message.content if completion.choices else None
        if not content:
            raise GenerationError(
                "the generation provider returned no answer",
                attempts=attempts,
            )

        try:
            draft = AnswerDraft.model_validate_json(content)
        except ValidationError as exc:
            raise GenerationError(
                "the generation provider returned an invalid answer",
                attempts=attempts,
            ) from exc

        usage = completion.usage
        if usage is None:
            raise GenerationError(
                "the generation provider returned no usage metadata",
                attempts=attempts,
            )
        return GenerationResult(
            model=self.model_name,
            draft=draft,
            attempts=attempts,
            usage=GenerationUsage(
                input_tokens=usage.prompt_tokens,
                output_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
            ),
        )

    @staticmethod
    def _is_json_validation_failure(error: BadRequestError) -> bool:
        body = error.body
        if not isinstance(body, dict):
            return False
        details = body.get("error")
        return isinstance(details, dict) and details.get("code") == JSON_VALIDATION_ERROR_CODE

    @staticmethod
    def _provider_failure(error: APIError, *, attempts: int) -> GenerationError:
        provider_error_code = None
        body = error.body
        if isinstance(body, dict):
            details = body.get("error")
            if isinstance(details, dict) and isinstance(details.get("code"), str):
                provider_error_code = details["code"]
        return GenerationError(
            "the generation provider request failed",
            attempts=attempts,
            provider_status_code=getattr(error, "status_code", None),
            provider_error_code=provider_error_code,
        )
