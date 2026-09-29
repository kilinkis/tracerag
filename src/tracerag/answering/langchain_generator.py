"""LangChain answer-generator adapter over Groq's OpenAI-compatible endpoint.

Official integration sources:
https://docs.langchain.com/oss/python/integrations/chat/openai
https://docs.langchain.com/oss/python/langchain/models#structured-output
https://console.groq.com/docs/openai
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import cached_property
from typing import Literal, cast

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from openai import APIError, BadRequestError

from tracerag.answering.generator import (
    JSON_VALIDATION_ERROR_CODE,
    GenerationError,
    GenerationUnavailableError,
)
from tracerag.answering.models import AnswerDraft, GenerationResult, GenerationUsage
from tracerag.answering.prompt import SYSTEM_PROMPT, build_user_message
from tracerag.retrieval.models import RetrievalMatch

GROQ_OPENAI_BASE_URL = "https://api.groq.com/openai/v1"
STRUCTURED_OUTPUT_ERROR_CODE = "structured_output_parse_failed"


class LangChainGroqAnswerGenerator:
    """Generate the same ruling contract through LangChain's standard chat-model interface."""

    def __init__(
        self,
        *,
        api_key: str | None,
        model_name: str,
        timeout_seconds: float,
        max_completion_tokens: int,
        reasoning_effort: Literal["low", "medium", "high"] = "medium",
        structured_output_retries: int = 1,
        model: BaseChatModel | None = None,
    ) -> None:
        if structured_output_retries < 0:
            raise ValueError("structured_output_retries cannot be negative")
        self._api_key = api_key
        self._model_name = model_name
        self._timeout_seconds = timeout_seconds
        self._max_completion_tokens = max_completion_tokens
        self._reasoning_effort = reasoning_effort
        self._structured_output_retries = structured_output_retries
        self._provided_model = model

    @property
    def model_name(self) -> str:
        return self._model_name

    @cached_property
    def _model(self) -> BaseChatModel:
        if self._provided_model is not None:
            return self._provided_model
        if not self._api_key:
            raise GenerationUnavailableError(
                "LangChain answer generation requires TRACERAG_GROQ_API_KEY"
            )

        # ChatOpenAI documents explicit base_url support, and Groq documents this endpoint as
        # OpenAI-compatible. use_responses_api=False keeps this on Chat Completions for parity with
        # the direct Groq adapter.
        return ChatOpenAI(
            model=self.model_name,
            api_key=self._api_key,
            base_url=GROQ_OPENAI_BASE_URL,
            temperature=0,
            timeout=self._timeout_seconds,
            max_retries=2,
            max_completion_tokens=self._max_completion_tokens,
            reasoning_effort=self._reasoning_effort,
            use_responses_api=False,
            extra_body={"citation_options": "disabled"},
        )

    @cached_property
    def _structured_model(self) -> Runnable[object, object]:
        # Pydantic supplies runtime validation; include_raw preserves the AIMessage usage metadata.
        return cast(
            Runnable[object, object],
            self._model.with_structured_output(
                AnswerDraft,
                method="json_schema",
                include_raw=True,
                strict=True,
            ),
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
                response = self._structured_model.invoke(
                    [
                        ("system", SYSTEM_PROMPT),
                        ("human", build_user_message(question, evidence)),
                    ]
                )
            except BadRequestError as exc:
                if (
                    not self._is_json_validation_failure(exc)
                    or attempts > self._structured_output_retries
                ):
                    raise self._provider_failure(exc, attempts=attempts) from exc
                continue
            except APIError as exc:
                raise self._provider_failure(exc, attempts=attempts) from exc

            if not isinstance(response, dict):
                raise GenerationError(
                    "LangChain returned an unexpected structured-output envelope",
                    attempts=attempts,
                )

            parsing_error = response.get("parsing_error")
            draft = response.get("parsed")
            if parsing_error is not None or not isinstance(draft, AnswerDraft):
                if attempts <= self._structured_output_retries:
                    continue
                raise GenerationError(
                    "LangChain returned an invalid structured answer",
                    attempts=attempts,
                    provider_error_code=STRUCTURED_OUTPUT_ERROR_CODE,
                ) from (parsing_error if isinstance(parsing_error, BaseException) else None)

            raw = response.get("raw")
            if not isinstance(raw, AIMessage) or raw.usage_metadata is None:
                raise GenerationError(
                    "LangChain returned no usage metadata",
                    attempts=attempts,
                )
            usage = raw.usage_metadata
            return GenerationResult(
                model=self.model_name,
                draft=draft,
                attempts=attempts,
                usage=GenerationUsage(
                    input_tokens=usage["input_tokens"],
                    output_tokens=usage["output_tokens"],
                    total_tokens=usage["total_tokens"],
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
            "the LangChain generation provider request failed",
            attempts=attempts,
            provider_status_code=getattr(error, "status_code", None),
            provider_error_code=provider_error_code,
        )
