"""Construction tests for replaceable runtime adapters."""

from pydantic import SecretStr
from pytest import MonkeyPatch

from tracerag import dependencies
from tracerag.answering.generator import GroqAnswerGenerator
from tracerag.answering.langchain_generator import LangChainGroqAnswerGenerator
from tracerag.config import Settings


def test_answer_generator_defaults_to_direct_groq_adapter(monkeypatch: MonkeyPatch) -> None:
    settings = Settings(_env_file=None, groq_api_key=SecretStr("test-key"))
    monkeypatch.setattr(dependencies, "get_settings", lambda: settings)
    dependencies.get_answer_generator.cache_clear()
    try:
        assert isinstance(dependencies.get_answer_generator(), GroqAnswerGenerator)
    finally:
        dependencies.get_answer_generator.cache_clear()


def test_answer_generator_can_select_langchain_adapter(monkeypatch: MonkeyPatch) -> None:
    settings = Settings(
        _env_file=None,
        groq_api_key=SecretStr("test-key"),
        generation_backend="langchain",
    )
    monkeypatch.setattr(dependencies, "get_settings", lambda: settings)
    dependencies.get_answer_generator.cache_clear()
    try:
        assert isinstance(dependencies.get_answer_generator(), LangChainGroqAnswerGenerator)
    finally:
        dependencies.get_answer_generator.cache_clear()
