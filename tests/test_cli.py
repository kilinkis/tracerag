"""Behavioral coverage for operational command orchestration."""

import json
import sys
from types import SimpleNamespace

import pytest

from tracerag import cli


class StubReport:
    def __init__(self, top_k: int) -> None:
        self.top_k = top_k

    def model_dump(self, *, mode: str) -> dict[str, int]:
        assert mode == "json"
        return {"top_k": self.top_k}


class StubRunner:
    def __init__(self) -> None:
        self.calls: list[tuple[int, bool]] = []

    def run(
        self,
        dataset: object,
        *,
        top_k: int,
        evaluate_answers: bool,
    ) -> StubReport:
        self.calls.append((top_k, evaluate_answers))
        return StubReport(top_k)


def test_retrieval_sweep_deduplicates_depths_and_never_builds_answerer(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset = SimpleNamespace(name="test-dataset", season=2026)
    runner = StubRunner()

    monkeypatch.setattr(sys, "argv", ["tracerag-evaluate", "--top-k-sweep", "1", "3", "3"])
    monkeypatch.setattr(cli, "get_settings", lambda: SimpleNamespace(retrieval_top_k=5))
    monkeypatch.setattr(cli, "load_evaluation_dataset", lambda path: dataset)
    monkeypatch.setattr(cli, "get_retriever", lambda: object())
    monkeypatch.setattr(cli, "EvaluationRunner", lambda retriever, answerer: runner)

    def unexpected_answerer() -> object:
        pytest.fail("a retrieval sweep must not create an answer service")

    monkeypatch.setattr(cli, "get_answer_service", unexpected_answerer)

    cli.evaluate_corpus()

    assert runner.calls == [(1, False), (3, False)]
    assert json.loads(capsys.readouterr().out) == {
        "dataset": "test-dataset",
        "season": 2026,
        "reports": [{"top_k": 1}, {"top_k": 3}],
    }


def test_retrieval_sweep_rejects_answer_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["tracerag-evaluate", "--top-k-sweep", "1", "3", "5", "--answers"],
    )

    with pytest.raises(SystemExit, match="2"):
        cli.evaluate_corpus()
