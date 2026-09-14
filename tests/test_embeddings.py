"""Tests for the local embedding adapter without downloading model files."""

from collections.abc import Iterator
from pathlib import Path

from tracerag.embeddings import FastEmbedProvider


class FakeVector(list[float]):
    def tolist(self) -> list[float]:
        return list(self)


class FakeFastEmbedModel:
    def passage_embed(self, passages: list[str]) -> Iterator[FakeVector]:
        return iter(FakeVector([float(index), 0.0]) for index, _ in enumerate(passages))

    def query_embed(self, query: str) -> Iterator[FakeVector]:
        assert query == "catch rule"
        return iter([FakeVector([0.5, 0.5])])


def test_fastembed_adapter_keeps_query_and_passage_paths_distinct() -> None:
    provider = FastEmbedProvider(
        model_name="test-model",
        dimensions=2,
        cache_dir=Path("unused"),
    )
    provider.__dict__["_model"] = FakeFastEmbedModel()

    passages = provider.embed_passages(["first", "second"])
    query = provider.embed_query("catch rule")

    assert passages == [[0.0, 0.0], [1.0, 0.0]]
    assert query == [0.5, 0.5]


def test_empty_passage_batch_does_not_initialize_model() -> None:
    provider = FastEmbedProvider(
        model_name="test-model",
        dimensions=2,
        cache_dir=Path("unused"),
    )

    assert provider.embed_passages([]) == []
    assert "_model" not in provider.__dict__
