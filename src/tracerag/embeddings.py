"""Embedding interfaces and the local FastEmbed implementation.

FastEmbed usage source:
https://qdrant.github.io/fastembed/Getting%20Started/
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from fastembed import TextEmbedding


class EmbeddingProvider(Protocol):
    """Maps passages and questions into one shared vector space."""

    @property
    def model_name(self) -> str: ...

    @property
    def dimensions(self) -> int: ...

    def embed_passages(self, passages: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, query: str) -> list[float]: ...


class FastEmbedProvider:
    """Generate local ONNX embeddings with a supported FastEmbed model."""

    def __init__(
        self,
        *,
        model_name: str,
        dimensions: int,
        cache_dir: Path,
    ) -> None:
        self._model_name = model_name
        self._dimensions = dimensions
        self._cache_dir = cache_dir

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @cached_property
    def _model(self) -> TextEmbedding:
        # Keep model initialization lazy so health and corpus endpoints do not download model files.
        from fastembed import TextEmbedding

        model = TextEmbedding(
            model_name=self.model_name,
            cache_dir=str(self._cache_dir),
        )
        if model.embedding_size != self.dimensions:
            raise ValueError(
                f"embedding model exposes {model.embedding_size} dimensions; "
                f"configured for {self.dimensions}"
            )
        return model

    def embed_passages(self, passages: Sequence[str]) -> list[list[float]]:
        if not passages:
            return []
        return [vector.tolist() for vector in self._model.passage_embed(passages)]

    def embed_query(self, query: str) -> list[float]:
        vectors = list(self._model.query_embed(query))
        if len(vectors) != 1:
            raise RuntimeError(f"expected one query embedding, received {len(vectors)}")
        return vectors[0].tolist()
