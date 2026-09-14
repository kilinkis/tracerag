"""Idempotent indexing of the version-controlled corpus."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict

from tracerag.corpus.loader import chunk_corpus, load_corpus
from tracerag.corpus.models import RuleChunk
from tracerag.embeddings import EmbeddingProvider
from tracerag.retrieval.store import VectorStore


class IndexReport(BaseModel):
    """Summary of a completed corpus index operation."""

    model_config = ConfigDict(frozen=True)

    season: int
    documents: int
    chunks: int
    embedding_model: str
    dimensions: int


class CorpusIndexer:
    """Transforms corpus files into embeddings and synchronizes the vector store."""

    def __init__(self, embedder: EmbeddingProvider, store: VectorStore) -> None:
        self._embedder = embedder
        self._store = store

    def index(self, corpus_path: Path) -> IndexReport:
        documents = load_corpus(corpus_path)
        chunks = chunk_corpus(documents)
        embeddings = self._embedder.embed_passages(
            [_embedding_text(chunk) for chunk in chunks]
        )
        indexed = self._store.sync_chunks(
            chunks,
            embeddings,
            embedding_model=self._embedder.model_name,
        )
        if indexed != len(chunks):
            raise RuntimeError(f"store reported {indexed} indexed chunks; expected {len(chunks)}")

        return IndexReport(
            season=documents[0].season,
            documents=len(documents),
            chunks=indexed,
            embedding_model=self._embedder.model_name,
            dimensions=self._embedder.dimensions,
        )


def _embedding_text(chunk: RuleChunk) -> str:
    heading = " > ".join(chunk.heading_path)
    return f"{chunk.document_title}\n{heading}\n{chunk.text}"
