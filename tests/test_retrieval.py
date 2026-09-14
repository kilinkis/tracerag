"""Unit tests for embedding, indexing, and retrieval orchestration."""

from collections.abc import Sequence
from pathlib import Path

from tracerag.corpus.models import RuleChunk
from tracerag.retrieval.indexer import CorpusIndexer
from tracerag.retrieval.models import RetrievalMatch
from tracerag.retrieval.service import Retriever

CORPUS_ROOT = Path(__file__).parents[1] / "data" / "nfl"


class FakeEmbedder:
    model_name = "test-embedding-model"
    dimensions = 3

    def __init__(self) -> None:
        self.passages: list[str] = []
        self.queries: list[str] = []

    def embed_passages(self, passages: Sequence[str]) -> list[list[float]]:
        self.passages = list(passages)
        return [[1.0, 0.0, 0.0] for _ in passages]

    def embed_query(self, query: str) -> list[float]:
        self.queries.append(query)
        return [1.0, 0.0, 0.0]


class FakeStore:
    def __init__(self, matches: list[RetrievalMatch] | None = None) -> None:
        self.matches = matches or []
        self.synced_chunks: list[RuleChunk] = []
        self.synced_embeddings: list[Sequence[float]] = []
        self.search_arguments: dict[str, object] = {}

    def sync_chunks(
        self,
        chunks: Sequence[RuleChunk],
        embeddings: Sequence[Sequence[float]],
        *,
        embedding_model: str,
    ) -> int:
        self.synced_chunks = list(chunks)
        self.synced_embeddings = list(embeddings)
        assert embedding_model == "test-embedding-model"
        return len(chunks)

    def search(
        self,
        query_embedding: Sequence[float],
        *,
        embedding_model: str,
        season: int,
        limit: int,
    ) -> list[RetrievalMatch]:
        self.search_arguments = {
            "query_embedding": list(query_embedding),
            "embedding_model": embedding_model,
            "season": season,
            "limit": limit,
        }
        return self.matches[:limit]


def test_indexer_embeds_heading_context_and_syncs_every_chunk() -> None:
    embedder = FakeEmbedder()
    store = FakeStore()

    report = CorpusIndexer(embedder, store).index(CORPUS_ROOT)

    assert report.model_dump() == {
        "season": 2026,
        "documents": 6,
        "chunks": 21,
        "embedding_model": "test-embedding-model",
        "dimensions": 3,
    }
    assert len(store.synced_chunks) == 21
    assert len(store.synced_embeddings) == 21
    assert embedder.passages[0].startswith(
        "Clock Management and Ten-Second Runoffs\n"
        "Clock Management and Ten-Second Runoffs > Illegally conserving time\n"
    )


def test_retriever_embeds_normalized_question_and_preserves_metadata() -> None:
    embedder = FakeEmbedder()
    match = RetrievalMatch(
        chunk_id="0" * 16,
        document_id="completing-a-catch",
        document_title="Completing a Catch",
        season=2026,
        heading_path=("Completing a Catch", "Sideline catches"),
        text="A player must establish control and get both feet inbounds.",
        source_title="2026 NFL Rulebook",
        source_url="https://example.com/rules",
        rule_references=("8-1-3",),
        relative_path="completing-a-catch.md",
        score=0.91,
    )
    store = FakeStore([match])

    result = Retriever(embedder, store).search(
        "  What makes a sideline catch complete?  ",
        season=2026,
        limit=3,
    )

    assert result.question == "What makes a sideline catch complete?"
    assert result.matches == (match,)
    assert embedder.queries == ["What makes a sideline catch complete?"]
    assert store.search_arguments == {
        "query_embedding": [1.0, 0.0, 0.0],
        "embedding_model": "test-embedding-model",
        "season": 2026,
        "limit": 3,
    }
