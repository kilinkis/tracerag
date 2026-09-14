"""Question-to-evidence retrieval orchestration."""

from tracerag.embeddings import EmbeddingProvider
from tracerag.retrieval.models import RetrievalResult
from tracerag.retrieval.store import VectorStore


class Retriever:
    """Embed one question and rank matching rule passages."""

    def __init__(self, embedder: EmbeddingProvider, store: VectorStore) -> None:
        self._embedder = embedder
        self._store = store

    def search(self, question: str, *, season: int, limit: int) -> RetrievalResult:
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("question cannot be blank")

        query_embedding = self._embedder.embed_query(normalized_question)
        matches = self._store.search(
            query_embedding,
            embedding_model=self._embedder.model_name,
            season=season,
            limit=limit,
        )
        return RetrievalResult(
            question=normalized_question,
            season=season,
            embedding_model=self._embedder.model_name,
            matches=tuple(matches),
        )
