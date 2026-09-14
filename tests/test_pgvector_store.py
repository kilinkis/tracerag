"""Integration coverage for exact cosine retrieval in PostgreSQL."""

import os
from pathlib import Path

import psycopg
import pytest

from tracerag.corpus.loader import chunk_corpus, load_corpus
from tracerag.retrieval.store import EMBEDDING_DIMENSIONS, PgVectorStore

CORPUS_ROOT = Path(__file__).parents[1] / "data" / "nfl"
TEST_DATABASE_URL = os.getenv("TRACERAG_TEST_DATABASE_URL")


@pytest.mark.integration
@pytest.mark.skipif(TEST_DATABASE_URL is None, reason="test database is not configured")
def test_sync_and_exact_cosine_search() -> None:
    assert TEST_DATABASE_URL is not None
    chunks = [
        chunk.model_copy(update={"chunk_id": f"{'f' * 15}{index}", "season": 9999})
        for index, chunk in enumerate(chunk_corpus(load_corpus(CORPUS_ROOT))[:2])
    ]
    first_embedding = [1.0, *([0.0] * (EMBEDDING_DIMENSIONS - 1))]
    second_embedding = [0.0, 1.0, *([0.0] * (EMBEDDING_DIMENSIONS - 2))]
    store = PgVectorStore(TEST_DATABASE_URL)

    try:
        indexed = store.sync_chunks(
            chunks,
            [first_embedding, second_embedding],
            embedding_model="integration-test-model",
        )
        matches = store.search(
            first_embedding,
            embedding_model="integration-test-model",
            season=9999,
            limit=2,
        )

        assert indexed == 2
        assert [match.chunk_id for match in matches] == [chunks[0].chunk_id, chunks[1].chunk_id]
        assert matches[0].score == pytest.approx(1.0)
        assert matches[1].score == pytest.approx(0.0)
    finally:
        with psycopg.connect(TEST_DATABASE_URL) as connection:
            connection.execute(
                "DELETE FROM rule_chunks WHERE season = %s AND embedding_model = %s",
                (9999, "integration-test-model"),
            )
