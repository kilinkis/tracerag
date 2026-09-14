"""PostgreSQL/pgvector persistence for embedded rule chunks.

Official integration and cosine-query sources:
https://github.com/pgvector/pgvector-python#psycopg-3
https://github.com/pgvector/pgvector#querying
https://www.psycopg.org/psycopg3/docs/basic/params.html
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

from tracerag.corpus.models import RuleChunk
from tracerag.retrieval.models import RetrievalMatch

EMBEDDING_DIMENSIONS = 384


class VectorStore(Protocol):
    """Persists embedded chunks and ranks them against query vectors."""

    def sync_chunks(
        self,
        chunks: Sequence[RuleChunk],
        embeddings: Sequence[Sequence[float]],
        *,
        embedding_model: str,
    ) -> int: ...

    def search(
        self,
        query_embedding: Sequence[float],
        *,
        embedding_model: str,
        season: int,
        limit: int,
    ) -> list[RetrievalMatch]: ...


class PgVectorStore:
    """Exact cosine retrieval over PostgreSQL vector columns."""

    def __init__(self, database_url: str, *, dimensions: int = EMBEDDING_DIMENSIONS) -> None:
        if dimensions != EMBEDDING_DIMENSIONS:
            raise ValueError(f"PostgreSQL schema requires {EMBEDDING_DIMENSIONS} dimensions")
        self._database_url = database_url
        self._dimensions = dimensions

    def sync_chunks(
        self,
        chunks: Sequence[RuleChunk],
        embeddings: Sequence[Sequence[float]],
        *,
        embedding_model: str,
    ) -> int:
        if not chunks:
            raise ValueError("cannot index an empty corpus")
        if len(chunks) != len(embeddings):
            raise ValueError("each chunk must have exactly one embedding")
        if len({chunk.season for chunk in chunks}) != 1:
            raise ValueError("all chunks in an index operation must use one corpus season")
        self._validate_embeddings(embeddings)

        with self._connect(ensure_schema=True) as connection, connection.cursor() as cursor:
            cursor.executemany(
                """
                INSERT INTO rule_chunks (
                    chunk_id,
                    document_id,
                    document_title,
                    season,
                    heading_path,
                    content,
                    source_title,
                    source_url,
                    rule_references,
                    relative_path,
                    embedding_model,
                    embedding
                )
                VALUES (
                    %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s
                )
                ON CONFLICT (chunk_id) DO UPDATE SET
                    document_id = EXCLUDED.document_id,
                    document_title = EXCLUDED.document_title,
                    season = EXCLUDED.season,
                    heading_path = EXCLUDED.heading_path,
                    content = EXCLUDED.content,
                    source_title = EXCLUDED.source_title,
                    source_url = EXCLUDED.source_url,
                    rule_references = EXCLUDED.rule_references,
                    relative_path = EXCLUDED.relative_path,
                    embedding_model = EXCLUDED.embedding_model,
                    embedding = EXCLUDED.embedding,
                    indexed_at = CURRENT_TIMESTAMP
                """,
                [
                    (
                        chunk.chunk_id,
                        chunk.document_id,
                        chunk.document_title,
                        chunk.season,
                        list(chunk.heading_path),
                        chunk.text,
                        chunk.source.title,
                        str(chunk.source.url),
                        list(chunk.rule_references),
                        chunk.relative_path,
                        embedding_model,
                        Vector(embedding),
                    )
                    for chunk, embedding in zip(chunks, embeddings, strict=True)
                ],
            )
            cursor.execute(
                """
                DELETE FROM rule_chunks
                WHERE season = %s
                  AND NOT (chunk_id = ANY(%s))
                """,
                (chunks[0].season, [chunk.chunk_id for chunk in chunks]),
            )
        return len(chunks)

    def search(
        self,
        query_embedding: Sequence[float],
        *,
        embedding_model: str,
        season: int,
        limit: int,
    ) -> list[RetrievalMatch]:
        self._validate_embeddings([query_embedding])
        if not 1 <= limit <= 20:
            raise ValueError("limit must be between 1 and 20")

        query_vector = Vector(query_embedding)
        with self._connect() as connection, connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                """
                SELECT
                    chunk_id,
                    document_id,
                    document_title,
                    season,
                    heading_path,
                    content AS text,
                    source_title,
                    source_url,
                    rule_references,
                    relative_path,
                    1 - (embedding <=> %s) AS score
                FROM rule_chunks
                WHERE season = %s
                  AND embedding_model = %s
                ORDER BY embedding <=> %s, chunk_id
                LIMIT %s
                """,
                (query_vector, season, embedding_model, query_vector, limit),
            )
            return [RetrievalMatch.model_validate(row) for row in cursor.fetchall()]

    def _connect(self, *, ensure_schema: bool = False) -> psycopg.Connection:
        connection = psycopg.connect(self._database_url)
        if ensure_schema:
            self._ensure_schema(connection)
        register_vector(connection)
        return connection

    @staticmethod
    def _ensure_schema(connection: psycopg.Connection) -> None:
        connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS rule_chunks (
                chunk_id CHAR(16) PRIMARY KEY,
                document_id TEXT NOT NULL,
                document_title TEXT NOT NULL,
                season INTEGER NOT NULL,
                heading_path TEXT[] NOT NULL,
                content TEXT NOT NULL,
                source_title TEXT NOT NULL,
                source_url TEXT NOT NULL,
                rule_references TEXT[] NOT NULL,
                relative_path TEXT NOT NULL,
                embedding_model TEXT NOT NULL,
                embedding VECTOR(384) NOT NULL,
                indexed_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

    def _validate_embeddings(self, embeddings: Sequence[Sequence[float]]) -> None:
        if any(len(embedding) != self._dimensions for embedding in embeddings):
            raise ValueError(f"every embedding must have {self._dimensions} dimensions")
