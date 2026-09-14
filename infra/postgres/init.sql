CREATE EXTENSION IF NOT EXISTS vector;

-- Exact nearest-neighbor search is intentional for the small baseline corpus.
-- Approximate indexes will be introduced only when benchmarks justify them:
-- https://github.com/pgvector/pgvector#querying
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
);
