"""Operational commands for the TraceRAG service."""

import json

from tracerag.config import get_settings
from tracerag.dependencies import get_embedder, get_vector_store
from tracerag.retrieval.indexer import CorpusIndexer


def index_corpus() -> None:
    """Embed and synchronize the configured corpus into PostgreSQL."""

    settings = get_settings()
    report = CorpusIndexer(get_embedder(), get_vector_store()).index(settings.corpus_path)
    print(json.dumps(report.model_dump(), indent=2))
