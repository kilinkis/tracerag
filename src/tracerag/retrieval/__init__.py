"""Dense retrieval over the versioned NFL rules corpus."""

from tracerag.retrieval.indexer import CorpusIndexer, IndexReport
from tracerag.retrieval.models import RetrievalMatch, RetrievalResult
from tracerag.retrieval.service import Retriever
from tracerag.retrieval.store import PgVectorStore, VectorStore

__all__ = [
    "CorpusIndexer",
    "IndexReport",
    "PgVectorStore",
    "RetrievalMatch",
    "RetrievalResult",
    "Retriever",
    "VectorStore",
]
