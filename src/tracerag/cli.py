"""Operational commands for the TraceRAG service."""

import argparse
import json
from pathlib import Path

from tracerag.config import get_settings
from tracerag.dependencies import (
    get_answer_service,
    get_embedder,
    get_retriever,
    get_vector_store,
)
from tracerag.evaluation import EvaluationRunner, load_evaluation_dataset
from tracerag.retrieval.indexer import CorpusIndexer


def index_corpus() -> None:
    """Embed and synchronize the configured corpus into PostgreSQL."""

    settings = get_settings()
    report = CorpusIndexer(get_embedder(), get_vector_store()).index(settings.corpus_path)
    print(json.dumps(report.model_dump(), indent=2))


def evaluate_corpus() -> None:
    """Run the versioned retrieval benchmark and optional answer evaluation."""

    parser = argparse.ArgumentParser(description="Evaluate TraceRAG against a question dataset.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("evals/nfl-rules.json"),
        help="path to the evaluation dataset",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="number of passages to retrieve per question",
    )
    parser.add_argument(
        "--answers",
        action="store_true",
        help="also evaluate abstention and citations using one generation call per case",
    )
    arguments = parser.parse_args()

    settings = get_settings()
    dataset = load_evaluation_dataset(arguments.dataset)
    top_k = settings.retrieval_top_k if arguments.top_k is None else arguments.top_k
    runner = EvaluationRunner(
        get_retriever(),
        get_answer_service() if arguments.answers else None,
    )
    report = runner.run(dataset, top_k=top_k, evaluate_answers=arguments.answers)
    print(json.dumps(report.model_dump(mode="json"), indent=2))
