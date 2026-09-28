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
    top_k_group = parser.add_mutually_exclusive_group()
    top_k_group.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="number of passages to retrieve per question",
    )
    top_k_group.add_argument(
        "--top-k-sweep",
        type=int,
        nargs="+",
        metavar="K",
        help="run retrieval evaluation at multiple passage limits, for example 1 3 5",
    )
    parser.add_argument(
        "--answers",
        action="store_true",
        help="also evaluate abstention and citations using one generation call per case",
    )
    arguments = parser.parse_args()

    if arguments.top_k_sweep and arguments.answers:
        parser.error("--top-k-sweep cannot be combined with --answers")
    if arguments.top_k is not None and arguments.top_k < 1:
        parser.error("--top-k must be at least one")
    if arguments.top_k_sweep and any(top_k < 1 for top_k in arguments.top_k_sweep):
        parser.error("every --top-k-sweep value must be at least one")

    settings = get_settings()
    dataset = load_evaluation_dataset(arguments.dataset)
    runner = EvaluationRunner(
        get_retriever(),
        get_answer_service() if arguments.answers else None,
    )
    if arguments.top_k_sweep:
        top_ks = tuple(dict.fromkeys(arguments.top_k_sweep))
        reports = [
            runner.run(dataset, top_k=top_k, evaluate_answers=False).model_dump(mode="json")
            for top_k in top_ks
        ]
        output: dict[str, object] = {
            "dataset": dataset.name,
            "season": dataset.season,
            "reports": reports,
        }
    else:
        top_k = settings.retrieval_top_k if arguments.top_k is None else arguments.top_k
        output = runner.run(
            dataset,
            top_k=top_k,
            evaluate_answers=arguments.answers,
        ).model_dump(mode="json")
    print(json.dumps(output, indent=2))
