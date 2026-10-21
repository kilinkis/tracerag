"""Run the existing evaluator with progress on stderr; no grading or metric changes."""

import argparse
import sys
import time
from pathlib import Path

from tracerag.answering.generator import GenerationError
from tracerag.dependencies import get_answer_service, get_retriever
from tracerag.evaluation import EvaluationRunner, load_evaluation_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="evals/nfl-rules.json")
parser.add_argument("--top-k", type=int, default=3)
parser.add_argument("--pause-seconds", type=float, default=0)
arguments = parser.parse_args()

dataset = load_evaluation_dataset(Path(arguments.dataset))
case_ids = {case.question: case.id for case in dataset.cases}
service = get_answer_service()


class ProgressAnswerer:
    def answer(self, question, *, season, limit):
        case_id = case_ids[question]
        if arguments.pause_seconds:
            time.sleep(arguments.pause_seconds)
        print(f"Starting {case_id}", file=sys.stderr, flush=True)
        try:
            result = service.answer(question, season=season, limit=limit)
        except GenerationError as error:
            print(
                f"Failed {case_id}: HTTP {error.provider_status_code}, "
                f"code={error.provider_error_code}",
                file=sys.stderr,
                flush=True,
            )
            if error.provider_status_code in {401, 403, 404}:
                raise RuntimeError("Provider configuration failure; stopping benchmark") from error
            raise
        print(
            f"Completed {case_id}: abstained={result.abstained}, "
            f"citations={[c.document_id for c in result.citations]}",
            file=sys.stderr,
            flush=True,
        )
        return result


report = EvaluationRunner(get_retriever(), ProgressAnswerer()).run(
    dataset,
    top_k=arguments.top_k,
    evaluate_answers=True,
)
print(report.model_dump_json(indent=2))
