"""Freeze retrieval, run resumable controlled comparisons, export standard review reports.

Groq SDK retries are disabled here so a 429 stops the run after one HTTP request:
https://github.com/groq/groq-python#retries
Run from the repository root with .venv/bin/python evals/experiments/run.py --help.
"""

import argparse
import hashlib
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from groq import Groq

from tracerag.answering.generator import SYSTEM_PROMPT, GenerationError, GroqAnswerGenerator
from tracerag.answering.models import GenerationResult
from tracerag.answering.service import AnswerService
from tracerag.config import Settings
from tracerag.evaluation import EvaluationRunner, load_evaluation_dataset
from tracerag.evaluation.review import create_review, fingerprint_dataset
from tracerag.retrieval.models import RetrievalResult


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def active_cooldown(output, model):
    """Honor saved limits across separate experiments using the same provider/model."""
    cooldown = 0
    for path in output.parent.parent.rglob("*.json"):
        value = json.loads(path.read_text())
        if not isinstance(value, dict) or value.get("configuration", {}).get("model") != model:
            continue
        for record in value.get("results", {}).values():
            if record.get("status") == 429:
                cooldown = max(cooldown, record.get("retry_at_epoch", 0))
    return cooldown


class FrozenRetriever:
    def __init__(self, snapshot):
        self.snapshot = snapshot

    def search(self, question, *, season, limit):
        if season != self.snapshot["dataset"]["season"] or limit != self.snapshot["top_k"]:
            raise ValueError("Retrieval settings differ from frozen inputs")
        return RetrievalResult.model_validate(self.snapshot["retrieval"][question])


def freeze(args):
    dataset = load_evaluation_dataset(args.dataset)
    if len({c.question for c in dataset.cases}) != len(dataset.cases):
        raise ValueError("Controlled snapshots require distinct question strings")
    if args.output.exists():
        raise ValueError("Refusing to overwrite frozen inputs")
    if args.from_report:
        report = json.loads(args.from_report.read_text())
        if report.get("dataset_fingerprint") != fingerprint_dataset(dataset):
            raise ValueError("Source report does not match frozen dataset")
        if report["top_k"] != args.top_k:
            raise ValueError("Source report uses a different retrieval limit")
        if any(c.get("generation_error") or not c.get("evidence") for c in report["answer_cases"]):
            raise ValueError("Source report has failures or missing evidence; use live retrieval")
        evidence = {c["case_id"]: c["evidence"] for c in report["answer_cases"]}
        retrieval = {
            c.question: dict(
                question=c.question,
                season=dataset.season,
                embedding_model=report["embedding_model"],
                matches=evidence[c.id],
            )
            for c in dataset.cases
        }
    else:
        from tracerag.dependencies import get_retriever

        retriever = get_retriever()
        retrieval = {
            c.question: retriever.search(
                c.question, season=dataset.season, limit=args.top_k
            ).model_dump(mode="json")
            for c in dataset.cases
        }
    save(
        args.output,
        dict(dataset=dataset.model_dump(mode="json"), top_k=args.top_k, retrieval=retrieval),
    )


def run(args):
    snapshot = json.loads(args.inputs.read_text())
    prompt = args.prompt.read_text() if args.prompt else SYSTEM_PROMPT
    config = dict(
        inputs_sha256=digest(snapshot),
        model=args.model,
        system_prompt=prompt,
        reasoning_effort="medium",
        max_completion_tokens=1400,
        temperature=0,
        http_retries=0,
        structured_output_retries=0,
    )
    state = (
        json.loads(args.output.read_text())
        if args.output.exists()
        else dict(configuration=config, results={})
    )
    if state["configuration"] != config:
        raise ValueError("Cannot resume with different inputs, model, or prompt")
    cooldown = active_cooldown(args.output, args.model)
    if time.time() < cooldown:
        state["blocked_until_epoch"] = cooldown
        state["status"] = "provider_retry_delay"
        save(args.output, state)
        print(f"Provider retry delay is active; resume after epoch {cooldown}")
        return
    state.pop("blocked_until_epoch", None)
    state["status"] = "running"
    settings = Settings()
    if settings.groq_api_key is None:
        raise ValueError("TRACERAG_GROQ_API_KEY is required")
    generator = GroqAnswerGenerator(
        api_key=None,
        model_name=args.model,
        timeout_seconds=60,
        max_completion_tokens=1400,
        structured_output_retries=0,
        system_prompt=prompt,
        client=Groq(api_key=settings.groq_api_key.get_secret_value(), timeout=60, max_retries=0),
    )
    retriever = FrozenRetriever(snapshot)
    for case in snapshot["dataset"]["cases"]:
        case_id = case["id"]
        # Failed attempts are retained; retries require an explicit flag.
        previous = state["results"].get(case_id)
        if previous is not None:
            eligible_retry = args.retry_errors or (
                args.retry_rate_limits and previous.get("status") == 429
            )
            if not eligible_retry or "error" not in previous:
                continue
            retry_at = previous.get("retry_at_epoch", 0)
            if time.time() < retry_at:
                print(f"Rate limit still active for {case_id}; resume after {retry_at}")
                break
            state.setdefault("previous_attempts", {}).setdefault(case_id, []).append(previous)
        print("Starting", case_id, args.model, flush=True)
        started = time.perf_counter()
        try:
            evidence = retriever.search(
                case["question"], season=snapshot["dataset"]["season"], limit=snapshot["top_k"]
            ).matches
            if evidence:
                result = generator.generate(case["question"], evidence)
                record = dict(generation=result.model_dump(mode="json"))
            else:
                record = dict(no_evidence=True)
        except GenerationError as error:
            record = dict(
                error=str(error),
                status=error.provider_status_code,
                code=error.provider_error_code,
                attempts=error.attempts,
                output_error_codes=error.output_error_codes,
            )
            cause = error.__cause__
            response = getattr(cause, "response", None)
            if response is not None:
                record["rate_limit_headers"] = {
                    k: v
                    for k, v in response.headers.items()
                    if k.lower().startswith(("retry-after", "x-ratelimit-"))
                }
        record["completed_at"] = datetime.now(UTC).isoformat()
        retry_after = record.get("rate_limit_headers", {}).get("retry-after")
        if retry_after and retry_after.isdigit():
            record["retry_at_epoch"] = time.time() + int(retry_after)
        record["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
        state["results"][case_id] = record
        save(args.output, state)
        if record.get("status") in {401, 403, 404, 429}:
            print("Stopped after provider HTTP", record["status"], file=sys.stderr)
            break
        if args.pause_seconds:
            time.sleep(args.pause_seconds)
    pending = len(snapshot["dataset"]["cases"]) - len(state["results"])
    state["status"] = "complete" if not pending else "interrupted"
    save(args.output, state)
    print(f"Saved {len(state['results'])} attempted; {pending} not attempted", flush=True)


def export(args):
    from tracerag.evaluation.models import EvaluationDataset

    snapshot = json.loads(args.inputs.read_text())
    review_path = args.output.with_name(args.output.stem + "-review.json")
    if review_path.exists() and json.loads(review_path.read_text()).get("reviewer"):
        raise ValueError("Refusing to overwrite an attributed review")
    state = json.loads(args.run.read_text())
    if state["configuration"]["inputs_sha256"] != digest(snapshot):
        raise ValueError("Run belongs to different frozen inputs")
    dataset = EvaluationDataset.model_validate(snapshot["dataset"])
    if set(state["results"]) != {c.id for c in dataset.cases}:
        raise ValueError("Incomplete run: unattempted cases must not be scored as model failures")
    by_question = {c.question: c.id for c in dataset.cases}

    class ReplayGenerator:
        model_name = state["configuration"]["model"]

        def generate(self, question, evidence):
            record = state["results"][by_question[question]]
            if "error" in record:
                raise GenerationError(
                    record["error"],
                    attempts=record["attempts"]
                    + sum(
                        r.get("attempts", 0)
                        for r in state.get("previous_attempts", {}).get(by_question[question], [])
                    ),
                    provider_status_code=record["status"],
                    provider_error_code=record["code"],
                    output_error_codes=tuple(record.get("output_error_codes", [])),
                )
            generation = GenerationResult.model_validate(record["generation"])
            previous = state.get("previous_attempts", {}).get(by_question[question], [])
            return generation.model_copy(
                update=dict(
                    attempts=generation.attempts + sum(r.get("attempts", 0) for r in previous)
                )
            )

    retriever = FrozenRetriever(snapshot)

    # Replay resolves citations through the production service, with recorded generation latency.
    class ReplayAnswerer:
        def answer(self, question, *, season, limit):
            record = state["results"][by_question[question]]
            previous = state.get("previous_attempts", {}).get(by_question[question], [])
            latency = record["latency_ms"] + sum(r["latency_ms"] for r in previous)
            clock = iter((0, latency / 1000))
            return AnswerService(retriever, ReplayGenerator(), clock=lambda: next(clock)).answer(
                question, season=season, limit=limit
            )

    report = EvaluationRunner(retriever, ReplayAnswerer()).run(
        dataset, top_k=snapshot["top_k"], evaluate_answers=True
    )
    save(args.output, report.model_dump(mode="json"))
    save(
        args.output.with_name(args.output.stem + "-provenance.json"),
        dict(
            source_run=str(args.run),
            source_run_sha256=digest(state),
            configuration=state["configuration"],
            previous_attempts=state.get("previous_attempts", {}),
            usage_note="Failed-call token use is unknown; final successful usage only.",
            latency_note="Latency includes archived failed attempts, excluding pacing/retrieval.",
        ),
    )
    review = create_review(dataset, report)
    save(args.output.with_name(args.output.stem + "-review.json"), review.model_dump(mode="json"))


parser = argparse.ArgumentParser(description=__doc__)
commands = parser.add_subparsers(dest="command", required=True)
f = commands.add_parser("freeze")
f.add_argument("--dataset", type=Path, required=True)
f.add_argument("--top-k", type=int, default=5)
f.add_argument("--from-report", type=Path)
f.add_argument("--output", type=Path, required=True)
r = commands.add_parser("run")
r.add_argument("--inputs", type=Path, required=True)
r.add_argument("--model", required=True)
r.add_argument("--prompt", type=Path)
r.add_argument("--pause-seconds", type=float, default=20)
r.add_argument("--retry-errors", action="store_true")
r.add_argument("--retry-rate-limits", action="store_true")
r.add_argument("--output", type=Path, required=True)
e = commands.add_parser("export")
e.add_argument("--inputs", type=Path, required=True)
e.add_argument("--run", type=Path, required=True)
e.add_argument("--output", type=Path, required=True)
if __name__ == "__main__":
    arguments = parser.parse_args()
    {"freeze": freeze, "run": run, "export": export}[arguments.command](arguments)
