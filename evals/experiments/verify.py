"""Evaluate a verifier independently on labeled accepted/rejected drafts; save every verdict."""

import argparse
import json
import time
from pathlib import Path

from groq import Groq

# Shared immutable configuration/checkpoint format with the generation runner.
from run import active_cooldown, digest, save

from tracerag.answering.generator import GenerationError
from tracerag.answering.models import AnswerDraft, GenerationResult, GenerationUsage
from tracerag.answering.verification import (
    VERIFICATION_PROMPT,
    GroqAnswerVerifier,
    VerifiedAnswerGenerator,
)
from tracerag.config import Settings
from tracerag.retrieval.models import RetrievalMatch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--challenges", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--model", default="openai/gpt-oss-120b")
parser.add_argument("--pause-seconds", type=float, default=20)
args = parser.parse_args()
challenges = json.loads(args.challenges.read_text())
config = dict(
    challenges_sha256=digest(challenges),
    model=args.model,
    prompt=VERIFICATION_PROMPT,
    quote_policy="whitespace-normalized-v1",
)
state = (
    json.loads(args.output.read_text())
    if args.output.exists()
    else dict(configuration=config, results={})
)
if state["configuration"] != config:
    raise ValueError("Challenge set, prompt or model differs from existing run")
cooldown = active_cooldown(args.output, args.model)
if time.time() < cooldown:
    raise SystemExit(f"Provider retry delay active; resume after epoch {cooldown}")
settings = Settings()
if settings.groq_api_key is None:
    raise ValueError("Groq key required")
verifier = GroqAnswerVerifier(
    Groq(api_key=settings.groq_api_key.get_secret_value(), timeout=60, max_retries=0), args.model
)


class FixtureGenerator:
    model_name = "labeled-draft-fixture"

    def __init__(self, draft):
        self.draft = draft

    def generate(self, question, evidence):
        return GenerationResult(
            model=self.model_name,
            draft=self.draft,
            attempts=1,
            usage=GenerationUsage(input_tokens=0, output_tokens=0, total_tokens=0),
        )


class RecordedVerifier:
    def __init__(self, result):
        self.result = result

    def verify(self, question, draft, evidence):
        return self.result


for case in challenges["cases"]:
    if case["id"] in state["results"]:
        continue
    print("Verifying", case["id"], flush=True)
    started = time.perf_counter()
    try:
        result = verifier.verify(
            case["question"],
            AnswerDraft.model_validate(case["draft"]),
            tuple(RetrievalMatch.model_validate(m) for m in case["evidence"]),
        )
        gated = VerifiedAnswerGenerator(
            FixtureGenerator(AnswerDraft.model_validate(case["draft"])), RecordedVerifier(result)
        ).generate(
            case["question"], tuple(RetrievalMatch.model_validate(m) for m in case["evidence"])
        )
        record = dict(
            gated_draft=gated.draft.model_dump(mode="json"),
            verification=result.model_dump(mode="json"),
            correct=result.decision.accepted == case["expected_accepted"],
        )
    except GenerationError as error:
        record = dict(
            error=str(error),
            status=error.provider_status_code,
            code=error.provider_error_code,
            rate_limit_headers=error.rate_limit_headers,
        )
    retry_after = record.get("rate_limit_headers", {}).get("retry-after")
    if retry_after and retry_after.isdigit():
        record["retry_at_epoch"] = time.time() + int(retry_after)
    record["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
    state["results"][case["id"]] = record
    save(args.output, state)
    if record.get("status") in {401, 403, 404, 429}:
        break
    time.sleep(args.pause_seconds)
# Separate coverage and operational failures from false accepts/rejects.
scored = [(c, state["results"].get(c["id"], {})) for c in challenges["cases"]]
scored = [(c, r) for c, r in scored if "verification" in r]
summary = dict(
    planned=len(challenges["cases"]),
    judged=len(scored),
    not_attempted=len(challenges["cases"]) - len(state["results"]),
    operational_errors=sum("error" in r for r in state["results"].values()),
    false_accepts=sum(
        not c["expected_accepted"] and r["verification"]["decision"]["accepted"] for c, r in scored
    ),
    false_rejects=sum(
        c["expected_accepted"] and not r["verification"]["decision"]["accepted"] for c, r in scored
    ),
)
save(args.output.with_name(args.output.stem + "-summary.json"), summary)
print(summary)
