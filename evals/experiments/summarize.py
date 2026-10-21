"""Describe run coverage, abstention, latency and usage without inventing text accuracy."""

import argparse
import json
from pathlib import Path
from statistics import mean

from run import digest

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--inputs", type=Path, required=True)
parser.add_argument("--run", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
snapshot = json.loads(args.inputs.read_text())
state = json.loads(args.run.read_text())
if state["configuration"]["inputs_sha256"] != digest(snapshot):
    raise ValueError("Run belongs to different frozen inputs")
cases = {c["id"]: c for c in snapshot["dataset"]["cases"]}
valid = [(cases[k], r) for k, r in state["results"].items() if "generation" in r]
errors = [r for r in state["results"].values() if "error" in r]
summary = dict(
    planned=len(cases),
    attempted=len(state["results"]),
    valid=len(valid),
    generation_failures=len(errors),
    provider_http_errors=sum(r.get("status") is not None for r in errors),
    invalid_outputs=sum(
        r.get("error") == "the generation provider returned an invalid answer" for r in errors
    ),
    not_attempted=len(cases) - len(state["results"]),
    expected_abstentions=sum(c["expected_abstained"] for c in cases.values()),
    correct_abstentions=sum(
        c["expected_abstained"] and r["generation"]["draft"]["abstained"] for c, r in valid
    ),
    false_abstentions=sum(
        not c["expected_abstained"] and r["generation"]["draft"]["abstained"] for c, r in valid
    ),
    incorrect_answers_on_abstention_cases=sum(
        c["expected_abstained"] and not r["generation"]["draft"]["abstained"] for c, r in valid
    ),
    mean_successful_generation_latency_ms=mean([r["latency_ms"] for _, r in valid])
    if valid
    else None,
    abstention_scope="Drafts before server citation validation; use reports for final behavior.",
    text_accuracy="Requires an attributed rubric review; not derived from citations.",
    token_usage_note="Successful final results only; failed-call usage unknown.",
)
for key in ["input_tokens", "output_tokens", "total_tokens"]:
    summary[key] = sum(r["generation"]["usage"][key] for _, r in valid)
args.output.write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
