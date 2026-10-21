"""Run stage four sequentially, respecting shared rate limits and preserving every attempt."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from run import digest, save

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--retry-rate-limits", action="store_true")
args = parser.parse_args()
root = Path("evals/runs/2026-10-21-improvements/04-holdout")
runner = Path("evals/experiments/run.py")
plan = [("full", "full-inputs.json"), ("holdout-1", "inputs.json"), ("holdout-2", "inputs.json")]
summary = {"model": "openai/gpt-oss-120b", "prompt": "baseline", "semantic_gate": False, "runs": {}}
for label, inputs in plan:
    output = root / (label + ".json")
    subprocess.run(
        [
            sys.executable,
            str(runner),
            "run",
            "--inputs",
            str(root / inputs),
            "--model",
            summary["model"],
            "--output",
            str(output),
        ]
        + (["--retry-rate-limits"] if args.retry_rate_limits else []),
        check=True,
    )
    state = json.loads(output.read_text())
    snapshot = json.loads((root / inputs).read_text())
    records = list(state["results"].values())
    summary["runs"][label] = dict(
        planned=len(snapshot["dataset"]["cases"]),
        attempted=len(records),
        valid=sum("generation" in r for r in records),
        errors=sum("error" in r for r in records),
        status=state.get("status"),
        blocked_until_epoch=state.get("blocked_until_epoch"),
    )
    save(root / "campaign-summary.json", summary)
    if len(records) == len(snapshot["dataset"]["cases"]):
        report = root / (label + "-report.json")
        if report.exists():
            provenance = report.with_name(report.stem + "-provenance.json")
            old_hash = json.loads(provenance.read_text()).get("source_run_sha256")
            if old_hash != digest(state):
                report = root / (label + "-report-" + digest(state)[:12] + ".json")
        summary["runs"][label]["report"] = str(report)
        if not report.exists():
            subprocess.run(
                [
                    sys.executable,
                    str(runner),
                    "export",
                    "--inputs",
                    str(root / inputs),
                    "--run",
                    str(output),
                    "--output",
                    str(report),
                ],
                check=True,
            )
save(root / "campaign-summary.json", summary)
print(json.dumps(summary, indent=2))
