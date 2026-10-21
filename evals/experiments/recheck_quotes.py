"""Reapply the whitespace-only quote fix to saved judgments, without new model calls.

The original guard only changed an ACCEPT into a rejection with the exact sentinel reason below.
Consequently its original acceptance bit can be recovered; its overwritten rationale cannot.
No semantic judgment or label is changed in this replay.
"""

import argparse
import json
from pathlib import Path

from run import digest

from tracerag.answering.verification import quote_present

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--challenges", type=Path, required=True)
parser.add_argument("--run", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
challenges = json.loads(args.challenges.read_text())
run = json.loads(args.run.read_text())
if run["configuration"]["challenges_sha256"] != digest(challenges):
    raise ValueError("Saved judgments belong to a different frozen challenge set")
if run["configuration"].get("quote_policy") != "exact-substring-v0":
    raise ValueError("Replay requires the saved exact-substring baseline policy")
rows = []
for case in challenges["cases"]:
    record = run["results"].get(case["id"], {})
    if "verification" not in record:
        continue
    decision = record["verification"]["decision"]
    was_quote_rejection = decision["reason"] == (
        "Verifier supplied missing or unresolvable evidence quotes"
    )
    raw_acceptance = decision["accepted"] or was_quote_rejection
    cited = {
        m["chunk_id"]: m["text"]
        for m in case["evidence"]
        if m["chunk_id"] in case["draft"]["cited_chunk_ids"]
    }
    quotes_valid = bool(decision["supporting_quotes"]) and all(
        q["chunk_id"] in cited and quote_present(q["quote"], cited[q["chunk_id"]])
        for q in decision["supporting_quotes"]
    )
    after = raw_acceptance and quotes_valid
    rows.append(
        dict(
            case_id=case["id"],
            expected_accepted=case["expected_accepted"],
            before=decision["accepted"],
            after=after,
            raw_acceptance_recovered_from_guard=was_quote_rejection,
            reason_recovery_note="Original rationale overwritten by old guard; unavailable."
            if was_quote_rejection
            else decision["reason"],
        )
    )
summary = {
    "cases": rows,
    "scoring_provenance": "Same saved model judgments and frozen labels; "
    "only deterministic whitespace normalization changes. No new API calls.",
}
for label in ["before", "after"]:
    summary[label] = dict(
        judged=len(rows),
        false_accepts=sum(not r["expected_accepted"] and r[label] for r in rows),
        false_rejects=sum(r["expected_accepted"] and not r[label] for r in rows),
    )
args.output.write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps({k: v for k, v in summary.items() if k != "cases"}, indent=2))
