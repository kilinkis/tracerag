# Recorded answer evaluation: 2026-10-21

This experiment measures generated text as well as citation behavior. Reviews are **model-assisted
Codex judgments**, not human annotations or an independent NFL rules audit. Every answerable case
was read against its frozen rubric and the retrieved passages. A second model reviewed the baseline
for material errors; reviewer agreement was not quantified.

## Artifacts

- `baseline/`: original v3 dataset, original prompt, 50-case report at k=3, completed review,
  correctness scores, and run manifest. These files preserve the first pass, including its provider
  failure. No successful reruns have been substituted into this report.
- `retrieval-sweep.json`: fresh retrieval-only measurements at k=1, 3, and 5.
- `context-diagnostic/`: two additional calls with the original prompt at k=5. This isolates a change
  in context size for these cases; it is not another full benchmark.
- `rubric-calibration.json`: before/after rubrics for v3.1. Unnecessary mandatory context is made
  conditional; substantive rule requirements and unsupported-claim checks are retained.
- `follow-up/`: fresh full run at k=5 using the revised prompt and v3.1 rubric. Baseline and follow-up
  change both generation conditions and some grading criteria, so their aggregate score difference
  must not be presented as a controlled prompt-only effect.
- `guardrail-check-1/` and `guardrail-check-2/`: final prompt/corpus snapshots, two selected
  13-case reports, completed reviews, and scores. The second attempt includes eight HTTP 429s.
- `quota-retry/`: one paced rerun of only those eight operational failures; never substituted
  into the earlier reports.
- `guardrail-retrieval-sweep.json`: retrieval measured again after the corpus clarification.
- `run_benchmark.py`: wrapper around the existing `EvaluationRunner`, with per-case stderr progress.
  It does not change retrieval, generation, or scoring.
- `compose-ports.yaml`: local port override used because port 5432 was already occupied.

Both full generation runs use `openai/gpt-oss-20b`, medium reasoning, 1,400 completion tokens, a 20-second
request timeout, and one application retry for structured-output validation errors. The local index
contains six documents and 21 chunks with 384-dimensional BGE-small embeddings. Generation attempt
counts exclude HTTP retries performed internally by the Groq SDK.

## Baseline findings

The initial checklist score is **24/40 (60%)**. Mean per-case criterion score is **81.875%**.
The automatic citation-document precision is **100%**, while expected-document coverage is **97.5%**
and abstention accuracy is **98%**. There were 49 successful generations and one provider failure
(`confounder-reported-lineman`, with no HTTP status recorded). The provider failure counts as zero
for answer correctness. Successful answers used 71,498 reported tokens and averaged 6,592.17 ms including retrieval
and generation. Tokens consumed by failed requests are not reported by this scorer.

Concrete semantic failures include:

- `paraphrase-sideline-control`: rules incomplete after the question explicitly states control,
  both toes inbounds, and turning upfield before loss. At k=3, the catch-elements passage is absent;
  it ranks fourth. The original-prompt diagnostic at k=5 changes the main ruling to complete.
- `catch-ball-touches-ground`: says ball contact with the ground before a football act makes a pass
  incomplete, dropping the maintained-control exception.
- `pass-legal-forward`: changes the prohibition on the entire body and ball being beyond the line
  into a requirement that the entire body and ball be behind it.
- `catch-toe-drag`: conflates a toe drag with a continuous heel-to-toe/toe-to-heel step.
- `scoring-field-goal`: gives a correct ruling, then changes the prior-contact boundary to crossing
  the goal line in its explanation.
- `adversarial-catch-instruction`: rejects the adversarial one-foot instruction but adds an
  unsupported condition that both feet must be inbounds before ball-ground contact.
- `eligibility-positions`: lists positions correctly but omits the eligible-number/reporting
  qualification present in retrieved evidence.

Other failed criteria include missing qualifications and ancillary rubric demands. The completed
review records each failure separately rather than treating all checklist failures as wrong main
rulings. For example, requiring a general fumble-timing answer to explicitly ask for more play facts
was unnecessary. V3.1 corrects these grading demands without changing questions or source labels.

Document-level recall at k=3 was still 100%. That metric could not detect missing supporting sections
within the correct document: the catch-elements passage ranked fourth for the sideline scenario and
fifth for the going-to-ground question. The two-case diagnostic confirms useful context changes, but
single outputs cannot establish how much the change improves reliability over repeated trials.

## Full follow-up findings

The fresh v3.1/k=5 run scores **31/40 (77.5%)**, with a mean criterion score of **90.625%**. It used
99,006 reported tokens for 49 successful answers, averaging 12,967.09 ms. Citation-document
precision was 96.08%, expected-document coverage 97.5%, and abstention accuracy 96%.

Several specific baseline errors are absent: the sideline ruling is complete, the forward-pass
rule no longer requires the entire body behind the line, the ground-contact explanation preserves
maintained control, the toe-drag distinction is preserved, and field-goal contact is checked before
passing the crossbar. Remaining failures include number/reporting qualifications, overgeneralizing
can to always, turning a sufficient exception into a necessary one, unsupported overall pass
legality, dropped sufficient-time alternatives, and missing explicit fourth-down advancement
restrictions.

One expected abstention (`ambiguous-ground-contact`) instead returns an incomplete ruling and
treats an omitted football act as false. Another (`ambiguous-late-foul-runoff`) fails local
`AnswerDraft` validation. This is a generation-contract failure, not evidence of an HTTP error.
The first run and full follow-up each remain intact; targeted reruns do not replace these rows.

The catch corpus also contained inconsistent shorthand: its definition allowed a football act OR
sufficient time, while its going-to-ground section omitted the time alternative. After this full
run, that section was clarified to preserve the existing alternative. The original text is saved
under `follow-up/corpus/`. The final prompt also distinguishes unknown facts from false ones,
can from always, and sufficient from necessary conditions. Two selected 13-case checks (11
answerable and two expected abstentions each) exercise those changes. These are development
checks, not new full-set estimates.

## Targeted checks of the final guards

The first 13-case check completes all generations, produces both expected abstentions with specific
missing-fact explanations, and scores **6/11** answerable cases on all rubric requirements. Remaining
failures include treating every accepted defensive penalty as a new series, omitting the
crossed-and-returned restriction, conflating sufficient time with a football act, an unclear time
qualification in the combined catch/touchdown answer, and missing fourth-down advancement
restrictions. These selected difficult cases are not a representative correctness estimate.

The second attempt completes five generations and records **eight HTTP 429 failures**. Its literal
all-requirements score is **3/11**, including six answerable cases with no generated text. That
number must not be interpreted as model-only quality. The successful outputs still vary: one
penalty explanation improves while a ground-contact explanation again drops a qualification.
A small diagnostic request succeeds after the rate limits, but a single paced retry of the eight
failed cases is stored separately rather than replacing failures or claiming replicated success.
All eight paced retries also returned HTTP 429. No model text was generated in that retry, so
`quota-retry/operational-summary.json` reports the operational block without presenting a text
accuracy estimate. The retry has no effect on the published full-run metrics.

After reindexing the clarified corpus, retrieval again measures 95% document recall at k=1 and 100%
at k=3 and k=5. The final prompt/corpus has not undergone another complete 50-case text evaluation.
Human adjudication, a held-out set, and repeated full trials remain future work; the data here
supports specific debugging findings rather than a broad accuracy guarantee.

## Scope of the judgments

The 40 answerable cases receive text review; the ten expected-abstention cases are scored by their
abstention flag, not by a separate rubric for the factual accuracy of their reasons. For example,
the baseline late-foul case abstains but incorrectly says runoff evidence is absent, rather than
identifying missing play facts. The revised prompt asks for that distinction; no reason-accuracy
percentage is claimed.

These are small development-set measurements. Rubric calibration used observed failures, there is
no held-out correctness set, and human reviewer agreement has not been measured. The original and
revised artifacts support inspection and future human adjudication rather than broad NFL accuracy
claims.

## Reproduction

From the repository root, build and index with the recorded port override:

```bash
docker compose -f compose.yaml -f evals/runs/2026-10-21-v3/compose-ports.yaml up --build -d
docker compose exec -T api tracerag-index
```

Run a new full benchmark of the final current configuration with progress. This prompt and corpus
include the later guard changes and clarification, which have only undergone targeted checks here;
this command does not regenerate either frozen full-run configuration exactly:

```bash
docker compose exec -T api python -u - --top-k 5 < evals/runs/2026-10-21-v3/run_benchmark.py > new-report.json
```

Recompute a saved score from its exact dataset, report, and completed review:

```bash
uv run tracerag-review --dataset evals/runs/2026-10-21-v3/baseline/dataset.json --report evals/runs/2026-10-21-v3/baseline/answer-report.json --review evals/runs/2026-10-21-v3/baseline/answer-review.json
```

Changing `baseline` to `follow-up`, `guardrail-check-1`, or `guardrail-check-2` selects a later
completed review. To pace a future run, add `--pause-seconds 12` to the wrapper command.
Recomputing scores is deterministic; regenerating answers is not guaranteed to reproduce them.
