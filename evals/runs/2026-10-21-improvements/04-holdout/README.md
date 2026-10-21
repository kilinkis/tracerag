# 4. Full benchmark and newly authored holdout

`full-inputs.json` freezes all 50 existing v3.1 questions and current top-five retrieved passages.
`dataset.json` and `inputs.json` freeze 20 newly authored questions: 12 answerable and eight expected
abstentions. Cases cover altered timelines, alternate exceptions, reporting/alignment, different
runoff types, sufficient time for a catch, and missing scenario facts. Rubrics/labels were written
before generation and independently reviewed against the local corpus. This is assistant-authored
and assistant-reviewed holdout material, not a human-curated external benchmark. The author knew the
project's failure categories; “new” refers to the exact questions, not unfamiliar rule topics.

Candidate fixed before any holdout outputs: **120B, baseline prompt, medium reasoning, 1,400-token
completion cap, temperature zero, k=5, semantic gate off**. The concise prompt did not improve all-criteria accuracy on the
development set; the gate has not yet earned promotion. The incomplete 20B comparison means this is
an exploratory 120B candidate, not a demonstrated model winner or a new production default.

Planned runs: full 50 once; holdout 20 twice. Record provider failures and unattempted cases
separately. Report all-rubric answer accuracy, correct abstentions, false abstentions, citation
metadata, generation latency and provider-reported tokens. Keep the repeated outputs even when
they fail. Do not revise the prompt, rubric or labels after seeing holdout results.

Rate limits may prevent completion. A frozen dataset or an attempted run is not a completed
benchmark. Consult per-run coverage/status; partial runs have no full-set accuracy claim.

## Recorded results

| Measure | Existing full set | New holdout, run 1 | New holdout, run 2 |
| --- | ---: | ---: | ---: |
| Planned cases | 50 | 20 | 20 |
| Attempted | 50 | 20 | 6 |
| Valid outputs | 48 | 17 | 5 |
| Invalid answer objects | 2 | 3 | 0 observed |
| HTTP failures | 0 | 0 | 1 (429) |
| Unattempted | 0 | 0 | 14 |
| All-rubric correctness on answerable cases | **36/40 (90%)** | **11/12 (91.7%)** | Incomplete; no full score |
| Mean criterion score | 96.46% | 97.22% | Incomplete |
| Proper expected abstentions | **8/10** | **4/8** | None of the eight abstention cases reached |
| Wrong definite answers on expected-abstention cases | 0 | 1 | Not evaluated |
| False abstentions on answerable cases | 0 | 0 | 0 observed |
| Mean generation latency, valid outputs | 1,842 ms | 1,648 ms | 2,073 ms (five outputs only) |
| Input tokens, valid outputs | 78,523 | 28,658 | 8,632 |
| Output tokens, valid outputs | 20,818 | 8,368 | 2,871 |
| Total tokens, valid outputs | 99,341 | 37,026 | 11,503 |

All answer-text grades are **model-assisted rubric reviews**, not human gold adjudication.
An independent assistant reviewed the complete full-set score and holdout judgments. A holdout
ordinary-measurement answer was initially graded too strictly for omitting unrelated penalty
alternatives; adjudication corrected that grade (10/12 → 11/12). This corrected a judgment, with
no change to the frozen question, rubric, evidence or output. See `holdout-1-review-adjudication.json`.
The same review process corrected a conditional football-act judgment in stage 2 (8/11 → 9/11).

The full set still gets **100% expected-document coverage and citation-document precision** while
four answerable outputs fail their factual rubrics. Failures include lost continuous-foot-contact
qualifications, a stricter entire-body-behind rule, and requiring a football act before ground
contact. Document matching is not answer correctness.

The new holdout catches a different problem: the model gives the right no-grounding ruling but
then wrongly says an outside-pocket passer must reach the line, despite a nearby eligible receiver.
On the catch with explicitly unknown football-act/time facts, it treats missing as false and calls
an incompletion. The three other failed abstention cases return locally invalid answer objects;
they are not successful abstentions or provider outages. The original raw invalid objects were not
retained, so their exact field violations cannot be reconstructed. New `output_error_codes`
diagnostics distinguish inconsistent abstention/support fields without logging response text;
these diagnostics were added before run 2, preserving schema, prompt and acceptance rules. Run 2
was rate limited before reaching those cases, so it did not establish their precise failure codes.

Retrieval is scored independently from frozen matches: document recall@5 is 100% on both the
40 existing and 12 holdout answerable cases. MRR is 0.9875 and 1.0 respectively. These document
metrics do not establish that every required sentence was retrieved or faithfully applied.

## Partial repetition and provider limits

Run 2 stopped with HTTP 429 after five valid outputs. Its saved headers report `Retry-After: 557`
seconds. The five shared valid cases already show one rubric outcome change: the receiver-target
explanation fails in run 1 and passes in run 2. Both rulings say no grounding. There is no complete
20-case repeat score or reliable variance estimate. `holdout-2-observed-review.json` grades only the
five observed outputs; the failure and fourteen unattempted questions are not silently scored.

The fixed checker also has a newly frozen ten-draft diagnostic set made from **actual** saved
outputs (seven known bad drafts, three supported), including the missing-fact catch and the 20B
penalty overgeneralization. That additional check is blocked before any model call by the same
quota. Stage 3's completed synthetic-control result must not be presented as a completed check on
these real drafts. The blocked plan is saved in `03-semantic-validation/actual-120b.json`.

After provider capacity is available, resume without regenerating valid outputs or selectively
rerunning invalid model outputs:

```sh
.venv/bin/python evals/experiments/campaign.py --retry-rate-limits
.venv/bin/python evals/experiments/verify.py \
  --challenges evals/runs/2026-10-21-improvements/03-semantic-validation/actual-draft-challenges.json \
  --output evals/runs/2026-10-21-improvements/03-semantic-validation/actual-120b.json
```

The campaign preserves prior provider attempts and creates a new report name if recovered outputs
change a previously exported report. Newly completed review templates still need attributed grading.
`--retry-rate-limits` leaves invalid model-output failures intact. The 20B comparison can separately
resume with that same flag using the stage-1 inputs/model/output. A retry delay is not proof that
all remaining requests will fit; later calls can hit another limit.

No model, concise prompt or semantic gate has been promoted to production defaults. These results
still justify further work on missing-fact decisions and consistent structured abstentions.
