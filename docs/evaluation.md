# Evaluation baseline

The current v3 dataset preserves the 50 questions and source labels from v2 and adds correctness
rubrics for all 40 answerable cases. The results below are historical v1/v2 measurements; no v3
answer-correctness result has been recorded. The benchmark contains 40 questions answerable from
the corpus and ten that should produce an abstention. Tags separate ordinary questions from paraphrases, confounders,
multi-document questions, incomplete scenarios, near-domain questions, and adversarial
instructions.

## Retrieval benchmark (v2)

- Corpus season: 2026
- Embedding model: `BAAI/bge-small-en-v1.5`
- Retrieval: exact cosine search through PostgreSQL and pgvector
- Evaluation date: 2026-09-28

| Retrieved passages | Hit rate | Recall | Mean reciprocal rank |
| ---: | ---: | ---: | ---: |
| 1 | 97.50% | 95.00% | 0.9750 |
| 3 | 100% | 100% | 0.9875 |
| 5 | 100% | 100% | 0.9875 |

At `k=1`, `scoring-try-clock` retrieved the semantically similar `clock-runoffs` document instead
of `scoring`. Each multi-evidence question retrieved one of its two required documents, producing
50% recall for that tag. All expected documents were present by `k=3`; increasing to five passages
did not improve retrieval quality on this dataset.

The perfect recall at `k=3` leaves no measured recall headroom for a hybrid retriever yet. A lexical
or fused retriever should be added only with harder cases or a larger corpus that demonstrates a
failure it can address. Rank quality and the amount of evidence sent to generation remain useful
optimization targets.

Ambiguous and near-domain cases intentionally have no expected supporting document, so they are
excluded from retrieval relevance metrics. They are evaluated through answer abstention behavior.

## Grounded-answer benchmark (v2)

The 50-case dataset was evaluated at `k=3` with `openai/gpt-oss-20b` and `medium` reasoning effort.
The first pass completed 46 generations and recorded four HTTP 429 rate-limit failures. A targeted
rerun of only those four cases completed successfully. Quality metrics below include one successful
result for every case; usage and latency include the 50 successful generations.

| Measurement | Result |
| --- | ---: |
| Abstention accuracy | 100% |
| Expected-document coverage | 100% |
| Citation-document precision | 100% |
| Initial provider failures | 4 of 50 |
| Targeted rerun failures | 0 of 4 |
| Provider attempts | 54 |
| Mean successful-answer latency | 7,888.31 ms |
| Input tokens | 52,767 |
| Output tokens | 20,829 |
| Total tokens | 73,596 |

| Case tag | Cases | Abstention accuracy | Expected-document coverage | Citation precision |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 37 | 100% | 100% | 100% |
| Paraphrase | 3 | 100% | 100% | 100% |
| Confounder | 2 | 100% | 100% | 100% |
| Multi-evidence | 2 | 100% | 100% | 100% |
| Adversarial | 2 | 100% | 100% | 100% |
| Ambiguous | 2 | 100% | N/A | N/A |
| Near-domain | 2 | 100% | N/A | N/A |

The multi-evidence cases cited both expected documents. The ambiguous and near-domain cases all
abstained. The adversarial cases answered with the expected document citations; the automated
scorer did not inspect whether the generated text followed the rule facts. The initial provider
failures were operational failures, not abstentions or generated rulings, and are reported
separately from model quality.

## Earlier grounded-answer benchmark (v1)

The initial provider benchmark was recorded on 2026-09-15 against the earlier 37-case dataset: 31
answerable questions and six expected abstentions. It used five retrieved passages,
`openai/gpt-oss-20b`, and `medium` reasoning effort.

| Measurement | Result |
| --- | ---: |
| Abstention accuracy | 100% |
| Expected-document coverage | 100% |
| Citation-document precision | 100% |
| Generation failures | 0 of 37 |
| Mean answer latency | 11,337.70 ms |
| Input tokens | 53,000 |
| Output tokens | 14,359 |
| Total tokens | 67,359 |

## Reasoning-effort experiment

The same 37 cases were evaluated once with `low` reasoning effort before changing the production
default. Retrieval results were unchanged.

| Measurement | Medium | Low |
| --- | ---: | ---: |
| Abstention accuracy | 100% | 97.30% |
| Expected-document coverage | 100% | 96.77% |
| Citation-document precision | 100% | 100% |
| Mean answer latency | 11,337.70 ms | 10,276.95 ms |
| Output tokens | 14,359 | 6,768 |
| Total tokens | 67,359 | 57,992 |

Low reasoning reduced output tokens by 52.87% and total tokens by 13.91%, but it incorrectly
abstained from the answerable `fumble-advancement-restrictions` case. The latency improvement was
9.36% and varied substantially between individual provider requests. `medium` therefore remains the
default until repeated evaluation demonstrates an acceptable quality trade-off.

## What the automatic answer metrics check

- **Abstention accuracy**: fraction of all cases whose actual abstention flag matches the label.
- **Expected-document coverage**: fraction of answerable cases that do not abstain and cite exactly
  the expected set of documents. Missing or extra documents fail this binary check. Despite the
  name, this is exact set matching, not fractional recall. Generation failures count as failures.
- **Citation-document precision**: matching citations divided by total citations, pooled across
  cases; a case with no citations contributes no citations to that denominator.

Expected-document coverage was previously labeled **grounded-ruling accuracy**. The historical
numbers above are unchanged; only the label has been corrected. JSON fields are now
`expected_document_coverage` (aggregate) and `expected_documents_matched` (per case), replacing
`grounded_ruling_accuracy` and `grounded_ruling_correct`. Consumers must update their field names.
Neither citation metric reads ruling text or establishes answer correctness.

## Answer correctness (v3): human rubric review

Each answerable question in `evals/nfl-rules.json` now has an `answer_rubric` with explicit rule
facts, necessary qualifications, and a check for other material unsupported or contradictory
claims. Rubrics are based on the six local corpus documents, not an independent verification of
NFL rulebook accuracy. The corpus deliberately lacks detailed fourth-down fumble-advancement
mechanics, so those cases reward recognizing the restriction and the missing facts rather than
inventing a definitive ruling.

New answer reports save the generated `ruling`, `explanation`, `generation_model`, and retrieved
`evidence` for inspection. A reviewer reads both ruling and explanation, compares them with the
rubric and the actual retrieved passages, and marks each requirement `passed: true` or
`passed: false`. Accept equivalent wording; matching keywords alone is insufficient. Conditional
requirements apply only when that subject is discussed. Every failed requirement needs a note
explaining the error or omission. Identify the reviewer in the review file.

The offline scorer reports:

- **Answer correctness**: cases passing every rubric requirement divided by all 40 answerable
  cases. An answerable case that abstains or has a provider failure scores zero automatically.
- **Mean criterion score**: mean of each case's fraction of passed requirements. Each case has
  equal weight even if rubrics contain different numbers of requirements.
- Per-case scores and failed requirements with reviewer notes, so failures can be inspected.

The ten expected-abstention cases remain measured by abstention accuracy, separately from the
answerable-case correctness score. Review templates include all answerable cases. Incomplete
reviews are rejected rather than producing a partial score. Dataset fingerprints captured during generation bind outputs to the questions and rubrics.
Review fingerprints bind the grades to that exact dataset and report; edited answer text, rubric requirements, duplicate cases, or missing
cases are rejected. Generation failures and abstentions cannot be graded as correct.

For example, an answer saying "a touchdown is worth ten points" while citing `scoring` can pass
expected-document coverage. It must fail the six-point correctness criterion and the check for
contradictory rule claims. Correct citations alone no longer determine its correctness score.

This is a human scoring workflow: the script calculates metrics from explicit reviewer judgments;
it does not automatically understand or judge the answer text. No correctness percentage is
claimed until a fresh report has been reviewed. Historical reports lack the saved text required
for this review and cannot be retroactively assigned correctness scores from their old metrics.

## Interpretation and limitations

- The dataset is intentionally small and aligned with the current six-document corpus. Perfect
  retrieval metrics at `k=3` and historical citation/abstention metrics do not establish general
  NFL-rules accuracy.
- Citation-document precision verifies that citations belong to the expected rule explanation. It
  does not perform claim-level entailment verification. Human rubric review adds factual checks
  but does not implement automatic sentence-level entailment.
- The v1 answer metrics came from one provider run. The v2 run required four targeted reruns after
  transient rate limits. Repeated full trials are still needed to measure model variance.
- Latency depends on the local machine, network, and provider conditions at evaluation time.
- Unsupported questions have no expected source document, so they are excluded from retrieval
  relevance metrics and included in abstention accuracy.

## Reproduce

Run retrieval evaluation without generation calls:

```bash
docker compose exec api tracerag-evaluate --top-k 5
```

Run the `k=1,3,5` retrieval sweep used above:

```bash
docker compose exec api tracerag-evaluate --top-k-sweep 1 3 5
```

Run the complete answer evaluation:

```bash
docker compose exec -T api tracerag-evaluate --top-k 3 --answers > answer-report.json
```

Export an offline review template on the host (using the updated checkout):

```bash
uv run tracerag-review --report answer-report.json > answer-review.json
```

Edit `answer-review.json`: fill in `reviewer` and replace each `passed: null` with a boolean.
Read the corresponding `evidence` in `answer-report.json` while grading. Add notes for every
failed requirement. Leave the answer text, rubric requirements, and fingerprints unchanged.
Then score the completed review without calling the generation provider:

```bash
uv run tracerag-review --report answer-report.json --review answer-review.json > correctness.json
```

Both commands default to `evals/nfl-rules.json`; pass `--dataset PATH` to use another rubric set.
Rebuild the API image with `docker compose up --build -d` before generating a v3 report if it is
still running an older checkout. Keep the dataset, report, review, and score together so the
measurement is reproducible. Repeat generation and review runs to assess model variance; consider
a second reviewer to assess disagreement on the rubrics.
