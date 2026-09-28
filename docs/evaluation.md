# Evaluation baseline

The current benchmark uses 50 versioned questions: 40 answerable from the corpus and ten that
should produce an abstention. Tags separate ordinary questions from paraphrases, confounders,
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

## Grounded-answer benchmark (v1)

The initial provider benchmark was recorded on 2026-09-15 against the earlier 37-case dataset: 31
answerable questions and six expected abstentions. It used five retrieved passages,
`openai/gpt-oss-20b`, and `medium` reasoning effort.

| Measurement | Result |
| --- | ---: |
| Abstention accuracy | 100% |
| Grounded-ruling accuracy | 100% |
| Citation-document precision | 100% |
| Generation failures | 0 of 37 |
| Mean answer latency | 11,337.70 ms |
| Input tokens | 53,000 |
| Output tokens | 14,359 |
| Total tokens | 67,359 |

The v2 answer benchmark has not yet been recorded. Its additional cases are specifically intended
to test multi-document citation coverage, abstention on incomplete scenarios, and resistance to
instructions that conflict with the evidence.

## Reasoning-effort experiment

The same 37 cases were evaluated once with `low` reasoning effort before changing the production
default. Retrieval results were unchanged.

| Measurement | Medium | Low |
| --- | ---: | ---: |
| Abstention accuracy | 100% | 97.30% |
| Grounded-ruling accuracy | 100% | 96.77% |
| Citation-document precision | 100% | 100% |
| Mean answer latency | 11,337.70 ms | 10,276.95 ms |
| Output tokens | 14,359 | 6,768 |
| Total tokens | 67,359 | 57,992 |

Low reasoning reduced output tokens by 52.87% and total tokens by 13.91%, but it incorrectly
abstained from the answerable `fumble-advancement-restrictions` case. The latency improvement was
9.36% and varied substantially between individual provider requests. `medium` therefore remains the
default until repeated evaluation demonstrates an acceptable quality trade-off.

## Interpretation and limitations

- The dataset is intentionally small and aligned with the current six-document corpus. Perfect
  retrieval metrics at `k=3` and v1 answer metrics do not establish general NFL-rules accuracy.
- Citation-document precision verifies that citations belong to the expected rule explanation. It
  does not perform claim-level entailment verification.
- Answer metrics come from one provider run. Repeated trials are needed to measure model variance
  and intermittent generation failures.
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
docker compose exec api tracerag-evaluate --top-k 5 --answers
```
