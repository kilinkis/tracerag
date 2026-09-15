# Evaluation baseline

This baseline measures the initial dense-retrieval and grounded-answer pipeline against
`evals/nfl-rules.json`. The dataset contains 37 questions: 31 answerable from the corpus and six
that should produce an abstention.

## Configuration

- Corpus season: 2026
- Embedding model: `BAAI/bge-small-en-v1.5`
- Generation model: `openai/gpt-oss-20b`
- Reasoning effort: `medium`
- Retrieved passages per question: 5
- Retrieval: exact cosine search through PostgreSQL and pgvector
- Evaluation date: 2026-09-15

## Results

| Measurement | Result |
| --- | ---: |
| Retrieval hit rate at 5 | 100% |
| Retrieval recall at 5 | 100% |
| Mean reciprocal rank | 0.9839 |
| Abstention accuracy | 100% |
| Grounded-ruling accuracy | 100% |
| Citation-document precision | 100% |
| Generation failures | 0 of 37 |
| Mean answer latency | 11,337.70 ms |
| Input tokens | 53,000 |
| Output tokens | 14,359 |
| Total tokens | 67,359 |

The `scoring-try-clock` case was the only answerable question whose expected document did not rank
first. Its expected `scoring` document ranked second behind `clock-runoffs`, which shares strong
clock-related language. The expected document was still retrieved within the configured top five.

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
  answer metrics on one run do not establish general NFL-rules accuracy.
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

Run the complete answer evaluation:

```bash
docker compose exec api tracerag-evaluate --top-k 5 --answers
```
