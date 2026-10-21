# Four separate answer-quality experiments

Read these directories in numerical order. This work does **not** redefine document coverage as
answer correctness. Every text accuracy figure requires the frozen rubric review. Reviews authored
by the Codex assistant are labeled `model_assisted`, not human gold judgments.

1. **[01-model-comparison](01-model-comparison/README.md)**: same 13 development cases, same exact top-five passages, baseline
   prompt, temperature zero, medium reasoning, 1,400 completion tokens. Compare 20B and 120B.
2. **[02-concise-explanations](02-concise-explanations/README.md)**: same frozen inputs and 120B model; only the prompt changes.
   Remove unnecessary commentary while preserving all material conditions. Compare accuracy,
   output tokens and latency against stage 1's 120B run.
3. **[03-semantic-validation](03-semantic-validation/README.md)**: independently labeled good/bad draft pairs test an optional checker.
   Report false accepts and false rejects separately. Checker decisions never become rubric gold.
4. **[04-holdout](04-holdout/README.md)**: freeze the original full 50-case set and 20 newly authored questions before
   generating holdout outputs. Run the selected candidate on both, with two holdout repetitions.
   Do not revise labels or tune the candidate after viewing holdout answers.

## Experimental controls and operational limits

`evals/experiments/run.py freeze` saves the whole dataset and exact retrieved text/metadata.
`run` fingerprints those inputs and stores the full system prompt/model/settings with each run.
Each result is saved atomically. `--retry-errors` retains prior failed attempts; successful outputs
are never regenerated during resume. Groq SDK HTTP retries and structured-output retries are both
zero for controlled experiments. Production retry settings are unchanged.

A 429 stops the run and saves allowlisted rate-limit headers. Saved Retry-After delays apply across
experiments using the same model. No incomplete run can be exported as a full benchmark; unattempted
questions are not model failures. Exported latency/attempt counts include archived failed attempts;
pacing and retrieval are excluded. Failed-call token usage is unknown. These are generation
benchmarks on frozen retrieval, not live end-to-end latency measurements.

Rate limits have interrupted these runs. A failed provider request is not evidence of a wrong
model answer, nor is a checker outage a successful abstention. No model, prompt or semantic gate is
promoted to the API default without enough evidence. The gate is opt-in and remains fallible.

## Commands

Run from the repository root using `.venv/bin/python evals/experiments/run.py --help`.
Use `freeze --dataset DATASET --output INPUTS`, then:

```sh
.venv/bin/python evals/experiments/run.py run \
  --inputs INPUTS --model openai/gpt-oss-120b --output RUN
# Add --prompt PATH for a saved prompt variant; --retry-errors explicitly retries errors.
.venv/bin/python evals/experiments/run.py export \
  --inputs INPUTS --run RUN --output REPORT
```

Export creates the standard report and an **ungraded** review template. Complete and attribute the
review, then score with `tracerag-review --dataset DATASET --report REPORT --review REVIEW`.
Do not overwrite a completed review by exporting a run twice.

The runtime uses Groq SDK 1.7.0 and Pydantic Settings 2.15.0 from the existing lockfile. Both candidate
models support [Groq strict structured outputs](https://console.groq.com/docs/structured-outputs);
that constrains JSON shape, not factual correctness. Explicit retry control follows the
[official SDK](https://github.com/groq/groq-python#retries).

## Current outcome

| Step | Result | Remaining |
| --- | --- | --- |
| 1. Model comparison | 120B completed 13 development attempts, rubric 9/11; four valid paired 20B outputs saved | 20B rate-limited; full head-to-head incomplete |
| 2. Concise explanations | 9/11, tied with baseline after independent grading adjudication; no demonstrated accuracy gain | Remains experimental |
| 3. Semantic gate | 12 labeled controls; whitespace-only quote fix changes false rejects 4/5 → 0/5, false accepts stay 0/7 | Ten real-draft checks blocked by quota; gate remains opt-in |
| 4. Full set and holdout | Full set 36/40 answerable; holdout run 1 11/12 answerable | Holdout repeat saved five valid outputs, one HTTP 429, fourteen unattempted |

Correct abstentions are a separate weakness: 8/10 on the full set and 4/8 on the new holdout.
The latter also has one unsupported definite catch ruling and three invalid output objects. These
remain documented failures. All answer-text grades are model-assisted and require human review
before claiming human-verified accuracy. Read stage four for complete operational counts and the
commands that resume only rate-limited requests.

Attributed reports and reviews are protected from automatic export overwrite; recovered runs get
new report filenames.
Report provenance hashes the source run. Failed-call token use is unknown, and all latency figures
here measure generation on fixed retrieval, excluding retrieval and pacing.
