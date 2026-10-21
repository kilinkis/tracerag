# 3. Optional semantic validation

The generator still emits typed drafts and the server still resolves citation metadata from
retrieved records. `VerifiedAnswerGenerator` adds an independent call checking the **whole ruling
and explanation**, using only the draft's cited passages. The checker gets no evaluation rubric or
expected answer. A rejection becomes an abstention. A provider failure remains an operational
failure. Existing generator abstentions and invalid citation IDs skip this call.

A positive verdict must include exact quotes, and code checks quote membership and chunk identity.
Those deterministic checks do not establish that quotes entail every sentence. The model still
judges semantic relevance, omitted conditions and missing scenario facts, and can be wrong.

The runtime and standard evaluation reports preserve the verifier model, decision, reason, quotes
and provider usage. Trace token totals include generator and verifier calls on successful results;
the nested verifier usage is a breakdown, not additional tokens to add again. Generation attempts
count the primary generator calls; verification is an additional call. Failed-call usage is unknown.

`challenge-set.json` contains assistant-labeled valid drafts and controlled corruptions of the same
rules: “can”→“every”, dropped sufficient-time alternatives, invented forward-pass permission,
missing number/reporting qualifications, and definite rulings on missing facts. The labels were
written before verifier outputs. They are development checks, not an independent human holdout.
`verify.py` saves separate false-accept/false-reject counts and the actual post-gate draft. It never
uses a checker acceptance as the gold criterion.

Enable for an explicit candidate runtime with:

```sh
TRACERAG_VERIFICATION_ENABLED=true
TRACERAG_VERIFICATION_MODEL=openai/gpt-oss-120b
```

The gate remains off by default until it demonstrates acceptable rejection/acceptance behavior.
Adding a checker is not evidence that the documented model failures have all been fixed.

## Measured result and the fix required

All 12 labeled draft checks completed with 120B: five supported drafts and seven bad drafts.
The initial exact-character substring guard rejected **four of five good drafts** because the
model joined Markdown line wraps into spaces. It accepted **zero of seven bad drafts**.

The fix normalizes whitespace for quote lookup and leaves words, case and punctuation unchanged.
Reapplying just that deterministic guard to the **same saved model judgments** yields:

| Guard policy | False accepts / 7 bad drafts | False rejects / 5 good drafts |
| --- | ---: | ---: |
| Exact character substring | 0/7 | 4/5 |
| Whitespace-normalized substring | 0/7 | 0/5 |

See `whitespace-fix-replay.json`. No labels, rubrics, prompts or semantic judgments were revised to
obtain the improved result. The original code overwrote the positive model rationale when the
quote guard rejected it; the replay recovers only the known original acceptance bit from that
specific branch's sentinel reason. It does not invent the lost rationale. New runtime results
retain `model_decision` separately from the effective guarded `decision`, and the experiment runner
versions the quote policy to prevent mixed-policy resumes.

This is a small development control set. Zero errors here does not establish sentence-level
entailment accuracy or justify calling all generated answers correct. The full benchmark still
uses independent rubric review. Runtime quote matching now means whitespace-normalized literal
matching, not exact character identity.
