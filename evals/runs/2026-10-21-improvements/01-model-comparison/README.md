# 1. Controlled model comparison

`inputs.json` freezes the existing 13-case development subset (11 answerable, two ambiguous),
including exact top-five passages from the final catch corpus. Both models get identical prompt,
question, evidence, schema, temperature 0, medium reasoning and 1,400-token cap. Only the model
identifier changes. Completion cap includes reasoning, so equal caps are a resource control rather
than a claim of equal reasoning allocation.

`20b.json` and `120b.json` retain per-case outputs, errors, rate-limit headers and previous attempts.
The 20B run was rate limited before completing. The 120B baseline completed all 13 attempts: 12 valid
outputs, one locally invalid answer. It answered one ambiguous question with a proper abstention;
the other ambiguous case had an invalid output, not a successful abstention.

The attributed model-assisted rubric review scores **9/11 answerable cases (81.8%)**, mean criterion
score **94.7%**. Two failures remain: the forward-pass answer adds an entire-body-behind requirement;
the ground-contact answer uses an overbroad “only if” condition. This is qualitative assistant review
against frozen corpus rubrics, not human gold evaluation.

Do not compare this 9/11 with the older 6/11 targeted run as a controlled model effect. Only the
new paired runs share this experiment's frozen inputs and retry configuration. Until both models
finish, there is no complete head-to-head result and no justified automatic model promotion.
