# 2. Constrain explanations, measured separately

Only the system prompt changes from the stage-1 120B run. It asks for a one-sentence ruling where
possible and an explanation limited to necessary rule conditions and their application. The exact
addition lives in `answering/prompts.py`; the complete submitted prompt is saved here.

Same frozen 13 development questions/passages, medium reasoning, temperature zero and 1,400-token
cap. The existing 11 answer rubrics are unchanged. Review remains explicitly model-assisted.

| Measurement | Baseline | Concise variant |
| --- | ---: | ---: |
| Answerable cases passing every criterion | 9/11 | 8/11 |
| Correct abstentions / two ambiguous cases | 1/2; other output invalid | 1/2; other incorrectly answered |
| Valid outputs / 13 attempts | 12/13 | 13/13 |
| Output tokens, valid calls | 6,803 | 7,265 |
| Input tokens, valid calls | 19,954 | 22,934 |
| Total tokens, valid calls | 26,757 | 30,199 |
| Mean generation latency, valid calls | 1,582 ms | 1,951 ms |

Token totals have unequal valid-output denominators, so they are **not** a paired cost comparison.
Output tokens include model reasoning, so they do not directly measure visible explanation length.
The baseline latency includes an archived 429 attempt on an eventually successful case. Neither
latency includes pacing or retrieval. These single small runs do not establish latency superiority.

The concise variant drops the crossed-and-returned forward-pass restriction, overgeneralizes
incompletion on ground contact, and explains the football act without prior qualifying contact.
It also turns the omitted football-act/sufficient-time fact into a definite incompletion on the
ambiguous ground-contact case. This experiment did not improve accuracy and is not the default.

For further experiments only: `TRACERAG_GENERATION_PROMPT_PROFILE=concise` (wired with stage 3).
