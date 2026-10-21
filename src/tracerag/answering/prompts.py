"""Versioned additions to the baseline prompt for controlled experiments."""

CONCISE_EXPLANATION = """
Response discipline:
- Give the ruling in one sentence where possible.
- Explain only the rule conditions needed to answer this question, then how the stated facts
  satisfy them. A short answer must still preserve every material exception or alternative.
- For a general rule question, include all material restrictions in the supplied rule passage.
- Do not add a tutorial, new definitions, hypothetical plays, or claims about unrelated fouls.
- Before submitting, remove any sentence that adds no necessary support for the ruling.
"""
