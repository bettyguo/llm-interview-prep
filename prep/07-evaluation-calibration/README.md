# 07 — Evaluation & Calibration

The topic where interviewers find out whether you can ship something *and tell whether it's working*. This is also where most production AI projects quietly fail — the team built a thing without an eval.

## What you should walk in able to do

- Reason about **benchmark contamination**: how to detect it (n-gram overlap, exact-match, perplexity comparison), how to mitigate it.
- Reason critically about **LLM-as-judge**: position bias, length bias, self-preference bias; the mitigations (pairwise comparisons with randomized order, multiple judges, reference-based grading).
- Contrast **pairwise vs. single-grade evaluation** and when each is appropriate.
- Reason about **calibration**: ECE, reliability diagrams, temperature scaling.
- Reason about **hallucination measurement**: TruthfulQA, FActScore, FaithBench, their limits.
- Design an **eval set** for a real task — representativeness, sample size, stratification, statistical significance.
- Contrast **offline vs. online evals** and reason about shadow traffic, A/B, gated rollout.

## Questions

See [`questions.md`](questions.md).

## References (aggregated)

See [`references.md`](references.md).
