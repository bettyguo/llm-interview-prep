# 06 — Agents & Harnesses

The newest interview topic and the one most candidates handle worst — partly because the field's vocabulary is still settling. Interviewers want to see whether you can reason about *reliability across multi-step LLM workflows*, not whether you can name 12 frameworks.

For depth beyond what's here, see the peer repo `harness-engineer-roadmap`.

## What you should walk in able to do

- Explain **tool use**: schema design, parallel vs. sequential calls, error recovery, retries.
- Contrast **ReAct, Plan-and-Execute, Reflexion, Tree-of-Thoughts** as planning patterns — what each is good for.
- Reason about **multi-step reliability**: failure-mode taxonomy, verifier patterns, rollback.
- Reason about **context-window management** in a long-running agent: summarization, episodic memory, retrieval over agent history.
- Explain **agent evals**: SWE-bench (Verified / Live), TAU-bench, GAIA, AgentBench — what each measures, what each misses.
- Reason about **cost and latency control** in agentic flows: when to short-circuit, when to escalate model size, when to cache.
- State the empirical case for **single-agent over multi-agent** for most tasks, and the few cases where multi-agent wins.

## Questions

See [`questions.md`](questions.md).

## References (aggregated)

See [`references.md`](references.md).
