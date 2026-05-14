# D7 — AI coding assistant / autocomplete

A worked drill following the [6-step structure](../README.md#the-canonical-6-step-structure).

---

### Q: Design a coding-assistant / autocomplete system (Copilot / Cursor / Claude Code style).

**Category:** system-design
**Difficulty:** senior
**Tags:** [coding-assistant, autocomplete, code-llm, retrieval]

**Short answer.** Two interfaces: **inline completion** (very low latency, small model, prefix+suffix context) and **chat / agent mode** (higher latency, strong model, agentic loop with file-search + code-execution tools). Both rely on a **code-aware retriever** that finds relevant project files via embeddings + symbol-graph + recent-edits. Eval on HumanEval+/LiveCodeBench (for the LM) and SWE-bench Verified (for the agent). Latency is the binding constraint on inline; capability is the binding constraint on agent mode.

**Expansion / why this is the answer.**

**1. Clarify requirements.**
Functional:
- **Inline completion**: as the user types, suggest the next chunk; accept-on-Tab.
- **Chat / agent mode**: free-text request; can read files, run tests, edit code.
- **In-editor explanation**, refactoring suggestions, doc generation.
- Multi-language support.

Non-functional:
- **Inline latency**: p50 < 200 ms, p99 < 500 ms. Token-by-token streaming.
- **Chat latency**: first-token < 2 s; full response within reasonable time.
- **Throughput**: 10k+ QPS (autocomplete is high-volume).
- **Privacy**: user code may not leave certain perimeters (enterprise tier).
- **Acceptance rate**: a UX metric — how often suggestions are accepted.

Clarifying Qs:
- "Inline-only or also chat mode?"
- "Enterprise / on-prem requirements?"
- "Which languages prioritized?"
- "What's the latency budget on inline?"

**2. Define metrics.**

Online:
- **Acceptance rate** (Tab-accept on inline; 'accept change' on agent).
- **Lines accepted per session**.
- **Retention** (DAU/MAU).
- **Time-to-first-suggestion** (latency).

Offline:
- **HumanEval+ / MBPP+ / LiveCodeBench** for raw LM coding capability.
- **Per-language pass-rate** on internal benchmarks.
- **SWE-bench Verified** for agent mode.
- **Repository-level pass@k** (e.g., RepoBench).

Online ↔ offline: bench wins correlate with acceptance rate but not perfectly; A/B online.

**3. Data and labels.**

For the LM (pretraining / continued pretraining):
- Public code (GitHub) with permissive licenses.
- Internal code (with consent, for enterprise / on-prem).
- Docs, issues, PRs (for chat).

Labels:
- For SFT: high-quality `(intent, code)` pairs — often filtered or generated.
- For DPO / RLHF: preference pairs `(intent, accepted, rejected)`.
- For acceptance prediction: real acceptance logs (with strict privacy).

Privacy:
- Code is sensitive. Default: no logging of user code without consent.
- Enterprise: on-prem inference, no telemetry.

**4. Modeling.**

**4a. Inline completion**:
- **Model**: code-specialized small-to-mid LLM (e.g. Codestral-22B, DeepSeek-Coder-V2-Lite, custom-trained model in the few-billion range).
- **Latency-critical**: aggressive optimizations — speculative decoding, INT4 quantization, prefix caching.
- **Context**: editor prefix + suffix (`<|fim_prefix|> ... <|fim_suffix|> ... <|fim_middle|>` for fill-in-the-middle).
- **Retrieval-augmented**: pull in relevant files / function defs from the project.

**4b. Chat / agent mode**:
- **Model**: a strong frontier-class LLM (Claude, GPT, custom big model).
- **Loop**: ReAct or similar with tools: `read_file`, `edit_file`, `run_tests`, `grep`, `execute_code`.
- **Context management**: compaction; selective file inclusion.

**4c. Retriever (code-aware)**:
- **Embeddings**: code-trained encoder (Voyage code-2, CodeRankEmbed, internal).
- **Symbol graph**: function calls, imports, type relationships from a language server / tree-sitter parse.
- **Recent edits**: prioritize files the user just touched.
- **Open-tab boost**: files currently open in the editor are high-signal.

**4d. Fine-tuning per organization (Enterprise)**:
- Small SFT on the organization's coding style / patterns.
- LoRA-served via multi-LoRA.

**5. Serving.**

Inline:
- Co-located inference (close to user) for latency.
- Speculative decoding for token throughput.
- INT4 / FP8 quantization.
- Streaming token-by-token.
- KV-cache for ongoing sessions.
- **Cancel-on-keystroke**: if the user types another character, cancel the in-flight completion.

Chat / agent:
- Standard LLM serving stack (vLLM, internal).
- Tool execution sandbox: containers per session.

Privacy / on-prem:
- Self-hosted models for enterprise tier.
- No data egress.

**6. Monitoring + iteration.**

Acceptance rate as the headline online metric.

Drift:
- New language features, framework versions.
- User-coding-style shifts.

Retraining:
- Base model: quarterly to yearly (expensive).
- Fine-tunes: monthly with new data.
- Retrieval-specific tuning: as the codebase changes.

Evals:
- Per-language pass-rate dashboarded.
- Repository-level evals on representative codebases.
- SWE-bench Verified for agent mode; track per-model release.

A/B every change.

User feedback:
- Explicit thumbs-up/down + comments.
- Implicit: accept-rate, edit-after-accept rate (a low edit-after-accept signals good suggestions).

---

**Common follow-ups.**

- "How do you reduce inline-completion latency further?" → Smaller model + speculative decoding + INT4 weights + co-located inference.
- "How do you handle the privacy / enterprise tier?" → On-prem inference; air-gapped option; no telemetry without opt-in.
- "How do you balance the base LLM quality vs. retrieval quality?" → Both matter; retrieval is necessary for project-specific code; base LLM determines floor.

**Common mistakes.**

- Single-mode design (only chat or only inline).
- No retrieval; the model has no project context.
- Skipping latency budget on inline.
- No fine-tune per enterprise — generic model misses internal patterns.

**References.**

- [Chen et al. — "Evaluating Large Language Models Trained on Code" (HumanEval)](https://arxiv.org/abs/2107.03374) — HumanEval.
- [Jimenez et al. — SWE-bench](https://arxiv.org/abs/2310.06770) — SWE-bench.
- [Bavarian et al. — "Efficient Training of Language Models to Fill in the Middle"](https://arxiv.org/abs/2207.14255) — FIM.
- [DeepSeek-Coder paper](https://arxiv.org/abs/2401.14196) — code-specialized model.
- [Anthropic — Claude Code overview](https://docs.claude.com/en/docs/claude-code/overview) — coding-agent architecture.
