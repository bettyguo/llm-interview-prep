# Changelog

All notable changes to `llm-interview-prep` are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.2.0] — 2026-05-14

Question bank expansion — now 209 total entries (up from 147), comfortably past the 200-question bar for a qualified interview-prep resource.

Added per topic:
- **T1 ML & DL Fundamentals**: +10 → 25 entries (softmax+CE gradient derivation, bagging vs boosting, EM algorithm, backprop derivation, dropout, SGD variants, early stopping, RF vs GBM, SVMs + kernel trick, logistic vs linear regression).
- **T2 Transformers & LLM Internals**: +8 → 24 entries (Mamba / SSMs, multimodal LLMs / vision-language alignment, residual stream + interpretability, multi-token prediction, tokenization arithmetic effects, weight tying / sharing, grouped-query MoE, softmax-bottleneck).
- **T3 Training & Fine-Tuning**: +7 → 22 entries (PEFT comparison incl prefix / prompt / IA³, rejection sampling fine-tuning, tool-use fine-tuning, RAG vs fine-tuning vs continued pretraining, process reward modeling (PRM), multi-task vs single-task, off-policy vs on-policy RL).
- **T4 Inference & Serving**: +6 → 21 entries (Splitwise / disaggregated serving, model routing / cascades, 405B GPU memory math, TPU vs GPU, dynamic batching, chunked prefill).
- **T5 Retrieval & RAG**: +6 → 21 entries (agentic RAG, multilingual retrieval, code search system design, embedding-as-a-service providers, Self-RAG / CRAG / Adaptive RAG, sparse vs dense vs multi-vector).
- **T6 Agents & Harnesses**: +6 → 20 entries (computer-use agents, agent jailbreak defense, framework comparison incl LangChain / LangGraph / CrewAI / AutoGen / DSPy, agent state persistence, multimodal-vs-text-only agents, deep research agents).
- **T7 Evaluation & Calibration**: +6 → 18 entries (Chatbot Arena methodology, creative-task eval, tool-use accuracy eval, capability vs UX eval, medical/legal domain eval, statistical power and A/B sample size).
- **T9 ML/AI Coding**: +6 → 20 entries (DPO loss in PyTorch, beam search, min-p sampling, KV-cache decode-step update, gradient accumulation, sliding-window attention).
- **T10 Research Discussion**: +4 → 13 entries (DeepSeek-V3 deep-dive, FlashAttention deep-dive, Mixtral / MoE deep-dive, 30-min job-talk structure).
- **T11 Behavioral**: +3 → 15 entries (AI-misuse response, disagreeing with leadership on AI strategy, productive workflows during long training runs).

Every new entry carries an authoritative reference and passes the validator. Linkcheck: all URLs OK (3 additional bot-protected sites — JSTOR, MIT Press — added to ignore list).

## [0.1.0] — 2026-05-14

Initial public release.

- Full 12-topic taxonomy populated end to end: ML/DL fundamentals through behavioral.
- ML system design section with 10 worked drills.
- Three study plans: 8-week default, 4-week experienced, 1-week cram.
- Tooling: `validate_entries.py` (enforces the reference-required rule), `linkcheck.py`, `build.py`.
- CI: validation on every PR, scheduled weekly linkcheck.
- Every non-trivial answer carries an authoritative reference (the answer-correctness protocol).
