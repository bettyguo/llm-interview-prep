# Changelog

All notable changes to `llm-interview-prep` are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.3.0] — 2026-05-14

Question bank expansion to 501 entries (from 209). Per topic:

- T1 ML & DL Fundamentals: 25 → 62 (+37)
- T2 Transformers & LLM Internals: 24 → 63 (+39)
- T3 Training & Fine-Tuning: 22 → 61 (+39)
- T4 Inference & Serving: 21 → 63 (+42)
- T5 Retrieval & RAG: 21 → 55 (+34)
- T6 Agents & Harnesses: 20 → 49 (+29)
- T7 Evaluation & Calibration: 18 → 48 (+30)
- T9 ML/AI Coding: 20 → 42 (+22)
- T10 Research Discussion: 13 → 23 (+10)
- T11 Behavioral: 15 → 25 (+10)

New coverage areas: information theory primitives, fairness metrics + DP, multi-armed bandits, meta-learning, lottery-ticket, mode collapse, Mamba/SSMs, multimodal LLMs, residual stream, MTP, attention sinks, FlashAttention v1/v2/v3 differences, FFN-as-KV-memory, mixture-of-depths, ring attention, Muon optimizer, RLVR/reasoning RL, model collapse on synthetic data, self-play / STaR / RAFT, ZeRO sharding details, PCGrad, FP8 training, muP, iterative DPO, model merging / SLERP / TIES / DARE, disaggregated serving / Splitwise / Mooncake, CUDA Graphs, BitNet 1.58-bit, decode-time KV pruning (H2O/SnapKV), structured generation cost, mixed-batch prefill, TPU vs GPU, throughput vs goodput, EAGLE-2/3 spec streaming, agentic RAG / Self-RAG / CRAG, contextual retrieval, late chunking, ColBERT multi-vector, code-search system design, multilingual retrieval / BGE-M3, table-heavy document ingest, computer-use agents, agent jailbreak defense, framework comparison, agent state persistence, deep-research agents, parallel tool calling, subagent delegation, skill libraries / Voyager, vibe-driven dev anti-pattern, Chatbot Arena methodology, RewardBench, RULER, ARC-AGI, MMMU multimodal eval, IFEval, LiveCodeBench, GSM-Plus adversarial, G-Eval rubrics, MT-Bench, Arena Hard, BBH, harm-rate measurement, hallucination at production scale, behavior change detection across versions, BLEU/ROUGE retrospective, plus many more.

All entries validated via tools/validate_entries.py with the reference-required rule. Linkcheck: 493 URLs checked, all OK (27 known bot-protected URLs ignored).

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
