# Default 8-week study plan

For someone prepping cold for an AI/ML/LLM engineering interview loop. Assumes ~12–15 hr/week. End state: exam-ready across every topic on the spine.

## Week 1 — ML & DL Fundamentals (Topic 01)

**Goal.** Refresh classical ML and DL fundamentals to "no hesitation" level.

| Day | Focus | Output |
|-----|-------|--------|
| Mon | Bias–variance, regularization, supervised vs. self-supervised | self-quiz, 5 questions out loud |
| Tue | Loss functions, optimization (SGD → Adam → AdamW), LR schedules | self-quiz, derive cross-entropy from MLE |
| Wed | Activation functions, vanishing/exploding gradients, initialization | self-quiz |
| Thu | Normalization (BatchNorm vs. LayerNorm vs. RMSNorm) | one whiteboard explanation, timed 5 min |
| Fri | Classical algorithms (linear, GBM, SVM, kNN, k-means, PCA) | "when would you use X over Y" drills |
| Sat | Evaluation metrics, leakage, class imbalance | one mock with a friend |
| Sun | Weekly review, retake quiz | quiz score ≥ 80% |

**End of week 1 check.** Ten Topic-1 questions in a row without looking. If not, repeat the weak day.

## Week 2 — Transformers & LLM Internals (Topic 02)

**Goal.** Internalize attention math, KV-cache implications, scaling laws. This is the highest-yield week.

| Day | Focus |
|-----|-------|
| Mon | Scaled dot-product attention; multi-head; derivation |
| Tue | Positional encodings (sinusoidal, RoPE, ALiBi) |
| Wed | MQA / GQA and KV-cache memory math |
| Thu | Encoder vs. decoder; pre-norm vs. post-norm |
| Fri | MoE: gating, top-k, load balancing, fine-grained experts |
| Sat | FlashAttention sketch; scaling laws (Kaplan vs. Chinchilla); emergent abilities critique |
| Sun | Implement attention from scratch in NumPy — see [`prep/09-coding/snippets/attention.py`](../prep/09-coding/snippets/attention.py) — without notes |

## Week 3 — Training & Fine-Tuning + coding interleave (Topics 03 + 09)

| Day | Focus |
|-----|-------|
| Mon | Pretraining objectives; SFT dataset design |
| Tue | RLHF (PPO), DPO, IPO, KTO, GRPO — what each optimizes |
| Wed | LoRA, QLoRA, DoRA — math + parameter savings |
| Thu | Distributed training: ZeRO 1/2/3, FSDP, TP, PP |
| Fri | bf16 vs. fp16 stability; gradient accumulation; gradient checkpointing |
| Sat | Coding: write a minimal training loop, then nucleus sampling |
| Sun | Coding: implement k-means and logistic regression from scratch |

## Week 4 — Inference & Serving + Retrieval & RAG (Topics 04 + 05)

| Day | Focus |
|-----|-------|
| Mon | KV cache math; paged attention |
| Tue | Speculative decoding; continuous batching; prefill vs. decode |
| Wed | Quantization (INT8/INT4, GPTQ, AWQ, FP8); serving stacks |
| Thu | Embeddings; chunking; HNSW vs. IVF; hybrid retrieval |
| Fri | Reranking; HyDE / query rewriting |
| Sat | Build a toy RAG locally (use the peer repo `build-your-own-ai` for a reference build) |
| Sun | RAG evaluation: retrieval metrics + generation metrics + end-to-end |

## Week 5 — Agents & Evaluation (Topics 06 + 07)

| Day | Focus |
|-----|-------|
| Mon | Tool use, parallel tool calls, error recovery |
| Tue | Planning patterns (ReAct, Plan-and-Execute, Reflexion) |
| Wed | Agent harness design; context management; agent evals (SWE-bench, TAU-bench, GAIA) |
| Thu | LLM-as-judge biases and mitigations |
| Fri | Benchmark contamination; hallucination measurement |
| Sat | Design an eval set for a small task (1-hour exercise) |
| Sun | Offline vs. online evals; shadow traffic; gated rollout |

## Week 6 — ML System Design (drills 1–5)

One drill per day, timed 45 minutes, then read the worked solution. Do not peek.

| Day | Drill |
|-----|-------|
| Mon | D1 — Video recommendation |
| Tue | D2 — Ad CTR |
| Wed | D3 — Content moderation |
| Thu | D4 — E-commerce semantic search |
| Fri | D5 — News feed ranking |
| Sat–Sun | Review weak drills; mock with a friend |

## Week 7 — ML System Design (drills 6–10) + Research Discussion (Topic 10)

| Day | Focus |
|-----|-------|
| Mon | D6 — RAG customer support |
| Tue | D7 — AI coding assistant |
| Wed | D8 — Enterprise agent |
| Thu | D9 — Fraud detection |
| Fri | D10 — Feature store |
| Sat | Pick 3 papers; write one-pagers for each (contribution / method / eval / threats / take) |
| Sun | Rehearse 30-second / 5-minute / 30-minute version of your own work |

## Week 8 — Behavioral + mocks + gap fill (Topic 11)

| Day | Focus |
|-----|-------|
| Mon | Draft 7 behavioral stories in STAR form |
| Tue | AI-specific behavioral (model failed in prod; chose not to use ML; communicated model risk) |
| Wed | Mock interview — technical |
| Thu | Mock interview — system design |
| Fri | Mock interview — behavioral |
| Sat | Gap fill: re-do the 5 questions you bombed across mocks |
| Sun | Rest, light review, sleep |

## What to expect from this plan

- A candidate who follows the plan honestly walks into a loop with full topic coverage, two timed system-design drills under their belt, 5–7 polished behavioral stories, and a coding warmup that includes attention-from-scratch.
- The plan errs on **depth + repetition** over breadth. Skipping or skimming days has compounding cost — week 6's drills assume week 2–4's internals.

## Variants

- **Slipping schedule.** Drop week 9 (Behavioral) compression by 2 days into weeks 1–8 by trimming weekly review.
- **Loop is heavy on system design.** Move drills D6–D10 to week 6 alongside D1–D5; compress week 7's paper work.
- **Loop is heavy on agents.** Expand week 5 to 8 days; pull from `harness-engineer-roadmap`.
