# 1-week cram

Triage. Your loop is in 7 days. You will not master new material; you will refresh, sharpen openers for the most-likely questions, and rehearse one system-design drill. Aim ~3–5 hr/day.

## Mon — Transformers internals + coding warmup

- AM: T2 questions list. Drill: attention math, MHA vs. MQA vs. GQA, KV cache math, RoPE, MoE.
- PM: Implement scaled dot-product attention from scratch in NumPy. Aim 30 min, no peeking.

## Tue — Training & Fine-tuning

- AM: T3 — RLHF vs. DPO; LoRA / QLoRA; distributed-training names.
- PM: T3 hard-mode drills: reward hacking; mixed-precision pitfalls; ZeRO-3 vs. FSDP one-liner.

## Wed — Inference & Serving + Retrieval & RAG

- AM: T4 — KV cache math, paged attention, speculative decoding, quantization.
- PM: T5 — chunking, hybrid retrieval, reranking, lost-in-the-middle, RAG eval.

## Thu — Agents + Evaluation

- AM: T6 — tool use; ReAct; agent eval suites (SWE-bench, TAU-bench).
- PM: T7 — LLM-as-judge biases; contamination detection; hallucination metrics.

## Fri — One system-design drill, timed

- Pick the drill whose shape most resembles what the company you're interviewing at builds:
  - Consumer product → D1 (video recs) or D5 (feed ranking).
  - LLM-app shop → D6 (RAG customer support) or D8 (enterprise agent).
  - Coding-AI shop → D7 (coding assistant).
- Timed 45 min, no peek; then read the worked drill and self-grade.

## Sat — Your-own-work + Behavioral

- AM: Rehearse the 30-second, 5-minute, and 30-minute version of your most relevant project. Out loud. Recorded.
- PM: Draft 5 STAR stories: conflict, ambiguous spec, missed deadline, hardest bug, biggest mistake. Plus 2 AI-specific: chose-not-to-use-ML, model-failed-in-prod.

## Sun — Mock + sleep

- AM: One full mock (45 min technical + 45 min system design + 30 min behavioral) with a friend or self-recorded.
- PM: Gap fill on the 5 things you bombed. Then **stop**. Sleep matters more than the marginal hour at this point.

## What to skip in a 1-week cram

- T1 fundamentals deep-dive (skim only).
- T9 coding beyond the attention warmup (you'll likely get one or two; trust your existing reflexes).
- T10 paper deep-dives (you'll get a paper question; lean on the 3 you already know).
- T8 drills 1–10 in sequence (one drill only).

## What absolutely not to skip

- The attention-from-scratch coding warmup.
- One full system-design drill timed.
- 5–7 STAR stories written down.
