# 4-week experienced plan

For a senior engineer with prior ML exposure who has 4 weeks. Assumes ~15–20 hr/week and that you can skim T1 fundamentals at speed. Front-loads modern LLM-era topics.

## Week 1 — Fundamentals refresh (fast) + Transformers (deep)

| Day | Focus |
|-----|-------|
| Mon | T1 ML/DL fundamentals — read the questions list; deep-dive only on weak spots |
| Tue | T2 attention math + multi-head; implement attention from scratch |
| Wed | T2 positional encodings; MQA/GQA + KV cache implications |
| Thu | T2 pre-norm vs. post-norm; MoE |
| Fri | T2 FlashAttention; scaling laws (Chinchilla); emergent abilities critique |
| Sat | T9 coding: stable softmax, layernorm, transformer block |
| Sun | Self-assessment quiz on T2; gap fill |

## Week 2 — Training + Inference

| Day | Focus |
|-----|-------|
| Mon | T3 pretraining objectives + SFT |
| Tue | T3 RLHF / DPO / GRPO; reward hacking |
| Wed | T3 LoRA / QLoRA; distributed training |
| Thu | T4 KV cache math; paged attention; quantization |
| Fri | T4 speculative decoding; continuous batching; serving stacks |
| Sat | T9 coding: training loop, nucleus sampling |
| Sun | Self-assessment on T3 + T4 |

## Week 3 — RAG + Agents + Evaluation + a System-Design drill

| Day | Focus |
|-----|-------|
| Mon | T5 retrieval, chunking, indexes, hybrid + reranking |
| Tue | T5 query rewriting / HyDE; RAG eval |
| Wed | T6 tool use; ReAct / Plan-and-Execute / Reflexion |
| Thu | T6 agent evals; harness design |
| Fri | T7 LLM-as-judge biases; contamination; hallucination measurement |
| Sat | T8 drill D6 (RAG customer support) — timed 45 min |
| Sun | T8 drill D8 (enterprise agent) — timed 45 min |

## Week 4 — System Design + Research + Behavioral + Mocks

| Day | Focus |
|-----|-------|
| Mon | T8 drill of your choice — timed |
| Tue | T8 drill of your choice — timed |
| Wed | T10 — three paper one-pagers; rehearse your-own-work talks |
| Thu | T11 — draft 7 behavioral stories; AI-specific behavioral |
| Fri | Technical mock |
| Sat | System-design mock + behavioral mock |
| Sun | Gap fill; rest |

## Variants

- **Loop is heavy on classical ML.** Spend more of week 1 on T1; trim T2 internals to one day.
- **You already work on transformers daily.** Skip the T2 implementation day; gain a day for system design.
