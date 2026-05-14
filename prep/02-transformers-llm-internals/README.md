# 02 — Transformers & LLM Internals

The single highest-signal topic for any 2026 AI/LLM role. Interviewers in this space probe internals because they're trying to distinguish "I've used the HuggingFace API" candidates from "I understand what the API is doing" candidates.

## What you should walk in able to do

- Derive **scaled dot-product attention** on a whiteboard and explain every term.
- Contrast **MHA vs. MQA vs. GQA** and explain the KV-cache implications.
- Explain **positional encodings** — sinusoidal, learned, ALiBi, RoPE — and when each is used.
- Explain **encoder-only vs. decoder-only vs. encoder–decoder** architectures and which problem each fits.
- Explain **pre-norm vs. post-norm** and why almost all modern LLMs are pre-norm.
- Explain **MoE** routing, load balancing, top-k, expert capacity, and the modern fine-grained variants.
- Explain **FlashAttention** at a sketch level: what it actually changes vs. naive attention.
- Explain **scaling laws** — Kaplan's original recipe vs. the Chinchilla revision — and what each says about compute-optimal training.
- Engage critically with claims about **emergent abilities** post-Schaeffer (2023).

## Self-assessment quiz

If you cannot, right now, sketch the Q/K/V math and explain why we divide by √d_k, stop here and spend a focused day. This is the question.

## Questions

See [`questions.md`](questions.md).

## References (aggregated)

See [`references.md`](references.md).
