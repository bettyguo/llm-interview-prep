# 04 — Inference & Serving

Where AI engineering meets systems engineering. Interview probes here separate the candidate who can ship an LLM service at $/Mtok the business can afford from the one who only knows `pipeline(...)`.

## What you should walk in able to do

- Explain the **KV cache** — what it stores, why it grows with sequence length, the memory math (2 × n_layers × n_heads × head_dim × seq_len × bytes).
- Explain how **GQA / MQA** reduce KV-cache memory and what the quality tradeoff is.
- Explain **paged attention** (vLLM): the page-table analogy, why it eliminates KV-cache fragmentation.
- Explain **speculative decoding**: the draft-model–target-model dance, acceptance probability, Medusa/EAGLE variants.
- Explain **continuous (inflight) batching** vs. static batching.
- Reason about the **prefill vs. decode** phases — compute-bound vs. memory-bound, and what that implies for batch sizing.
- Compare **quantization** schemes (INT8/INT4 weight-only, GPTQ, AWQ, SmoothQuant, FP8) and their quality/latency tradeoffs.
- Compare **serving stacks**: vLLM, TensorRT-LLM, SGLang, TGI.
- Do **cost math** for an LLM deployment given a GPU, model size, and traffic shape.
- Explain **prompt caching / prefix caching** and when it wins.

## Questions

See [`questions.md`](questions.md).

## References (aggregated)

See [`references.md`](references.md).
