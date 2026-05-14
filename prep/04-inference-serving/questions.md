# Inference & Serving — questions

Entries follow the [Q&A schema](../../CONTRIBUTING.md#the-qa-entry-schema).

---

### Q: Why is LLM decode memory-bandwidth-bound, and what does that imply for throughput?

**Category:** concept
**Difficulty:** mid
**Tags:** [decode, memory-bandwidth, throughput]

**Short answer.** Decode produces one token at a time. Per decode step, the model reads its full weights (and the KV cache) but does only `O(d²)` of compute per layer — `≈ 1` token's worth — while modern GPUs have a compute-to-bandwidth ratio of ~300+ FLOPs per byte. So memory traffic, not FLOPs, sets the speed. Implication: batching is the only thing that pushes a memory-bound kernel into the compute-bound regime, which is why throughput-oriented LLM serving stacks (vLLM, TGI, SGLang) live and die by their batching strategies.

**Expansion / why this is the answer.**
- **Arithmetic intensity** = FLOPs / bytes-loaded. For a decode step:
  - FLOPs per matmul ≈ `2 · m · n · k` for output `(m, n)`.
  - Bytes loaded ≈ weight size + activations.
- Decode reads the entire weight matrix to multiply by a single-token activation — high bytes, few FLOPs → low arithmetic intensity, memory-bound.
- **Roofline model**: throughput = min(compute / FLOP_rate, bytes_to_move / bandwidth). H100 has ~989 TFLOPs/s bf16 and ~3.3 TB/s HBM3 bandwidth → break-even arithmetic intensity ~300 FLOPs/byte.
- **Why batching helps**: doubling batch size doubles FLOPs but only marginally increases bytes loaded (weights stay the same; activations and KV cache scale). Arithmetic intensity rises linearly with batch size until you saturate the compute side.
- **Prefill is different**: prefill processes `n_prompt` tokens at once, so it's already at high arithmetic intensity even at batch 1 — prefill is compute-bound, decode is memory-bound.
- **What this implies for serving**:
  - Throughput-per-GPU rises sharply with batch size; vLLM and others target very large effective batch via continuous batching.
  - Latency per token does *not* scale linearly with batch — adding requests to a memory-bound kernel is mostly free up to a point.

**Common follow-ups.**
- "When does decode become compute-bound?" → At very large batch sizes (depending on model: dozens to hundreds), or when activations are quantized aggressively so per-step compute is the dominant cost.
- "Why is first-token-latency dominated by prefill but inter-token latency by decode?" → Different regimes; prefill is compute-bound and dominates initial response time; decode is memory-bound and dominates streaming speed.

**Common mistakes.**
- Saying "more FLOPs = slower decode." More FLOPs from batching are nearly free on a memory-bound kernel.
- Forgetting that the KV cache also has to be read every decode step.

**References.**
- [Pope et al. — "Efficiently Scaling Transformer Inference"](https://arxiv.org/abs/2211.05102) — the canonical analysis of LLM-serving regimes.
- [NVIDIA — "Hopper Architecture Whitepaper"](https://resources.nvidia.com/en-us-tensor-core) — HBM/compute spec for arithmetic intensity math.
- [Kwon et al. — "vLLM" paper](https://arxiv.org/abs/2309.06180) — applied batching analysis.

---

### Q: Walk me through paged attention. What problem does it solve?

**Category:** concept
**Difficulty:** senior
**Tags:** [paged-attention, vllm, kv-cache, memory-fragmentation]

**Short answer.** Paged attention (Kwon et al. 2023, vLLM) splits the KV cache into fixed-size blocks ("pages") and uses an indirection table so each request's KV cache need not be contiguous in GPU memory. Solves the **KV-cache memory fragmentation** problem: traditional contiguous-allocation serving wastes 60–80% of KV memory on padding/over-reservation; paged attention drops waste to <4%, enabling much higher batch sizes and throughput.

**Expansion / why this is the answer.**
- **The fragmentation problem in pre-vLLM serving stacks**:
  - You must reserve KV memory for `max_seq_len` per request.
  - Most requests don't fill it → internal fragmentation.
  - Allocating contiguously over many concurrent requests of different lengths → external fragmentation (no slot fits a long request even though enough free memory exists).
  - Result: 60–80% of KV memory wasted.
- **Paged attention solution** (analogous to virtual memory in OSes):
  - Split KV cache into fixed-size blocks (e.g. 16 tokens).
  - Per-request **block table** maps logical positions to physical block addresses.
  - Allocate blocks on demand as new tokens are generated.
  - Free blocks immediately when requests complete.
- **Attention kernel**: must follow the block table to gather K and V from non-contiguous locations. vLLM ships a custom CUDA kernel for this.
- **Wins**:
  - Memory waste drops from 60–80% to <4%.
  - 2–4× higher serving throughput on real workloads in the paper.
  - Enables **prefix sharing**: when many requests share a prompt prefix, their block tables can point to the *same* physical blocks for that prefix — copy-on-write semantics.
- **Caveats**: kernel complexity; minor latency overhead vs. perfectly contiguous (but the throughput win pays for it many times over).

**Common follow-ups.**
- "What's prefix caching?" → Application-level use of paged attention: a system prompt's KV cache is materialized once, then shared across many concurrent requests.
- "How does this compare to FlashAttention?" → Orthogonal: FlashAttention is a tile-based attention kernel; paged attention is a memory-layout scheme. vLLM uses both.

**Common mistakes.**
- Saying paged attention "approximates" attention — it's exact; only the memory layout changes.
- Calling it a "GPU-level paging" — it's at the LLM-serving layer, not the GPU driver.

**References.**
- [Kwon et al. — "Efficient Memory Management for Large Language Model Serving with PagedAttention"](https://arxiv.org/abs/2309.06180) — vLLM paper.
- [vLLM project](https://github.com/vllm-project/vllm) — primary implementation.

---

### Q: Explain speculative decoding. What does the draft model do?

**Category:** concept
**Difficulty:** senior
**Tags:** [speculative-decoding, draft-model, medusa, eagle]

**Short answer.** Speculative decoding (Leviathan et al. 2023) uses a small fast "draft" model to predict K future tokens, then verifies them in parallel with a single forward pass of the large "target" model. Tokens the target would have generated are accepted; mismatches trigger a rollback. Result: same final output as the target alone, with often 2–3× lower latency, because expensive target-model decode steps are reduced to one verifying pass per K speculated tokens.

**Expansion / why this is the answer.**
- The algorithm:
  1. Draft model proposes `K` tokens: `d_1, d_2, ..., d_K`.
  2. Target model runs one forward pass on `[context, d_1, ..., d_K]`, getting target distributions `p_1, ..., p_K` at each position.
  3. For each `d_i`, with probability `min(1, p_target(d_i) / p_draft(d_i))`, accept; otherwise reject and stop accepting from here on.
  4. If `d_i` is rejected, replace it with a sample from a corrected distribution `(p_target − p_draft)_+` to maintain exact target-distribution output.
- **Correctness**: the math (Leviathan / Chen et al. 2023) shows that the output distribution is exactly the target's — speculative decoding is **lossless**.
- **Why it speeds up**: the target's expensive forward pass produces K logits-positions in one go. If the draft is right `n` times in a row, you got `n+1` tokens for one target-forward.
- **Acceptance rate** depends on draft-target alignment; typical 60–90% per token at K=4–8.
- **Variants**:
  - **Medusa** (Cai et al. 2024): no separate draft model; add `M` LM heads to the target that predict `M` future positions. Self-speculation.
  - **EAGLE** (Li et al. 2024): predict the *features* (hidden states) of future positions, not tokens — better acceptance rate.
  - **Lookahead decoding** (Fu et al. 2024): no draft model; use Jacobi iteration to refine future tokens in parallel.

**Common follow-ups.**
- "Is the output identical to greedy decoding?" → Yes, statistically. With temperature sampling, output is sampled from the exact target distribution; with greedy, it's identical.
- "What ruins acceptance rate?" → Draft and target disagreeing often — usually because the draft model is too small / poorly aligned. Worse acceptance ⇒ less speedup.
- "Does it cost more compute?" → Slightly: the draft model runs `K` extra forward passes, and the target's forward has a slightly longer sequence. But wall-clock latency drops.

**Common mistakes.**
- Calling speculative decoding "approximate." It is not; it's exact under the algorithm in the paper.
- Saying it "always" speeds up. If the draft acceptance rate is low, it can be a wash.

**References.**
- [Leviathan, Kalman, Matias — "Fast Inference from Transformers via Speculative Decoding"](https://arxiv.org/abs/2211.17192) — Google's version.
- [Chen et al. — "Accelerating Large Language Model Decoding with Speculative Sampling"](https://arxiv.org/abs/2302.01318) — DeepMind's contemporaneous version.
- [Cai et al. — "Medusa"](https://arxiv.org/abs/2401.10774) — Medusa.
- [Li et al. — "EAGLE"](https://arxiv.org/abs/2401.15077) — EAGLE.

---

### Q: Walk me through continuous (inflight) batching. Why is it crucial for LLM serving?

**Category:** concept
**Difficulty:** mid
**Tags:** [batching, continuous-batching, throughput]

**Short answer.** Continuous (or inflight) batching adds new requests to the running batch *between decode iterations*, rather than waiting for the whole batch to finish. Because LLM decode is sequential and request lengths are heterogeneous, static batching wastes GPU time waiting for the longest request; continuous batching cuts that idle time and lifts throughput by 2–5×.

**Expansion / why this is the answer.**
- **Static batching pathology**:
  - You batch N requests together at time T_0.
  - Each request generates a variable number of tokens (1 to thousands).
  - The batch can't complete until the longest request finishes.
  - Shorter requests' slots sit idle once they hit EOS.
- **Continuous batching** (Yu et al. 2022, Orca; popularized in production by vLLM):
  - Every decode step, the scheduler can evict completed requests and admit new ones from the queue.
  - The "batch" is a continuously-changing set; the GPU is always full.
  - Pre-fill (the new request's prompt) can be interleaved or done in a dedicated stream.
- **Why this matters at scale**:
  - With many concurrent users and variable prompt/response lengths, idle slots in static batching can be 50%+ of total compute.
  - Continuous batching turns 50% idle into near-100% utilization.
- **Tradeoffs / complications**:
  - **Prefill vs. decode interference**: a long prefill blocks decode tokens. Modern stacks use **chunked prefill** (split prefill into multiple steps, interleaved with decode) to bound the latency hit.
  - **Latency variance**: per-token latency rises slightly compared to dedicated-stream serving, because each step now has more work.
  - **Memory pressure**: more concurrent requests → more KV cache → needs paged attention to fit.
- **Stack picks** (2026): vLLM, TGI, SGLang, TensorRT-LLM all implement continuous batching.

**Common follow-ups.**
- "What's chunked prefill?" → Split a long prompt's prefill into chunks (e.g. 512 tokens) and interleave with decode steps; bounds first-token-latency for users without big-prefill backlog.
- "When does continuous batching hurt?" → Very low traffic — overhead of scheduling > benefit; static-batch one-request-at-a-time would be marginally faster.

**Common mistakes.**
- Confusing continuous batching with dynamic batching (a more general term used in CV serving) — continuous is the LLM-specific version that handles per-token scheduling.
- Forgetting that continuous batching needs paged attention to scale; static-allocation KV caches don't accommodate variable concurrency well.

**References.**
- [Yu et al. — "Orca: A Distributed Serving System for Transformer-Based Generative Models"](https://www.usenix.org/conference/osdi22/presentation/yu) — Orca / inflight batching.
- [Kwon et al. — "vLLM" paper](https://arxiv.org/abs/2309.06180) — continuous batching + paged attention together.
- [vLLM docs — Continuous batching](https://docs.vllm.ai/en/latest/) — primary docs.

---

### Q: Compare INT8, INT4 (GPTQ / AWQ), and FP8 quantization. What does each cost?

**Category:** concept
**Difficulty:** senior
**Tags:** [quantization, gptq, awq, fp8]

**Short answer.** **INT8** weight-only: cut weight memory in half (vs. bf16); near-zero quality loss; standard for most inference today. **INT4** (GPTQ, AWQ): 4× memory reduction; small quality drop on most tasks at 70B+; trickier at 7B-class. **FP8** (E4M3 / E5M2): newer floating-point format; on H100+ tensor cores, drop-in for weights *and* activations; near-zero quality loss; can be used for training (DeepSeek-V3 trained in FP8). Choice depends on hardware (older GPUs lack FP8), accuracy budget, and whether you need quantized activations too.

**Expansion / why this is the answer.**
- **Quantization fundamentals**:
  - "Weight-only": activations stay in higher precision (bf16); only weights are quantized.
  - "Activation quantization" (SmoothQuant, etc.): quantize activations too; harder because activations have outliers.
- **INT8 weight-only**: per-channel or per-tensor scaling; basically a memory-saving win for free at most scales. Default for many production deployments.
- **GPTQ** (Frantar et al. 2022): post-training INT4 weight-only via second-order error compensation. Calibration set required (a few hundred samples).
- **AWQ** (Lin et al. 2023): activation-aware quantization — protect "salient" weight channels (the ones with large activation magnitudes); per-channel scale. Generally outperforms GPTQ at the same bit width.
- **FP8** (Micikevicius et al. 2022):
  - **E4M3** (4-bit exponent, 3-bit mantissa): more precision, less range — for weights and forward-pass activations.
  - **E5M2**: more range, less precision — for gradients during training.
  - On H100+, tensor cores support FP8 natively; throughput is ~2× bf16.
- **Quality bench** (rough; consult per-model):
  - INT8 weight-only: <0.5 MMLU-pt drop on most models.
  - INT4 (AWQ): ~1–2 pt drop at 70B-class; larger at 7B.
  - FP8: comparable to bf16 with good per-tensor scaling.
- **Activation quantization** is harder because of **outlier features** — a few activation channels have very large values; naive per-tensor scaling loses precision elsewhere. SmoothQuant (Xiao et al. 2022) shifts the outlier magnitude into weights, then quantizes both.

**Common follow-ups.**
- "Why does AWQ beat GPTQ?" → AWQ protects activation-salient channels; GPTQ minimizes weight-reconstruction error, which doesn't always correspond to model-quality minima.
- "What's KV-cache quantization?" → Quantize K and V *as you store them* — FP8 / INT8 / INT4. Cuts KV-cache memory linearly; needs care at INT4 to avoid quality drop on long context.
- "When can't you use FP8?" → Pre-Hopper GPUs (A100, etc.) don't have FP8 tensor cores.

**Common mistakes.**
- Reporting "INT4 with no quality loss" — there's always some loss; the question is whether it's tolerable.
- Confusing post-training quantization (no training) with quantization-aware training (calibration during finetune).

**References.**
- [Frantar et al. — "GPTQ"](https://arxiv.org/abs/2210.17323) — GPTQ.
- [Lin et al. — "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration"](https://arxiv.org/abs/2306.00978) — AWQ.
- [Xiao et al. — "SmoothQuant"](https://arxiv.org/abs/2211.10438) — activation-outlier handling.
- [Micikevicius et al. — "FP8 Formats for Deep Learning"](https://arxiv.org/abs/2209.05433) — FP8 spec.

---

### Q: What is prompt caching / prefix caching, and what does it require?

**Category:** concept
**Difficulty:** mid
**Tags:** [prefix-caching, prompt-caching, kv-cache]

**Short answer.** Prefix caching materializes the KV cache for a common prompt prefix once, then reuses it across requests that share that prefix — typically a long system prompt, retrieved context, or chat history. Requires (1) the serving stack to support persistent KV slabs, (2) a way to detect shared prefixes (often by hashing the token-id sequence), and (3) for the underlying attention kernel to gather KV from non-contiguous storage (paged attention is the natural enabler).

**Expansion / why this is the answer.**
- The cost saved: prefilling a 5,000-token system prompt is expensive. If 100 requests share it, prefilling once saves 99× the prefill cost.
- **vLLM**: prefix caching via the paged-attention block table — hash the token-id prefix, dedupe blocks, reference-count.
- **Anthropic Prompt Caching API**: explicit API control over what is cached, including beta features for cache TTL. Cost: writes are slightly more expensive than uncached prefills; reads are much cheaper.
- **OpenAI / Google**: implicit prompt caching (no API knob for some, explicit for others); the system caches recently-seen prefixes automatically.
- **What invalidates a cache hit**:
  - Any divergence in token ids at the start of the prefix.
  - Tokenizer changes.
  - Some KV-quantization mismatches.
- **Cache hit shape**: complete prefix-match from byte zero is the safe assumption. Some advanced systems do partial / longest-common-prefix matching across requests.

**Common follow-ups.**
- "Does this reduce *total* compute or just per-request latency?" → Both, if the cache is hot. The first request's prefill is unchanged; the second and beyond reuse the cached blocks.
- "What's the memory cost?" → A cached prefix takes the full KV memory of a prefix-length prompt. Trade compute-vs-memory.

**Common mistakes.**
- Treating prefix caching as a no-op feature — it requires architecture support (paged attention) to do well.
- Forgetting that cache eviction strategy matters under memory pressure.

**References.**
- [Kwon et al. — "vLLM" paper](https://arxiv.org/abs/2309.06180) — paged attention enables block-level prefix sharing.
- [Anthropic — Prompt caching docs](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) — primary docs for the Claude API.
- [OpenAI — prompt caching docs](https://platform.openai.com/docs/guides/prompt-caching) — OpenAI's version.

---

### Q: How would you do back-of-envelope cost math for an LLM deployment?

**Category:** concept
**Difficulty:** senior
**Tags:** [cost, serving, capacity-planning]

**Short answer.** Cost / Mtok = (GPU rental $/hr × number of GPUs) / (output tokens / hr). To get output tokens / hr at steady state: throughput = batch × tokens-per-second per request, where the batch is set by KV-cache memory and the latency budget. Walk through: pick model & GPU; compute weight memory; compute per-token KV; figure out how many concurrent requests fit; estimate decode TPS per request at that batch; multiply.

**Expansion / why this is the answer.**
- Cost = (cluster $ / hr) ÷ (throughput tokens / hr).
- **Worked example**: serve LLaMA-3-70B (140 GB bf16 weights) on 2× H100 (80 GB each), each $4/hr cloud → cluster cost $8/hr.
  - Weights: 140 GB; spread over 2 GPUs via tensor parallel → 70 GB/GPU. 10 GB left per GPU for KV + activations.
  - KV per token (GQA n_kv_heads=8, n_layers=80, d_head=128, bf16 = 2B): `2·80·8·128·2 = 327,680 bytes ≈ 320 KB/token`. Cluster sees 2× because TP shards it: ~160 KB/token-per-GPU.
  - 10 GB free per GPU at peak ≈ 60k tokens of total context (sum over batched requests).
  - At 4k context per request → ~15 concurrent requests fit. Realistic operating point: 8–10 to leave headroom.
  - Decode TPS per request at batch 8 on 2× H100 for 70B GQA: ~50 tok/s (rough, depends on stack).
  - Aggregate output throughput: 8 × 50 = 400 tok/s = 1.44 Mtok / hr.
  - Cost: $8 / 1.44 Mtok = **$5.55 / Mtok output**.
- **Quantization shifts the math**: INT4 weights → 35 GB/GPU; much more KV budget; ~5–10× more concurrent batch → much lower $/Mtok.
- **Speculative decoding** can ~2× decode TPS → halve cost.
- **Tradeoff lines**:
  - Larger batch → more KV memory, lower $/Mtok, but higher per-request latency.
  - Smaller model → lower memory, higher batch, but worse quality.
- **In production**, real numbers come from load-testing your specific stack, not from BoE — but the BoE should be within 2×.

**Common follow-ups.**
- "What's the input-token cost?" → Prefill is compute-bound; per-input-token cost is ~10× cheaper than per-output-token. APIs price accordingly.
- "When does it make sense to use a closed API vs. self-host?" → If your traffic is bursty / low (<<10M tokens/day), API is cheaper. At sustained high traffic with consistent load, self-hosting wins on $/Mtok, especially with INT4 + spec-decoding.

**Common mistakes.**
- Treating GPU $ as the only cost — networking, storage, ops are non-trivial in production.
- Using batch=1 throughput numbers; batch matters a lot.

**References.**
- [Pope et al. — "Efficiently Scaling Transformer Inference"](https://arxiv.org/abs/2211.05102) — formal LLM-inference cost analysis.
- [Anyscale — "Continuous Batching" blog](https://www.anyscale.com/blog/continuous-batching-llm-inference) — practical benchmarks.

---

### Q: Compare nucleus (top-p), top-k, and temperature sampling. When do you use each?

**Category:** concept
**Difficulty:** intro
**Tags:** [sampling, decoding, top-p]

**Short answer.** **Temperature** scales logits before softmax: `T < 1` sharpens (more deterministic), `T > 1` flattens (more diverse). **Top-k** truncates to the k highest-probability tokens before sampling. **Top-p (nucleus)** truncates to the smallest set of tokens whose cumulative probability ≥ p; size of the set adapts per step. In practice, modern stacks use temperature + top-p together (e.g. T=0.7, top_p=0.9), with top-k as a safety cap (e.g. top_k=50).

**Expansion / why this is the answer.**
- **Greedy** (T → 0): always pick argmax. Deterministic but repetitive; bad for open-ended generation.
- **Pure temperature sampling**: sample from `softmax(logits / T)`. Issue: the long tail still contains rare-but-possible tokens that can derail generation.
- **Top-k** (Fan et al. 2018): keep the k highest-probability tokens, set the rest to 0, renormalize. Hyperparameter `k` doesn't adapt to entropy: if the top-1 is 99% likely, top-k=50 still considers 49 nearly-impossible tokens.
- **Top-p / nucleus** (Holtzman et al. 2019): keep the smallest token set with cumulative probability ≥ p. Adapts: when distribution is peaked, you keep few; when flat, you keep many.
- **Min-p sampling** (newer; Nguyen et al. 2024): keep tokens with probability ≥ `min_p · p_top`. Better tail control than top-p in some cases.
- **Modern stack defaults**:
  - Code / math: temperature near 0 (more deterministic).
  - Creative writing: temperature ~0.7–1.0, top-p 0.9.
  - Chatbot reply: T=0.7, top-p=0.9, top-k=50 as a cap, plus repetition penalty.
- **Beam search** is mostly absent from modern LLM generation — it produces dull, generic text; useful in translation but not in open-ended generation (the Holtzman paper popularized this insight).

**Common follow-ups.**
- "What's repetition penalty?" → Apply a multiplier `< 1` to logits of already-seen tokens. Crude but effective at preventing the "ChatGPT-loop" failure.
- "Why does beam search give bad open-ended text?" → It maximizes total log-prob, which favors generic, "safe" continuations.

**Common mistakes.**
- Using top-k=1 to mean "greedy" — technically equivalent, but conceptually muddled.
- Mixing up temperature placement: it scales logits before any truncation.

**References.**
- [Holtzman et al. — "The Curious Case of Neural Text Degeneration"](https://arxiv.org/abs/1904.09751) — top-p, beam-search critique.
- [Fan, Lewis, Dauphin — "Hierarchical Neural Story Generation"](https://arxiv.org/abs/1805.04833) — top-k.

---

### Q: What's the difference between prefill and decode? Why do they need different optimizations?

**Category:** concept
**Difficulty:** mid
**Tags:** [prefill, decode, throughput-latency]

**Short answer.** **Prefill** = process the whole prompt in one forward pass. Produces all the K/V tensors for the prompt at once. Compute-bound. **Decode** = generate one token at a time, attending to the growing KV cache. Memory-bandwidth-bound. They differ in arithmetic intensity, in latency-vs-throughput characteristics, and so in what kernels and batching strategies serve them best.

**Expansion / why this is the answer.**
- **Prefill**:
  - Input: prompt tokens.
  - Cost per layer: `O(n_prompt² · d)` for attention + `O(n_prompt · d²)` for FFN.
  - For non-trivial prompts (>100 tokens), this is compute-bound on modern GPUs.
  - High arithmetic intensity → fewer batching gains needed.
  - Latency: dominates first-token-latency.
- **Decode**:
  - Input: one new token; attends back over the full KV cache.
  - Cost per layer: `O(n · d)` for attention + `O(d²)` for FFN.
  - Memory-bound (see the memory-bandwidth question).
  - Batching is essential.
  - Latency: dominates inter-token-latency.
- **Implications for serving**:
  - **Don't interfere**: a giant prefill blocks decode tokens. **Chunked prefill** splits prefill into ~512-token chunks interleaved with decode steps.
  - **Different SLOs**: first-token-latency (governed by prefill) and inter-token-latency (governed by decode) are tracked separately.
  - **Hardware partitioning**: some stacks (Splitwise, Mooncake) literally split prefill and decode across separate GPUs/nodes to avoid resource conflicts.
- **Speculative decoding** turns decode steps into a more-prefill-like operation (a single forward verifying many tokens) — partially compute-bound when acceptance is high.

**Common follow-ups.**
- "What's Splitwise?" → Patel et al. 2023 — separate GPU pools for prefill vs. decode; each tuned for its workload.
- "Is decode ever compute-bound?" → At very large batch sizes, or with extensive activation quantization, yes.

**Common mistakes.**
- Treating decode as a small version of prefill — they're qualitatively different (one matmul shape, one arithmetic intensity).
- Forgetting that long prompts make first-token-latency dominate user experience.

**References.**
- [Pope et al. — "Efficiently Scaling Transformer Inference"](https://arxiv.org/abs/2211.05102) — prefill/decode regimes.
- [Patel et al. — "Splitwise: Efficient Generative LLM Inference Using Phase Splitting"](https://arxiv.org/abs/2311.18677) — physical separation.
- [Agrawal et al. — "Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve" (chunked prefill)](https://arxiv.org/abs/2403.02310) — chunked prefill.

---

### Q: Compare vLLM, TensorRT-LLM, SGLang, and TGI as serving stacks.

**Category:** concept
**Difficulty:** mid
**Tags:** [vllm, tensorrt-llm, sglang, tgi, serving-stacks]

**Short answer.** **vLLM**: open-source, paged attention + continuous batching, the most widely-deployed OSS option. **TensorRT-LLM**: NVIDIA's stack; deeply optimized CUDA kernels and FP8/INT4 support; faster on NVIDIA hardware but heavier to integrate. **SGLang**: research-leaning open-source; RadixAttention for prefix sharing, fast structured generation, good for agent / multi-turn workloads. **TGI** (Text Generation Inference): HuggingFace's serving stack; good integration with HF ecosystem; less feature-leading vs. the others lately.

**Expansion / why this is the answer.**
- **vLLM** (Berkeley / community, OSS):
  - Paged attention; continuous batching; prefix caching; multi-LoRA; speculative decoding; FP8.
  - Excellent breadth of model support.
  - The de-facto default for self-hosted OSS models.
- **TensorRT-LLM** (NVIDIA, OSS-ish):
  - Highly optimized CUDA kernels per-GPU-arch.
  - Strong FP8 / INT4 support, including FP8 on Hopper.
  - Build-time compilation per model (less flexible than vLLM).
  - Best wall-clock on NVIDIA hardware *if* you invest in the integration.
- **SGLang** (Zheng et al. 2023, LMSYS):
  - **RadixAttention**: a tree-structured prefix cache for highly-branching workloads (agent loops, multi-turn).
  - First-class **structured generation** (compiled grammar / JSON-mode at high throughput).
  - **xGrammar / FSM-guided decoding**: very fast constrained decoding.
  - Strong picks for agent / RAG workloads, fast-moving project.
- **TGI** (HuggingFace, OSS):
  - Good first-mover; some features (continuous batching, paged attention) have caught up to vLLM but lag in others.
  - Strongest if you live in the HF ecosystem.
- **Other notable**: Triton + custom kernels; mistral.rs / llama.cpp for CPU-edge; DeepSpeed-Inference (now mostly subsumed by vLLM patterns).
- **Decision rubric**:
  - Self-host any OSS model fast: **vLLM**.
  - Max NVIDIA performance, willing to engineer: **TensorRT-LLM**.
  - Agent loops, structured output, prefix-share heavy: **SGLang**.
  - Already-HF-pipeline shop: **TGI**.

**Common follow-ups.**
- "What's RadixAttention?" → SGLang's data structure: a radix tree of KV-cache blocks indexed by prefix. Handles many-branching prompts (e.g. tree-of-thought) efficiently.
- "Does TensorRT-LLM support continuous batching?" → Yes, in current versions (inflight batching).

**Common mistakes.**
- Picking based on raw benchmark numbers without considering integration cost.
- Forgetting that the model-quality bar comes from the model, not the stack — stacks affect latency/throughput, not output quality.

**References.**
- [vLLM project](https://github.com/vllm-project/vllm) — primary.
- [TensorRT-LLM project](https://github.com/NVIDIA/TensorRT-LLM) — primary.
- [SGLang project](https://github.com/sgl-project/sglang) and [SGLang paper](https://arxiv.org/abs/2312.07104) — primary.
- [TGI project](https://github.com/huggingface/text-generation-inference) — primary.

---

### Q: What is structured generation / constrained decoding? How does it work?

**Category:** concept
**Difficulty:** mid
**Tags:** [constrained-decoding, structured-output, grammar, json-mode]

**Short answer.** Structured generation forces the model's output to conform to a schema (JSON, regex, grammar). It works by masking the next-token distribution at each decode step: tokens that would violate the schema are set to `−inf` before softmax, so the model can only sample valid continuations. Modern implementations (Outlines, xGrammar, lm-format-enforcer, Guidance) compile the schema into a finite-state machine for fast per-step masking.

**Expansion / why this is the answer.**
- **Why constrain at all**: production LLM apps usually need parseable output. Prompt-only "please output JSON" works sometimes; fails under high concurrency, weird inputs, or smaller models.
- **The mechanism**:
  1. Compile the schema (JSON schema, regex, context-free grammar) into a FSM.
  2. At each decode step, the FSM determines which next tokens are valid given the prefix-so-far.
  3. Set logits of invalid tokens to `−inf`.
  4. Sample as normal from the masked distribution.
- **Per-step cost**: very small with a well-designed FSM. xGrammar reports ~1µs per step.
- **Failure modes**:
  - The schema is so restrictive that the model can't complete it — early refusal or empty output.
  - The model wanted to say something but the schema forced a different token sequence; can produce semantically off content.
  - Token-boundary mismatches: a JSON value like `"true"` may tokenize as `["tr", "ue"]`, and the FSM has to track partial-token state.
- **Modern implementations**:
  - **Outlines** (dottxt-ai): widely used; supports regex and JSON-schema.
  - **xGrammar** (now part of SGLang): high-perf grammar-guided decoding.
  - **lm-format-enforcer**: similar.
  - **OpenAI structured outputs** (2024): server-side, model-aware; very reliable for OpenAI models.
  - **Anthropic tool use**: schema-driven, with the model trained for it.

**Common follow-ups.**
- "Does constrained decoding change the model's quality?" → It can: if the model's "preferred" continuation isn't in the grammar, you trade fluency for schema-validity.
- "Why is JSON output hard?" → Token boundaries don't align with JSON tokens. A robust implementation tracks partial-token state.

**Common mistakes.**
- Calling structured generation "just prompting" — it's a kernel-level mask, not a prompt technique.
- Assuming it has no quality cost.

**References.**
- [Willard & Louf — "Efficient Guided Generation for Large Language Models" (Outlines)](https://arxiv.org/abs/2307.09702) — Outlines.
- [Dong et al. — "XGrammar"](https://arxiv.org/abs/2411.15100) — xGrammar.
- [OpenAI — Structured Outputs docs](https://platform.openai.com/docs/guides/structured-outputs) — primary docs for OpenAI's version.

---

### Q: How does the choice of n_kv_heads / GQA affect serving cost? Walk through with numbers.

**Category:** derivation
**Difficulty:** senior
**Tags:** [gqa, kv-cache, cost-math]

**Short answer.** Halving `n_kv_heads` halves KV-cache memory, halves KV-read bandwidth per decode step, and (because decode is memory-bound) roughly doubles decode throughput per GPU. Concretely, a 70B GQA-8 model has 8× smaller KV cache than the same-shape MHA-64 model, so it admits ~8× more concurrent tokens of context at the same memory budget, dropping $/Mtok proportionally.

**Expansion / why this is the answer.**
- KV bytes per token = `2 · n_layers · n_kv_heads · d_head · bytes`.
- For LLaMA-3-70B (n_layers=80, d_head=128, bf16):
  - MHA (n_kv_heads=64): `2 · 80 · 64 · 128 · 2 = 2.62 MB/token`.
  - GQA-8 (n_kv_heads=8): `2 · 80 · 8 · 128 · 2 = 327 KB/token`. **8× less**.
- **Throughput impact** (memory-bound regime):
  - Decode step reads weights (constant per step) + KV cache (per active token across batch).
  - KV cache grows linearly with batch × context.
  - At long context, KV traffic dominates → halving KV roughly halves bandwidth, ~doubling throughput.
- **Memory impact**:
  - On 2× H100 (160 GB total), 70B in bf16 is 140 GB weights; 20 GB free for KV + activations.
  - GQA-8 lets you fit ~60 concurrent tokens, i.e. ~15 simultaneous 4k-context requests.
  - MHA-64 at the same shape: ~7 concurrent tokens — barely 2 requests fit. **Different deployment economy**.
- **Quality**: GQA-8 vs. MHA-64 quality gap on benchmarks is empirically small (single-digit pt or less at this scale). The compute saved goes into more parameters elsewhere, often yielding *better* quality.
- **MLA** (DeepSeek-V2/V3) takes this further: K and V are projected from a small shared latent vector, reducing cache to factor of `n_kv_heads · d_head → d_latent`.

**Common follow-ups.**
- "What's the bandwidth math?" → H100: 3.35 TB/s HBM3. At GQA-8 70B with 4k-context per request × 10 concurrent ≈ 13 GB/decode-step → ~250 decode-steps/sec across the batch. Rough; modulated by kernel efficiency.
- "Why don't all models pick GQA-1 (MQA)?" → Quality drops more sharply at MQA than GQA-8 in long-context tasks; GQA-8 is the modern sweet spot.

**Common mistakes.**
- Forgetting to use `n_kv_heads`, not `n_heads`.
- Treating GQA as quality-free — there's a small drop; the value is in the tradeoff.

**References.**
- [Ainslie et al. — "GQA"](https://arxiv.org/abs/2305.13245) — GQA paper.
- [DeepSeek-V2 paper](https://arxiv.org/abs/2405.04434) — MLA.
- [Pope et al. — "Efficiently Scaling Transformer Inference"](https://arxiv.org/abs/2211.05102) — bandwidth-vs-throughput math.

---

### Q: What's quantized KV cache, and what does it cost?

**Category:** concept
**Difficulty:** mid
**Tags:** [kv-cache-quantization, fp8-kv, int4-kv]

**Short answer.** Storing the KV cache in lower precision (FP8 / INT8 / INT4) instead of bf16. Cuts KV memory by 2–4×, allowing larger batch / longer context. Quality cost is generally small for FP8 and INT8; INT4 KV starts to bite on long context. Most modern serving stacks (vLLM, TensorRT-LLM, SGLang) support FP8 KV out of the box.

**Expansion / why this is the answer.**
- The math: KV bytes scale with bytes/element. bf16 = 2 B, FP8 = 1 B, INT4 = 0.5 B.
- **FP8 KV cache** (E4M3 typical): near-free quality; the FP8 dynamic range is wide enough that K and V magnitudes fit without much issue. Supported on H100+.
- **INT8 KV cache**: per-tensor or per-token scaling. Small quality drop on short context; can compound on long context.
- **INT4 KV cache**: aggressive; KIVI (Liu et al. 2024) and others show small drops with careful per-channel quantization, but long-context tasks (long-doc QA, summarization) start to degrade.
- **Tradeoff with KV access patterns**:
  - Decode reads the full KV cache every step; lower precision = less bandwidth, faster decode.
  - But lower precision also reduces numerical fidelity of attention scores; cumulative error over long contexts.
- **In practice**: most production deployments at 2026 use FP8 KV if hardware supports it; INT8 as a fallback on older GPUs; INT4 only when memory is the binding constraint and quality drop is acceptable.

**Common follow-ups.**
- "Can you mix precisions per-layer?" → Yes — some research stacks use higher precision on the most-impactful layers (e.g. first and last few).
- "How does KV quantization interact with paged attention?" → Cleanly: pages can store quantized blocks; the gather kernel needs to dequantize on load.

**Common mistakes.**
- Confusing weight quantization with KV-cache quantization (independent axes — you can do either, or both).
- Reporting "no quality loss" without measuring on long-context tasks.

**References.**
- [Liu et al. — "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache"](https://arxiv.org/abs/2402.02750) — aggressive KV quantization.
- [NVIDIA — TensorRT-LLM FP8 KV cache docs](https://docs.nvidia.com/deeplearning/tensorrt-llm/) — primary docs.

---

### Q: What does prefill latency depend on? How do you reduce time-to-first-token (TTFT)?

**Category:** concept
**Difficulty:** mid
**Tags:** [ttft, prefill, latency]

**Short answer.** TTFT ≈ prefill cost + queuing. Prefill cost is `O(n_prompt² · n_layers · d)` for attention plus `O(n_prompt · d²)` for FFN — quadratic in prompt length. Reduce TTFT by: chunked prefill (avoid blocking decode), prefix caching (skip re-compute for shared prefixes), parallel pre-fill across GPUs (TP), speculative decoding for the first few tokens, and pruning the prompt (only what the model needs to see).

**Expansion / why this is the answer.**
- **TTFT decomposition**:
  - Queuing time (how long before your request gets a GPU slot).
  - Prefill compute time.
  - Network / framework overhead.
- **Prefill cost** (single layer): `~6 · n_prompt² · d_model + 12 · n_prompt · d_model²` for the standard MHA+FFN with the GPT-3 flop count formula. Attention is quadratic; FFN is linear.
- **Levers**:
  - **Prefix caching**: massive TTFT cut when prompts share a system prompt.
  - **Chunked prefill** (Sarathi-Serve): split prefill into chunks (e.g. 512 tokens), interleaved with decode. Bounds TTFT for shorter prompts in a multi-tenant system.
  - **Tensor parallel**: parallelize the matmuls; reduce prefill time roughly linearly in GPUs.
  - **Speculative decoding** for first few tokens: cheap win.
  - **Prompt compression**: shrink the prompt (LLMLingua, etc.). Trade quality for latency.
  - **Use a smaller model** for routing decisions, larger only when needed (model routing / cascades).
- **Common production targets**: TTFT < 300 ms for chat UX; < 1 s for long-context.

**Common follow-ups.**
- "When is queuing the dominant cost?" → Saturated cluster, low priority requests. Capacity-plan around p99 queue depth.
- "What's LLMLingua?" → Microsoft Research's prompt compressor — uses a small LM to mark/remove low-information tokens. 2–10× compression at modest quality cost.

**Common mistakes.**
- Treating TTFT as decode latency — they're different phases.
- Reporting median TTFT without p95/p99 in a system with variable prompt sizes.

**References.**
- [Agrawal et al. — "Sarathi-Serve" (chunked prefill)](https://arxiv.org/abs/2403.02310) — chunked prefill.
- [Jiang et al. — "LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models"](https://arxiv.org/abs/2310.05736) — prompt compression.

---

### Q: What is multi-LoRA serving, and why is it useful?

**Category:** concept
**Difficulty:** mid
**Tags:** [multi-lora, peft, serving]

**Short answer.** Multi-LoRA serving runs many LoRA-fine-tuned models on a single base model in one serving stack, switching adapters per-request (or per-batch). It's useful because LoRA adapters are tiny (~MB vs. GB for the base); a single GPU running one base model can serve dozens of customer-specific adapters simultaneously, sharing the base weights and KV-cache infrastructure.

**Expansion / why this is the answer.**
- The setup:
  - Base model `W` lives on the GPU.
  - Each tenant has a LoRA `(A_i, B_i)`.
  - At forward time, `y = Wx + (α/r) · B_i · A_i · x` — pick the per-request adapter.
- **Punica** (Chen et al. 2024): efficient batched LoRA kernels that process multiple adapters in one batch via grouped GEMM. Throughput close to running the base alone.
- **vLLM multi-LoRA**: production-ready multi-LoRA serving with paged attention; can pin / unpin adapters per request.
- **S-LoRA** (Sheng et al. 2023): research project; tens of thousands of LoRAs on one base, with unified paging for adapter weights.
- **When this matters**:
  - SaaS LLM serving: each customer gets a tailored model without paying for a dedicated base.
  - Multi-tenant evaluation: A/B test many adapters cheaply.
  - Personalization at scale: per-user adapters.
- **Tradeoffs**: kernel complexity; some quality drops at very-high-concurrency multi-LoRA batches when adapters diverge sharply in rank.

**Common follow-ups.**
- "What's a typical LoRA size for a 70B base?" → r=16 LoRA on Q+V projections: low-tens-of-MB. Hundreds of adapters fit per GB.
- "Can you mix LoRA adapters at inference?" → Yes, with weighted blends — but rarely useful in production; usually one adapter per request.

**Common mistakes.**
- Confusing multi-LoRA serving with full-model multi-tenancy (much heavier).
- Forgetting that adapter switching has a (small) per-request cost.

**References.**
- [Chen et al. — "Punica: Multi-Tenant LoRA Serving"](https://arxiv.org/abs/2310.18547) — Punica.
- [Sheng et al. — "S-LoRA"](https://arxiv.org/abs/2311.03285) — S-LoRA.
- [vLLM multi-LoRA docs](https://docs.vllm.ai/en/latest/models/lora.html) — primary docs.

---
