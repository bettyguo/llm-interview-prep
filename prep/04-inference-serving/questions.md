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
- [NVIDIA — TensorRT-LLM project](https://github.com/NVIDIA/TensorRT-LLM) — primary repo + docs.

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
- [vLLM project (multi-LoRA support)](https://github.com/vllm-project/vllm) — primary repo.

---

### Q: What is disaggregated / phase-split serving (Splitwise, Mooncake)? When is it the right move?

**Category:** concept
**Difficulty:** senior
**Tags:** [disaggregated-serving, splitwise, prefill-decode]

**Short answer.** Disaggregated serving runs prefill and decode on *different* pools of GPUs, transferring the KV cache between them. Justified because prefill is compute-bound (wants different hardware tuning) while decode is memory-bandwidth-bound. Splitwise (Microsoft, 2023) and Mooncake (Moonshot AI, 2024) showed substantial throughput gains at high load. Worth it when (a) traffic is high and steady, (b) prefill-vs-decode resource contention is a measurable problem, and (c) the cluster is large enough that the KV-transfer cost is amortized.

**Expansion / why this is the answer.**
- The interference problem:
  - In a *collocated* serving stack (vLLM default), prefill and decode share GPUs.
  - A long prefill blocks decode steps → tail-latency spikes for streaming users.
  - Chunked prefill mitigates but doesn't eliminate.
- **Disaggregation** physically separates them:
  - **Prefill pool**: GPUs optimized for compute throughput; process incoming prompts; emit KV cache.
  - **Decode pool**: GPUs optimized for memory bandwidth; receive KV cache; generate tokens.
  - **KV transfer**: copy the cache between pools — typically over high-speed interconnect (NVLink, InfiniBand).
- **Splitwise** (Patel et al. 2023): proposed and benchmarked the design; gains depend on the prefill-vs-decode ratio in workloads.
- **Mooncake** (Qin et al. 2024, Moonshot AI): production-scale disaggregated serving with KV-cache pool and chunked transfer; cited 75% lower cost at high load.
- **When NOT to disaggregate**:
  - Low traffic — overhead dominates.
  - Single-node deployments — pool separation is meaningless.
  - Workloads with mostly-short prompts and long generations (decode dominates anyway).
- **What's required**:
  - KV-cache serialization across nodes.
  - A scheduler aware of both pools.
  - Fast interconnect.

**Common follow-ups.**
- "Is the KV cache transfer cheap?" → Not free — it's MB-to-GB per request. The win requires the transfer to be much smaller than the alternative cost.
- "Does vLLM disaggregate?" → Standard vLLM is collocated; disaggregation experiments exist in research forks.

**Common mistakes.**
- Calling chunked prefill the same as disaggregation — chunked prefill stays in one pool.
- Assuming disaggregation always wins — at low load it's overhead.

**References.**
- [Patel et al. — "Splitwise: Efficient Generative LLM Inference Using Phase Splitting"](https://arxiv.org/abs/2311.18677).
- [Qin et al. — "Mooncake"](https://arxiv.org/abs/2407.00079).

---

### Q: How does model cascading / routing reduce inference cost?

**Category:** concept
**Difficulty:** mid
**Tags:** [cascade, routing, cost-optimization]

**Short answer.** Send the easy fraction of queries to a cheap small model; escalate hard queries to an expensive large model. A small "router" (rules, classifier, or another LLM) decides which path. Done well, 70%+ of queries answered by the cheap model at 10× lower cost; only the residual hard set hits the expensive model. RouteLLM (Ong et al. 2024) showed up to 85% cost cut at near-frontier quality. The risk is router miscalibration — too aggressive routing degrades quality.

**Expansion / why this is the answer.**
- **The setup**:
  - Pool: one cheap model (e.g. Haiku, GPT-4o-mini, Llama-3.1-8B) + one expensive model (Sonnet/Opus, GPT-5, Llama-3.1-405B).
  - Router decides per query.
- **Router options**:
  - **Rules**: regex / heuristic based on prompt features (length, has-math, has-code, structured).
  - **Trained classifier**: a small model trained on `(query, which_model_won)` pairs.
  - **LLM-judge router**: a cheap LLM is asked "would the small model handle this well?" — itself a small-model decision.
  - **Self-confidence routing**: small model attempts; if its confidence is low (entropy, log-prob threshold), escalate.
- **Optimization metric**: cost subject to quality SLA (or quality subject to cost budget).
- **Empirical**:
  - RouteLLM (Ong et al. 2024) showed routers can recover 85% of frontier-model quality at 25–50% of cost.
  - Common production use case: route by query length, structure, or detected language.
- **Risks**:
  - Router error: a hard query routed to the small model → bad answer → user pain.
  - Stale router: the underlying models change; routing decisions drift.

**Common follow-ups.**
- "How do you train a router?" → Need a labeled dataset where each query was scored by both models. Either use logged production traffic or hand-grade a sample.
- "What's the right metric to optimize?" → Some combination of cost and downstream quality; depends on the deployment SLA.

**Common mistakes.**
- Routing only on cheap-to-compute features (length); misses semantic difficulty.
- Static rules in a fast-moving model landscape.

**References.**
- [Ong et al. — "RouteLLM"](https://arxiv.org/abs/2406.18665).
- [Chen et al. — "FrugalGPT"](https://arxiv.org/abs/2305.05176) — early cascading work.

---

### Q: Walk through GPU memory math for serving a 405B model.

**Category:** derivation
**Difficulty:** senior
**Tags:** [memory, large-models, tensor-parallel]

**Short answer.** Weights for Llama-3.1-405B in bf16 are ~810 GB. Doesn't fit on a single 80GB H100 — need tensor-parallel across at least 12 GPUs (1.5 nodes), realistically 16 (2 H100 nodes). Add KV cache: with GQA-8 and 16k context, ~70 MB per request × concurrent requests. INT4 quantization cuts weights to ~200 GB, fits in one H100 80GB node with 8 GPUs and TP=8. FP8 inference on H100 doubles throughput further.

**Expansion / why this is the answer.**
- **Weight memory**:
  - 405B params × 2 bytes (bf16) = 810 GB.
  - 405B × 1 byte (FP8) = 405 GB.
  - 405B × 0.5 bytes (INT4) = ~200 GB (with metadata).
- **Per-GPU footprint** depends on parallelism:
  - 8× H100 (80GB) = 640 GB total. bf16 doesn't fit; FP8 just barely; INT4 fits with headroom.
  - 16× H100 (1.28 TB) = bf16 fits with TP=16.
- **Activation + KV cache budget** (per GPU after weights):
  - At 8× H100 / INT4 weights: ~80 GB - 25 GB weights/GPU = ~55 GB free.
  - KV cache per token (Llama 3.1 405B, GQA-8, n_layers=126, d_head=128, bf16 = 2): `2·126·8·128·2 = 516 KB/token`. With TP=8 sharding KV → ~65 KB/token-per-GPU.
  - 55 GB / 65 KB ≈ ~850k tokens-of-context across all batched requests, per GPU.
  - In practice 32 concurrent requests × 16k context = 512k tokens — fits.
- **Throughput** (rough):
  - INT4 + FP8 + spec-decoding on 8× H100: ~30–50 output tokens/sec per request at batch 32 (workload-dependent).
- **Comparison**:
  - 70B fits on 1 GPU at INT4; 405B does not — that's a meaningful operational difference.
- **What an interviewer wants you to know**: the order-of-magnitude math; that quantization, tensor parallel, and GQA all shift the math; that real numbers come from load-test, not BoE.

**Common follow-ups.**
- "How do you scale to 1M context?" → KV cache dominates; need quantized KV (FP8/INT4), longer-context-trained weights, and likely sequence-parallel inference.
- "What about CPU offload?" → DeepSpeed-Inference / FlexGen offload weights to CPU/disk; works but latency tanks.

**Common mistakes.**
- Forgetting activations and KV cache eat into the per-GPU budget.
- Treating MoE total params as the memory cost (Mixtral 8x22B has 141B total but ~39B active; weight memory is *total* though).

**References.**
- [Llama 3.1 release page](https://ai.meta.com/blog/meta-llama-3-1/) — model specs.
- [Pope et al. — "Efficiently Scaling Transformer Inference"](https://arxiv.org/abs/2211.05102) — TP scaling math.

---

### Q: TPU vs. GPU for inference — what are the trade-offs?

**Category:** concept
**Difficulty:** mid
**Tags:** [tpu, gpu, accelerators]

**Short answer.** **GPUs (NVIDIA H100/H200)** have the broader software ecosystem (CUDA, all major serving stacks), strong fp8 throughput, and flexible programmability. **TPUs (Google v5p/v6e)** offer comparable peak compute per accelerator, very high inter-chip bandwidth via the TPU pod fabric, and tight integration with JAX. Both are competitive; the deciding factors are usually software ecosystem (CUDA dominance), cloud availability, and software-stack lock-in. Most public OSS LLMs target GPUs first.

**Expansion / why this is the answer.**
- **Per-chip peak compute** (bf16): H100 ~989 TFLOPs; TPU v5p ~459 TFLOPs (per chip; pods scale to thousands). H100 is more compute-dense per chip; TPU pods scale further with the fabric.
- **Memory**:
  - H100 SXM5: 80 GB HBM3, 3.35 TB/s.
  - H200: 141 GB HBM3e, 4.8 TB/s.
  - TPU v5p: 95 GB HBM, ~2.7 TB/s.
- **Inter-chip bandwidth**:
  - NVLink (intra-node, up to 8 GPUs): ~900 GB/s per GPU.
  - TPU OCI (optical circuit interconnect): pod fabric scales to thousands of chips with near-uniform bandwidth.
  - For very-large-scale training/inference, TPU pod fabric is advantageous.
- **Software**:
  - GPU: CUDA, PyTorch, vLLM, TensorRT-LLM, the entire OSS LLM ecosystem.
  - TPU: JAX, Flax, TFRT; some PyTorch support via PyTorch/XLA but with rough edges.
- **Practical considerations**:
  - Most OSS LLMs are released as PyTorch checkpoints; running on TPU requires conversion.
  - TPU is Google-Cloud-only (you can't buy them).
  - GPU pricing varies; H100 has been supply-constrained.
- **Apple/Other**:
  - AWS Inferentia/Trainium, Microsoft Maia, Cerebras, Groq, etc. — niche; primarily for specific cost or latency goals.

**Common follow-ups.**
- "Why does Google use TPUs internally?" → Vertical integration, cost, and the JAX ecosystem built around them.
- "Are AMD MI300 GPUs viable?" → Increasingly yes; ROCm has matured; some labs run on MI300X. CUDA dominance still skews choice.

**Common mistakes.**
- Citing peak TFLOPs alone — memory bandwidth and software stack matter more in practice.

**References.**
- [NVIDIA H100 datasheet](https://www.nvidia.com/en-us/data-center/h100/) — H100 spec.
- [Google TPU v5p](https://cloud.google.com/tpu/docs/v5p) — TPU spec.

---

### Q: What is asynchronous batching / dynamic batching in inference?

**Category:** concept
**Difficulty:** mid
**Tags:** [batching, asynchronous, dynamic-batching]

**Short answer.** Dynamic / asynchronous batching collects multiple in-flight requests in a short waiting window and groups them into a single batch before running the model. The wait increases per-request latency slightly but improves throughput by amortizing the GPU's memory traffic. In LLM serving specifically, continuous batching (per decode-iteration) is the LLM-shaped version; dynamic batching is more typical for CV models or one-shot inference.

**Expansion / why this is the answer.**
- **Static batching**: client controls batch size; rare in production.
- **Dynamic batching**: server collects requests for up to N ms; runs together. Standard for CV (image classification, detection); used in Triton Inference Server.
- **Continuous batching**: LLM-specific; admits new requests mid-decode. See [Continuous batching question](#q-walk-me-through-continuous-inflight-batching-why-is-it-crucial-for-llm-serving) in T4.
- **Why dynamic batching matters for non-LLM**: many CV models are heavily compute-bound at small batch; batching to >32 doubles or triples throughput at minimal latency cost.
- **Why LLMs are different**:
  - Variable output length means request lifetimes vary wildly; static-batch waste is high.
  - Continuous batching is the LLM answer to this.
  - Dynamic batching as a strict per-request scheme is still useful for *prefill* batching (group several prompts together).
- **Practical considerations**:
  - Wait time `T_wait` is a hyperparameter; too low loses batching benefit, too high adds latency.
  - Different model versions can be batched together if they share weights (e.g., LoRA serving with shared base).

**Common follow-ups.**
- "How do you decide `T_wait`?" → Empirical: simulate on your traffic. Typical: 10–50 ms for batch-friendly workloads.
- "What about variable input length in dynamic batching?" → Pad to max length in the batch; padding cost is a known overhead, partially fixed by sequence packing.

**Common mistakes.**
- Conflating dynamic batching (per-request waiting) with continuous batching (per-iteration admission).

**References.**
- [NVIDIA Triton Inference Server — Dynamic Batcher docs](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#dynamic-batcher).
- [Yu et al. — "Orca"](https://www.usenix.org/conference/osdi22/presentation/yu) — continuous batching.

---

### Q: How does prefill chunking interact with continuous batching?

**Category:** concept
**Difficulty:** senior
**Tags:** [chunked-prefill, continuous-batching, sarathi]

**Short answer.** Chunked prefill splits a long prompt into ~512-token chunks and processes one chunk per scheduling step, interleaved with decode steps from other requests. This prevents a single long prefill from blocking decode (and spiking tail latency), while letting continuous batching admit and progress many requests simultaneously. Sarathi-Serve (Agrawal et al. 2024) is the canonical reference; vLLM and SGLang ship it.

**Expansion / why this is the answer.**
- **The problem without chunked prefill**:
  - A 32k-prompt request enters the system.
  - Prefill computes the entire prompt in one forward pass (seconds).
  - During those seconds, decode of *other* requests is blocked or slow.
  - Tail latency for streaming users spikes.
- **Chunked prefill** (Sarathi-Serve):
  - Split the prompt into chunks (e.g. 512 tokens).
  - Each scheduling iteration: do one chunk of prefill + decode steps for other in-flight requests in the same forward pass.
  - The single iteration processes mixed prefill + decode work.
- **Effect**:
  - First-token-latency for the long-prompt request rises a bit (it now takes ~`n_prompt/chunk_size` scheduling steps).
  - Inter-token-latency for all *other* requests stays steady.
  - Tail latency p99 drops dramatically.
- **Tuning**:
  - Chunk size: 512 typical. Larger = fewer iterations for the long request, but more decode blocking per iteration.
  - The scheduler's decode/prefill ratio per iteration.

**Common follow-ups.**
- "Does chunked prefill help when there's no concurrent decode load?" → No; it adds overhead for no benefit. Beneficial only under multi-tenant load.
- "Difference between chunked prefill and splitwise?" → Chunked prefill stays in one GPU pool; Splitwise/disaggregation puts prefill and decode on *separate* GPUs entirely.

**Common mistakes.**
- Confusing with "speculative decoding."
- Setting chunk size too small (overhead dominates).

**References.**
- [Agrawal et al. — "Sarathi-Serve"](https://arxiv.org/abs/2403.02310) — chunked prefill canonical paper.

---

### Q: How does request scheduling work in LLM serving?

**Category:** concept
**Difficulty:** senior
**Tags:** [scheduling, request-priority, fairness]

**Short answer.** Modern LLM servers (vLLM, SGLang, TensorRT-LLM) implement a scheduler that picks at each iteration which prefills and decodes to advance. Priorities: latency-SLO requests, FIFO fairness, request size matching for cache reuse, preemption when memory tight. The scheduler decides per-step: which requests get prefill chunks, which advance decode, when to evict KV cache.

**Expansion / why this is the answer.**
- **Per-iteration decisions**:
  - Admit new requests from the queue (if memory permits).
  - Advance decode for in-flight requests.
  - Advance prefill (chunked) for one or more new requests.
  - Evict / preempt when memory pressure.
- **Policies**:
  - **FIFO**: simple; can starve long requests behind short ones with priority inversion.
  - **Priority**: latency-sensitive requests get jumped ahead.
  - **Size-aware**: cluster requests by expected length for better cache reuse.
- **Preemption**:
  - When memory is exhausted, swap out a low-priority request's KV cache (to CPU memory or disk); resume later.
  - Costs the swap-out time; only used under pressure.

**Common follow-ups.**
- "How does vLLM's scheduler work?" → Iteration-level; each step picks new requests to admit + decodes to advance; prefix-cache aware.
- "What's request batching latency?" → The added wait time for batching; tunable.

**Common mistakes.**
- Treating the scheduler as a black box; it's where serving-system performance lives.

**References.**
- [Kwon et al. — "vLLM"](https://arxiv.org/abs/2309.06180).
- [Yu et al. — "Orca"](https://www.usenix.org/conference/osdi22/presentation/yu).

---

### Q: How does prefix caching share between requests in production?

**Category:** concept
**Difficulty:** senior
**Tags:** [prefix-caching, kv-sharing, multi-tenant]

**Short answer.** Prefix caching matches by token-id prefix: if two requests share a common starting sequence of tokens, their KV cache blocks for that prefix can be shared (copy-on-write). The match must be exact at the token-id level — minor whitespace or tokenizer differences break it. Implementation: hash each KV-cache block's token-prefix; reference-count; deallocate when no requests reference it. vLLM does this via paged-attention block tables.

**Expansion / why this is the answer.**
- The mechanism (vLLM):
  - Hash each block of tokens (e.g. 16 tokens per block).
  - Lookup: does any cached block have this hash?
  - If yes: reuse the physical KV-cache block; bump ref count.
  - If no: allocate, compute prefill.
- **Common shared prefixes**:
  - Long system prompts ("You are a helpful assistant. Follow these rules: ...").
  - Few-shot examples.
  - Conversation history within a session.
- **Hit rate**: in production, typically 30–70% of total prefill tokens are cache hits when system prompts are large.
- **Cost savings**: linear in the cache-hit prefill tokens.
- **Anthropic / OpenAI** offer explicit prompt-caching APIs that expose this.

**Common follow-ups.**
- "Why must matches be exact?" → Token-ids must align; partial overlap doesn't help because KV computations depend on absolute position (post-RoPE) and exact preceding tokens.
- "What's RadixAttention?" → SGLang's data structure: a radix tree of token prefixes; efficient when many requests share branchy prefixes (tree-of-thoughts patterns).

**Common mistakes.**
- Expecting cache hits across slightly-different prompts.

**References.**
- [Kwon et al. — "vLLM"](https://arxiv.org/abs/2309.06180).
- [SGLang paper](https://arxiv.org/abs/2312.07104).

---

### Q: How do you compute LLM inference cost from first principles?

**Category:** derivation
**Difficulty:** senior
**Tags:** [cost, derivation, capacity-planning]

**Short answer.** Cost per million tokens = (GPU $ per hour × number of GPUs) / (tokens generated per hour). Tokens-per-hour = batch × tokens-per-second-per-request × 3600. The batch is bounded by KV-cache memory; tokens-per-second-per-request depends on model size, GPU memory bandwidth, and degree of quantization / speculative decoding. Walk through pin-down: model weight memory → KV per token → batch capacity → throughput → $/Mtok.

**Expansion / why this is the answer.**
- See also T4 cost-math entry; this is the extended derivation.
- Decode-step time = (weight reads + KV reads) / GPU memory bandwidth.
  - Llama 3 70B in bf16: 140 GB weights, 320 KB/token KV (GQA-8).
  - At batch 8, context 4k: 8 × 4096 × 320 KB = 10.5 GB KV.
  - Per decode-step memory traffic: 140 GB (weights) + 10.5 GB (KV) = ~150 GB read.
  - H100 bandwidth 3.35 TB/s → step time ~45 ms.
  - Decode throughput: ~22 step/s; at batch 8 = ~176 tokens/s aggregate.
- $/Mtok math:
  - 2× H100 cluster: $8/hr.
  - Throughput: 176 tok/s × 3600 = 633k tok/hr.
  - Cost: $8 / 0.633 Mtok ≈ $12.6/Mtok.
- INT4 weights cut weight memory to 35 GB → step time ~13 ms → batch 32 fits → throughput rises sharply.

**Common follow-ups.**
- "Why does the math break at very large batch?" → Compute eventually saturates; arithmetic intensity rises above the roofline.
- "MoE adjustment?" → Active params dominate decode; total params matter for memory.

**Common mistakes.**
- Forgetting KV memory; weights aren't the only thing read.

**References.**
- [Pope et al. — "Efficiently Scaling Transformer Inference"](https://arxiv.org/abs/2211.05102).

---

### Q: What is "request-level vs token-level" SLOs in LLM serving?

**Category:** concept
**Difficulty:** mid
**Tags:** [slo, latency, ttft, tpot]

**Short answer.** **Token-level SLOs**: time-to-first-token (TTFT) and time-per-output-token (TPOT). These describe per-request streaming experience. **Request-level SLOs**: total completion time, end-to-end latency. Token-level matters for interactive UX; request-level for batch / agent workloads. Define both; track p50/p95/p99 for each.

**Expansion / why this is the answer.**
- **TTFT**: time until the first output token. Dominated by prefill.
  - Target: < 300 ms for interactive chat.
- **TPOT** (or "inter-token latency"): time per subsequent token.
  - Target: < 50 ms / token for "smooth" streaming.
- **Total latency**: TTFT + (num_output_tokens × TPOT).
- **Why both**:
  - User starts seeing output at TTFT; the first impression.
  - Streaming experience depends on TPOT.
- **Tradeoffs**:
  - Bigger batch → lower per-token cost but higher TPOT.
  - Quantization → lower latency, slight quality drop.
  - Spec decoding → much better TPOT, slightly higher TTFT.

**Common follow-ups.**
- "What's p99 TPOT target in production?" → < 100 ms typical.
- "How does spec decoding affect TPOT?" → Variable: TPOT becomes "time per N tokens" where N is acceptance rate.

**Common mistakes.**
- Optimizing TTFT and TPOT in conflict (e.g. small batches help TTFT but hurt TPOT throughput).

**References.**
- [Anyscale — "Continuous Batching"](https://www.anyscale.com/blog/continuous-batching-llm-inference).

---

### Q: What's the role of CUDA Graphs in LLM inference?

**Category:** concept
**Difficulty:** senior
**Tags:** [cuda-graphs, performance, decode]

**Short answer.** CUDA Graphs capture a sequence of GPU operations once and replay them efficiently. For LLM decode (which repeats the same op sequence per token), CUDA Graphs eliminate the per-step CPU launch overhead — important when each step is microseconds-long (small models, single-batch decode). vLLM and TensorRT-LLM use CUDA Graphs for decode steps; not used for prefill (which has variable shapes).

**Expansion / why this is the answer.**
- **The problem**: launching a CUDA kernel has CPU overhead (~5-20 µs per kernel). If a decode step has ~100 kernels and each is short, CPU launch can be the bottleneck.
- **CUDA Graph**: record the kernel sequence once; replay with a single launch.
- **Best for**: decode with fixed shapes (batch size, context bucket). Multiple graphs for different shapes.
- **Limits**:
  - Fixed shapes required; dynamic shapes (variable context) break it.
  - Setup cost amortized over many uses.
- **Production**: vLLM enables CUDA Graphs for decode by default; significant speedup at small models / single requests.

**Common follow-ups.**
- "Why not use CUDA Graphs for prefill?" → Variable prompt lengths; would need many graphs per shape bucket; setup cost dominates.
- "How does this interact with continuous batching?" → CUDA Graphs require fixed shapes per replay; continuous batching has dynamic shapes — use multi-graph buckets.

**Common mistakes.**
- Overestimating CUDA Graph benefit at large models / batches (where compute dominates anyway).

**References.**
- [NVIDIA CUDA Graphs documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs).

---

### Q: How does INT8 vs INT4 weight-only quantization actually work?

**Category:** derivation
**Difficulty:** senior
**Tags:** [int8, int4, quantization, awq]

**Short answer.** Weight-only quantization: scale each weight to fit in 8 or 4 bits, store per-channel (or per-group) scales; dequantize back to bf16 at the point of matrix multiply. INT8 halves weight memory; INT4 quarters it. Activations stay in bf16. The challenge: 4 bits is so coarse that naive quantization hurts; methods like GPTQ (second-order error compensation) and AWQ (activation-aware salience preservation) restore quality.

**Expansion / why this is the answer.**
- **The math**:
  - Per channel: `s_c = max(|w_c|) / 127` for INT8.
  - Quantize: `q_c = round(w_c / s_c)`.
  - Dequantize: `w_c ≈ s_c · q_c`.
- **Granularity**:
  - Per-tensor: one scale for the whole matrix; cheapest, lowest quality.
  - Per-channel: one scale per output channel; standard.
  - Per-group: one scale per group of N input dims (e.g. N=128); finer; used in AWQ/GPTQ.
- **GPTQ**: minimizes weight reconstruction error using the Hessian (inverse covariance of activations); needs calibration data.
- **AWQ**: protect salient channels (those with large activation magnitudes) by giving them lower-quantization-error treatment; needs calibration data.
- **INT4 quality**:
  - Typical: 0.5–1.5 pt drop on MMLU at 70B; larger drop at 7B.

**Common follow-ups.**
- "Why not INT2 or INT1?" → Quality drops sharply at <4 bits; some research (BitNet) explores it, but production rarely.
- "Symmetric vs asymmetric quantization?" → Asymmetric (zero-point ≠ 0) is more general; symmetric is simpler and common for weights.

**Common mistakes.**
- Quantizing all weights including embeddings (often kept higher precision).

**References.**
- [Frantar et al. — "GPTQ"](https://arxiv.org/abs/2210.17323).
- [Lin et al. — "AWQ"](https://arxiv.org/abs/2306.00978).

---

### Q: What is BitNet / 1.58-bit quantization?

**Category:** concept
**Difficulty:** senior
**Tags:** [bitnet, extreme-quantization, 1.58-bit]

**Short answer.** BitNet b1.58 (Ma et al. 2024): weights restricted to {-1, 0, +1}, giving log₂(3) ≈ 1.58 bits per weight. The matmul becomes integer addition (no multiplies). Quality competitive with bf16 at the same parameter count (according to the paper) at certain scales, with dramatically lower memory and compute. Research-stage as of 2024–2026; not yet a production default.

**Expansion / why this is the answer.**
- **The idea**:
  - Quantize each weight to {-1, 0, +1}.
  - Matmul becomes integer addition: weighted sum of activations (where the weight is just a sign + skip).
  - Compute can use specialized hardware (BitNet-aware kernels).
- **Memory**: 1.58 bits/weight vs. 16 (bf16) = ~10× smaller weight memory.
- **Compute**: addition is much cheaper than multiplication.
- **Result claim**: BitNet b1.58 matches bf16 LLaMA quality at the same scale.
- **Caveats**:
  - Trained from scratch with BitNet's specific recipe; not a post-training quantization.
  - At larger scale (>100B), behavior is less validated.
  - Hardware support is custom; commodity GPUs don't have BitNet-specific kernels yet.

**Common follow-ups.**
- "How is this different from binary weight networks?" → Earlier binary nets used {-1, +1}; BitNet b1.58 adds 0, which empirically matters a lot.
- "Hardware implication?" → Custom inference accelerators benefit hugely; current GPUs less so.

**Common mistakes.**
- Treating BitNet as a drop-in replacement for bf16; it requires retraining.

**References.**
- [Ma et al. — "BitNet b1.58"](https://arxiv.org/abs/2402.17764).

---

### Q: What is page-eviction policy in paged attention?

**Category:** concept
**Difficulty:** senior
**Tags:** [paged-attention, eviction, memory-pressure]

**Short answer.** When the KV-cache memory is full and a new request needs blocks, the scheduler must evict an existing request's blocks. Common policies: **preempt** (evict the latest-arrived or lowest-priority request; swap its blocks to CPU memory; restart later); **terminate** (rare; abort the request); **partial eviction** (least-recently-used blocks). vLLM and similar use preempt-and-swap; rarely terminate.

**Expansion / why this is the answer.**
- **The decision**: which request to victim?
  - **FIFO of arrival**: evict latest; arrivals farthest from completion.
  - **Priority**: evict lowest-priority.
  - **LIFO** with low-priority requests at the back.
- **What to do with the evicted request**:
  - **Swap to CPU**: copy KV blocks to host memory; reload later. Costs swap time.
  - **Discard and restart**: lose the KV; recompute from scratch.
- **Trigger**: memory utilization threshold (e.g. 95% full).
- **Cost**: swapping a request's KV is `O(seq_len × d)` bytes over PCIe; can be hundreds of MB.
- **Production**: vLLM swaps low-priority requests; rarely discards.

**Common follow-ups.**
- "What's the swap rate in production?" → Workload-dependent; some systems target <1%.
- "Why does discard-and-restart sometimes win?" → If the request's prompt was short, recomputing is cheap; saves the PCIe swap.

**Common mistakes.**
- Treating eviction as rare; under load it's continuous.

**References.**
- [Kwon et al. — "vLLM"](https://arxiv.org/abs/2309.06180).

---

### Q: How do you serve LLMs at the edge (mobile, on-device)?

**Category:** concept
**Difficulty:** senior
**Tags:** [edge, on-device, llama-cpp, gguf]

**Short answer.** Edge LLM serving needs aggressive quantization (INT4/Q4_0 at minimum), small models (1–8B typical), and runtimes optimized for CPU/GPU on consumer hardware. **llama.cpp** (CPU-first, GGUF format) is the canonical OSS choice; **Core ML / MLX** on Apple; **ONNX Runtime** on Windows. Tradeoffs: latency vs. cost vs. privacy. Modern edge models: Phi-3-mini, Gemma 2B, Llama 3.2 1B/3B.

**Expansion / why this is the answer.**
- **Targets**:
  - Mobile (phones).
  - Laptops (Apple Silicon, CPU).
  - Embedded (IoT, edge devices).
- **Constraints**:
  - **Memory**: 4–16 GB typical; INT4 quantization brings 7B into this budget.
  - **Compute**: limited; no NVIDIA-class GPU.
  - **Power**: battery and thermal limits.
- **Runtimes**:
  - **llama.cpp**: CPU-first; supports CUDA, Metal, ROCm backends; GGUF model format.
  - **MLX**: Apple's native ML framework for Apple Silicon.
  - **Core ML**: Apple's mobile inference.
  - **ONNX Runtime**: cross-platform.
  - **Mistral.rs / candle**: Rust-based, very fast on CPU.
- **Edge models**:
  - Phi-3-mini (3.8B): designed for on-device.
  - Gemma 2B / Llama 3.2 1B/3B: small enough to run on a phone with INT4.
- **Quality vs cloud**: meaningful gap; edge models are weaker than 7B+ cloud LLMs.

**Common follow-ups.**
- "Why use on-device LLMs?" → Privacy, offline use, latency.
- "GGUF vs other formats?" → llama.cpp's quantization-friendly binary format; ubiquitous in OSS edge LLM ecosystem.

**Common mistakes.**
- Assuming edge models match cloud quality; they don't.

**References.**
- [llama.cpp project](https://github.com/ggerganov/llama.cpp).
- [Apple MLX](https://github.com/ml-explore/mlx).

---

### Q: What is "Continuous Batching" specifically (vs. naive batching)?

**Category:** concept
**Difficulty:** mid
**Tags:** [continuous-batching, in-flight-batching]

**Short answer.** Already covered in T4 base; specific contrast: **naive batching** waits for the slowest request in a batch to finish before starting a new batch — GPU idle for the variable-length tail. **Continuous batching** admits new requests at every decode iteration as in-flight ones finish, keeping the GPU busy continuously. Critical for LLM throughput; supported by vLLM, TGI, TensorRT-LLM, SGLang.

**Expansion / why this is the answer.**
- See base T4 entry on continuous batching for the full discussion.
- This entry is a quick contrast for an interviewer's "is your batching naive?" question.

**Common follow-ups.**
- "How much speedup?" → 2–5× in throughput on realistic workloads.

**Common mistakes.**
- Calling dynamic batching = continuous batching (related but different).

**References.**
- [Yu et al. — "Orca"](https://www.usenix.org/conference/osdi22/presentation/yu).
- [Anyscale continuous batching blog](https://www.anyscale.com/blog/continuous-batching-llm-inference).

---

### Q: What's the "engine bucket" / "TensorRT compilation bucket" tradeoff?

**Category:** concept
**Difficulty:** senior
**Tags:** [tensorrt, compilation, shape-bucket]

**Short answer.** TensorRT-LLM compiles kernels for *specific* input shapes (batch, seq-length) for max performance. A "bucket" is a (batch, max-seq-length) combination compiled once. More buckets = better shape coverage but more compile time and binary size. Production: a few buckets (e.g. batches 1, 4, 16, 32; seq-lengths 1k, 4k, 16k); fall back to dynamic shapes if outside.

**Expansion / why this is the answer.**
- **TensorRT compiles** per-shape; static shapes give best kernel selection.
- **Buckets**: shape combinations you commit to compiling.
- **Tradeoffs**:
  - More buckets: better coverage; compile time and binary size grow.
  - Dynamic-shape fallback: works for any shape but slower.
- **Production pattern**: compile a handful of common buckets; route requests; dynamic-shape for outliers.
- **vLLM is more flexible** (PyTorch-based) at the cost of some peak throughput.

**Common follow-ups.**
- "Why does vLLM not need buckets?" → It uses JIT-compiled PyTorch kernels; dynamic by design.
- "When is TensorRT-LLM worth it?" → Stable, high-volume workloads where peak throughput matters and shape distribution is narrow.

**Common mistakes.**
- Over-bucketing; tens of buckets cost too much compile time.

**References.**
- [TensorRT-LLM documentation](https://github.com/NVIDIA/TensorRT-LLM).

---

### Q: What's the difference between throughput and goodput in LLM serving?

**Category:** concept
**Difficulty:** mid
**Tags:** [throughput, goodput, slo-attainment]

**Short answer.** **Throughput** = tokens generated per second by the system. **Goodput** = tokens delivered within SLO. A system can have high throughput but low goodput if the latency distribution has heavy tails (some requests slow). Optimizing for goodput is the production-relevant metric; throughput alone misses the SLO-violation tail.

**Expansion / why this is the answer.**
- **Throughput**: pure system measure; sum of all token output / time.
- **Goodput**: tokens delivered to requests that *meet* their SLO (e.g. TTFT < 1s).
- The gap: high-utilization batched serving often improves throughput at the cost of tail latency, dropping goodput.
- **Operational practice**:
  - Optimize for goodput, not raw throughput.
  - Set SLOs first; size the cluster to keep p99 within.
- **Distrib serving** (Splitwise, Mooncake) explicitly target goodput by separating prefill (latency for TTFT) from decode (throughput for TPOT).

**Common follow-ups.**
- "How do you measure goodput?" → SLO is defined per-request; count tokens delivered within SLO.
- "Can goodput exceed throughput?" → No; goodput is a subset.

**Common mistakes.**
- Reporting throughput in benchmarks and ignoring the SLO failures it conceals.

**References.**
- [Qin et al. — "Mooncake"](https://arxiv.org/abs/2407.00079) — goodput discussion.

---

### Q: What is "speculative streaming" / EAGLE-2 / Eagle-3?

**Category:** concept
**Difficulty:** senior
**Tags:** [eagle, speculative-decoding, draft-model]

**Short answer.** EAGLE family (Li et al. 2024+): self-speculation methods that predict the *features* (hidden states) of future positions, not just tokens. EAGLE-2 adds adaptive draft tree shaping; EAGLE-3 improves acceptance rate further. Higher acceptance than Medusa or speculative decoding with a separate small draft model; comparable to MTP-trained models like DeepSeek-V3.

**Expansion / why this is the answer.**
- **Standard spec decode**: draft model predicts tokens; target verifies.
- **EAGLE**: draft model predicts hidden states of the next positions; project to logits with the target model's LM head; verify.
- **Advantages**:
  - Predicting features is "easier" than predicting tokens directly (less collapse to top-1 token).
  - Higher acceptance rate (often 70–90%).
- **EAGLE-2** (Li et al. 2024): adaptive draft tree shape; explore more branches when uncertain.
- **EAGLE-3** (Li et al. 2024): trained with multi-step feature prediction; improved acceptance further.
- **Use in production**: vLLM, SGLang, TGI integrate EAGLE family.

**Common follow-ups.**
- "How does EAGLE compare to Medusa?" → Often better acceptance; Medusa is simpler to deploy but EAGLE wins on quality.
- "Does this work with any base model?" → Yes; train the EAGLE head as a post-hoc addition.

**Common mistakes.**
- Calling EAGLE "lossy" — it's lossless (target distribution preserved by the verification step).

**References.**
- [Li et al. — "EAGLE"](https://arxiv.org/abs/2401.15077).
- [Li et al. — "EAGLE-2"](https://arxiv.org/abs/2406.16858).

---

### Q: What is the role of the LLM router / model routing in production?

**Category:** concept
**Difficulty:** mid
**Tags:** [router, model-routing, cascade]

**Short answer.** A router decides which model handles each request: small cheap model for simple queries, big expensive model for hard ones. Saves cost at near-frontier quality. Routers can be rule-based (length, language detection), classifier-based (small ML model), or LLM-based (cheap LLM as decision-maker). RouteLLM (Ong et al. 2024) showed 85% cost reduction at near-frontier quality.

**Expansion / why this is the answer.**
- See T4 base entry on cascades for the deeper treatment.

**Common follow-ups.**
- "What signal trains the router?" → Logged data of (query, which model produced the better output) pairs; or quality-vs-cost scoring.

**Common mistakes.**
- Routing purely on length; semantic difficulty matters too.

**References.**
- [Ong et al. — "RouteLLM"](https://arxiv.org/abs/2406.18665).

---

### Q: What is "TGI inflight batching" vs vLLM continuous batching?

**Category:** concept
**Difficulty:** mid
**Tags:** [tgi, vllm, batching]

**Short answer.** Same concept, different names. TGI (Text Generation Inference, HuggingFace) calls its variant "inflight batching"; vLLM calls its "continuous batching." Both refer to per-iteration admission of new requests during decode. Implementation details differ (TGI has its own scheduler; vLLM has its own with paged attention) but the user-facing behavior is similar.

**Expansion / why this is the answer.**
- Pure terminology disambiguation.
- See base continuous-batching entry.

**Common follow-ups.**
- "Is TGI's implementation as good?" → Historically vLLM has been ahead on features (paged attention, prefix caching); TGI has caught up over time.

**References.**
- [HuggingFace TGI](https://github.com/huggingface/text-generation-inference).
- [vLLM project](https://github.com/vllm-project/vllm).

---

### Q: How does the batch size affect throughput vs. latency tradeoff?

**Category:** derivation
**Difficulty:** mid
**Tags:** [batch-size, throughput-latency-tradeoff]

**Short answer.** Larger batch: lower per-token cost (compute amortized over more tokens); higher per-request latency (more memory pressure, longer step time). Smaller batch: faster per-request response; higher cost per token. The sweet spot depends on workload: latency-sensitive UX prefers small batches; batch-processing workloads (offline summarization, evaluation) prefer large.

**Expansion / why this is the answer.**
- **The math**:
  - Decode step time grows weakly with batch (memory-bound: weights are reread anyway, only KV traffic grows).
  - Per-token cost ≈ step_time / batch_size — drops as batch grows.
  - Per-request latency = step_time × number_of_decode_steps — rises slightly with batch (step time grows).
- **At the sweet spot**:
  - Batch around 32–64 for a 70B class model on H100.
  - Beyond that: KV memory becomes the bottleneck; latency rises faster than throughput.
- **Different for different workloads**:
  - Interactive chat: small batches (4–16) for low latency.
  - Batch jobs: large batches (32–256) for cost.
- **Continuous batching** dynamically balances; the scheduler picks per-iteration.

**Common follow-ups.**
- "Why does increasing batch help so much in memory-bound regimes?" → The weight reads are amortized; only activation/KV reads grow.
- "When does increasing batch hurt latency more than throughput?" → When you saturate compute (arithmetic intensity exceeds the roofline).

**Common mistakes.**
- Treating "batch size" as a fixed knob; it's dynamic in continuous batching.

**References.**
- [Pope et al. — "Efficiently Scaling Transformer Inference"](https://arxiv.org/abs/2211.05102).

---

### Q: What's the difference between RAM, VRAM, HBM, and GPU caches?

**Category:** concept
**Difficulty:** mid
**Tags:** [memory-hierarchy, gpu-architecture]

**Short answer.** **System RAM** (DDR4/5): CPU's main memory; ~100 GB/s. **VRAM = HBM** on modern GPUs: GPU's high-bandwidth memory; H100: 3.35 TB/s. **L2 cache** on GPU: ~50 MB; ~10 TB/s. **L1 / shared memory**: per-SM; ~200+ TB/s. The memory hierarchy is what makes FlashAttention's SRAM optimization meaningful — keeping data in fast on-chip memory saves HBM traffic.

**Expansion / why this is the answer.**
- **System RAM (DDR)**: CPU's working memory; ~100 GB/s on modern desktops; ~400 GB/s on server CPUs.
- **PCIe**: link between CPU RAM and GPU; ~64 GB/s on PCIe Gen5 ×16. Bottleneck for CPU↔GPU transfer.
- **VRAM (HBM)**:
  - H100 80GB HBM3: 3.35 TB/s.
  - H200 141GB HBM3e: 4.8 TB/s.
  - The "main memory" of the GPU.
- **L2 cache** (on GPU chip): 40-50 MB on H100; much faster than HBM.
- **L1 / shared memory** (SRAM, per-SM): 228 KB per SM on H100; closest to compute.
- **Registers**: fastest; per-thread.
- **FlashAttention exploits this**: keep tiles in SRAM (L1/shared); operate; avoid round trips to HBM.

**Common follow-ups.**
- "Why does HBM bandwidth matter for LLM decode?" → Decode reads the full weight matrix every step; HBM traffic = bandwidth × step time.
- "What's NVLink?" → Inter-GPU interconnect, ~900 GB/s per GPU in NVLink 4 (H100); much faster than PCIe but slower than HBM.

**Common mistakes.**
- Confusing GPU memory hierarchy with CPU's; the relative speeds are different.

**References.**
- [NVIDIA Hopper Architecture Whitepaper](https://resources.nvidia.com/en-us-tensor-core).

---

### Q: What is "weight offload" in LLM serving?

**Category:** concept
**Difficulty:** senior
**Tags:** [offload, cpu-memory, large-models]

**Short answer.** Weight offload moves model weights from GPU VRAM to CPU RAM (or disk); the weights are transferred to GPU on demand for each forward pass. Lets you run larger models than fit in VRAM, at huge latency cost. Useful for one-off inference, research, or batch jobs where latency is irrelevant. DeepSpeed-Inference and accelerate support this.

**Expansion / why this is the answer.**
- **Why**:
  - You have a 70B model but only one H100 (80 GB).
  - Weights: 140 GB in bf16; doesn't fit.
  - Offload some/all to CPU RAM; transfer the needed weights at forward time.
- **Performance hit**: PCIe at 64 GB/s vs HBM at 3.35 TB/s — orders-of-magnitude slowdown.
- **For batch processing where latency doesn't matter**: usable; sequence forward through layer-by-layer, transferring weights ahead of compute.
- **Production use**: rare; most serving stacks reject offload for latency reasons.
- **Apple Silicon caveat**: unified memory means no offload needed at moderate scales.

**Common follow-ups.**
- "Disk offload?" → Even slower; only for one-off / research scenarios.
- "FlexGen?" → A research framework specifically targeting offload-heavy serving.

**Common mistakes.**
- Trying offload for latency-sensitive workloads; the speed hit is brutal.

**References.**
- [DeepSpeed ZeRO-Inference](https://www.deepspeed.ai/inference/) — offload support.

---

### Q: What is grouped-query attention's KV-cache memory savings at concrete numbers?

**Category:** derivation
**Difficulty:** mid
**Tags:** [gqa, kv-cache, derivation]

**Short answer.** GQA-`g` (with `g` KV heads vs `h` query heads) reduces KV-cache memory by factor `h/g`. For LLaMA-3 70B: `h=64, g=8`, so 8× less KV memory than MHA. Concrete: 4k-context request KV = 1.3 GB (MHA) vs 160 MB (GQA-8). At batch 16, that's 21 GB vs 2.6 GB — the difference between fitting and not fitting on a single GPU.

**Expansion / why this is the answer.**
- See T4 GQA-derivation entry; this is a quick-recall concrete-numbers version.

**Common follow-ups.**
- "MQA vs GQA-1?" → Same thing.
- "Why not GQA-1 in modern models?" → Quality drops more at GQA-1 than GQA-8 at the same model scale.

**Common mistakes.**
- Forgetting GQA only affects KV; Q is still `h` heads.

**References.**
- [Ainslie et al. — "GQA"](https://arxiv.org/abs/2305.13245).

---

### Q: What's the role of NVLink in distributed LLM training and serving?

**Category:** concept
**Difficulty:** mid
**Tags:** [nvlink, interconnect, distributed]

**Short answer.** NVLink: NVIDIA's high-bandwidth GPU-to-GPU interconnect. H100 NVLink 4: 900 GB/s per GPU (aggregate to other GPUs in the same node). Much faster than PCIe (~64 GB/s) or Infiniband (~50 GB/s). Critical for tensor parallelism (high all-reduce traffic) and expert-parallel MoE (all-to-all traffic). Used for intra-node communication; inter-node uses Infiniband.

**Expansion / why this is the answer.**
- **NVLink topology**:
  - H100 SXM5: 4 NVLink 4.0 links per pair, 900 GB/s aggregate per GPU.
  - HGX H100 8-GPU board: full mesh via NVSwitch.
- **Distributed use**:
  - **Tensor parallelism**: matmul-split requires all-reduce per layer; only feasible at NVLink speeds.
  - **Expert parallelism (MoE)**: all-to-all token shuffle; NVLink critical.
  - **Inter-node**: drops to Infiniband; tensor parallelism doesn't extend well across nodes.
- **Practical guidance**:
  - TP within a node (over NVLink).
  - PP across nodes (over Infiniband).
  - DP / FSDP can cross both.

**Common follow-ups.**
- "What's NVSwitch?" → A switch chip that fully connects 8 GPUs over NVLink within a node.
- "Infiniband vs Ethernet?" → Infiniband: lower latency, more expensive; standard for HPC.

**Common mistakes.**
- Assuming TP scales across nodes; it doesn't at frontier speeds.

**References.**
- [NVIDIA NVLink documentation](https://www.nvidia.com/en-us/data-center/nvlink/).

---

### Q: What's the difference between Llama.cpp and vLLM for serving?

**Category:** concept
**Difficulty:** intro
**Tags:** [llama-cpp, vllm, serving]

**Short answer.** **llama.cpp**: CPU-first (with optional GPU backends); GGUF model format; designed for edge/local inference; uses memory-mapped quantized models. **vLLM**: GPU-first; paged attention; continuous batching; designed for cloud-scale serving. Same models, different deployment targets. Use llama.cpp for laptops/phones; vLLM for cloud servers.

**Expansion / why this is the answer.**
- **llama.cpp strengths**:
  - Runs on CPU (and Apple Metal, CUDA, ROCm).
  - GGUF format with INT4/INT8/etc quantization built-in.
  - Memory-mapped models (large models on small RAM).
  - Many community quantizations available.
- **vLLM strengths**:
  - Paged attention.
  - Continuous batching.
  - Multi-LoRA serving.
  - Frontier-scale models.
  - Production HTTP API.
- **No overlap in target deployment**: serve at scale → vLLM; serve on a laptop → llama.cpp.

**Common follow-ups.**
- "GGUF vs safetensors?" → GGUF: llama.cpp-native, quantization-friendly. Safetensors: HuggingFace native, fp16/bf16.
- "Can vLLM run on CPU?" → Limited; designed for GPU.

**Common mistakes.**
- Choosing llama.cpp for cloud-scale (terrible throughput).

**References.**
- [llama.cpp](https://github.com/ggerganov/llama.cpp).
- [vLLM](https://github.com/vllm-project/vllm).

---

### Q: How would you load-test an LLM serving stack?

**Category:** concept
**Difficulty:** mid
**Tags:** [load-testing, benchmarking, slo]

**Short answer.** (1) Simulate realistic traffic: vary prompt length, output length, request rate, concurrent users. (2) Use a load-gen tool (Locust, k6, or LLM-specific like vLLM's `benchmark_serving.py`). (3) Measure: TTFT p50/p95/p99, TPOT p50/p95/p99, throughput, error rate, GPU utilization. (4) Sweep request rate to find SLO break-points. (5) Validate at sustained load (hours, not minutes) to catch slow leaks.

**Expansion / why this is the answer.**
- **Realistic traffic generation**:
  - Sample prompt lengths from a distribution matching production.
  - Sample output lengths similarly.
  - Vary inter-arrival times (Poisson-like, with bursts).
  - Vary concurrency (5, 20, 50, 100 concurrent users).
- **Tools**:
  - `vllm.entrypoints.benchmark_serving`: vLLM's official.
  - `locust`: general-purpose HTTP load tester.
  - `k6`: similar.
  - `genai-perf` (NVIDIA): LLM-specific.
- **Metrics to record**:
  - Per-request: TTFT, TPOT, total latency.
  - Aggregate: throughput, error rate.
  - System: GPU memory, GPU util, CPU util.
- **What to look for**:
  - The point where p95 latency starts climbing — that's your SLO break.
  - Memory leaks at sustained load.
  - Queue length under burst.

**Common follow-ups.**
- "Synthetic vs replay?" → Replay is realistic but harder to script; synthetic gives controlled variation.
- "Why measure p99 not just p50?" → User-impacting tail; p50 hides 1% disasters.

**Common mistakes.**
- Load-testing for 5 minutes; thermal / memory effects emerge after hours.

**References.**
- [vLLM benchmark_serving.py](https://github.com/vllm-project/vllm/blob/main/benchmarks/benchmark_serving.py).

---

### Q: What is "request priority" / SLO-aware scheduling?

**Category:** concept
**Difficulty:** mid
**Tags:** [priority, slo, scheduling]

**Short answer.** Production LLM serving distinguishes request priorities: interactive UI requests (tight SLO), agent loops (moderate), batch jobs (lax). The scheduler preempts low-priority requests' KV cache to make room for high-priority ones; evicts based on priority; routes based on SLO. Common in multi-tenant systems where one customer's batch job shouldn't starve another's chat.

**Expansion / why this is the answer.**
- **Priority levels**:
  - **Interactive**: user typing in chat; TTFT < 500 ms required.
  - **Background interactive**: agent's tool calls; ~2s OK.
  - **Batch**: offline; minutes OK.
- **Scheduler behavior**:
  - Preempt batch to admit interactive.
  - Reserve fraction of memory for high-priority.
  - Use separate queues per priority.
- **Tradeoffs**:
  - Higher priority for some → starvation risk for batch jobs.
  - Strict SLOs → idle capacity at low load (over-provisioning).

**Common follow-ups.**
- "How do you communicate priority?" → Per-request header / metadata; mapped to scheduler priority.
- "Multi-tenant fair-share?" → Weighted round-robin or DRF (dominant resource fairness).

**Common mistakes.**
- One-size-fits-all scheduling; batch jobs slow interactive.

**References.**
- [Kwon et al. — "vLLM"](https://arxiv.org/abs/2309.06180).

---

### Q: What's RAGAS / RAG-eval specifically for serving?

**Category:** concept
**Difficulty:** intro
**Tags:** [ragas, rag-eval, monitoring]

**Short answer.** Not strictly serving — but production RAG systems use RAGAS-style continuous evaluation: sample 1% of traffic; LLM-judge faithfulness/relevance; alert on drift. Becomes part of the serving stack's quality monitoring. See T7 for the eval-side detail.

**Expansion / why this is the answer.**
- See base T7 RAGAS question.

**Common follow-ups.**
- "Why integrate eval into serving?" → Continuous quality tracking; catches regressions in real time.

**References.**
- [Es et al. — "RAGAS"](https://arxiv.org/abs/2309.15217).

---

### Q: How does multi-turn (chat) serving differ from single-turn?

**Category:** concept
**Difficulty:** mid
**Tags:** [multi-turn, chat, kv-cache, session]

**Short answer.** Multi-turn chat re-uses the conversation history: each turn appends user message + assistant response to the context. Serving efficiency depends on **prefix caching** — the prior turns' KV cache is reused, only the new tokens are processed. Sessions sometimes spread across hours; the scheduler must decide when to evict idle sessions' KV.

**Expansion / why this is the answer.**
- **Per-turn flow**:
  - Turn N: prior context + new user message → prefill new tokens (using cached prior) → decode response.
  - The cached KV from turn N-1 is the prefix.
- **Without prefix caching**: every turn re-prefills the entire context — quadratic cost over many turns.
- **With prefix caching**: each turn only prefills the new user message; the LM head decodes the new assistant tokens.
- **Session management**:
  - When to evict: TTL (e.g. 10 min idle), memory pressure.
  - Re-population: re-prefill from raw history when needed.
- **Modern APIs**:
  - Anthropic / OpenAI prompt-caching APIs: explicit cache control.

**Common follow-ups.**
- "Connection to stateless serving?" → If you don't cache, each turn re-prefills; works but expensive.
- "What's the typical session length distribution?" → Heavy-tailed; most short, some very long (debugging sessions, research chats).

**Common mistakes.**
- Treating chat as stateless and re-prefilling every turn.

**References.**
- [Kwon et al. — "vLLM"](https://arxiv.org/abs/2309.06180) — prefix caching.
- [Anthropic — Prompt Caching docs](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching).

---

### Q: How does CPU offload help in serving 405B+ models?

**Category:** concept
**Difficulty:** senior
**Tags:** [cpu-offload, large-models, latency]

**Short answer.** CPU offload stages parts of the model in CPU RAM and transfers to GPU on-demand. For a 405B model that doesn't fit in 8× H100 in bf16 even with TP, offloading some layers / weights to CPU RAM lets the model run, at order-of-magnitude latency cost. Used for research / batch / one-off serving where latency is irrelevant. Production: usually quantize to INT4 instead.

**Expansion / why this is the answer.**
- Already covered above in weight-offload entry; this is a model-specific framing.

**Common follow-ups.**
- "When is offload preferred over quantization?" → Almost never in production; quantization gives much better latency at modest quality cost.

**References.**
- [DeepSpeed ZeRO-Inference](https://www.deepspeed.ai/inference/).

---

### Q: What's the difference between "online serving" and "batch inference"?

**Category:** concept
**Difficulty:** intro
**Tags:** [online, batch, inference-modes]

**Short answer.** **Online serving**: low-latency real-time inference per request; SLOs in milliseconds-to-seconds; high throughput needed. **Batch inference**: process a large set of inputs offline; latency irrelevant; throughput-only. Different stacks: vLLM/TensorRT-LLM/SGLang for online; same stacks or simpler scripts for batch (no need for continuous batching, no SLO).

**Expansion / why this is the answer.**
- **Online**:
  - Per-request latency matters.
  - Variable load; need capacity planning.
  - Standard chat / API workloads.
- **Batch**:
  - Process N inputs; report N outputs.
  - Process in a single large batch (or chunks).
  - No SLO; can run overnight.
  - Examples: bulk classification, summarization, embedding generation for a large corpus.
- **Why both stacks**:
  - Online needs continuous batching for adaptive load.
  - Batch can use simpler static batching.

**Common follow-ups.**
- "Cost difference?" → Batch is ~5–10× cheaper per token (better utilization, simpler stack, no idle capacity).
- "Anthropic Batch API / OpenAI Batch API?" → Explicitly: 50% discount in exchange for 24h SLA.

**Common mistakes.**
- Using an online stack for a batch job; misses the cost savings.

**References.**
- [OpenAI Batch API docs](https://platform.openai.com/docs/guides/batch).
- [Anthropic Batch API docs](https://docs.anthropic.com/en/docs/build-with-claude/batch-processing).

---

### Q: What is "thinking-time" / "test-time compute" scaling at inference?

**Category:** concept
**Difficulty:** senior
**Tags:** [test-time-compute, o1, reasoning, inference]

**Short answer.** Modern reasoning models (o1-style, DeepSeek-R1, Claude Extended Thinking) trade inference compute for quality by generating long internal "thinking" traces before the final answer. Inference cost rises by 5–100×; quality on reasoning tasks rises substantially. Production tradeoff: when reasoning matters more than latency/cost, enable thinking; when latency matters, disable.

**Expansion / why this is the answer.**
- **The pattern**:
  - Model emits a long "thinking" trace internally.
  - Then emits the final answer.
  - Both consume tokens; user only sees the final answer in some APIs.
- **Cost implications**:
  - Token count: thinking can be 10×–100× the final answer.
  - Inference cost scales accordingly.
- **Quality**:
  - Reasoning benchmarks (GPQA, MATH, ARC-AGI) improve substantially with more thinking.
  - Diminishing returns past a point.
- **APIs**:
  - OpenAI o1/o3: thinking budget controllable.
  - Anthropic Extended Thinking: explicit thinking-budget parameter.
  - DeepSeek-R1: thinking visible by default.
- **Production**: route easy queries to non-thinking models; thinking for hard ones.

**Common follow-ups.**
- "When does thinking not help?" → Simple lookup, classification, short factual recall.
- "How is this related to chain-of-thought?" → CoT is the prompt-level analog; thinking models train the model to do this internally.

**Common mistakes.**
- Enabling thinking universally; massive cost increase for many queries that don't need it.

**References.**
- [OpenAI — o1 system card](https://openai.com/index/openai-o1-system-card/).
- [Anthropic — Extended Thinking docs](https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking).
- [DeepSeek-R1 paper](https://arxiv.org/abs/2501.12948).

---

### Q: How does "structured output" affect serving performance?

**Category:** concept
**Difficulty:** mid
**Tags:** [structured-output, fsm, grammar]

**Short answer.** Constrained generation (JSON, regex, grammar) adds per-step masking of the logits via a finite-state machine, costing ~µs per step. Net effect: minimal latency overhead; slightly fewer effective compute steps (model can sometimes generate longer rejections). Modern FSM-based implementations (Outlines, xGrammar) are well-optimized; rarely the bottleneck.

**Expansion / why this is the answer.**
- See T4 base structured-generation entry.

**Common follow-ups.**
- "Performance penalty?" → Negligible with xGrammar.
- "Quality penalty?" → Yes; sometimes meaningful if grammar is restrictive.

**References.**
- [Dong et al. — "XGrammar"](https://arxiv.org/abs/2411.15100).

---

### Q: What's the role of mixed-batch prefill in modern servers?

**Category:** concept
**Difficulty:** senior
**Tags:** [mixed-batch, prefill, decode]

**Short answer.** Modern servers run prefill and decode steps *in the same forward pass* — mixed-batch. A single iteration may include: prefill chunks for one or two new requests, decode steps for many in-flight ones. Improves GPU utilization (avoids switching between prefill-only and decode-only kernels) and decouples per-request progress. Sarathi-Serve (Agrawal et al. 2024) and later vLLM versions support this.

**Expansion / why this is the answer.**
- **The problem with separating phases**:
  - Pure-decode iterations: low arithmetic intensity; under-utilize compute.
  - Pure-prefill iterations: high latency; block streaming.
- **Mixed batch solution**:
  - Each iteration: bundle prefill chunks + decode positions into one forward pass.
  - Schedule prefill chunks to fill compute slack during decode.
- **Implementation**: attention kernel must handle variable shapes within a single batch (varlen FlashAttention supports this).

**Common follow-ups.**
- "How is this different from disaggregation?" → Mixed batch keeps everything on one GPU pool; disaggregation has separate pools.
- "Does this work with paged attention?" → Yes; paged attention handles the variable shapes.

**Common mistakes.**
- Treating prefill and decode as separate kernels; modern stacks fuse.

**References.**
- [Agrawal et al. — "Sarathi-Serve"](https://arxiv.org/abs/2403.02310).

---

### Q: What's "warmup" in LLM serving?

**Category:** concept
**Difficulty:** intro
**Tags:** [warmup, cold-start, kernel-compilation]

**Short answer.** First-request latency is often much higher than steady-state because of (a) kernel JIT compilation (PyTorch, TensorRT), (b) CUDA Graphs capture, (c) memory pre-allocation. Production serving warms up the engine with dummy requests at startup so first real requests don't pay the cold-start tax. Standard practice; usually 1–5 dummy requests cover the common shape buckets.

**Expansion / why this is the answer.**
- **What gets compiled / cached on first run**:
  - PyTorch JIT'd kernels.
  - TensorRT engine selection.
  - CUDA Graphs capture (per shape bucket).
  - Memory allocator warmup.
- **Cold-start cost**: 100s of ms to seconds.
- **Mitigation**:
  - Warmup script: send dummy prompts of representative shapes at startup.
  - Probe-then-serve: don't accept real traffic until warmup completes.

**Common follow-ups.**
- "Does cloud auto-scaling preserve warmup?" → Each new instance pays the cold-start cost; pre-warm in the readiness probe.
- "Why does this matter for serverless LLM?" → Cold-starts can be unacceptable for interactive UX.

**Common mistakes.**
- Reporting "first-request latency" as production-typical.

**References.**
- [TensorRT-LLM documentation](https://github.com/NVIDIA/TensorRT-LLM).

---

### Q: How does vLLM's chunked prefill work specifically?

**Category:** concept
**Difficulty:** senior
**Tags:** [chunked-prefill, vllm, sarathi]

**Short answer.** vLLM's chunked prefill (since v0.5): split long prefills into chunks (e.g. 512 tokens) processed across multiple scheduler iterations. Each iteration's batch can mix one chunk of prefill from a new request with decode steps for other requests. Inspired by Sarathi-Serve; controlled by `--max-num-batched-tokens`.

**Expansion / why this is the answer.**
- See T4 base chunked-prefill entry.

**Common follow-ups.**
- "Default chunk size?" → 512 tokens typical.
- "Does it help in single-request workloads?" → No; only multi-tenant.

**References.**
- [vLLM docs — chunked prefill](https://docs.vllm.ai/en/latest/).

---

### Q: What is "tensor parallelism degree" choice based on?

**Category:** concept
**Difficulty:** senior
**Tags:** [tp-degree, distributed, model-parallel]

**Short answer.** TP degree (number of GPUs the model is split across) is chosen to make the model fit while keeping intra-node NVLink as the dominant communication path. For a 70B model in bf16 (140 GB): TP=2 on 2× H100 fits. For 405B (810 GB bf16): TP=8 on 8× H100. Choose TP = ceil(model_size / per_gpu_memory) — minimal degree that fits.

**Expansion / why this is the answer.**
- **The math**:
  - Per-GPU memory budget = (GPU VRAM - KV cache - activations - overhead).
  - TP=N spreads weights across N GPUs.
  - Per-GPU weight footprint = total_weights / N.
- **Communication scaling**:
  - TP-N has ~`(N-1)/N` of the all-reduce bandwidth.
  - Beyond 8 GPUs (one NVLink-connected node), bandwidth drops sharply; cross-node TP rarely worthwhile.
- **Recipe**:
  - Pick smallest TP that fits.
  - If still doesn't fit at TP=8 (one node), use TP=8 + PP across nodes.

**Common follow-ups.**
- "Why not always TP=8?" → More TP = more communication overhead. Smaller TP is faster if you can fit.
- "Pipeline parallelism across nodes?" → Yes; common at 405B+ scale.

**Common mistakes.**
- Picking TP based on convenience rather than fit; either wastes memory or fails to fit.

**References.**
- [Shoeybi et al. — "Megatron-LM"](https://arxiv.org/abs/1909.08053).

---

### Q: What is "GPU sharing" / fractional GPU?

**Category:** concept
**Difficulty:** mid
**Tags:** [gpu-sharing, mig, time-sharing]

**Short answer.** Two ways: (1) **MIG (Multi-Instance GPU)**: NVIDIA's hardware partitioning — split one A100/H100 into multiple isolated instances (e.g. 7×A100-10G). Each looks like a smaller GPU. (2) **Time-sharing**: multiple processes share the same GPU; the driver schedules. MIG is for strict isolation; time-sharing is for opportunistic use. For LLM serving, MIG is rare because models often need the full GPU.

**Expansion / why this is the answer.**
- **MIG**:
  - A100 supports up to 7 instances of different memory configs.
  - Strong isolation: separate memory, separate compute SMs.
  - Useful for serving many small models.
- **Time-sharing**:
  - Default CUDA behavior: kernels from multiple processes interleave.
  - No memory isolation; processes share VRAM.
  - Risk: one process OOMs, another fails.
- **For LLM serving**:
  - Modern LLMs typically saturate a full GPU; MIG less useful.
  - Multi-LoRA on one base model: shares the GPU efficiently without MIG (logical multi-tenancy in the serving layer).

**Common follow-ups.**
- "Is MIG used in production for LLMs?" → Rare; most production LLMs use full GPUs.
- "How does Kubernetes share GPUs?" → MIG, time-sharing, or nvidia-plugin abstractions.

**Common mistakes.**
- Assuming MIG helps for serving one big LLM; it just makes the GPU smaller.

**References.**
- [NVIDIA MIG documentation](https://www.nvidia.com/en-us/technologies/multi-instance-gpu/).

---

### Q: What is "compile-once, run-many" in TensorRT-LLM?

**Category:** concept
**Difficulty:** mid
**Tags:** [tensorrt, compilation, deployment]

**Short answer.** TensorRT-LLM compiles a model + shape buckets into an optimized engine file once (takes minutes). The engine then loads quickly and runs efficiently. Production deploys the compiled engine, not the model source. Re-compilation needed when shapes change, GPU type changes, or model is updated.

**Expansion / why this is the answer.**
- Compile time: minutes for a 7B model; tens of minutes for 70B.
- Engine output: optimized binary, GPU-architecture-specific (different for A100 vs H100).
- Re-deploy: ship the engine + a runtime; no PyTorch model needed at serve time.
- Iteration cost: re-compile when needed; CI/CD pipeline handles this.

**Common follow-ups.**
- "Why not JIT like vLLM?" → Slightly less optimized; vLLM trades some peak perf for flexibility.
- "Does ONNX Runtime work like this?" → Yes; similar AOT-compile pattern.

**Common mistakes.**
- Re-compiling at every deploy; wastes time.

**References.**
- [TensorRT-LLM docs](https://github.com/NVIDIA/TensorRT-LLM).

---

### Q: What's "cache reuse" across different requests with similar contexts?

**Category:** concept
**Difficulty:** senior
**Tags:** [cache-reuse, semantic-cache]

**Short answer.** Two senses: (1) **KV-cache reuse** (exact-prefix matching; see prefix caching). (2) **Semantic cache** (e.g. GPTCache): hash the *semantic content* of the prompt; if a previous prompt is semantically similar, return its cached response. The second is application-level, not infrastructure; trades response freshness for cost.

**Expansion / why this is the answer.**
- **KV-cache reuse**: token-level, exact-match — infrastructure layer.
- **Semantic cache**: query-level, semantic-match — application layer.
  - Embed the prompt; lookup in a vector store of past `(prompt_embedding, response)` pairs.
  - If similarity > threshold, return cached response.
  - GPTCache, Redis-with-vectors, custom solutions.
- **Tradeoff**:
  - Cost savings: 0 LLM calls for cache hits.
  - Quality risk: cached response might not be perfect for the slightly-different new prompt.
  - Freshness risk: the cached response may be stale if the underlying data has changed.
- **When to use**:
  - FAQ-style apps with high prompt repetition.
  - Cost-sensitive workloads.

**Common follow-ups.**
- "Semantic threshold?" → 0.85–0.95 cosine typical; tune by acceptable hit rate.
- "Why not always use semantic cache?" → Cache hits are wrong sometimes; user notices.

**Common mistakes.**
- Treating semantic cache as a free win; quality is at risk.

**References.**
- [GPTCache project](https://github.com/zilliztech/GPTCache).

---

### Q: How does Mooncake's KV-cache pool work?

**Category:** concept
**Difficulty:** senior
**Tags:** [mooncake, kv-pool, disaggregated]

**Short answer.** Mooncake (Qin et al. 2024, Moonshot AI): a disaggregated LLM serving system that includes a centralized KV-cache pool (across nodes), allowing prefill and decode nodes to share cache fragments. Optimizes for high prefix-cache hit rate at scale. Reported 75% lower cost than collocated baselines.

**Expansion / why this is the answer.**
- **Architecture**:
  - Prefill pool (compute-heavy GPUs).
  - Decode pool (memory-bandwidth-heavy GPUs).
  - **Centralized KV pool** in shared memory (CPU RAM + RDMA-attached NVMe).
  - Prefill writes KV to the pool; decode reads from it.
- **Why this matters**:
  - At very large scale, KV reuse across requests pays for the transport cost.
  - Decoupling prefill and decode pools lets each scale independently.
- **Performance** (Mooncake paper): 75% cost reduction vs. collocated under specific workloads.

**Common follow-ups.**
- "Is this in production?" → Moonshot AI uses it; Mooncake-style pooling is a research direction; not yet OSS-standard.
- "Tradeoff with paged attention?" → Different: paged is intra-GPU; KV pool is cross-GPU/cross-node.

**Common mistakes.**
- Treating KV-cache pool as universal; only justified at large scale with heavy prefix reuse.

**References.**
- [Qin et al. — "Mooncake"](https://arxiv.org/abs/2407.00079).

---

### Q: What is "KV cache transfer" cost in disaggregated serving?

**Category:** derivation
**Difficulty:** senior
**Tags:** [kv-transfer, disaggregated, networking]

**Short answer.** Transferring a KV cache between prefill and decode nodes costs `seq_len × KV_per_token` bytes over the interconnect. For Llama 3 70B GQA-8 at 16k context: ~5 GB per request. Over PCIe Gen5 (64 GB/s) or RDMA Infiniband (200 Gbps = 25 GB/s), transfer time is 0.2–1 second — significant. Justifies disaggregation only at high load where prefill/decode contention costs more.

**Expansion / why this is the answer.**
- **Math**: KV per token (Llama 3 70B GQA-8, bf16) = 320 KB. At 16k context: 16k × 320 KB = 5 GB.
- **Transfer paths**:
  - **PCIe Gen5 ×16**: 64 GB/s; transfer in ~80 ms.
  - **Infiniband HDR (200 Gbps)**: 25 GB/s; ~200 ms.
  - **NVLink intra-rack**: 900 GB/s; ~5 ms.
- **Trade-off**:
  - Cost: 5–500 ms of added latency per request.
  - Benefit: avoid prefill/decode contention.
- Worth it when contention adds more than transfer time saves.

**Common follow-ups.**
- "FP8 / INT8 KV?" → Linear reduction in transfer size.
- "Why is intra-rack so much faster?" → NVLink between racks via NVSwitch.

**Common mistakes.**
- Ignoring transfer cost; it's substantial.

**References.**
- [Patel et al. — "Splitwise"](https://arxiv.org/abs/2311.18677).
- [Qin et al. — "Mooncake"](https://arxiv.org/abs/2407.00079).

---

### Q: What is "decode-time pruning" of KV cache?

**Category:** concept
**Difficulty:** senior
**Tags:** [kv-pruning, h2o, decode]

**Short answer.** During decode, identify and evict less-important KV-cache entries to bound memory. Methods: **H2O** (Zhang et al. 2023) keeps "heavy hitters" (high-attention tokens); **SnapKV** (Li et al. 2024) selects relevant tokens once at the start of generation. Quality cost is modest; memory savings can be 5–10×. Useful for very long contexts where KV memory is the binding constraint.

**Expansion / why this is the answer.**
- The intuition: most tokens in a long context receive little attention; we can drop their KV without much quality loss.
- **H2O** (Zhang et al. 2023): "heavy hitter oracle" — keep top-K by accumulated attention score.
- **SnapKV** (Li et al. 2024): observe attention patterns over the first N decode steps; permanently drop low-attention tokens.
- **StreamingLLM** (Xiao et al. 2024): keep attention sinks + sliding window.
- **Quality cost**: typically <5pt on benchmarks at 5× KV reduction; rises sharply at 10×+.

**Common follow-ups.**
- "How does this interact with prefix caching?" → Pruned tokens lose their cache value; needs care.
- "Is this lossless?" → No — pruning is approximate.

**Common mistakes.**
- Treating KV pruning as lossless.

**References.**
- [Zhang et al. — "H2O"](https://arxiv.org/abs/2306.14048).
- [Li et al. — "SnapKV"](https://arxiv.org/abs/2404.14469).

---

### Q: What is "JIT vs AOT compilation" for LLM kernels?

**Category:** concept
**Difficulty:** mid
**Tags:** [jit, aot, compilation]

**Short answer.** **JIT (Just-In-Time)**: kernels compiled at runtime when first called (PyTorch eager + torch.compile). Flexible, slower startup. **AOT (Ahead-Of-Time)**: kernels compiled before deployment (TensorRT, ONNX Runtime). Faster startup, less flexible. Tradeoff: AOT for stable production; JIT for development and dynamic workloads.

**Expansion / why this is the answer.**
- **JIT**:
  - PyTorch eager + `torch.compile` (Triton-backed).
  - Compiles on first call; cached for subsequent.
  - Easy to iterate; fewer deployment artifacts.
- **AOT**:
  - TensorRT, ONNX Runtime.
  - Compile to optimized engine file.
  - Shipped with the application.
  - Specific to GPU architecture; needs recompile per target.
- **vLLM**: primarily JIT.
- **TensorRT-LLM**: AOT.

**Common follow-ups.**
- "torch.compile?" → PyTorch's JIT entry; falls back to eager on graph breaks.
- "Production preference?" → Depends; many use vLLM (JIT) for flexibility; TensorRT-LLM (AOT) for peak perf.

**Common mistakes.**
- Forgetting AOT engines are GPU-architecture-specific.

**References.**
- [PyTorch torch.compile docs](https://pytorch.org/docs/stable/torch.compiler.html).

---

### Q: How does inference benefit from sparsity / pruning?

**Category:** concept
**Difficulty:** senior
**Tags:** [sparsity, pruning, deja-vu]

**Short answer.** **Structured sparsity** (2:4 NVIDIA pattern, channel pruning) gives real speedup on modern GPUs (~2× for 2:4). **Unstructured sparsity** rarely gives speedup on dense hardware. **Dynamic / runtime sparsity** (Déjà Vu, Liu et al. 2023): predict which MLP neurons activate per token; skip the rest — gives speedup at inference. Net: sparsity helps when the hardware supports it.

**Expansion / why this is the answer.**
- **Static structured sparsity**:
  - NVIDIA 2:4 pattern (every 4 weights have 2 zeros).
  - Tensor cores natively support; ~2× throughput.
  - Compatible with quantization.
- **Unstructured sparsity**:
  - Random zeros; doesn't map to GPU tensor cores; rarely speeds up.
- **Dynamic sparsity (Déjà Vu)**:
  - Predict which FFN neurons / attention heads contribute meaningfully per token.
  - Skip the others.
  - Speedups 2–5× on FFN-dominated forward pass.
- **MoE is dynamic sparsity** at the expert granularity.

**Common follow-ups.**
- "Compatibility with quantization?" → Yes; sparsity + INT4 stacks.
- "Production use?" → 2:4 sparsity on some NVIDIA stacks; Déjà Vu-style not yet standard.

**Common mistakes.**
- Pruning randomly; doesn't translate to throughput.

**References.**
- [Liu et al. — "Déjà Vu"](https://arxiv.org/abs/2310.17157).
- [NVIDIA 2:4 sparsity guide](https://developer.nvidia.com/blog/accelerating-inference-with-sparsity-using-ampere-and-tensorrt/).

---

### Q: How does ZeRO-3 inference work vs ZeRO-3 training?

**Category:** concept
**Difficulty:** senior
**Tags:** [zero-inference, deepspeed]

**Short answer.** ZeRO-3 inference (DeepSpeed-Inference): parameters sharded across GPUs; gather on demand per layer for forward pass; re-shard. Memory: per-GPU = `total / N`. Latency cost: gather adds communication per layer. Compared to TP: ZeRO-3 sharding doesn't split matmuls (so each GPU runs the whole layer in turn); TP splits matmuls (so all GPUs work in parallel). TP is faster; ZeRO-3 inference is more memory-flexible.

**Expansion / why this is the answer.**
- **TP inference**: matmul-split across GPUs; full parallelism per layer.
- **ZeRO-3 inference**: parameter-shard; gather to one GPU per layer (or sub-batch); compute; re-shard.
- **Latency**: TP wins for throughput; ZeRO-3 wins for the largest models that don't fit even with TP.
- **Use case**: ZeRO-3 inference rare in production; mostly research / batch.

**Common follow-ups.**
- "Why is ZeRO-3 slower?" → Communication for each layer; less parallelism.
- "FSDP inference?" → Similar to ZeRO-3 inference; not common.

**Common mistakes.**
- Treating ZeRO-3 as universally optimal; for inference, TP usually wins.

**References.**
- [DeepSpeed-Inference docs](https://www.deepspeed.ai/inference/).

---
