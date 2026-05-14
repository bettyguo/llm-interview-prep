# Transformers & LLM Internals — questions

The most-asked topic in 2026 AI interview loops. Entries follow the [Q&A schema](../../CONTRIBUTING.md#the-qa-entry-schema).

---

### Q: Derive scaled dot-product attention. Why divide by √d_k?

**Category:** derivation
**Difficulty:** mid
**Tags:** [attention, scaling, softmax]

**Short answer.** `Attention(Q, K, V) = softmax(QKᵀ / √d_k) V`. We divide by `√d_k` because the dot product `QKᵀ` between vectors of dimension `d_k` has variance proportional to `d_k` under random initialization, so the unscaled softmax saturates into one-hot for large `d_k`, killing gradients. Dividing by `√d_k` keeps the variance of pre-softmax logits ~constant and preserves a useful gradient signal.

**Expansion / why this is the answer.**
- Assume `q, k ∈ ℝ^{d_k}` with components i.i.d. zero-mean unit-variance. Then `q·k = Σ q_i k_i` has mean 0 and **variance `d_k`** (sum of `d_k` variance-1 random variables). So the standard deviation is `√d_k`.
- Without scaling, for `d_k = 64` say, logits have std 8; softmax of values with std 8 is highly peaked — gradient saturation.
- With scaling by `√d_k`, logits stay with std ~1, softmax stays well-spread, gradients flow.
- The full mechanism (Vaswani et al. 2017):
  1. Project inputs to Q, K, V via `W_Q, W_K, W_V`.
  2. Compute `scores = QKᵀ / √d_k`.
  3. Apply mask if causal (set future positions to `-inf` before softmax).
  4. Softmax along the key dimension.
  5. Multiply by V.
- **Multi-head**: split `d_model` into `h` heads each of size `d_k = d_model/h`; compute attention in parallel per head; concatenate; project. The "split" lets different heads attend to different things.
- **Complexity**: `O(n² · d)` for sequence length `n` — the quadratic cost is *the* problem long-context techniques (FlashAttention, sparse attention) try to address.

**Common follow-ups.**
- "Why softmax instead of sigmoid?" → Softmax gives a probability distribution over key positions (sums to 1); sigmoid would let attention "split attention 1.0 everywhere," which destroys selectivity.
- "What's the difference between cross-attention and self-attention?" → Self-attention: Q, K, V from the same sequence. Cross-attention: Q from the decoder, K and V from the encoder (e.g. in T5).

**Common mistakes.**
- Forgetting the `√d_k` scale (catastrophic — the answer fails immediately).
- Saying it's just "to normalize" without the variance argument.
- Confusing `d_k` with `d_model` (in multi-head, they differ).

**References.**
- [Vaswani et al. — "Attention Is All You Need"](https://arxiv.org/abs/1706.03762) — the original transformer paper, with the scaling explanation in §3.2.1.
- [Annotated Transformer (Harvard NLP)](https://nlp.seas.harvard.edu/2018/04/03/attention.html) — line-by-line.

---

### Q: Walk me through MHA, MQA, and GQA. What's the KV-cache implication?

**Category:** concept
**Difficulty:** mid
**Tags:** [attention, kv-cache, gqa, mqa]

**Short answer.** **MHA** (multi-head attention): each head has its own Q, K, V projections — `h` separate sets of K/V. **MQA** (multi-query attention): one shared K and V across all heads; `h×` smaller KV cache, modest quality drop. **GQA** (grouped-query attention): a middle ground — split the `h` heads into `g` groups, each group shares one K/V; `h/g`× smaller KV cache, near-MHA quality. **The KV-cache implication**: MQA and GQA dramatically reduce inference memory and bandwidth, which is why almost every modern LLM (LLaMA 2/3, Mistral, Mixtral, Qwen, DeepSeek) uses GQA.

**Expansion / why this is the answer.**
- **MHA**: KV cache size per token = `2 · n_layers · n_heads · d_head` bytes (×2 for fp16 vs. fp32). The factor of `n_heads` is the dominant cost.
- **MQA** (Shazeer 2019): one K and one V per layer, shared across all heads. KV cache = `2 · n_layers · 1 · d_head`. `n_heads`× reduction. Some quality loss in long-context tasks.
- **GQA** (Ainslie et al. 2023): `g` groups, each with its own K, V. KV cache = `2 · n_layers · g · d_head`. Typical `g` = 8 with `n_heads` = 64 → 8× cache reduction with near-MHA quality.
- **Why this matters for serving**:
  - Decode is **memory-bandwidth-bound**, not compute-bound. The KV cache is read for every decode token; smaller cache = higher tokens/sec.
  - The KV cache **scales linearly with batch × sequence length** — a 70B-class model with long context can have a cache that is many tens of GB per batch.
- **Modern picks** (as of 2026): LLaMA 3, Mistral, Mixtral, Qwen 2/3, Gemma 2 all use GQA. Some research models use MLA (multi-head latent attention, DeepSeek-V2/3) which factorizes the K/V projections — even smaller cache.

**Common follow-ups.**
- "What's MLA?" → Multi-head latent attention (DeepSeek-V2): K and V are derived from a small shared latent vector via a learned projection, making the cache even smaller than GQA's.
- "Why doesn't training quality suffer from GQA?" → Empirically, the quality drop is small (GQA paper: ~negligible at GQA-8 vs. MHA on a 30B model); the parameters saved go into width/depth elsewhere.

**Common mistakes.**
- Saying MQA is "always worse" than MHA — it's a tradeoff, and at modern scale GQA gets you ~MHA quality at a fraction of the cache.
- Confusing GQA with sliding-window attention (different axis: GQA reduces head sharing, SWA limits context).

**References.**
- [Shazeer — "Fast Transformer Decoding: One Write-Head is All You Need" (MQA)](https://arxiv.org/abs/1911.02150) — MQA.
- [Ainslie et al. — "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"](https://arxiv.org/abs/2305.13245) — GQA.
- [DeepSeek-V2 paper](https://arxiv.org/abs/2405.04434) — MLA.

---

### Q: Explain positional encodings — sinusoidal vs. learned vs. ALiBi vs. RoPE.

**Category:** concept
**Difficulty:** mid
**Tags:** [positional-encoding, rope, alibi]

**Short answer.** Attention is permutation-equivariant, so positional information must be injected explicitly. **Sinusoidal** (original transformer): fixed sin/cos of position, added to embeddings — generalizes to longer sequences. **Learned**: a position-embedding table, learned; limited to training-time max length. **ALiBi**: add a linear bias to attention logits proportional to relative distance; extrapolates well. **RoPE**: rotate Q and K by an angle proportional to position; encodes relative position multiplicatively; the modern default (LLaMA, Mistral, Qwen, DeepSeek).

**Expansion / why this is the answer.**
- **Sinusoidal** (Vaswani et al. 2017): `PE(pos, 2i) = sin(pos / 10000^(2i/d))`; `PE(pos, 2i+1) = cos(...)`. Added to token embeddings. Allegedly extrapolates; in practice does not extrapolate well past training length.
- **Learned absolute**: a `(max_len, d)` parameter matrix; what BERT uses. Bounded by training-time max length.
- **ALiBi** (Press et al. 2022): instead of adding to embeddings, add a per-head linear penalty `−m · |i − j|` to attention logits. No new parameters; **extrapolates** to lengths longer than seen at train time. Used by some models (e.g. MPT).
- **RoPE** (Su et al. 2021): rotate Q and K by `θ_i = pos · 10000^(−2i/d)` (a per-dimension rotation in the complex plane). The dot product `q·k` then depends only on the **relative** position `pos_q − pos_k`. The modern default — LLaMA 1/2/3, Mistral, Qwen, Mixtral, Gemma, DeepSeek.
- **Extending RoPE context length** (post-training): NTK-aware scaling, YaRN (Peng et al. 2023), Linear scaling — each modifies the base 10000 or the position scaling to extend usable context without retraining from scratch.

**Common follow-ups.**
- "Why does RoPE encode relative position despite being applied to Q and K independently?" → The rotation algebra: `(R_a q) · (R_b k) = q · R_{b−a} k` (relative rotation only). The dot product depends only on the difference.
- "What's YaRN?" → "Yet another RoPE extension" — a method for extending RoPE-trained models to longer context with minimal fine-tuning, by adjusting the per-dimension rotation frequency.
- "Why don't modern LLMs use sinusoidal anymore?" → RoPE empirically gives better perplexity at the same parameter count; relative-position-aware; integrates cleanly with extension techniques.

**Common mistakes.**
- Saying "sinusoidal extrapolates well" — it was *claimed* to in the original paper; in practice, models trained on length 2k degrade rapidly past length 2k without help.
- Confusing RoPE with adding a vector to embeddings — RoPE is a *rotation* applied inside the attention layer.
- Treating ALiBi and RoPE as interchangeable — they're different mechanisms; ALiBi modifies logits, RoPE modifies Q/K.

**References.**
- [Vaswani et al. — "Attention Is All You Need"](https://arxiv.org/abs/1706.03762) — sinusoidal.
- [Su et al. — "RoFormer: Enhanced Transformer with Rotary Position Embedding"](https://arxiv.org/abs/2104.09864) — RoPE.
- [Press et al. — "Train Short, Test Long: Attention with Linear Biases" (ALiBi)](https://arxiv.org/abs/2108.12409) — ALiBi.
- [Peng et al. — "YaRN: Efficient Context Window Extension of Large Language Models"](https://arxiv.org/abs/2309.00071) — YaRN.

---

### Q: Encoder-only vs. decoder-only vs. encoder–decoder. When to use which?

**Category:** concept
**Difficulty:** intro
**Tags:** [architecture, bert, gpt, t5]

**Short answer.** Encoder-only (BERT-style): bidirectional self-attention, no causal mask, pretrained on masked LM; best for understanding tasks (classification, NER, embeddings). Decoder-only (GPT-style): causal self-attention, pretrained on next-token prediction; the dominant architecture for generative LLMs. Encoder–decoder (T5/BART-style): encode source bidirectionally, decode target causally with cross-attention; the natural fit for sequence-to-sequence tasks (translation, summarization in its strict form). In 2026, decoder-only has won the general-purpose race; encoder-only persists for embeddings and classification; encoder-decoder is niche.

**Expansion / why this is the answer.**
- **Encoder-only**: BERT, RoBERTa, DeBERTa, ModernBERT (2024). Output a contextualized representation per token. Used as the backbone for retrievers (SBERT), rerankers, classifiers.
- **Decoder-only**: GPT family, LLaMA, Mistral, Claude, Gemini, DeepSeek. Causal mask means each position only attends to past. Pretrained on next-token prediction. Can do everything via prompting + generation.
- **Encoder–decoder**: T5, BART, Flan-T5; cross-attention from decoder to encoder. The classical seq-to-seq picture. Still strong on some translation/summarization benchmarks; less general than decoder-only.
- **Why decoder-only won**: a single objective (next-token prediction) handles every task by prompting; scaling laws hold cleanly; in-context learning falls out naturally; serving is simpler (no encoder pass).
- **Practical 2026 picks**:
  - Embeddings: encoder-only (E5, BGE, GTE) or LLM-based dense encoders.
  - Classification: encoder-only (BERT/ModernBERT) for cost; decoder-only for quality with cost no object.
  - Generation: decoder-only.
  - Translation: still some encoder-decoder; mostly decoder-only too.

**Common follow-ups.**
- "Why can't BERT generate text?" → It's pretrained with a fill-in-the-blank objective; not autoregressive; non-trivial to coerce into generation (and worse than a decoder-only model that was trained for it).
- "What's ModernBERT?" → A 2024 refresh of BERT with longer context (8192), GeGLU activations, RoPE, and a modern training recipe; faster encoder embeddings.
- "If decoder-only is so flexible, why pay for an encoder at all?" → Cost. A small encoder is much cheaper for embeddings/classification than a comparable-quality decoder.

**Common mistakes.**
- Calling GPT "encoder-decoder."
- Confusing "encoder" with the inference role of producing embeddings (which decoder-only models also do).

**References.**
- [Vaswani et al. — "Attention Is All You Need"](https://arxiv.org/abs/1706.03762) — original encoder-decoder.
- [Devlin et al. — "BERT"](https://arxiv.org/abs/1810.04805) — encoder-only.
- [Radford et al. — "Language Models are Unsupervised Multitask Learners" (GPT-2)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — decoder-only.
- [Raffel et al. — "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer" (T5)](https://arxiv.org/abs/1910.10683) — encoder–decoder.
- [Warner et al. — "ModernBERT"](https://arxiv.org/abs/2412.13663) — modern encoder.

---

### Q: Pre-norm vs. post-norm — which do modern LLMs use, and why?

**Category:** concept
**Difficulty:** mid
**Tags:** [layernorm, pre-norm, training-stability]

**Short answer.** Modern LLMs use **pre-norm**: `x + Sublayer(LN(x))`. Post-norm (`LN(x + Sublayer(x))`) was the original transformer's choice but suffers training instability at depth — gradients can blow up because the residual stream is normalized after the sublayer, making the gradient magnitude depend on the sublayer's output scale. Pre-norm puts the LayerNorm before the sublayer, leaving the residual stream untouched; this is mathematically equivalent to a much more stable iteration and is what GPT-2/3, LLaMA, Mistral, Mixtral, and essentially all 2024+ LLMs use.

**Expansion / why this is the answer.**
- **Post-norm** (original Vaswani 2017): `x_{l+1} = LN(x_l + F_l(x_l))`. Requires a learning-rate warmup to train stably at depth.
- **Pre-norm** (Xiong et al. 2020 formalized it; widespread since GPT-2): `x_{l+1} = x_l + F_l(LN(x_l))`. The residual stream `x_l` carries unchanged through each layer; the sublayer reads a normalized version. Trains without warmup at depth, more stable.
- Why pre-norm wins at depth: the residual stream acts like a "highway" — gradients can flow back through `∂x_l/∂x_{l-1} ≈ I`, regardless of `F_l`. Post-norm wraps `F_l`'s output in another normalization, which scrambles the residual identity.
- Note: some recent models use a **Sandwich / DeepNorm** variant (Wang et al. 2022) that tries to combine the stability of pre-norm with some quality benefits sometimes observed for post-norm. Not dominant.
- "RMSNorm + pre-norm" is the modal modern choice (LLaMA, Mistral, etc.).

**Common follow-ups.**
- "Why does pre-norm need a final LN after the last block?" → Because the residual stream is never normalized otherwise; the LN before the LM head normalizes the final activations.
- "What goes wrong at 100 layers without pre-norm?" → Loss spikes, training divergence; warmup helps but isn't always sufficient.

**Common mistakes.**
- Drawing the diagram with pre-norm but calling it post-norm.
- Forgetting the final LN before the LM head.

**References.**
- [Xiong et al. — "On Layer Normalization in the Transformer Architecture"](https://arxiv.org/abs/2002.04745) — the canonical pre-norm vs. post-norm analysis.
- [Wang et al. — "DeepNet: Scaling Transformers to 1,000 Layers"](https://arxiv.org/abs/2203.00555) — DeepNorm.

---

### Q: Explain Mixture-of-Experts. What is the gating function, and why is load balancing necessary?

**Category:** concept
**Difficulty:** senior
**Tags:** [moe, gating, load-balancing, mixtral, deepseek]

**Short answer.** MoE replaces each (or some) dense FFN sublayer with **E** parallel FFNs ("experts") and a small **gating network** that, per token, routes the token to the top-k experts. Only the chosen experts are activated, so total parameters can be much larger than per-token compute. The router can collapse — sending all tokens to the same expert — so a **load-balancing auxiliary loss** is added to push the router to distribute traffic across experts.

**Expansion / why this is the answer.**
- The math: gating computes `s = softmax(W_g x)` of logits for each expert; pick top-k experts (typically k=1 or k=2); weight their outputs by their gating scores; sum. The FFN cost is `k/E` of the dense equivalent for the same total parameters.
- **Why MoE**: parameter count and compute decouple. You can ship a model with 10× parameters at the same forward FLOPs.
- **Load balancing**: without an auxiliary loss, the router converges to picking the same expert (Shazeer et al. 2017). The standard fix is the load-balancing auxiliary loss `L_aux = α · E · Σ_i f_i p_i` where `f_i` is the fraction of tokens routed to expert `i` and `p_i` is the mean gating score for `i`. Pushes both to `1/E`.
- **Expert capacity**: a hard cap on how many tokens a given expert can take per batch, with overflowing tokens dropped or routed to a backup; trades quality for predictability of compute.
- **Fine-grained experts** (DeepSeek-V2/V3): many small experts + a shared expert that processes every token. Empirically improves quality at the same active-parameter count vs. coarse experts (like Mixtral's 8x7B).
- **Practical examples**: Switch Transformer (Fedus et al. 2021, k=1), Mixtral 8x7B (k=2 of 8), DeepSeek-V3 (256 routed + 1 shared, k=8 of 256).

**Common follow-ups.**
- "Why is MoE harder to train than dense?" → Routing decisions are non-differentiable (top-k); auxiliary losses, optimizer state explosion (lots of params), communication overhead for expert-parallel training.
- "What is router z-loss?" → A regularizer on the gating logits' log-sum-exp; stabilizes training (used in ST-MoE).
- "Where does MoE fail?" → Tasks needing long-context coherent reasoning sometimes underperform dense models of equal active params, but the gap shrinks with better routing.

**Common mistakes.**
- Saying "MoE has 8× the compute of dense" — false; per-token compute is roughly the active expert count.
- Forgetting load balancing — the modal interview gotcha.
- Confusing routing decisions with attention.

**References.**
- [Shazeer et al. — "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer"](https://arxiv.org/abs/1701.06538) — foundational.
- [Fedus, Zoph, Shazeer — "Switch Transformer"](https://arxiv.org/abs/2101.03961) — top-1 routing.
- [Jiang et al. — "Mixtral of Experts"](https://arxiv.org/abs/2401.04088) — Mixtral 8x7B.
- [DeepSeek-AI — "DeepSeek-V3 Technical Report"](https://arxiv.org/abs/2412.19437) — fine-grained MoE.

---

### Q: Explain FlashAttention. What does it actually change vs. naive attention?

**Category:** concept
**Difficulty:** senior
**Tags:** [flash-attention, kernels, memory]

**Short answer.** FlashAttention is an **IO-aware exact attention kernel**: it computes the same softmax(QKᵀ/√d_k)V as standard attention but tiles the computation so the intermediate `n×n` attention matrix never lives in HBM (GPU main memory). Tiles of Q, K, V are loaded into on-chip SRAM, partial softmax statistics are maintained online, and the output is accumulated tile by tile. Result: same numerical output, dramatically less HBM traffic, much faster wall time for long sequences. It is the standard kernel in modern training stacks (PyTorch SDPA's "flash" backend, vLLM, TensorRT-LLM).

**Expansion / why this is the answer.**
- The naive attention bottleneck: the `n × n` attention matrix `A = softmax(QKᵀ / √d)` is materialized in HBM. For sequence length 8k that's 64M entries per head per batch; HBM bandwidth becomes the wall, not compute.
- **FlashAttention** (Dao et al. 2022):
  - Tile Q into blocks `Q_i`, K and V into blocks `K_j`, `V_j`.
  - For each `Q_i`, iterate over `K_j, V_j` blocks; compute partial logits, maintain a running softmax statistic (max + sum) so the streaming-softmax is numerically exact.
  - Accumulate into the output `O_i` tile.
  - Backward pass is similarly tiled (no need to store `A`; recompute on the fly).
- **What changed in FlashAttention-2/3**:
  - **v2** (Dao 2023): better parallelization across heads and sequence dim; ~2× faster than v1.
  - **v3** (Shah et al. 2024): Hopper-specific (FP8, async warp specialization, ping-pong scheduling); leverages H100 features.
- **What it does *not* change**: numerical output (it's exact, up to floating-point reassociation). It is not approximate attention.

**Common follow-ups.**
- "How does the backward pass avoid storing the attention matrix?" → Recomputes the softmax during the backward, using the saved softmax denominator and max statistics.
- "What's the speedup empirically?" → Depends; on long sequences and modern GPUs, 2–5× wall-clock speedup vs. the naive PyTorch kernel.
- "Why is FlashAttention not always the default in PyTorch?" → It is, since PyTorch 2.0, via `scaled_dot_product_attention` selecting the flash backend when conditions are met (dtype, head_dim, mask compatibility).

**Common mistakes.**
- Calling FlashAttention "approximate." It's exact.
- Saying it reduces FLOPs. It doesn't reduce FLOPs; it reduces **memory traffic** (and so wall time).
- Conflating with sparse attention (Longformer, Sparse Transformer) — those *do* reduce FLOPs by computing fewer entries.

**References.**
- [Dao et al. — "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"](https://arxiv.org/abs/2205.14135) — v1.
- [Dao — "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning"](https://arxiv.org/abs/2307.08691) — v2.
- [Shah et al. — "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision"](https://arxiv.org/abs/2407.08608) — v3.
- [PyTorch — torch.nn.functional.scaled_dot_product_attention](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) — the integrated PyTorch API.

---

### Q: Walk me through scaling laws — Kaplan's vs. Chinchilla. Which is the modern view?

**Category:** concept
**Difficulty:** senior
**Tags:** [scaling-laws, chinchilla, compute-optimal]

**Short answer.** Kaplan et al. (OpenAI, 2020) measured that LM loss falls as a power law in compute, parameters, and data, and concluded that compute-optimal training favors very large models on relatively little data (recipe: increase model size faster than dataset size). Chinchilla (Hoffmann et al., DeepMind 2022) re-did the experiment more carefully and concluded the opposite: for compute-optimal training, **data and parameters should scale roughly in equal proportion**. Roughly: tokens ≈ 20× parameters. Chinchilla is the modern view, and explains why Chinchilla-70B beat Gopher-280B at the same compute.

**Expansion / why this is the answer.**
- The Kaplan power laws are still broadly right at a high level — loss scales smoothly in `(N, D, C)`. The disagreement is in the **exponents** and consequently the recipe.
- **Kaplan recipe**: given a compute budget, train a model with `N ∝ C^0.73, D ∝ C^0.27` — bias toward more parameters.
- **Chinchilla recipe**: `N ∝ C^0.5, D ∝ C^0.5` — roughly equal scaling. Concretely: a model trained on `D ≈ 20·N` tokens is compute-optimal.
- Why the discrepancy: Kaplan fixed LR schedule across runs (over-trained small models suffered); Chinchilla retuned per run.
- **The post-Chinchilla LLM-training norms**:
  - LLaMA 1 (Feb 2023): 7B/13B trained on 1T tokens, 33B/65B on 1.4T tokens — explicitly Chinchilla-influenced and even *past* the compute-optimal point (more tokens than compute-optimal) to reduce inference cost.
  - LLaMA 2 (Jul 2023): 2T tokens.
  - LLaMA 3 (Apr 2024): 15T+ tokens — way past compute-optimal; pays training cost to lower inference cost.
- **Modern caveat**: real-world models often train *past* compute-optimal because **inference is cheaper at small parameters even if training cost is sub-optimal**. The Chinchilla recipe is the right answer for compute-optimal *loss*, not for compute-optimal *deployment economics*.
- **Emergent abilities** (Wei et al. 2022): claim that some capabilities appear sharply at scale. Critiqued by Schaeffer, Miranda, Koyejo (2023) — "Are Emergent Abilities a Mirage?" — arguing many emergence claims are artifacts of discontinuous metrics.

**Common follow-ups.**
- "Why train past Chinchilla-optimal then?" → Inference cost. LLaMA chose smaller models trained longer to reduce serving cost.
- "Does Chinchilla apply to MoE?" → The exact exponents differ; the qualitative shape (more data matters more than Kaplan claimed) still holds.
- "What does Schaeffer say about emergence?" → That emergence claims often depend on the metric: replace exact-match (discontinuous) with a continuous metric and many "emergent" capabilities show smooth scaling.

**Common mistakes.**
- Citing Kaplan as the current recipe.
- Conflating Chinchilla-optimal training with deployment-optimal training (they diverge; modern models intentionally over-train data).
- Citing "emergent abilities" uncritically without the Schaeffer pushback.

**References.**
- [Kaplan et al. — "Scaling Laws for Neural Language Models"](https://arxiv.org/abs/2001.08361) — Kaplan.
- [Hoffmann et al. — "Training Compute-Optimal Large Language Models" (Chinchilla)](https://arxiv.org/abs/2203.15556) — Chinchilla.
- [Wei et al. — "Emergent Abilities of Large Language Models"](https://arxiv.org/abs/2206.07682) — the emergence claim.
- [Schaeffer, Miranda, Koyejo — "Are Emergent Abilities of Large Language Models a Mirage?"](https://arxiv.org/abs/2304.15004) — the critique.

---

### Q: Walk me through tokenization. What is BPE, and what are its known failure modes?

**Category:** concept
**Difficulty:** mid
**Tags:** [tokenization, bpe, sentencepiece]

**Short answer.** Tokenization splits text into discrete units (tokens) that the model sees. **BPE** (Byte-Pair Encoding) starts from characters/bytes and greedily merges the most-frequent adjacent pair until a target vocab size is reached. Modern variants (byte-level BPE in GPT-2, SentencePiece-Unigram in T5/LLaMA) handle arbitrary text including Unicode. **Failure modes**: numbers split arbitrarily (hurts arithmetic), multilingual coverage is biased toward training data, glitch tokens (rare strings that decode to nothing useful and confuse the model — SolidGoldMagikarp), and identical text in different scripts/casings producing different tokens.

**Expansion / why this is the answer.**
- **BPE** (Sennrich et al. 2016 for NMT, popularized by GPT-2 in byte-level form):
  - Initialize with single-byte (or character) tokens.
  - Count pair frequencies on training text.
  - Merge the most-frequent pair into a new token.
  - Repeat until vocab size hits target (typically 32k–128k).
- **Byte-level BPE** (Radford et al. 2019, GPT-2): start from raw UTF-8 bytes, so the tokenizer can never produce an OOV token. The GPT-2/3/4 tokenizer family is byte-level BPE.
- **SentencePiece** (Kudo & Richardson 2018): library implementing both BPE and Unigram LM; used by T5 and LLaMA. Unigram-LM picks merges differently — selects the segmentation maximizing a unigram likelihood under a fixed vocab.
- **Failure modes**:
  - **Numbers**: "1234" might tokenize as "12" "34", or "1" "234". Inconsistent splits hurt arithmetic and date handling. LLaMA 3 added a "digit-by-digit" rule for numbers.
  - **Glitch tokens** (Rumbelow & Watkins 2023): rare strings (e.g. `" SolidGoldMagikarp"`) that ended up in the vocabulary but were essentially absent from training data; the model has no learned behavior for them and emits nonsense.
  - **Cross-lingual cost**: English text typically tokenizes to ~4 chars/token; some languages (Burmese, Telugu) tokenize to ~1 char/token at common vocab sizes — same content, 4× the cost.
  - **Code**: code-heavy tokenizers (codellama, deepseek-coder) have specialized vocabularies that compress code better; using a general-purpose tokenizer on code is expensive.

**Common follow-ups.**
- "Why not just use character-level?" → Sequence becomes very long (compute-quadratic in attention); vocab is tiny so each token carries little info.
- "Why is GPT bad at arithmetic on long numbers?" → Tokenization inconsistency + the digit being split unpredictably.
- "What's tiktoken?" → OpenAI's BPE tokenizer library, the canonical reference implementation for the GPT-3.5/4 cl100k_base and o200k_base tokenizers.

**Common mistakes.**
- Saying BPE is "word-level." It's sub-word.
- Conflating sentencepiece-BPE with sentencepiece-Unigram.
- Forgetting that the tokenizer is part of the model's contract — swapping tokenizers requires retraining the embedding table at minimum.

**References.**
- [Sennrich, Haddow, Birch — "Neural Machine Translation of Rare Words with Subword Units"](https://arxiv.org/abs/1508.07909) — BPE for NMT.
- [Radford et al. — GPT-2 paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — byte-level BPE.
- [Kudo & Richardson — "SentencePiece"](https://arxiv.org/abs/1808.06226) — SentencePiece library.
- [OpenAI tiktoken](https://github.com/openai/tiktoken) — primary tokenizer source.

---

### Q: Explain how KV-cache memory grows. Why does it limit batch size at long context?

**Category:** derivation
**Difficulty:** mid
**Tags:** [kv-cache, memory, inference]

**Short answer.** Per token, KV-cache memory is `2 · n_layers · n_kv_heads · d_head · dtype_bytes` (2 for K and V). The cache scales **linearly with sequence length and batch size**. At long context (say 100k tokens) for a 70B GQA model with 8 KV heads, this can be tens of GB per request — so the cache forces small batches or aggressive paging.

**Expansion / why this is the answer.**
- Per-layer, per-token K and V storage: each has shape `(n_kv_heads, d_head)`.
- Total per-token bytes (across all layers): `2 · n_layers · n_kv_heads · d_head · bytes_per_elem`.
- Worked example for LLaMA-3-70B (n_layers=80, n_kv_heads=8, d_head=128, bf16 → 2 bytes):
  - Per token: `2 · 80 · 8 · 128 · 2 = 327,680 bytes ≈ 320 KB/token`.
  - 100k context: 32 GB **per request**.
- For MHA (no GQA) at the same model dims, `n_kv_heads = n_heads`, which would be `64`, so 8× larger: ~2.6 MB/token, ~256 GB at 100k. This is why all modern long-context models use GQA or stronger (MLA).
- This dominates batch sizing during decode — even on an H100 (80 GB), a 70B model in bf16 takes ~140 GB just for weights (over 2 GPUs) plus KV cache; long-context batching becomes constrained.
- **What helps**:
  - GQA / MLA / MQA (covered separately).
  - **Quantized KV cache** (FP8, INT8, INT4): cuts memory linearly; quality cost usually small.
  - **Paged attention** (vLLM): eliminates fragmentation, so memory you do have is fully used.
  - **Prefix caching** (sharing the cache for shared system prompts): not memory savings on a per-token basis, but it lets many concurrent requests share a prefix.

**Common follow-ups.**
- "What's the KV cache during prefill?" → During prefill (process the prompt), all `n_prompt` K/V pairs are computed at once; during decode, one new K/V pair is appended per generated token.
- "Why is decode memory-bandwidth-bound?" → Each decode step reads the entire KV cache to compute one new token; FLOPs are small (~`O(n · d)`), but memory traffic is large.

**Common mistakes.**
- Forgetting the `2×` for K and V both.
- Forgetting to use `n_kv_heads` (after GQA) rather than `n_heads`.
- Mixing up per-token vs. per-request memory.

**References.**
- [Pope et al. — "Efficiently Scaling Transformer Inference"](https://arxiv.org/abs/2211.05102) — KV cache memory math and serving regimes.
- [Kwon et al. — "Efficient Memory Management for Large Language Model Serving with PagedAttention" (vLLM)](https://arxiv.org/abs/2309.06180) — paged attention.

---

### Q: How does in-context learning work? Why is it surprising?

**Category:** concept
**Difficulty:** senior
**Tags:** [in-context-learning, few-shot, gpt-3]

**Short answer.** In-context learning (ICL) is the ability of a large LM to perform a new task from a few demonstrations placed in the prompt — without parameter updates. It was the headline result of GPT-3 (Brown et al. 2020). The surprise is that the model "learns" from examples at inference time, despite no weight change, suggesting the pretraining objective implicitly produces a meta-learner. Mechanistically, ICL appears to be (partly) implemented as an attention-driven analogue of gradient descent on the example pairs in-context, but the empirics are still active research.

**Expansion / why this is the answer.**
- The GPT-3 result: showing 0/1/few in-context examples on a downstream task; performance improves with more examples, scaling roughly with model size.
- Why it's surprising:
  - No weight update — pretraining alone produced a meta-learner.
  - The behavior emerges (sharply, on some metrics) as scale increases.
  - The prompt format (instructions vs. exemplars) matters.
- **Mechanistic hypotheses**:
  - **Induction heads** (Olsson et al. 2022): a specific attention pattern that copies/completes sequences; arguably the substrate of much ICL behavior.
  - **Implicit gradient descent** (Akyürek et al. 2022; von Oswald et al. 2023): transformers can simulate gradient descent on linear regression with a few attention layers, suggesting ICL is a kind of in-network meta-learning.
- **Empirical caveats**:
  - Order of demonstrations matters; the model is sensitive to seemingly irrelevant prompt details (Lu et al. 2022).
  - Min et al. (2022, "Rethinking the Role of Demonstrations"): for some tasks, *label correctness in the demonstrations* matters less than expected — what matters is the input distribution and the label space.
- **Modern context**: ICL has been partially supplanted by chain-of-thought prompting and instruction tuning, but it's still the foundation interviewers expect you to be able to explain.

**Common follow-ups.**
- "What's the difference between zero-shot, one-shot, and few-shot?" → Number of examples in the prompt; n-shot means n labeled examples.
- "How does chain-of-thought interact with ICL?" → Adding "let's think step by step" or reasoning exemplars elicits step-by-step generation, improving multi-step task accuracy (Wei et al. 2022).
- "Why does ICL break down on long contexts?" → Lost-in-the-middle (Liu et al. 2023): mid-prompt content is underused by some models.

**Common mistakes.**
- Saying the model "fine-tunes itself on examples" — no weights change.
- Treating ICL as a magic property of "any LLM" — it's strongly scale-dependent.

**References.**
- [Brown et al. — "Language Models are Few-Shot Learners" (GPT-3)](https://arxiv.org/abs/2005.14165) — the ICL paper.
- [Olsson et al. — "In-context Learning and Induction Heads"](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) — mechanistic.
- [Akyürek et al. — "What learning algorithm is in-context learning?"](https://arxiv.org/abs/2211.15661) — implicit GD.
- [Min et al. — "Rethinking the Role of Demonstrations"](https://arxiv.org/abs/2202.12837) — empirical surprise.

---

### Q: What is chain-of-thought prompting? When does it help, when doesn't it?

**Category:** concept
**Difficulty:** mid
**Tags:** [chain-of-thought, prompting, reasoning]

**Short answer.** Chain-of-thought (CoT) prompting elicits step-by-step reasoning before the final answer — either by including step-by-step exemplars (few-shot CoT, Wei et al. 2022) or by appending "let's think step by step" (zero-shot CoT, Kojima et al. 2022). It substantially improves multi-step reasoning tasks (math, symbolic, logical) at sufficient model scale. **It does not reliably help** on tasks that don't decompose into discrete reasoning steps, and it can hurt latency/cost without quality gain on straightforward tasks.

**Expansion / why this is the answer.**
- **Few-shot CoT**: include a few `(question, reasoning, answer)` exemplars; the model imitates the pattern.
- **Zero-shot CoT**: append "Let's think step by step." After step-by-step output, append "Therefore, the answer is" and parse.
- **Self-consistency** (Wang et al. 2022): sample multiple CoT trajectories and majority-vote — robustly improves over greedy CoT.
- **Where CoT helps**: GSM8K, MATH, multi-step logical / symbolic tasks; tasks with a chain of operations where intermediate state matters.
- **Where CoT doesn't help**: simple lookup, short classification, tasks where the answer is direct.
- **Caveats** (2024–2025 research):
  - **Faithfulness**: the model's stated reasoning is not always the causal path it took (Turpin et al. 2023) — the chain can be post-hoc rationalization.
  - **Scale-dependence**: small models often degrade with CoT; the effect emerges at scale.
- **Modern context**: post-training stages now bake reasoning into the model directly (RLHF/RLAIF reasoning, GRPO + verifier reward, o1-style "thinking" tokens) — CoT is increasingly an internal mechanism rather than an explicit prompting trick.

**Common follow-ups.**
- "What's tree-of-thoughts?" → Yao et al. 2023: extend CoT to a tree search; useful for tasks with backtracking. Higher cost.
- "Why does self-consistency help?" → Marginalizing over reasoning paths denoises errors; the most-common answer is the most reliable one.
- "What's the difference between CoT and o1-style reasoning?" → CoT is prompt-level; o1-style models are trained (often with RL) to produce a long thinking trace internally.

**Common mistakes.**
- Treating CoT as universally beneficial.
- Confusing CoT with retrieval-augmented generation (different mechanism, different problem).

**References.**
- [Wei et al. — "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"](https://arxiv.org/abs/2201.11903) — few-shot CoT.
- [Kojima et al. — "Large Language Models are Zero-Shot Reasoners"](https://arxiv.org/abs/2205.11916) — zero-shot CoT.
- [Wang et al. — "Self-Consistency Improves Chain of Thought Reasoning"](https://arxiv.org/abs/2203.11171) — self-consistency.
- [Turpin et al. — "Language Models Don't Always Say What They Think"](https://arxiv.org/abs/2305.04388) — faithfulness.

---

### Q: What is sliding-window / local attention, and when do you use it?

**Category:** concept
**Difficulty:** mid
**Tags:** [attention, sliding-window, mistral]

**Short answer.** Sliding-window (SWA) attention restricts each token to attend only to a fixed-size window of recent tokens (e.g. 4096). The KV cache size is bounded by the window, not the full context, so memory is constant in sequence length. SWA is used in Longformer (Beltagy et al. 2020) and in Mistral 7B (window = 4096). The tradeoff: long-range dependencies must propagate through stacked layers ("information bottleneck"), and important tokens outside the window are inaccessible directly.

**Expansion / why this is the answer.**
- Architecture: token at position `i` attends to positions `[i - w + 1, i]` for window `w`.
- Information at position `i - w - k` can still influence position `i` after `k` layers, because each layer's window slides — like a CNN with kernel size `w`. The effective receptive field grows with depth, but signal is diluted.
- **Longformer** (Beltagy et al. 2020): mix of local sliding window + global attention for select "anchor" tokens (e.g. `[CLS]`).
- **Mistral 7B** (Jiang et al. 2023): pure SWA with `w = 4096` and a context of 8192 tokens; uses RoPE; KV cache capped at the window.
- **Hybrid choices** in 2024–2026 frontier models: alternate local-attention layers with full-attention layers — Gemma 2 (Google) interleaves sliding-window and global-attention layers explicitly; the original Mistral 7B used pure SWA but its later checkpoints (v0.2+) and Mixtral dropped SWA in favor of full attention.
- Failure mode: tasks requiring direct attention to far-away tokens (e.g. needle-in-haystack at the prompt start) degrade — though Mistral's results showed SWA-only models still recover most long-range info via depth.

**Common follow-ups.**
- "How does this compare to attention-sink schemes?" → Streaming-LLM (Xiao et al. 2024) keeps SWA + a small number of "attention sink" tokens at the very start; lets the model run on near-infinite streams.
- "Is SWA the same as Big Bird?" → Big Bird (Zaheer et al. 2020) adds global + random + sliding patterns; SWA is just the sliding part.

**Common mistakes.**
- Calling SWA "approximate" — it computes a different but exact attention; the *model* is different from a full-attention one.
- Forgetting that even with SWA, signal can flow long-range via stacked depth.

**References.**
- [Beltagy, Peters, Cohan — "Longformer"](https://arxiv.org/abs/2004.05150) — sliding + global.
- [Jiang et al. — "Mistral 7B"](https://arxiv.org/abs/2310.06825) — SWA in a production LLM.
- [Xiao et al. — "Efficient Streaming Language Models with Attention Sinks" (StreamingLLM)](https://arxiv.org/abs/2309.17453) — attention sinks.

---

### Q: Why do transformers struggle with very long context, and what techniques extend it?

**Category:** concept
**Difficulty:** senior
**Tags:** [long-context, rope-scaling, yarn, lost-in-the-middle]

**Short answer.** Three obstacles: (1) **compute** is quadratic in sequence length (`O(n² · d)`); (2) **KV-cache memory** is linear in sequence length; and (3) **quality degrades** outside the trained context, both because positional encodings extrapolate poorly and because the model's attention patterns trained on short context fail at long ranges ("lost in the middle"). Techniques: FlashAttention (compute), GQA + paged + quantized KV cache (memory), RoPE scaling / YaRN / NTK-aware interpolation (positional extrapolation), and continued pretraining or fine-tuning on long sequences (quality).

**Expansion / why this is the answer.**
- **Compute**: attention's quadratic cost makes 128k+ context expensive without FlashAttention or sparse variants.
- **Memory**: KV cache linear in `n`, dominates long-context inference (see KV-cache question).
- **Positional extrapolation**: RoPE-trained models break above training length unless you adjust. Techniques:
  - **Position Interpolation (PI)** (Chen et al. 2023): scale positions by `L_train / L_target`. Requires a short fine-tune; quality drop is modest.
  - **NTK-aware scaling** (Reddit `/u/bloc97`): adjust the RoPE base such that high-frequency dims aren't compressed.
  - **YaRN** (Peng et al. 2023): combines NTK-aware with attention-temperature correction; the strongest pure-extension method.
  - **LongRoPE** (Ding et al. 2024): per-dim RoPE-frequency optimization; extends to 2M+.
- **Quality / lost-in-the-middle**: even with positional fixes, the model may not use mid-prompt content (Liu et al. 2023). Continued pretraining on long sequences + targeted data (e.g. needle-in-a-haystack-style supervised data) helps.
- **Architectural alternatives**: state-space models (Mamba), linear attention, hybrid SSM+attention models (Jamba, RecurrentGemma) — claim asymptotic linear cost. Mixed results in 2024–2026; transformers remain dominant.

**Common follow-ups.**
- "What is the needle-in-a-haystack evaluation?" → A small synthetic test: insert one sentence into a long context and ask about it; measures retrieval-from-context across positions.
- "Are Mamba/SSMs better than transformers at long context?" → On synthetic copying/retrieval tasks, often comparable; on general LM quality at scale, transformers still lead as of 2025.

**Common mistakes.**
- Saying "just increase RoPE base" — it's more subtle than that; PI / NTK / YaRN are doing principled corrections, not just a scale.
- Forgetting the *quality* axis — extending context to 1M tokens is worth little if the model can't actually use mid-prompt content.

**References.**
- [Chen et al. — "Extending Context Window of Large Language Models via Positional Interpolation"](https://arxiv.org/abs/2306.15595) — PI.
- [Peng et al. — "YaRN"](https://arxiv.org/abs/2309.00071) — YaRN.
- [Ding et al. — "LongRoPE"](https://arxiv.org/abs/2402.13753) — extending to 2M+.
- [Liu et al. — "Lost in the Middle"](https://arxiv.org/abs/2307.03172) — the quality failure.
- [Gu & Dao — "Mamba"](https://arxiv.org/abs/2312.00752) — SSM alternative.

---

### Q: SwiGLU vs. GELU vs. ReLU. What activation function do modern LLMs use, and why?

**Category:** concept
**Difficulty:** mid
**Tags:** [activations, swiglu, gelu, ffn]

**Short answer.** Most modern LLMs use **SwiGLU** in their FFN sublayer — `SwiGLU(x) = (Swish(xW_1)) ⊙ (xW_2)`, a gated linear unit with the Swish (= SiLU) activation. Empirically (Shazeer 2020) it beats GELU and ReLU at the same parameter count, at the cost of 1.5× FFN parameters because there are now two projections in the gate. LLaMA, Mistral, PaLM, Mixtral all use SwiGLU; BERT and the original transformer used GELU/ReLU.

**Expansion / why this is the answer.**
- **ReLU**: `max(0, x)`. Cheap, sharp; dominant in CNNs.
- **GELU** (Hendrycks & Gimpel 2016): `x · Φ(x)` (Gaussian CDF); smooth ReLU-like; used in original BERT and GPT-2/3.
- **Swish / SiLU** (Ramachandran et al. 2017): `x · σ(x)`; close cousin of GELU; smoother near zero.
- **Gated Linear Units** (Dauphin et al. 2016): `GLU(x) = (xW_1) ⊙ σ(xW_2)`. Two projections, multiply elementwise.
- **SwiGLU** (Shazeer 2020): replace the sigmoid in GLU with Swish: `(Swish(xW_1)) ⊙ (xW_2)`. To keep FFN parameter count roughly constant, the hidden dim is scaled by 2/3 (so total params ≈ original FFN).
- **GeGLU**: GLU with GELU. Used in ModernBERT, some T5 variants.
- **Why SwiGLU wins**: empirical — at the same parameter count, lower perplexity. The intuition (Shazeer): the gate gives the model multiplicative interactions that pure additive FFNs lack.

**Common follow-ups.**
- "Why scale hidden dim by 2/3 in SwiGLU?" → Naive SwiGLU has 50% more parameters than a vanilla FFN at the same hidden dim; scaling hidden by 2/3 restores parameter parity.
- "Is the FFN where most of the parameters are?" → Yes, for dense models: typically ~⅔ of model parameters are in the FFNs, ~⅓ in attention.

**Common mistakes.**
- Calling SwiGLU just a "non-linearity" — it's a gated structure, not a pointwise activation.
- Forgetting the hidden-dim scaling.

**References.**
- [Shazeer — "GLU Variants Improve Transformer"](https://arxiv.org/abs/2002.05202) — SwiGLU.
- [Hendrycks & Gimpel — "Gaussian Error Linear Units"](https://arxiv.org/abs/1606.08415) — GELU.
- [Dauphin et al. — "Language Modeling with Gated Convolutional Networks"](https://arxiv.org/abs/1612.08083) — GLU.

---

### Q: What's an induction head? Why does it matter?

**Category:** concept
**Difficulty:** senior
**Tags:** [interpretability, induction-heads, in-context-learning]

**Short answer.** An induction head is a specific two-attention-head circuit (a **previous-token head** followed by an **induction head**) that implements the pattern "if I saw `A B` earlier in the sequence and now I see `A`, predict `B` next." It was identified by Olsson et al. (2022) as the substrate of much of the in-context-learning behavior in transformers, and its emergence during training coincides with the loss curve's "ICL bump" — the point where the model starts to do in-context learning at all.

**Expansion / why this is the answer.**
- The circuit (two layers):
  1. A "previous-token head" at layer L copies the token at position `i-1` into the residual stream at position `i`.
  2. An "induction head" at layer L+1 uses that copied info as a key, so when it sees `A` at position `j` later, it attends back to the prior `A` and copies the token that followed it (`B`) into the residual stream at position `j`.
- The result: the model implements `Pr(next = B | "...A B... A_")` — token-level pattern matching across the prompt.
- This is a clean **mechanistic interpretability** result: a discoverable circuit that implements a specific capability.
- Olsson et al. show that:
  - Induction heads emerge at a specific point in training; that point is also when the model's in-context learning ability jumps.
  - Ablating the induction heads strongly degrades ICL.
- The result motivates a research program: identify other circuits (IOI — indirect object identification, etc.) and try to read off mechanism from weights.
- **What an interviewer wants you to know**: that mechanistic interpretability is a real research area, that induction heads are the most famous result so far, and that ICL has structural / circuit explanations rather than being pure magic.

**Common follow-ups.**
- "What's the IOI circuit?" → Wang et al. 2022, "Interpretability in the Wild" — a multi-head circuit in GPT-2 implementing indirect-object identification.
- "Can we use mech interp for safety?" → Active research direction; not yet a deployed safety technique.

**Common mistakes.**
- Saying ICL = induction heads. ICL is broader; induction heads explain a slice.
- Saying it's a single attention head — it's a two-head circuit.

**References.**
- [Olsson et al. — "In-context Learning and Induction Heads"](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) — the paper.
- [Wang et al. — "Interpretability in the Wild"](https://arxiv.org/abs/2211.00593) — IOI circuit.

---

### Q: Compare state-space models (Mamba) to transformers. What's the case for SSMs, and where do they still lag?

**Category:** concept
**Difficulty:** senior
**Tags:** [mamba, ssm, alternatives, long-context]

**Short answer.** State-space models (Mamba, Mamba-2) maintain a recurrent hidden state that processes tokens sequentially with O(n) compute and constant-per-token memory at inference — vs. transformer's O(n²) attention. The selective-scan trick in Mamba (Gu & Dao 2023) makes the state input-dependent, recovering content-addressable behavior that earlier SSMs lacked. SSMs are competitive on language modeling at smaller scale and on synthetic long-range tasks, but transformers still dominate at frontier scale; hybrid SSM-attention models (Jamba, Zamba, RecurrentGemma) are an active middle ground in 2025–2026.

**Expansion / why this is the answer.**
- **Why care**: transformer attention is O(n²) in seq length; SSMs are O(n). For very long context, this is a fundamentally better complexity class.
- **SSM basics**: an SSM has the form `h_t = A h_{t-1} + B x_t; y_t = C h_t`. With structure on `A` (HiPPO, Gu et al. 2020), it can capture long dependencies.
- **S4, S5, H3**: precursors; structured matrices for efficient computation.
- **Mamba** (Gu & Dao 2023):
  - Makes `B, C, Δt` input-dependent ("selective" SSM).
  - Selective scan computed with a custom hardware-aware kernel.
  - Matches transformers on LM perplexity at small-to-medium scale; sometimes exceeds on long-range tasks.
- **Mamba-2** (Dao & Gu 2024): reframes selective SSM as a matrix product, allowing parallel-friendly computation; competitive at billion-parameter scale.
- **Where SSMs lag**:
  - Frontier-scale language modeling (>30B): transformers remain the empirical winner.
  - In-context learning behavior is different (no induction heads in the standard form).
  - Tool / structured / code tasks: transformers' attention pattern remains the standard.
- **Hybrid models** (the modal 2025 design choice for long context):
  - **Jamba** (AI21, 2024): mix Mamba and attention layers (1:7 ratio).
  - **RecurrentGemma** (Google, 2024): Griffin-style recurrence + local attention.
  - **Zamba** (2024): SSM blocks with a shared attention every few layers.
- **What an interviewer wants you to know**: that SSMs exist, that the complexity-class argument is real but the quality gap at scale is also real, and that hybrid designs are the active frontier.

**Common follow-ups.**
- "Why no in-context-learning equivalent of induction heads in Mamba?" → Selective-scan can mimic some, but the attention-driven mechanism is different; an active research area.
- "Inference benefit?" → No KV cache; constant memory per layer regardless of context length.

**Common mistakes.**
- Saying "SSMs have replaced transformers" — they have not at scale.
- Calling Mamba "RNN-like" — it is recurrent, but the selective-scan kernel gives it transformer-like training parallelism.

**References.**
- [Gu & Dao — "Mamba"](https://arxiv.org/abs/2312.00752) — Mamba.
- [Dao & Gu — "Transformers are SSMs" (Mamba-2)](https://arxiv.org/abs/2405.21060) — Mamba-2.
- [Lieber et al. — "Jamba"](https://arxiv.org/abs/2403.19887) — hybrid model.
- [De et al. — "Griffin/RecurrentGemma"](https://arxiv.org/abs/2402.19427) — Griffin recurrence.

---

### Q: How do multimodal LLMs (LLaVA, Flamingo, GPT-4V) align vision with language?

**Category:** concept
**Difficulty:** senior
**Tags:** [multimodal, llava, vision-language, clip]

**Short answer.** Encode the image with a vision encoder (typically a ViT, often CLIP-pretrained); project the visual features into the LLM's token-embedding space with a small adapter (linear projection or MLP for LLaVA; Q-Former for BLIP-2; cross-attention layers for Flamingo); concatenate or interleave the resulting "image tokens" with text tokens in the LLM's input. Train end-to-end (or with the LLM frozen and only the projection trained, then unfreeze). The visual representation lives in the LLM's representational space alongside language.

**Expansion / why this is the answer.**
- **Components**:
  1. **Vision encoder**: produces image patch embeddings. ViT-Large, often initialized from CLIP (CLIP-ViT-L/14 is the canonical choice).
  2. **Projection / adapter**: maps the visual features into the LLM's embedding dim. Variants:
     - **Linear / MLP projector** (LLaVA, LLaVA-1.5/1.6): cheap; reuses every visual token; common in 2024.
     - **Q-Former** (BLIP-2, MiniGPT-4): a small transformer that queries the visual features and outputs a fixed number of tokens.
     - **Perceiver Resampler** (Flamingo): cross-attention down-sample.
  3. **LLM backbone**: any decoder-only LLM (LLaMA, Mistral, Vicuna in OSS; closed for GPT-4V, Gemini).
- **Training stages**:
  1. **Alignment / projector-only pretraining**: freeze the LLM and the vision encoder; train just the projector on `(image, caption)` data. Cheap.
  2. **Instruction tuning**: unfreeze the LLM (often only adapter weights via LoRA at first); fine-tune on multimodal instruction data (GPT-4-generated `(image, question, answer)` triples).
- **Architectural choice axes**:
  - **Late fusion** (LLaVA): visual tokens come in once at the input; the LLM processes them uniformly.
  - **Cross-attention fusion** (Flamingo): every (or some) LLM layer cross-attends to vision; more expressive, more params.
  - **Native multimodal** (Chameleon, Gemini, recent GPT-4o): trained from scratch on interleaved modalities; tokens for image + text share a vocabulary.
- **Eval benchmarks**: MMMU, MathVista, MMSTAR, ChartQA, DocVQA.

**Common follow-ups.**
- "Why does CLIP-pretraining help?" → CLIP's vision encoder is already aligned with text via contrastive training; saves alignment work.
- "What's a 'native multimodal' model?" → Trained from scratch with text and image tokens interleaved (not bolted on later) — GPT-4o, Gemini 1.5/2.x, Chameleon.

**Common mistakes.**
- Saying multimodal LLMs use the same parameters for vision and text — they don't; there's a separate vision encoder.
- Calling LLaVA "trained from scratch" — it's a projector-trained adapter atop frozen pretrained encoders.

**References.**
- [Liu et al. — "Visual Instruction Tuning" (LLaVA)](https://arxiv.org/abs/2304.08485) — LLaVA.
- [Alayrac et al. — "Flamingo"](https://arxiv.org/abs/2204.14198) — cross-attention fusion.
- [Li et al. — "BLIP-2"](https://arxiv.org/abs/2301.12597) — Q-Former.
- [Radford et al. — "CLIP"](https://arxiv.org/abs/2103.00020) — vision-language pretraining.
- [Chameleon team — "Chameleon"](https://arxiv.org/abs/2405.09818) — native multimodal.

---

### Q: What's the "residual stream" in a transformer, and why is it the right abstraction for interpretability?

**Category:** concept
**Difficulty:** senior
**Tags:** [residual-stream, interpretability, mech-interp]

**Short answer.** Each layer in a pre-norm transformer reads from and writes to a single shared **residual stream** of vectors `x_l`. The attention and FFN sub-layers each compute an update they add to the stream — they don't transform the stream in place. This makes the residual stream a "communication channel" through depth: each layer can read information any prior layer wrote. Mech-interp uses this abstraction because every intermediate computation has a stable basis (the residual stream's dimensions), and contributions decompose cleanly across layers and heads.

**Expansion / why this is the answer.**
- The structure of a pre-norm block:
  - `x_{l+1} = x_l + Attn(LN(x_l)) + FFN(LN(x_l + Attn(LN(x_l))))`
- All updates are *added* to the residual stream; the stream is the persistent state.
- **Reading vs. writing**:
  - Each attention head's **query** "reads" along its `W_Q` direction; its **value/output projection** writes along `W_OV`.
  - Each FFN reads along `W_in`; writes via `W_out`.
- This means an attention head from layer 3 can "read" something that an FFN at layer 1 "wrote" into the stream — long-range dependencies through depth.
- **Why this matters for mech-interp**:
  - Each head/FFN is a "circuit element"; the stream is the wire.
  - Contributions decompose linearly across heads and layers (in pre-norm).
  - You can ablate one head and see its effect on the downstream stream.
  - Anthropic's "Transformer Circuits" thread (Elhage et al. 2021) formalizes this view.
- **What this is NOT**: a description of activations as "concepts." It's a description of the architecture's data flow.
- **Implication for interpretability research**:
  - Linear-probe experiments target specific residual-stream directions.
  - Activation patching = swap residual-stream activations between two prompts and see what changes.
  - Sparse autoencoders try to find "feature directions" in the residual stream that correspond to interpretable concepts.

**Common follow-ups.**
- "Why does post-norm not have this clean structure?" → Post-norm applies LN *after* the residual add, breaking the linear decomposition.
- "What's a sparse autoencoder doing on the residual stream?" → Decomposing residual activations into a sparse dictionary of features, hoping each feature corresponds to a human-interpretable concept (Cunningham et al. 2023; Templeton et al. 2024).

**Common mistakes.**
- Confusing the residual stream with the input embedding (it evolves through layers).
- Calling it a "skip connection" without acknowledging the layer-by-layer accumulation.

**References.**
- [Elhage et al. — "A Mathematical Framework for Transformer Circuits"](https://transformer-circuits.pub/2021/framework/index.html) — the residual stream formalism.
- [Templeton et al. — "Scaling Monosemanticity"](https://transformer-circuits.pub/2024/scaling-monosemanticity/) — sparse autoencoders on the residual stream.

---

### Q: What is multi-token prediction (MTP), and how does DeepSeek-V3 use it?

**Category:** concept
**Difficulty:** senior
**Tags:** [mtp, multi-token-prediction, deepseek, training-objective]

**Short answer.** Multi-token prediction (MTP) adds auxiliary heads that predict the next 2, 3, ... tokens in parallel from the same hidden state — a denser learning signal per training step than vanilla next-token-only. DeepSeek-V3 uses MTP with sequential prediction modules during training to improve data efficiency (and speculative-decoding compatibility); only the main next-token head is used at inference unless speculative decoding is enabled.

**Expansion / why this is the answer.**
- **Standard objective**: predict `x_{t+1}` from `x_{≤t}`. One token of signal per position.
- **MTP** (Gloeckle et al. 2024, Meta; refined in DeepSeek-V3 2024):
  - Additional heads (or full transformer modules in DeepSeek-V3) predict `x_{t+2}`, `x_{t+3}`, ... from the same backbone state.
  - Auxiliary losses train these heads.
  - More training signal per token; faster convergence, better data efficiency.
- **DeepSeek-V3 specifics**:
  - Each MTP module is a small transformer that takes the main model's hidden state plus the next ground-truth token (chained) and predicts the token after.
  - Trained jointly with the main next-token loss, weighted.
  - At inference: the main head produces tokens; the MTP heads serve as **speculative-decoding draft heads** — the model is its own draft model.
- **Why this helps**:
  - Dense signal: every position teaches several positions of "what comes after."
  - At inference: speculative decoding speeds up tokens-per-second roughly proportional to the average acceptance rate of MTP heads.
- **What it does not do**: change the inference contract — the model still emits one token at a time unless spec-decoding is used.

**Common follow-ups.**
- "Why is MTP a better training objective than just longer context?" → MTP forces the model to maintain multi-step coherence in a single hidden state, not just temporally adjacent.
- "Connection to Medusa?" → Medusa (Cai et al. 2024) adds prediction heads at *inference time* to an already-trained model; MTP integrates them during training.

**Common mistakes.**
- Calling MTP "predicting all tokens at once" — it's still autoregressive at inference; the multi-token signal is a training trick.

**References.**
- [Gloeckle et al. — "Better & Faster Large Language Models via Multi-token Prediction"](https://arxiv.org/abs/2404.19737) — Meta MTP paper.
- [DeepSeek-AI — "DeepSeek-V3 Technical Report"](https://arxiv.org/abs/2412.19437) — production use of MTP.

---

### Q: Why does tokenization matter so much for chain-of-thought arithmetic?

**Category:** concept
**Difficulty:** mid
**Tags:** [tokenization, arithmetic, chain-of-thought]

**Short answer.** Most LLM tokenizers split multi-digit numbers inconsistently — "1234" might be one token, two tokens, or split differently from "1235." The model has to learn arithmetic separately for each tokenization pattern, fragmenting the signal. Chain-of-thought helps because it makes the model emit one digit at a time, recovering position-by-position computation; modern LLMs (LLaMA 3, Claude, GPT-4o) add tokenizer-level rules (e.g. always split numbers digit-by-digit) for the same reason.

**Expansion / why this is the answer.**
- The problem: BPE tokenizers learn merges based on frequency in training data. Common numbers ("100", "2024", "iPhone 15") get their own tokens; uncommon ones don't.
- Consequence: for arithmetic, "256 + 257" might tokenize as `[256][ + ][257]` (good) or `[2][56][ + ][2][57]` (bad, fragmented), depending on the specific tokenizer.
- Models trained on the bad tokenization must learn arithmetic separately for each fragmentation; they're worse at it than humans expect.
- **Chain-of-thought helps** because:
  - Forces the model to emit intermediate steps, breaking the number into manageable pieces.
  - The carry/borrow operation is exposed and can be computed digit-by-digit.
- **Tokenizer-level fixes**:
  - **Digit-by-digit tokenization**: LLaMA 3 introduced a rule splitting all sequences of digits into individual digit tokens. Improves arithmetic noticeably.
  - **Right-to-left tokenization** for numbers: ensures consistent fragmentation of long numbers from the least-significant digit.
- **Empirical results**: Singh & Strouse 2024 ("Tokenization counts: the impact of tokenization on arithmetic in frontier LLMs") quantifies the effect.

**Common follow-ups.**
- "Why don't we tokenize everything character-level then?" → Sequence length explodes; attention is quadratic; inefficient at scale.
- "What's the right tokenizer for code?" → Code-specialized BPE that handles symbols/operators as separate tokens; reduces fragmentation.

**Common mistakes.**
- Blaming "the model" for arithmetic errors when tokenization is the underlying cause.
- Assuming all tokenizers are equivalent; they aren't.

**References.**
- [Singh & Strouse — "Tokenization counts: the impact of tokenization on arithmetic in frontier LLMs"](https://arxiv.org/abs/2402.14903) — empirical study.
- [OpenAI tiktoken](https://github.com/openai/tiktoken) — the GPT tokenizer family.

---

### Q: What's the difference between weight tying and weight sharing? When is each used?

**Category:** concept
**Difficulty:** mid
**Tags:** [weight-tying, weight-sharing, embedding]

**Short answer.** **Weight tying** in transformers usually means sharing the parameters of the input token embedding table with the output (LM head) projection — `W_out = W_emb` (transposed). Saves parameters and empirically improves perplexity (Press & Wolf 2017). **Weight sharing** more broadly means any reuse of parameters across positions or layers — e.g. ALBERT shares parameters across transformer layers, dramatically shrinking the model. Modern decoder-only LLMs use input/output tying; few use cross-layer sharing because the depth-specific computation is too valuable.

**Expansion / why this is the answer.**
- **Embedding tying**:
  - Logic: the embedding matrix `W_emb ∈ ℝ^{V × d}` maps tokens to vectors; the LM head `W_out ∈ ℝ^{d × V}` maps the final hidden state to logits over the vocabulary.
  - Tied: `W_out = W_embᵀ`. Saves `V × d` parameters (substantial for large vocab).
  - Empirically improves perplexity slightly (Press & Wolf 2017, "Using the Output Embedding to Improve Language Models").
  - Most modern LLMs use this: GPT-2/3, LLaMA family, Mistral, etc.
- **Cross-layer parameter sharing** (ALBERT, Lan et al. 2019):
  - Same parameters used in every transformer block.
  - Drastically reduces parameter count; surprisingly competitive on benchmarks for ALBERT-size models.
  - Not used at frontier scale because the parameter budget is well-spent on per-layer specialization.
- **Other forms**:
  - **Multi-Query Attention**: shares K and V across heads (a form of attention-axis sharing).
  - **Tied positional encodings**: not really used; positional encodings are typically separate.

**Common follow-ups.**
- "Why does tying help perplexity?" → Empirically; intuition: the embedding learned for "the" should be the same vector the output projection expects when predicting "the." Tying enforces this constraint.
- "Cost?" → A single constraint; almost free.

**Common mistakes.**
- Conflating embedding tying (input-output) with attention parameter sharing.
- Assuming ALBERT-style sharing is always good — it's a quality tradeoff.

**References.**
- [Press & Wolf — "Using the Output Embedding to Improve Language Models"](https://arxiv.org/abs/1608.05859) — embedding tying.
- [Lan et al. — "ALBERT"](https://arxiv.org/abs/1909.11942) — cross-layer sharing.

---

### Q: What is grouped-query MoE (combining MoE with GQA)? Why does it matter for serving?

**Category:** concept
**Difficulty:** senior
**Tags:** [moe, gqa, hybrid, serving]

**Short answer.** Modern frontier-scale MoE models (DeepSeek-V3, Qwen-MoE, Mixtral) combine **MoE** in the FFN sublayer with **GQA** or **MLA** in the attention sublayer. MoE cuts active FFN compute (only top-k experts fire); GQA cuts KV-cache memory and bandwidth. The combination yields a model with large total parameters and small active compute *and* small KV cache — the right shape for cost-effective LLM serving at scale.

**Expansion / why this is the answer.**
- **MoE benefit**: total params can be 10×+ the active params; capacity grows without per-token compute growing.
- **GQA benefit**: KV cache shrinks; decode bandwidth drops; throughput rises.
- **Combined**:
  - Mixtral 8x7B: GQA-8 attention + MoE-8 FFN (k=2). 47B total params, ~13B active.
  - DeepSeek-V3: MLA attention (further cache reduction than GQA) + 256 routed experts + 1 shared expert (k=8 routed + 1 shared). 671B total params, ~37B active.
- **Why this is the right shape for serving**:
  - Decode is memory-bandwidth-bound.
  - MoE keeps active compute small.
  - GQA/MLA keeps KV-cache memory small.
  - Result: serve a 100B+ model at a per-token cost not far from a 30B dense model.
- **What this costs at training time**:
  - More total parameters → more communication for distributed training.
  - MoE adds expert-parallel communication and load-balancing complexity.
  - GQA reduces attention compute slightly compared to MHA.
- **What an interviewer wants you to know**: the architecture choices are tightly tied to serving economics; MoE alone or GQA alone is partial; the combination is what makes 600B-class models economically deployable.

**Common follow-ups.**
- "Why doesn't every MoE use GQA?" → Mostly does in 2024+; older designs (Switch Transformer) predate the GQA shift.
- "What about MLA?" → MLA goes further than GQA in cache reduction; used in DeepSeek-V2/V3.

**Common mistakes.**
- Treating MoE and GQA as alternatives — they address different axes (FFN compute vs. attention memory).

**References.**
- [Jiang et al. — "Mixtral of Experts"](https://arxiv.org/abs/2401.04088) — GQA + MoE.
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) — MLA + fine-grained MoE.
- [Ainslie et al. — "GQA"](https://arxiv.org/abs/2305.13245) — GQA.

---

### Q: What is the "softmax-bottleneck" problem, and is it still a concern?

**Category:** concept
**Difficulty:** senior
**Tags:** [softmax-bottleneck, lm-head, expressiveness]

**Short answer.** Yang et al. (2017) argued that the final softmax LM head bottlenecks expressiveness: the output distribution lives in a low-rank space determined by the embedding × LM-head matrix, while true natural-language conditional distributions are high-rank. Their proposed fix (Mixture of Softmaxes, MoS) gave modest gains on small LMs. At modern LLM scale, the bottleneck is rarely the binding constraint — capacity is abundant — and MoS is not standard. Knowing about it signals familiarity with classical LM theory.

**Expansion / why this is the answer.**
- The argument:
  - Softmax LM head: `P(x_t | x_<t) = softmax(W h_t)` where `h_t ∈ ℝ^d, W ∈ ℝ^{V × d}`.
  - The matrix of log-probabilities over all contexts × vocabulary has rank ≤ `d`.
  - Yang et al. argue real-language distributions have effective rank > `d`.
- **Mixture of Softmaxes (MoS)**: model `P = Σ π_k softmax(W h_t^{(k)})` for several heads; effective rank can exceed `d`.
- **Counter-argument** (Kasai et al. 2020 and the broader scaling argument): at sufficient model size, `d` is large enough that the bottleneck isn't the binding constraint; LM performance keeps improving with width.
- **Modern stance**: not used in production LLMs. The argument is academically interesting; the practical fix is "make the model bigger."

**Common follow-ups.**
- "Has anyone revisited this at LLM scale?" → A handful of papers; no clear positive result motivating MoS for production.
- "What is rank in this context?" → Effective rank of the (context × vocab) log-probability matrix.

**Common mistakes.**
- Citing MoS as "modern best practice" — it isn't.

**References.**
- [Yang et al. — "Breaking the Softmax Bottleneck"](https://arxiv.org/abs/1711.03953) — the canonical paper.

---

### Q: What is the gradient flow through attention? How are attention weights' gradients computed?

**Category:** derivation
**Difficulty:** senior
**Tags:** [attention, gradients, derivation]

**Short answer.** The gradient w.r.t. attention weights `A = softmax(S)` flows through the softmax via the Jacobian `∂A_ij/∂S_ik = A_ij(δ_jk − A_ik)`. The gradient w.r.t. `S = QKᵀ/√d` then splits into `Q` and `K` gradients via standard matrix-product chain rule. FlashAttention recomputes these on the backward pass from the saved softmax statistics (max + sum), avoiding the need to store the full attention matrix.

**Expansion / why this is the answer.**
- Forward: `S = QKᵀ/√d; A = softmax(S, dim=-1); O = AV`.
- Backward:
  - `∂L/∂V = AᵀP` where `P = ∂L/∂O`.
  - `∂L/∂A = PVᵀ`.
  - `∂L/∂S = J_softmax(∂L/∂A)` — the softmax Jacobian.
  - `∂L/∂Q = (∂L/∂S) K / √d`.
  - `∂L/∂K = (∂L/∂S)ᵀ Q / √d`.
- The softmax Jacobian is large (`n × n` per row); standard implementations materialize the attention matrix to backward through.
- **FlashAttention backward**: recompute `S` and `A` block by block using stored row-wise max and sum; saves the HBM read.

**Common follow-ups.**
- "Why is dropout in attention rare in modern LLMs?" → Adds noise to the gradient and to inference; pretraining at scale doesn't need the regularization.
- "What's the cost of the backward in attention?" → Same order as forward FLOPs; the memory savings from FlashAttention are the main benefit.

**Common mistakes.**
- Forgetting the `/√d` in both forward and backward.

**References.**
- [Dao et al. — "FlashAttention"](https://arxiv.org/abs/2205.14135) — backward pass design.
- [Vaswani et al. — "Attention Is All You Need"](https://arxiv.org/abs/1706.03762).

---

### Q: Why do transformers train so much more stably with pre-norm than post-norm?

**Category:** concept
**Difficulty:** senior
**Tags:** [pre-norm, post-norm, training-stability]

**Short answer.** In pre-norm (`x + Sublayer(LN(x))`), the residual stream is unchanged through every layer, providing an identity gradient path back to the input. In post-norm (`LN(x + Sublayer(x))`), the LayerNorm scales the sum by inverse-magnitude — so the residual contribution to the gradient is rescaled at every layer, compounding instability at depth. At 100+ layers, post-norm requires careful LR warmup; pre-norm trains stably without.

**Expansion / why this is the answer.**
- Pre-norm gradient flow: `∂x_{l+1}/∂x_l ≈ I + ∂/∂x_l Sublayer(LN(x_l))`. The identity dominates; the perturbation is small for any reasonable sublayer.
- Post-norm: `∂x_{l+1}/∂x_l = ∂LN/∂[·] · (I + ∂Sublayer/∂x_l)`. The LayerNorm Jacobian rescales — if `Sublayer(x)` has variable scale across layers/steps, the gradient norm fluctuates accordingly.
- At depth, post-norm without warmup can have gradient norms that blow up early in training, killing stability.
- Pre-norm + RMSNorm + SwiGLU is the modern recipe.

**Common follow-ups.**
- "Does pre-norm have any downsides?" → Some papers report slightly worse final loss vs. post-norm with careful warmup. Almost no one uses post-norm at scale anymore.

**Common mistakes.**
- Drawing the pre-norm diagram but describing post-norm behavior.

**References.**
- [Xiong et al. — "On Layer Normalization in the Transformer Architecture"](https://arxiv.org/abs/2002.04745).

---

### Q: Compare absolute position encodings with relative position encodings.

**Category:** concept
**Difficulty:** mid
**Tags:** [positional-encoding, absolute, relative]

**Short answer.** **Absolute**: each position has a fixed/learned vector added to the embedding (sinusoidal, BERT's learned embeddings). The model learns position-dependent representations. **Relative**: encode the *offset* between query and key positions (Shaw et al. 2018; T5's relative position bias; RoPE's rotation encodes relative position implicitly). Relative encodings generalize better to lengths unseen at training and are the modern default.

**Expansion / why this is the answer.**
- **Absolute** (sinusoidal, learned):
  - Encodes "this token is at position 17."
  - Doesn't generalize past training length.
- **Relative**:
  - Encodes "this query token is 5 positions after this key token."
  - Generalizes; multiple flavors:
    - **Shaw et al. 2018**: explicit learned relative-position embeddings added to keys.
    - **T5**: scalar relative-position bias added to attention scores.
    - **RoPE**: rotation of Q, K by angles proportional to position; dot product depends only on the difference.
    - **ALiBi**: linear penalty by relative distance added to scores.
- **Why relative matters**: long-context generalization, content-addressable behavior independent of absolute position.

**Common follow-ups.**
- "Why does BERT use absolute despite the limit?" → BERT's max length is 512; the limit wasn't binding. Modern long-context models need relative.
- "Why is RoPE the modern winner?" → Cheap; relative; extends with PI/YaRN; learned via standard attention.

**Common mistakes.**
- Treating sinusoidal as "relative" — it's absolute (per-position vectors).

**References.**
- [Shaw et al. — "Self-Attention with Relative Position Representations"](https://arxiv.org/abs/1803.02155) — relative.
- [Raffel et al. — "T5"](https://arxiv.org/abs/1910.10683) — T5's relative bias.
- [Su et al. — "RoFormer / RoPE"](https://arxiv.org/abs/2104.09864).

---

### Q: What is the prefill complexity for attention, and how does FlashAttention change it?

**Category:** derivation
**Difficulty:** senior
**Tags:** [prefill, flashattention, complexity]

**Short answer.** Standard prefill attention is `O(n² · d)` FLOPs and `O(n²)` HBM memory traffic for the attention matrix. FlashAttention keeps the FLOPs at `O(n² · d)` (it's still quadratic; the algorithm computes the same thing) but reduces HBM traffic to `O(n · d)` by tiling — the attention matrix never materializes. The wall-clock speedup comes from the memory-traffic reduction, not from fewer FLOPs.

**Expansion / why this is the answer.**
- Attention forward FLOPs: `O(n² · d)` for `QKᵀ` and another for `AV`.
- Naive HBM traffic: `O(n² + n · d)` — the `n²` for the attention matrix dominates at long `n`.
- FlashAttention: tile `Q, K, V` into blocks; compute partial output for each Q-tile by iterating over K/V tiles in SRAM; never write the full `n × n` matrix.
- HBM traffic: `O(n · d)`.
- For seq length 8k, head dim 128: naive HBM traffic ~64MB × num_heads × num_batches; FlashAttention ~1MB × num_heads × num_batches — 60×+ less.
- Wall clock: 2–5× faster on long sequences in practice.

**Common follow-ups.**
- "Does it reduce activation memory?" → Yes — no need to store the attention matrix for the backward.
- "How does it interact with KV cache during decode?" → Decode is `n × d` per step (Q is one token); the saving is mostly in the long-context prefill.

**Common mistakes.**
- Saying FlashAttention is `O(n)` — it's `O(n²)` FLOPs; the win is memory traffic.

**References.**
- [Dao et al. — "FlashAttention"](https://arxiv.org/abs/2205.14135).

---

### Q: What's a "circuit" in mechanistic interpretability?

**Category:** concept
**Difficulty:** senior
**Tags:** [interpretability, circuits, mech-interp]

**Short answer.** A circuit is a discoverable substructure inside a transformer — a collection of attention heads and MLP layers that, together, implement a specific computation (induction, indirect-object identification, modular arithmetic). Mechanistic interpretability tries to identify these circuits from weights and activations to explain what the model is doing in human terms. Foundational: Anthropic's transformer-circuits thread (Elhage et al. 2021; Olsson et al. 2022).

**Expansion / why this is the answer.**
- **The framing**: a circuit reads from the residual stream (via attention heads or MLP inputs) and writes back to it. Compose to implement higher-level capabilities.
- **Famous circuits**:
  - **Induction heads** (Olsson et al. 2022): two-layer attention pattern; the basis of much in-context learning.
  - **IOI** (Wang et al. 2022, "Interpretability in the Wild"): indirect object identification in GPT-2.
  - **Modular arithmetic** (Nanda et al. 2023): "grokking" — model goes from memorization to generalization, learning a specific algorithm (Fourier-based for mod-p addition).
- **Methods**:
  - **Activation patching**: swap activations between two prompts to localize causal effect.
  - **Causal scrubbing**: hypothesize a circuit; replace activations not on the circuit with mean-ablations; verify performance is preserved.
  - **Sparse autoencoders**: decompose residual-stream activations into a dictionary of "features" (Templeton et al. 2024).

**Common follow-ups.**
- "Is this useful for safety?" → Aspirational. The hope: identify dangerous circuits (deception, reward hacking) and ablate. Currently in research.
- "What's grokking?" → Phenomenon where a model trained on a small task suddenly transitions from memorizing to generalizing late in training (Power et al. 2022).

**Common mistakes.**
- Treating "feature" and "circuit" as synonyms — features are scalar properties; circuits are computational sub-structures.

**References.**
- [Elhage et al. — "Mathematical Framework for Transformer Circuits"](https://transformer-circuits.pub/2021/framework/index.html).
- [Wang et al. — "Interpretability in the Wild"](https://arxiv.org/abs/2211.00593).
- [Templeton et al. — "Scaling Monosemanticity"](https://transformer-circuits.pub/2024/scaling-monosemanticity/).

---

### Q: What's the difference between encoder-only and decoder-only embeddings?

**Category:** concept
**Difficulty:** mid
**Tags:** [embeddings, encoder, decoder]

**Short answer.** **Encoder-only** (BERT-style): bidirectional attention; pretrained on MLM; pooled output (CLS token or mean-pool) gives a sentence/passage embedding. **Decoder-only** (GPT-style): causal attention; pretrained on next-token prediction; final-token hidden state can be used as an embedding, but typically you use a separate instruction-tuned wrapper or a contrastive fine-tune. Encoder embeddings are typically *better* per unit compute for retrieval; decoder embeddings catch up at large scale.

**Expansion / why this is the answer.**
- **Encoder embeddings** (BERT, RoBERTa, ModernBERT, BGE, E5):
  - Bidirectional context: every token sees every other token.
  - Better at sentence-level semantic compression for the same parameter count.
  - Standard for retrieval (SBERT, BGE, E5).
- **Decoder embeddings**:
  - Causal — each token only sees its past.
  - Can extract from final hidden state or pool over the sequence.
  - **Instruction-tuned variants** (E5-Mistral, Mistral-based embeddings): take a strong decoder, fine-tune with contrastive loss; competitive on MTEB.
- **Why encoder is "naturally better"**:
  - Bidirectionality lets the embedding integrate context from both sides.
  - Trained objective (MLM) is closer to "produce a representation that captures meaning."
- **Why decoder catches up at scale**:
  - Better base models (general-purpose strength).
  - Instruction-tuned via contrastive loss specifically for retrieval.

**Common follow-ups.**
- "Can you use a decoder LLM directly without fine-tune as an embedding?" → Yes, but it's weaker than dedicated retrievers; the next-token-prediction objective doesn't optimize for sentence-level meaning.
- "ModernBERT vs LLM-based embedding?" → ModernBERT is cheap and strong; LLM-embedding is more expensive but tops MTEB.

**Common mistakes.**
- Treating any model's final-token hidden state as a good embedding without verification.

**References.**
- [Wang et al. — "E5-Mistral"](https://arxiv.org/abs/2401.00368) — decoder-based instruction-tuned embedding.
- [Warner et al. — "ModernBERT"](https://arxiv.org/abs/2412.13663).

---

### Q: How does causal masking interact with batched / variable-length sequences?

**Category:** concept
**Difficulty:** senior
**Tags:** [causal-mask, batching, packing]

**Short answer.** In a batch of variable-length sequences, you have to combine causal masking (no future) with padding-masking (no attention to pad tokens). The naive approach pads to max length and applies both masks. Better: **sequence packing** — concatenate multiple short sequences into one long sequence with per-sequence attention boundaries enforced by a block-diagonal mask. Eliminates padding waste; common in modern LLM training (e.g. Llama 3).

**Expansion / why this is the answer.**
- **Naive batching**: pad all sequences to max length; loss is masked out on pad tokens. Wastes compute on pad.
- **Sequence packing** (Krell et al. 2021): pack as many sequences as fit in the context window; use block-diagonal attention mask to prevent cross-sequence attention.
- The mask:
  - Within each "block" (one sequence): standard causal triangle.
  - Across blocks: completely masked.
- Implementation: track sequence boundaries; build the block-diagonal mask; apply.
- **Modern stacks**: FlashAttention supports variable-length / packed batches efficiently (varlen FlashAttention).

**Common follow-ups.**
- "How does packing affect loss computation?" → Compute loss per sequence; the pad-token loss-mask still applies if you use any.
- "Why doesn't this break in attention?" → The block-diagonal mask prevents leakage.

**Common mistakes.**
- Packing without setting the cross-sequence mask — sequences contaminate each other's attention.

**References.**
- [Krell et al. — "Efficient Sequence Packing without Cross-contamination"](https://arxiv.org/abs/2107.02027).
- [FlashAttention varlen support](https://github.com/Dao-AILab/flash-attention).

---

### Q: What is layer-wise learning rate decay (LLRD), and when is it useful?

**Category:** concept
**Difficulty:** mid
**Tags:** [llrd, fine-tuning, transformer]

**Short answer.** LLRD applies progressively smaller learning rates to lower (closer-to-input) layers during fine-tuning. The intuition: lower layers learn generic features; higher layers learn task-specific patterns. Decaying LR by depth preserves the pretrained low-level features while allowing the head to adapt. Common in BERT fine-tuning and some LLM PEFT setups.

**Expansion / why this is the answer.**
- Configuration:
  - Top layer LR: `η`.
  - Layer `l`: LR `η · γ^(L − l)`, where `γ ∈ (0, 1)` is the decay (typical 0.9–0.95).
- **When useful**:
  - Small-data fine-tuning of pretrained models.
  - Tasks where lower-layer features should be preserved.
- **When not used**:
  - LLM full pretraining (uniform LR across layers).
  - LoRA (we're only training the adapters, not the base).
- **Connection to differential learning rates**: LLRD is a specific scheduled form.

**Common follow-ups.**
- "Why does the head get the highest LR?" → It starts random (task-specific head); needs more learning. Backbone is pretrained.
- "When does LLRD hurt?" → When the lower layers also need to adapt (domain shift).

**Common mistakes.**
- Picking `γ` too small (lower layers frozen effectively); too large (no benefit over uniform).

**References.**
- [Howard & Ruder — "ULMFiT"](https://arxiv.org/abs/1801.06146) — original differential-LR fine-tuning.

---

### Q: How does temperature affect generation diversity?

**Category:** concept
**Difficulty:** intro
**Tags:** [temperature, sampling, diversity]

**Short answer.** Temperature `T` divides the logits before softmax. `T < 1` sharpens the distribution (more deterministic, higher probability on top token). `T > 1` flattens (more diversity, higher probability on tail). `T → 0` is greedy. `T → ∞` is uniform. Combined with top-p/top-k, temperature controls the tradeoff between diversity and quality for the entire generation.

**Expansion / why this is the answer.**
- Math: `p_i = exp(z_i / T) / Σ_j exp(z_j / T)`.
- Information-theoretic view: temperature scales the entropy of the distribution.
- Typical settings:
  - Code: `T = 0` (deterministic).
  - Math/reasoning: `T = 0.2–0.7`.
  - Creative writing: `T = 0.7–1.0`.
  - Brainstorming / variation: `T = 1.0–1.3`.
- **Beam search uses no temperature** (deterministic by construction).
- Temperature does NOT change the *ranking* of tokens; just the *probability mass*.

**Common follow-ups.**
- "Why combine temperature with top-p?" → Temperature controls how peaked the distribution is; top-p truncates the long tail. Together you get controlled diversity without long-tail noise.
- "Is `T = 0` the same as greedy?" → Numerically: as `T → 0`, the softmax becomes one-hot at the argmax — yes.

**Common mistakes.**
- Setting `T = 0` with non-greedy sampling and getting NaN (division by zero); use `T = 1e-6` or branch to greedy.

**References.**
- [Ackley, Hinton, Sejnowski — "A Learning Algorithm for Boltzmann Machines"](https://www.cs.toronto.edu/~hinton/absps/cogscibm.pdf) — temperature in softmax.

---

### Q: What is "weight decay" and how is it different from L2 regularization?

**Category:** concept
**Difficulty:** mid
**Tags:** [weight-decay, l2, adamw]

**Short answer.** **L2 regularization**: add `λ ||θ||²` to the loss; gradient gains `2λ θ`. **Weight decay**: directly shrink weights toward zero by `θ ← (1 - η · λ_wd) · θ` on each step. For vanilla SGD they are equivalent. For adaptive optimizers (Adam), they diverge — Adam's per-parameter scaling distorts L2's effective regularization. **AdamW** (Loshchilov & Hutter 2017) decouples weight decay from the adaptive update, restoring the intended behavior.

**Expansion / why this is the answer.**
- **L2 in Adam**: `g ← g + λθ` then Adam step. The `λθ` term gets divided by `√v` per parameter, so parameters with small accumulated gradient get *more* regularization. Surprising and harmful.
- **AdamW**: apply Adam step, then `θ ← θ - η · λ_wd · θ` separately. Each parameter is decayed identically (relative to its current value).
- Empirically: AdamW with proper weight decay generalizes better than Adam-with-L2.
- Default in modern LLM training (LLaMA, GPT-3, etc.): AdamW with `λ_wd = 0.1` typically.

**Common follow-ups.**
- "What's a sensible weight-decay value for LLM training?" → 0.01–0.1 typical; higher for smaller models.
- "Should you weight-decay LayerNorm parameters?" → Conventional practice is no; only the matmul weights.

**Common mistakes.**
- Using "Adam" when you mean "AdamW" — the difference matters at scale.

**References.**
- [Loshchilov & Hutter — "Decoupled Weight Decay Regularization"](https://arxiv.org/abs/1711.05101).

---

### Q: What is the role of attention sinks?

**Category:** concept
**Difficulty:** senior
**Tags:** [attention-sink, streaming-llm, long-context]

**Short answer.** "Attention sinks" (Xiao et al. 2024, StreamingLLM) are the first few tokens of a sequence that, in trained transformers, receive a disproportionate share of attention from every layer — they act as a "default destination" for unused attention mass. Critical for streaming-LLM inference: if you evict the sink tokens (e.g. via a sliding window without keeping the start), perplexity blows up. Modern long-context serving keeps a small number of sink tokens + a sliding window over recent tokens.

**Expansion / why this is the answer.**
- Empirical observation: in trained transformers, the first 1–4 tokens receive ~20%+ of attention weight across heads, regardless of content.
- **Why**: softmax must distribute 1.0 across the keys; if no key is a good match, the model parks attention at "always-present" positions. The early tokens fill this role.
- **Implication for streaming/infinite-context**:
  - Naive sliding-window attention (drop oldest tokens) destroys the sinks.
  - StreamingLLM: keep the first `k` tokens + the most recent `w` tokens; performance preserved.
- **Architectural fix**: train with explicit sink tokens (Xiao et al.); the model learns to use them properly without depending on the literal start-of-sequence.

**Common follow-ups.**
- "Why don't naive sliding-window models train this away?" → They don't; the behavior emerges from softmax + content distribution. Has to be addressed at training or serving.
- "Connection to MoE?" → Some MoE issues with router collapse are analogous: gradient flow biased toward default behaviors.

**Common mistakes.**
- Treating sinks as a "bug" — they're an emergent property of softmax.

**References.**
- [Xiao et al. — "Efficient Streaming Language Models with Attention Sinks"](https://arxiv.org/abs/2309.17453).

---

### Q: What's the difference between encoder cross-attention and decoder cross-attention?

**Category:** concept
**Difficulty:** mid
**Tags:** [cross-attention, encoder-decoder, seq2seq]

**Short answer.** Both refer to the decoder cross-attending to the encoder's output in an encoder-decoder transformer (T5, BART, original Vaswani). The encoder doesn't have cross-attention; only its own self-attention. The decoder has: (1) masked self-attention (only past target tokens), (2) cross-attention from decoder Q to encoder K/V. The "cross" is the decoder pulling source information.

**Expansion / why this is the answer.**
- Encoder layer: self-attention (bidirectional) + FFN.
- Decoder layer: masked self-attention + cross-attention + FFN.
- Cross-attention: Q from the decoder's current hidden state, K/V from the encoder's output. Lets the decoder "look at" the source while generating.
- **Used in**: T5, BART, original transformer (translation), Flamingo (vision-language cross-attention).
- **Not used in decoder-only LLMs** (GPT family) — there's no separate source to cross-attend to.

**Common follow-ups.**
- "Why don't decoder-only LLMs have it?" → They concatenate source + target in the prompt; self-attention over the whole sequence handles "looking at the source."
- "Cross-attention in Flamingo?" → Image features cross-attended from LLM layers; gives the LLM access to vision.

**Common mistakes.**
- Calling self-attention "cross-attention" when Q and K come from the same sequence — that's still self.

**References.**
- [Vaswani et al. — "Attention Is All You Need"](https://arxiv.org/abs/1706.03762).
- [Alayrac et al. — "Flamingo"](https://arxiv.org/abs/2204.14198) — vision-language cross-attention.

---

### Q: What is grokking?

**Category:** concept
**Difficulty:** senior
**Tags:** [grokking, generalization, training-dynamics]

**Short answer.** Grokking (Power et al. 2022): a phenomenon where a model trained on a small algorithmic task (e.g. modular arithmetic) memorizes the training set quickly but generalizes only much later in training — sometimes after 100× more steps. Suggests the optimization landscape contains multiple local minima of different generalization quality; weight decay and continued training are necessary to find the generalizing one.

**Expansion / why this is the answer.**
- The empirical curve: training accuracy hits 100% quickly; test accuracy stays at chance for many epochs; then suddenly jumps to 100%.
- Tasks: modular arithmetic, addition, copying patterns.
- **Mechanistic explanation** (Nanda et al. 2023): the model first memorizes via dense circuits; weight decay slowly drives the model toward a sparse, generalizing circuit (Fourier-based for mod-p addition).
- **Implications**:
  - Generalization can be "earned" via continued training even after training loss is zero.
  - Weight decay is important — without it, no transition.
- **Not just a curiosity**: suggests over-parameterized models have multiple regimes, only some of which generalize.

**Common follow-ups.**
- "Does grokking happen in LLMs?" → Not in the dramatic form; LLM pretraining is data-rich enough that the trivial-memorization minimum isn't reached.
- "Implications for fine-tuning?" → Suggests "train longer than you think" on small data with weight decay can find better-generalizing solutions.

**Common mistakes.**
- Conflating grokking with double descent (different phenomena).

**References.**
- [Power et al. — "Grokking"](https://arxiv.org/abs/2201.02177).
- [Nanda et al. — "Progress measures for grokking"](https://arxiv.org/abs/2301.05217).

---

### Q: What is the role of bias terms in transformers?

**Category:** concept
**Difficulty:** intro
**Tags:** [bias-terms, architecture, llama]

**Short answer.** Most modern LLMs (LLaMA, Mistral) **drop bias terms** from linear layers and use bias-free LayerNorm/RMSNorm. The decision is empirical: biases add parameters and rarely improve quality at scale; some papers report mild stability gains from removing them. Original BERT/GPT-2 used biases; LLaMA-family normalized this out.

**Expansion / why this is the answer.**
- Linear layer: `y = Wx + b`. The `b` adds `d` parameters per layer.
- Total bias parameters: ~`O(d · n_layers)` — small relative to weight matrices `O(d² · n_layers)`.
- Empirical findings:
  - At scale, biases contribute very little.
  - Some recipes (Chinchilla, Llama) drop them.
  - GPT-3 kept them; LLaMA dropped them.
- RMSNorm specifically drops the bias (the `β` in LayerNorm); this is the "RMS" part.

**Common follow-ups.**
- "Does it affect training stability?" → Some reports of mild improvement; not dramatic.
- "Why does the FFN still benefit from gating despite no biases?" → SwiGLU's multiplication is fundamentally different from a bias add.

**Common mistakes.**
- Calling LayerNorm "bias-free" — it has a bias (`β`); only RMSNorm doesn't.

**References.**
- [Touvron et al. — "LLaMA"](https://arxiv.org/abs/2302.13971) — bias-free linears.
- [Zhang & Sennrich — "RMSNorm"](https://arxiv.org/abs/1910.07467).

---

### Q: What is mixed expert parallelism (expert parallelism) for MoE training?

**Category:** concept
**Difficulty:** senior
**Tags:** [expert-parallel, moe, distributed]

**Short answer.** In MoE training, each expert is large enough that putting all of them on one GPU is infeasible. **Expert parallelism (EP)** distributes experts across devices: each device owns a subset of experts. Per token, routing decides which experts to call; tokens are **all-to-all** shuffled to the device hosting the chosen expert, processed, then **all-to-all** back. This adds communication overhead absent in dense training.

**Expansion / why this is the answer.**
- **The setup**: 64 experts; 64 GPUs; one expert per GPU.
- **Per-batch flow**:
  1. Forward gating: compute which expert each token routes to.
  2. **All-to-all dispatch**: send each token to the GPU hosting its chosen expert.
  3. Each GPU runs its expert on the received tokens.
  4. **All-to-all combine**: send results back to the originating GPU.
- **Communication cost**: linear in batch × token dim; significant.
- **Combining with TP, DP, PP**:
  - DP + EP: data-parallel replicas, each replicated across the expert pool.
  - TP + EP: tensor-parallel within each expert, expert-parallel across experts.
- **Load balancing**: critical to avoid expert hotspots that bottleneck the all-to-all.

**Common follow-ups.**
- "Why does MoE need fast interconnect?" → All-to-all is communication-intensive; high-bandwidth NVLink / Infiniband matter.
- "What's expert capacity?" → A hard cap on tokens per expert per batch; tokens beyond the cap are dropped or rerouted.

**Common mistakes.**
- Treating EP as "just another parallelism axis" without recognizing all-to-all cost.

**References.**
- [Fedus, Zoph, Shazeer — "Switch Transformer"](https://arxiv.org/abs/2101.03961).
- [Lepikhin et al. — "GShard"](https://arxiv.org/abs/2006.16668) — expert parallelism.

---

### Q: What is the relationship between transformer FFN and a key-value memory?

**Category:** concept
**Difficulty:** senior
**Tags:** [ffn, kv-memory, geva-2021]

**Short answer.** Geva et al. (2021): the transformer FFN sublayer can be interpreted as a **key-value memory**. The first matrix's rows are "keys" matched against the input; the activation pattern is sparse; the second matrix's columns are "values" associated with each key. Suggests FFN parameters encode discrete factual associations the model has learned, and modifying specific rows can edit specific facts (ROME, MEMIT).

**Expansion / why this is the answer.**
- The FFN: `y = W_2 σ(W_1 x)`.
- Interpretation: `W_1` rows are "keys" (probe vectors), `σ(...)` is a sparse activation pattern, `W_2` columns are "values" (output contributions).
- **Empirical evidence**:
  - FFN activations are sparse (many near zero).
  - Specific FFN rows correspond to specific factual associations (Geva et al.).
- **ROME** (Meng et al. 2022): edit a specific fact by modifying a small subset of FFN weights at a specific layer.
- **MEMIT** (Meng et al. 2023): scaled up — edit thousands of facts in batch.
- **Implication**: FFN ≈ a sparse retrieval-augmented memory; attention is the routing.
- Caveat: the picture is simplified; the model uses FFNs for more than just factual storage.

**Common follow-ups.**
- "Why does this matter for safety?" → Suggests model editing is feasible — but also that facts can be unlearned or modified in targeted ways.
- "Connection to MoE?" → MoE makes the key-value memory *explicit*: experts as memory shards, router as the addressing.

**Common mistakes.**
- Taking the KV-memory analogy literally — the FFN is a function approximator that *also* implements memory-like behavior; not exclusively memory.

**References.**
- [Geva et al. — "Transformer Feed-Forward Layers Are Key-Value Memories"](https://arxiv.org/abs/2012.14913).
- [Meng et al. — "ROME"](https://arxiv.org/abs/2202.05262).

---

### Q: What is speculative decoding's correctness proof?

**Category:** derivation
**Difficulty:** senior
**Tags:** [speculative-decoding, theory]

**Short answer.** Speculative decoding (Leviathan et al. 2023) accepts a draft token with probability `min(1, p_target/p_draft)`. If rejected, sample from a corrected distribution `(p_target − p_draft)_+ / normalizer`. This procedure is **provably unbiased** — the output distribution equals the target model's distribution exactly. The proof is a direct rejection-sampling argument.

**Expansion / why this is the answer.**
- Notation: `p` = target distribution; `q` = draft distribution.
- Draft samples `t ~ q`; accept with probability `min(1, p(t)/q(t))`.
- If accepted, output `t`.
- If rejected, sample `t' ~ r` where `r(x) = max(0, p(x) − q(x)) / normalizer`. Output `t'`.
- **Proof sketch**:
  - `P(output = x) = q(x) · min(1, p(x)/q(x)) + (1 − accept_rate) · r(x)`.
  - For `x` with `p(x) ≤ q(x)`: accepted with prob `p(x)/q(x)`; total prob `q(x) · p(x)/q(x) = p(x)`.
  - For `x` with `p(x) > q(x)`: accepted with prob 1; total prob `q(x)`. Plus the rejected-resampling contribution.
  - Sum: exactly `p(x)`.
- **Why this matters**: spec-decoding doesn't change the model's output distribution. It's a pure latency optimization.

**Common follow-ups.**
- "What's the acceptance rate?" → Depends on how well `q` approximates `p`; typical 60–90%.
- "Speedup math?" → On average, you produce more than one token per target-forward pass. Expected speedup ≈ (1 + acceptance_rate × draft_length).

**Common mistakes.**
- Treating spec-decoding as approximate — it's exact.

**References.**
- [Leviathan, Kalman, Matias — "Fast Inference from Transformers via Speculative Decoding"](https://arxiv.org/abs/2211.17192) — proof in §3.

---

### Q: What is parallel decoding (Jacobi / Lookahead)?

**Category:** concept
**Difficulty:** senior
**Tags:** [parallel-decoding, jacobi, lookahead]

**Short answer.** Parallel decoding methods (Lookahead, Fu et al. 2024; Jacobi decoding, Santilli et al. 2023) generate multiple tokens per forward pass *without* a separate draft model. The model itself proposes the next `n` tokens; the next forward pass verifies which ones are correct (Jacobi-iteration-style). Variants: Medusa (extra heads predict future positions), EAGLE (predict features), Lookahead (N-gram caching). Speedup similar to speculative decoding without the draft model.

**Expansion / why this is the answer.**
- **Jacobi decoding**: iteratively refine a sequence of `n` tokens; at each iteration, the model emits new candidates for each position; converge when stable.
- **Lookahead decoding** (Fu et al. 2024): cache N-gram patterns; predict ahead using them; verify with one forward pass.
- **Medusa** (Cai et al. 2024): add `M` extra LM heads to the base model; each predicts a future position.
- **EAGLE** (Li et al. 2024): predict hidden features of future positions, not tokens; better acceptance.
- **Trade-offs vs. speculative decoding**:
  - Pros: no separate draft model needed; simpler deployment.
  - Cons: requires model modification (Medusa, EAGLE) or specialized algorithms; speedups can be lower.
- **DeepSeek-V3's MTP**: trained-in version that doubles as self-speculation.

**Common follow-ups.**
- "Is Jacobi decoding lossless?" → Yes, when converged (last iteration produces the same tokens as autoregressive).
- "When does Medusa underperform spec decoding?" → If the extra heads aren't well-trained; the per-position accuracy directly determines speedup.

**Common mistakes.**
- Conflating parallel decoding with attention parallelism (different concept).

**References.**
- [Fu et al. — "Lookahead Decoding"](https://arxiv.org/abs/2402.02057).
- [Cai et al. — "Medusa"](https://arxiv.org/abs/2401.10774).
- [Li et al. — "EAGLE"](https://arxiv.org/abs/2401.15077).

---

### Q: What is the difference between encoder-decoder cross-attention KV cache and self-attention KV cache?

**Category:** concept
**Difficulty:** mid
**Tags:** [kv-cache, encoder-decoder, cross-attention]

**Short answer.** Encoder-decoder models have *two* KV caches: (1) the encoder's K/V (computed once from the source, reused for every decode step; static); (2) the decoder's self-attention K/V (grows with each generated target token; dynamic). The encoder's cache doesn't grow during decode; the decoder's does. This is a different cache shape than decoder-only models.

**Expansion / why this is the answer.**
- **Encoder-decoder** (T5, BART, original transformer):
  - Encoder K/V: shape `(n_layers, n_kv_heads, n_source, d_head)`. Computed once per source; constant during decode.
  - Decoder self-attention K/V: shape `(n_layers, n_kv_heads, n_target_so_far, d_head)`. Grows by 1 per step.
- **Decoder-only**:
  - Single cache; shape `(n_layers, n_kv_heads, n_total, d_head)`. Prompt + generated tokens combined.
- **Memory implication**:
  - Encoder-decoder: encoder cache constant; decoder grows.
  - Decoder-only: one big growing cache.
- **Why this matters for serving**:
  - Encoder-decoder: encoder pass amortized over many target tokens; great for "summarize this long document" workloads.
  - Decoder-only: prompt caching (T4) is the equivalent — cache the prompt's KV.

**Common follow-ups.**
- "Why has decoder-only mostly won?" → Simpler; single objective; in-context-learning is easy; serving is more uniform.

**Common mistakes.**
- Treating encoder-decoder and decoder-only as having the "same" KV cache.

**References.**
- [Vaswani et al. — "Attention Is All You Need"](https://arxiv.org/abs/1706.03762).
- [Raffel et al. — "T5"](https://arxiv.org/abs/1910.10683).

---

### Q: What is the "lost-in-the-middle" mitigation at the model level?

**Category:** concept
**Difficulty:** senior
**Tags:** [lost-in-the-middle, long-context, training]

**Short answer.** Mitigations at training: (1) train on **synthetic needle-in-haystack data** with the relevant info at varying positions; (2) **continued long-context pretraining** with documents that have important info distributed throughout; (3) **positional encoding fixes** (RoPE → YaRN / NTK extension); (4) **explicit attention pattern training** that rewards the model for attending to relevant mid-prompt context. Mitigations at serving: order retrieved passages by reranker score; truncate hard.

**Expansion / why this is the answer.**
- The training fix:
  - Liu et al. 2023 documents the problem.
  - Models like Claude 3, Gemini 1.5/2.0, GPT-4-turbo trained with long-context data including positional-diverse important info; flatten the U-curve substantially.
- **Synthetic data**:
  - Needle-in-haystack data construction; train so model accuracy is high across positions.
- **YaRN / NTK / PI**:
  - Extends RoPE to longer context; pairs with continued pretraining on long sequences.
- **Active research**: "lost in the middle" still appears in 2026 evals on real-world long-context tasks despite synthetic-NIH passing.

**Common follow-ups.**
- "How do you measure lost-in-the-middle yourself?" → Build a needle-in-haystack at varying positions; measure accuracy curve.
- "Why is real-world lost-in-the-middle harder than synthetic?" → Real-world long documents have many distractors; synthetic NIH is a single sentence to find.

**Common mistakes.**
- Reporting needle-in-haystack accuracy as proof the model handles real long-context.

**References.**
- [Liu et al. — "Lost in the Middle"](https://arxiv.org/abs/2307.03172).
- [Peng et al. — "YaRN"](https://arxiv.org/abs/2309.00071).

---

### Q: Compare GPT, BERT, T5 architectures briefly.

**Category:** concept
**Difficulty:** intro
**Tags:** [gpt, bert, t5, comparison]

**Short answer.** **GPT**: decoder-only; causal self-attention; pretrained on next-token prediction; great for generation. **BERT**: encoder-only; bidirectional self-attention; pretrained on masked LM + next-sentence prediction; great for understanding tasks. **T5**: encoder-decoder; bidirectional encoder, causal decoder with cross-attention; pretrained on span-corruption; treats all tasks as text-to-text. Modern: decoder-only (GPT-style) has won general-purpose; encoder-only persists for embeddings/classification; encoder-decoder is niche.

**Expansion / why this is the answer.**
- **GPT family** (Radford et al. 2018+):
  - Decoder-only with causal mask.
  - Pretrained: predict next token.
  - Scale: 117M (GPT) → 175B (GPT-3) → 1.8T+ (GPT-4-class est.).
- **BERT** (Devlin et al. 2018):
  - Encoder-only with bidirectional attention.
  - Pretrained: masked LM (mask 15% of tokens, predict).
  - Plus NSP (next-sentence prediction); later models (RoBERTa) drop NSP.
- **T5** (Raffel et al. 2019):
  - Encoder-decoder.
  - Span corruption: mask consecutive spans; predict them sequentially.
  - "Text-to-text" framing: every task is "input text → output text".
- **Why decoder-only won**:
  - Single objective (next-token) handles everything via prompting.
  - In-context learning falls out.
  - Serving is uniform.
  - Scaling laws hold cleanly.

**Common follow-ups.**
- "Why does BERT use [CLS]?" → A dedicated token whose final embedding is the sentence-level representation for downstream tasks.
- "T5's 'unified' framing?" → Cast classification as "predict the label string"; cast translation as text-to-text; etc.

**Common mistakes.**
- Saying GPT uses bidirectional attention.

**References.**
- [Devlin et al. — "BERT"](https://arxiv.org/abs/1810.04805).
- [Radford et al. — GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf).
- [Raffel et al. — "T5"](https://arxiv.org/abs/1910.10683).

---

### Q: What is "model surgery" and when is it done?

**Category:** concept
**Difficulty:** mid
**Tags:** [model-surgery, weight-edit, expansion]

**Short answer.** Model surgery = manually modifying trained weights for specific outcomes: (a) **width / depth expansion** (initialize a bigger model from a smaller one's weights — "Net2Net" style), (b) **layer pruning** for smaller deployment, (c) **weight editing** to fix specific facts (ROME, MEMIT), (d) **merging / interpolation** of fine-tuned variants (model souping, task arithmetic). All bypass standard training; useful for cost reduction and rapid iteration.

**Expansion / why this is the answer.**
- **Width/depth expansion** (Chen et al. 2015, Net2Net): map a small model's weights into a larger one preserving function; continue training. Used in some warm-start regimes.
- **Pruning** (lottery-ticket-style; structured pruning): cut weights below a threshold; fine-tune. Reduces model size for deployment.
- **Weight editing**:
  - **ROME** (Meng et al. 2022): locate-and-edit a single fact.
  - **MEMIT** (Meng et al. 2023): edit thousands of facts in batch.
  - **Task arithmetic** (Ilharco et al. 2022): `θ_pretrained + (θ_fine-tuned - θ_pretrained) = capability vector`; add/subtract these.
- **Model souping** (Wortsman et al. 2022): average weights of multiple fine-tunes of the same base; sometimes better than any individual.

**Common follow-ups.**
- "Does model souping work in production?" → Yes, especially across hyperparameter-different fine-tunes of the same task. Empirical risk: averaging across diverged tasks degrades.
- "Why is fact editing controversial?" → Edits can have unintended downstream effects; not a reliable way to delete information.

**Common mistakes.**
- Treating weight-edit as a substitute for fine-tuning at scale.

**References.**
- [Wortsman et al. — "Model Soups"](https://arxiv.org/abs/2203.05482).
- [Meng et al. — "ROME"](https://arxiv.org/abs/2202.05262).
- [Ilharco et al. — "Task Arithmetic"](https://arxiv.org/abs/2212.04089).

---

### Q: What's the "ringer attention" / "ring attention" idea?

**Category:** concept
**Difficulty:** senior
**Tags:** [ring-attention, long-context, sequence-parallel]

**Short answer.** Ring attention (Liu et al. 2023) is a sequence-parallel technique: each GPU holds a slice of the sequence; during attention, K/V blocks rotate around a ring of GPUs, with each device computing its Q-block's attention against received K/V. Enables training on contexts longer than fit on any single device. Used in Gemini 1.5's 1M-context training; key for very-long-context LLMs.

**Expansion / why this is the answer.**
- **Setup**: split sequence of length `n` across `P` GPUs; each holds `n/P` query tokens (and their corresponding K/V).
- **Algorithm**:
  - Each GPU computes attention of its `Q` block against its local `K, V` block.
  - Then `K, V` blocks rotate one position around the ring.
  - GPU computes attention against the next block.
  - Repeat `P` times — each Q block has attended over all K, V blocks.
- **Memory**: each GPU only holds `n/P` of the sequence; total sequence can be much larger than per-device memory.
- **Compute**: same total FLOPs as full attention; overlapped with the ring's communication.
- **Combines with**: FlashAttention (each block's local attention is FlashAttention).

**Common follow-ups.**
- "What's the communication overhead?" → The K/V rotation is one all-reduce/ring-shift per step. Overlapped with compute when possible.
- "Why is this for long context specifically?" → For short sequences, full attention on one GPU is fine; ring attention pays its communication cost only when needed.

**Common mistakes.**
- Confusing ring attention with ring-allreduce in DDP.

**References.**
- [Liu et al. — "Ring Attention with Blockwise Transformers for Near-Infinite Context"](https://arxiv.org/abs/2310.01889).

---

### Q: How does autoregressive sampling actually pick a token, step by step?

**Category:** concept
**Difficulty:** intro
**Tags:** [sampling, autoregressive, generation]

**Short answer.** (1) Model produces logits for the next token; (2) divide by temperature; (3) apply top-k or top-p truncation; (4) softmax to get a probability distribution; (5) sample from the multinomial. Append the sampled token to the context. Repeat until EOS or max-length. Greedy = step 5 is argmax; otherwise sample.

**Expansion / why this is the answer.**
- The full step-by-step:
  1. Forward pass on current context; output logits `z ∈ ℝ^V`.
  2. Apply repetition penalty (multiplicative on already-seen tokens).
  3. Divide `z / T`.
  4. Apply top-k: keep top `k` logits, set rest to `-inf`.
  5. Apply top-p: compute softmax; cumulative; truncate where cumulative > p.
  6. Compute final softmax over remaining tokens.
  7. Sample (multinomial) or take argmax (greedy).
  8. Append the sampled token; update KV cache.
- Order of operations matters: temperature before truncation; truncation before final softmax.

**Common follow-ups.**
- "What's the difference between top-k=1 and greedy?" → Equivalent (argmax).
- "Why apply repetition penalty before temperature?" → Convention; results are similar either way.

**Common mistakes.**
- Applying temperature *after* truncation (changes the truncation set).

**References.**
- [Holtzman et al. — "The Curious Case of Neural Text Degeneration"](https://arxiv.org/abs/1904.09751) — sampling strategies.

---

### Q: How does cross-entropy loss behave during LLM pretraining? What's a typical curve?

**Category:** concept
**Difficulty:** intro
**Tags:** [pretraining, loss-curve, scaling-laws]

**Short answer.** Loss starts high (around `log(V) ≈ ln(50000) ≈ 10.8` nats per token for a random model on a 50k-vocab tokenizer), drops rapidly in the first few % of training, then slows to a power-law decay. Typical end-of-training loss for a frontier LLM: 1.8–2.2 nats per token. Loss curve on log-log axes is approximately linear, matching the scaling laws.

**Expansion / why this is the answer.**
- Random baseline loss: `log V` where `V` is vocab size.
- Typical Llama-style 7B model loss at end of training: ~2.0 nats/token.
- Perplexity = `exp(loss)`; 2.0 nats/token ≈ perplexity 7.4 per token.
- Scaling laws (Kaplan, Chinchilla): loss vs. compute is approximately a power law.
- **Loss spikes**: see T3 — fp16 overflow, router collapse, bad batch.
- **Bumps**: occasionally observed when reaching new capabilities ("learning curves with phase transitions").

**Common follow-ups.**
- "Why is the curve approximately a power law in log-log?" → Empirical; matches the scaling-law fits.
- "What's a reasonable LR schedule?" → Linear warmup (few thousand steps) + cosine decay to ~10% of peak.

**Common mistakes.**
- Comparing loss across different tokenizers without normalizing for vocab size.

**References.**
- [Kaplan et al. — "Scaling Laws"](https://arxiv.org/abs/2001.08361).

---

### Q: What's the role of the unembedding (LM head) matrix?

**Category:** concept
**Difficulty:** mid
**Tags:** [unembedding, lm-head, weight-tying]

**Short answer.** The unembedding `W ∈ ℝ^{d × V}` projects the model's final hidden state into vocabulary logits: `logits = h · W`. With weight tying (input/output sharing), `W = W_emb^T` — the same matrix serves both as token-embed and as unembed. Without tying, separate parameter.

**Expansion / why this is the answer.**
- **Forward**: `h_final ∈ ℝ^d` → `logits = h_final W ∈ ℝ^V` → softmax → token probabilities.
- **With weight tying**: the same `W ∈ ℝ^{V × d}` is used for both embedding lookup and the linear projection. The token at position `i` shares its embedding-row with its unembedding-column.
- **Without tying**: a separate parameter; doubles vocabulary parameters. Empirically, weight tying gives slightly better perplexity at lower parameter count.
- **All modern open LLMs** use weight tying.

**Common follow-ups.**
- "How does this interact with the final LayerNorm?" → Pre-norm transformers apply a final RMSNorm before the LM head; ensures the unembedding sees normalized features.

**Common mistakes.**
- Confusing the embedding matrix with the unembedding matrix as separate entities when tied.

**References.**
- [Press & Wolf — "Using the Output Embedding to Improve LMs"](https://arxiv.org/abs/1608.05859).

---

### Q: What is mixture-of-depths (MoD)?

**Category:** concept
**Difficulty:** senior
**Tags:** [mod, conditional-compute, raposo-2024]

**Short answer.** Mixture-of-Depths (Raposo et al. 2024, DeepMind): instead of every token going through every layer, a routing network at each layer decides whether each token is "active" (processed) or "skipped" (passed through identity). Saves compute when many tokens don't need deep processing. A complementary axis to MoE (which routes between FFN experts within a layer).

**Expansion / why this is the answer.**
- The mechanism:
  - At each layer, gating network predicts a score per token.
  - Top-k tokens get processed by the layer's full computation.
  - The rest pass through unchanged (identity).
- **MoE vs MoD**:
  - MoE: which *expert* processes each token within a layer.
  - MoD: whether the *layer* processes each token at all.
  - Orthogonal and can be combined.
- **Benefit**: per-token compute can be much less than per-layer × n_layers.
- **Result**: MoD models report matching dense quality at 30–50% less compute.
- **Caveats**:
  - Routing adds complexity.
  - Top-k makes the operation non-differentiable in the strict sense; trained with auxiliary losses similar to MoE.

**Common follow-ups.**
- "Combining MoD with MoE?" → Yes; the Raposo et al. paper sketches it. Active research direction.
- "Why don't all transformers use this?" → Engineering complexity; MoE already gets most of the parameter-efficiency wins.

**Common mistakes.**
- Confusing MoD with depth-wise pruning (different concept).

**References.**
- [Raposo et al. — "Mixture-of-Depths"](https://arxiv.org/abs/2404.02258).

---

### Q: What's the difference between absolute, relative, and ALiBi position encoding in terms of how they extrapolate?

**Category:** concept
**Difficulty:** senior
**Tags:** [positional-encoding, extrapolation, alibi, rope]

**Short answer.** **Absolute** (sinusoidal, learned): each position is a distinct vector; extrapolation past training length is poor or impossible. **Relative** (Shaw / T5 bias): encodes pairwise differences; extrapolates only if the relative bias function is well-defined for unseen differences. **ALiBi**: linear penalty proportional to position difference; extrapolates "for free" — no new parameters needed for longer contexts. **RoPE**: rotation-based; extrapolates poorly out-of-the-box but extends well with PI/YaRN/NTK adjustments.

**Expansion / why this is the answer.**
- **Sinusoidal**: claimed extrapolation in Vaswani 2017 but empirically poor — model perplexity degrades quickly past training length.
- **Learned absolute** (BERT): zero extrapolation — positions past training are completely unseen.
- **Relative T5 bias**: extrapolates if you've trained the bias matrix to cover the new differences; bounded by training-time max difference.
- **ALiBi**: `−m · |i − j|` added to attention scores; the linear function generalizes to any difference. Best out-of-the-box extrapolation; the original paper showed model trained on 1k extrapolates to 2k+.
- **RoPE**: rotates each Q, K vector by an angle proportional to position; the *dot product* depends only on the difference. But the rotation angles for unseen positions land outside the model's trained range; needs PI/YaRN to fix.
- **Modern preference**: RoPE + YaRN/PI dominates frontier models despite the extrapolation friction; ALiBi is used in some niches (MPT, BLOOM).

**Common follow-ups.**
- "Why did RoPE win over ALiBi?" → Empirically, RoPE has better in-training quality (perplexity), and the extension methods (PI, YaRN) preserve quality at extended lengths.
- "ALiBi's `m` per head?" → Pre-determined geometric sequence per head; no learned parameters.

**Common mistakes.**
- Assuming RoPE extrapolates naturally — it doesn't without help.

**References.**
- [Press et al. — "ALiBi"](https://arxiv.org/abs/2108.12409).
- [Su et al. — "RoPE"](https://arxiv.org/abs/2104.09864).
- [Peng et al. — "YaRN"](https://arxiv.org/abs/2309.00071).

---

### Q: How does an LLM handle whitespace and special tokens?

**Category:** concept
**Difficulty:** intro
**Tags:** [tokenization, whitespace, special-tokens]

**Short answer.** Whitespace is a *first-class* part of token IDs for most LLM tokenizers (BPE byte-level). `"hello"` and `" hello"` are different tokens. Special tokens (`<|endoftext|>`, `<|im_start|>`, `<system>`) are dedicated vocabulary entries with reserved IDs the user can't accidentally produce. Modern chat templates (Anthropic, OpenAI, Llama 3) heavily rely on special tokens to delineate turns.

**Expansion / why this is the answer.**
- **Whitespace tokens**: BPE merges include leading-space-tokens. `" the"` is a different token than `"the"`. Critical for round-tripping text.
- **Special tokens**:
  - `<bos>`, `<eos>`: beginning/end of sequence.
  - `<|im_start|>`, `<|im_end|>`: ChatML format.
  - `<|user|>`, `<|assistant|>`, `<|system|>`: role markers.
  - Tool-call boundary tokens.
- **Chat templates**: each model has its own; LLaMA 3 has a different template than Mistral. The library `transformers` provides `apply_chat_template()`.
- **Why special tokens matter**:
  - Provide a parseable boundary that doesn't conflict with user input.
  - Train the model to respect them (e.g. never emit a system-role token in the middle of an assistant turn).
- **Failure mode**: user includes a special token in their input (intentional or otherwise); the model interprets it as a structural marker. Modern APIs strip these.

**Common follow-ups.**
- "Why does spacing in prompts matter?" → Token boundaries differ; `"Q: 5+3 = "` vs. `"Q: 5 + 3 = "` tokenize differently and yield different generations.
- "What's a chat template?" → A structured format wrapping (role, content) into the model's expected token sequence.

**Common mistakes.**
- Stripping whitespace before tokenization — usually breaks the expected format.

**References.**
- [HuggingFace Chat Templates](https://huggingface.co/docs/transformers/main/en/chat_templating).
- [OpenAI tiktoken](https://github.com/openai/tiktoken).

---

### Q: What does "next-token prediction" actually optimize?

**Category:** concept
**Difficulty:** intro
**Tags:** [pretraining, autoregressive, objective]

**Short answer.** Minimize the negative log-likelihood `−Σ_t log p(x_t | x_<t)` over a corpus. Equivalently, minimize cross-entropy between the model's predicted distribution and the empirical token distribution. The objective is dense (one signal per token), self-supervised (no labels needed), and produces an *autoregressive* model — at inference, you sample tokens one at a time conditioned on the history.

**Expansion / why this is the answer.**
- The autoregressive factorization: `p(x_1, ..., x_n) = Π p(x_t | x_<t)`.
- Loss: `−Σ_t log p(x_t | x_<t)` summed over the training corpus.
- **Why next-token works as a pretraining objective**:
  - Dense: every position is a training example.
  - Self-supervised: the labels are the inputs themselves (just shifted by one).
  - Universal: predicting the next token captures syntax, semantics, world knowledge, reasoning patterns.
- **The "next-token prediction is all you need" insight**: Brown et al. 2020 demonstrated that scale + this objective gives in-context learning, zero-shot capability, etc.
- **Limits**:
  - Pure next-token doesn't directly optimize for instruction-following (hence SFT).
  - Doesn't optimize for helpfulness or safety (hence RLHF / DPO).

**Common follow-ups.**
- "Why is this called 'self-supervised'?" → No human labels — the labels are derived from the data itself.
- "What's perplexity?" → `exp(cross-entropy loss)`; geometric mean of `1/p(token)` over the data.

**Common mistakes.**
- Treating next-token prediction as just imitation — it's a deep enough objective that capabilities emerge.

**References.**
- [Brown et al. — GPT-3](https://arxiv.org/abs/2005.14165).
- [Bengio et al. — "A Neural Probabilistic Language Model"](https://www.jmlr.org/papers/v3/bengio03a.html) — early autoregressive LM.

---

### Q: How does a transformer encode order if attention is permutation-equivariant?

**Category:** concept
**Difficulty:** mid
**Tags:** [permutation-equivariance, positional-encoding]

**Short answer.** Self-attention with no positional encoding is permutation-equivariant: shuffling the input tokens produces the same shuffled output. To break this symmetry — so the model "knows" order — you add positional information: sinusoidal/learned absolute embeddings (added to token embeddings), relative bias (added to attention scores), or RoPE (rotates Q, K by position-dependent angles). Without one of these mechanisms, the model treats input as a bag of tokens.

**Expansion / why this is the answer.**
- Permutation equivariance: `Attn(σ(X)) = σ(Attn(X))` for any permutation `σ` of token positions.
- The model has no way to distinguish "the dog bit the man" from "the man bit the dog" without positional info.
- **All transformer architectures add positional encoding** in some form:
  - Sinusoidal (original).
  - Learned absolute (BERT).
  - Relative bias (T5, Shaw et al.).
  - RoPE (LLaMA, Mistral, Qwen).
  - ALiBi (MPT, BLOOM).

**Common follow-ups.**
- "What about LSTMs / RNNs?" → They have inherent order via the recurrence.
- "Bag-of-words baseline?" → If you set positional encoding to zero, the transformer is approximately a bag-of-words model.

**Common mistakes.**
- Forgetting that attention itself is permutation-invariant — the embedding lookup is the only place "the" enters; without position, every "the" in the input is identical.

**References.**
- [Vaswani et al. — "Attention Is All You Need"](https://arxiv.org/abs/1706.03762).

---

### Q: What is "embedding shrinkage" / Matryoshka representation learning?

**Category:** concept
**Difficulty:** mid
**Tags:** [matryoshka, embeddings, dimensionality]

**Short answer.** Matryoshka Representation Learning (Kusupati et al. 2022): train an embedding model so that the first `k` dimensions are themselves usable embeddings, with quality degrading gracefully as `k` shrinks. The result: one embedding model produces nested embeddings at multiple dimensionalities — you pick `k` based on your storage/compute budget. Used in OpenAI text-embedding-3 (which supports "shortening" embeddings).

**Expansion / why this is the answer.**
- The training trick: aggregate the loss across multiple prefix lengths of the embedding. The first 128 dims must be useful; the first 256 even more useful; the full 1536 best.
- **Why it matters**:
  - Storage: smaller embeddings = less ANN-index memory.
  - Latency: shorter dot products in retrieval.
  - Single model serves multiple deployment configurations.
- **At inference**: just truncate the embedding to the desired dimensionality.
- **Trade-off**: minor quality drop at smaller `k`; ~95% quality at 50% dims is typical.

**Common follow-ups.**
- "Why not just train multiple embedding models at different sizes?" → Cost; one model is easier to maintain.
- "Connection to PCA?" → Different mechanism but similar shape: progressively-truncated dimensions, with the first few carrying most information.

**Common mistakes.**
- Truncating non-Matryoshka embeddings and expecting graceful degradation — they're not trained for it; quality drops sharply.

**References.**
- [Kusupati et al. — "Matryoshka Representation Learning"](https://arxiv.org/abs/2205.13147).
- [OpenAI text-embedding-3 announcement](https://openai.com/index/new-embedding-models-and-api-updates/).

---

### Q: How does temperature scaling at inference relate to RLHF's KL penalty?

**Category:** concept
**Difficulty:** senior
**Tags:** [temperature, kl, rlhf, sampling]

**Short answer.** RLHF's KL penalty `β · KL(π || π_ref)` constrains how far the trained policy can drift from the reference. Inference-time temperature `T` controls how peaked the sampled distribution is. They affect different things — the KL penalty shapes *what the model knows*; temperature shapes *how it samples*. A small `β` lets the policy diverge sharply; a small `T` makes inference more deterministic regardless of `β`. Both are levers for the "creativity vs. quality" tradeoff but at different layers.

**Expansion / why this is the answer.**
- **KL penalty**: training-time. `R(s) = r(s) − β · KL(π(·|x) || π_ref(·|x))`. Drives policy toward the reference where reward signal is weak.
- **Inference temperature**: `p_i = exp(z_i/T) / Σ exp(z_j/T)`. Shapes the sampling distribution at decode time.
- **Combined effect**:
  - Low `β` + high T: model trained to be opinionated, samples diversely.
  - High `β` + low T: model close to reference, samples greedily.
- **The "DPO β"** in the DPO loss is analogous to the KL penalty β; controls how aggressively the policy moves from reference.

**Common follow-ups.**
- "Can you use temperature 0 with high-β-trained policy?" → Yes; the deterministic mode is the policy's argmax.

**Common mistakes.**
- Conflating training-time and inference-time temperature scaling.

**References.**
- [Ouyang et al. — "InstructGPT"](https://arxiv.org/abs/2203.02155) — KL penalty.
- [Rafailov et al. — "DPO"](https://arxiv.org/abs/2305.18290) — DPO β.

---

### Q: What is the "compute-equivalent" comparison for MoE vs. dense models?

**Category:** concept
**Difficulty:** senior
**Tags:** [moe, compute-equivalent, scaling]

**Short answer.** "Compute-equivalent" MoE comparison fixes either training compute or inference compute and compares quality. Mixtral 8x7B has ~13B active params at inference (≈ compute-equivalent to 13B dense); but at training time it's closer to a 47B-total compute footprint. The right "compute-equivalent dense model" depends on whether you're optimizing for training cost (compare to ~25B-class dense) or inference cost (compare to ~13B-class dense). Frontier-lab papers carefully delineate which axis.

**Expansion / why this is the answer.**
- **MoE economic argument**:
  - Total params: large (capacity).
  - Active params: small (per-token inference compute).
  - Decouples capacity and compute.
- **Comparison challenge**: should we compare a 47B-total/13B-active MoE to a 13B dense or 47B dense?
- **Answer**: depends on what cost you're measuring:
  - **Inference $/Mtok**: compare to dense with similar *active* params.
  - **Training cost**: compare to dense with similar *training* FLOPs (somewhere between).
- **Empirical**: Mixtral 8x7B beats Llama 2 70B on most benchmarks — substantial win on the active-param-equivalent axis.
- **What papers should report**: both axes (Mixtral does this).

**Common follow-ups.**
- "Why is MoE training compute closer to total params than active?" → Routing + all-to-all + load balancing add overhead; training requires the full model to be in memory.

**Common mistakes.**
- Reporting only the active-params number; misleads on training cost.

**References.**
- [Jiang et al. — "Mixtral"](https://arxiv.org/abs/2401.04088).
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437).

---

### Q: What is the "scaling brain" / scaling-laws-second-order pattern from the BC / scaling-laws literature?

**Category:** concept
**Difficulty:** senior
**Tags:** [scaling-laws, biological-comparison, deep-learning]

**Short answer.** A speculative but recurring observation in scaling-laws literature: LLM loss vs. compute follows a power-law (Kaplan, Chinchilla); the exponents and crossover points are surprisingly similar across model classes (text, vision, multimodal) and even across animal-brain scales of dataset and parameter sizes. The pattern motivates "scaling is mostly about training efficiency" rather than architecture revolutions.

**Expansion / why this is the answer.**
- Multiple scaling-laws papers (Kaplan, Henighan, Hoffmann, Hoffmann revised, OpenAI's "scaling laws for autoregressive generative modeling") show similar shapes.
- **Implication**: architecture choices (within reasonable variants) matter less than picking the right data/compute mix.
- **Caveats**:
  - Not every architecture follows the same exponents (Mamba's scaling is different in some regimes).
  - The data quality dimension is harder to model.

**Common follow-ups.**
- "Are MoE scaling laws different?" → Yes, the active-params vs. total-params dimension complicates things. DeepSeek-V3 paper has updated MoE-specific scaling fits.

**Common mistakes.**
- Citing one scaling-laws paper as definitive — they revise each other regularly.

**References.**
- [Kaplan et al. — "Scaling Laws"](https://arxiv.org/abs/2001.08361).
- [Hoffmann et al. — "Chinchilla"](https://arxiv.org/abs/2203.15556).

---

### Q: How do you handle very small vocabulary (e.g. a domain-specific tokenizer)?

**Category:** concept
**Difficulty:** mid
**Tags:** [tokenizer, vocabulary, domain]

**Short answer.** Smaller vocab (a few thousand tokens) cuts the embedding table and LM head substantially but produces longer token sequences for the same text. Trade-off: vocab size × hidden_dim is a constant cost; sequence length is the per-input cost. For specialized domains (code, protein, music), domain-specific tokenizers (smaller vocab, more efficient for that domain) often win. For general purpose, large vocab (50k–200k) is standard.

**Expansion / why this is the answer.**
- **Vocab size tradeoff**:
  - Large vocab: shorter sequences per piece of text; bigger embedding table + LM head.
  - Small vocab: longer sequences; smaller embedding/LM head.
- **Memory math**:
  - Embedding params: `V · d` (50k × 4096 ≈ 200M for a 7B model).
  - LM head: same (often tied).
- **Specialized examples**:
  - Code tokenizers (Code Llama, DeepSeek-Coder): vocab tuned to code patterns; better compression for code.
  - Protein LMs: small alphabet (~25 amino acids); character-level tokenization.
- **Tiktoken (GPT-4)**: 100k vocab; well-tuned for English and code.

**Common follow-ups.**
- "Why don't we just use byte-level always?" → Sequence becomes very long; attention is quadratic. Subword balances expressiveness with sequence cost.
- "Can you swap a tokenizer post-pretraining?" → Painfully; the embedding table is conditioned on the original.

**Common mistakes.**
- Reporting "tokens" without specifying which tokenizer.

**References.**
- [Sennrich et al. — BPE](https://arxiv.org/abs/1508.07909).
- [DeepSeek-Coder paper](https://arxiv.org/abs/2401.14196).

---

### Q: What's the structural difference between FlashAttention v1, v2, v3?

**Category:** concept
**Difficulty:** senior
**Tags:** [flashattention, kernel, hopper]

**Short answer.** **v1** (Dao 2022): introduced the IO-aware tiled-softmax algorithm; reduces HBM traffic dramatically. **v2** (Dao 2023): improved parallelization — splits work across attention heads and sequence positions more efficiently; ~2× faster than v1. **v3** (Shah et al. 2024): Hopper-specific — uses async warp specialization, FP8 attention, ping-pong scheduling to leverage H100 features. Algorithm is the same as v1; v2 and v3 are GPU-architecture-optimized rewrites.

**Expansion / why this is the answer.**
- **v1**: tile Q, K, V; compute attention block by block; online softmax. Saves the `n × n` HBM materialization.
- **v2 improvements**:
  - Parallelize the inner loop across query positions (v1 was sequential).
  - Better work distribution across thread blocks.
  - ~2× wall-clock improvement on common workloads.
- **v3 improvements**:
  - Async warp specialization: dedicate warps to specific tasks (load, compute, store) in parallel.
  - FP8 attention: take advantage of Hopper's FP8 tensor cores; further speed gain at acceptable accuracy.
  - Targeted at H100 / H200 specifically.
- **Compatibility**:
  - v1 / v2: Ampere (A100) and beyond.
  - v3: Hopper (H100) and beyond.
- PyTorch 2.x's `scaled_dot_product_attention` dispatches to the best available backend.

**Common follow-ups.**
- "What's the v2 improvement story for backward?" → Better work distribution there too; recomputation cost is small.
- "Why does v3 specialize for Hopper?" → New hardware features (TMA, warp specialization, FP8) only available on H100+.

**Common mistakes.**
- Calling FlashAttention "approximate" in any version.

**References.**
- [Dao et al. — "FlashAttention" v1](https://arxiv.org/abs/2205.14135).
- [Dao — "FlashAttention-2"](https://arxiv.org/abs/2307.08691).
- [Shah et al. — "FlashAttention-3"](https://arxiv.org/abs/2407.08608).

---

### Q: What's "context-free" generation vs. "constrained" generation?

**Category:** concept
**Difficulty:** intro
**Tags:** [generation, constrained, structured-output]

**Short answer.** **Context-free / unconstrained**: the model emits any token in the vocabulary at each step. **Constrained**: a finite-state-machine or grammar masks invalid tokens at each step (e.g. JSON-only). Constrained generation is critical for production LLM apps that need machine-parseable output (function calls, JSON, regex-conforming strings). See T4 structured-generation question.

**Expansion / why this is the answer.**
- Unconstrained: standard generation.
- Constrained: at each step, set logits of grammar-invalid tokens to `-inf` before softmax.
- The grammar can be:
  - Regex (FSM compiled).
  - JSON schema (a more complex FSM).
  - Context-free grammar (more general).
- Modern implementations: Outlines, xGrammar, lm-format-enforcer, OpenAI structured outputs.
- Cost: per-token mask computation; with a precompiled FSM, ~1µs per step.

**Common follow-ups.**
- "Does constraining change the model's quality?" → Yes — the model may prefer different content that doesn't satisfy the grammar; quality degradation depends on the grammar's restrictiveness.
- "How is this related to JSON-mode?" → JSON-mode in OpenAI / Anthropic is constrained generation with a JSON-schema FSM.

**Common mistakes.**
- Calling this "prompting" — it's a kernel-level mechanism, not a prompt.

**References.**
- [Willard & Louf — "Outlines"](https://arxiv.org/abs/2307.09702).

---

### Q: How does the loss differ for instruction tuning vs. continued pretraining?

**Category:** concept
**Difficulty:** mid
**Tags:** [sft, continued-pretraining, loss]

**Short answer.** Both use next-token cross-entropy. The differences: **continued pretraining** computes loss on every token in the corpus (next-token, no masking). **Instruction tuning** typically masks the loss on the prompt tokens (computes loss only on the response). Conceptually: continued pretraining adds knowledge; SFT teaches a response format and behavior.

**Expansion / why this is the answer.**
- **Continued pretraining**:
  - Data: documents (no prompt/response structure).
  - Loss: per-token CE on every token.
  - Goal: acquire new knowledge / domain capabilities.
- **Instruction tuning (SFT)**:
  - Data: `(prompt, response)` pairs.
  - Loss: per-token CE on the response tokens only (mask out prompt).
  - Goal: teach the model to follow instructions in the desired format.
- **Why mask the prompt**: the prompt is *input*, not output; computing loss on it would teach the model to *predict the prompt*, which is degenerate.
- **Practical formats**: chat template — `<|user|>...<|assistant|>...`; loss starts at the `<|assistant|>` token.

**Common follow-ups.**
- "Do you ever NOT mask the prompt?" → Some research suggests including a small loss term on the prompt can help generalization, but it's not standard.
- "Can you combine CPT and SFT?" → Yes; common sequence: CPT for domain knowledge, then SFT for instruction-following.

**Common mistakes.**
- Training SFT without prompt masking; model becomes weirdly anchored on predicting the prompt.

**References.**
- [Ouyang et al. — "InstructGPT"](https://arxiv.org/abs/2203.02155).
- [Wei et al. — "FLAN"](https://arxiv.org/abs/2109.01652).

---
