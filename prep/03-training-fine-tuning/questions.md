# Training & Fine-Tuning — questions

Entries follow the [Q&A schema](../../CONTRIBUTING.md#the-qa-entry-schema).

---

### Q: Walk me through the standard LLM training pipeline — pretraining → SFT → preference optimization.

**Category:** concept
**Difficulty:** mid
**Tags:** [pretraining, sft, rlhf, dpo, pipeline]

**Short answer.** **Pretraining**: next-token prediction on a massive web-scale corpus (trillions of tokens), produces a "base model" that knows language and facts but isn't aligned to follow instructions. **Supervised fine-tuning (SFT)**: a smaller curated `(prompt, ideal completion)` dataset teaches instruction-following. **Preference optimization** (RLHF / DPO / GRPO): a preference dataset of `(prompt, chosen, rejected)` aligns the model toward human-preferred outputs and shapes refusal/safety behavior. The result is the chat-ready / instruct model.

**Expansion / why this is the answer.**
- **Pretraining**
  - Objective: minimize `−Σ log p(x_t | x_<t)` over a large corpus (Common Crawl + curated mixtures + code).
  - Tokens-to-parameters ratio (Chinchilla): ~20:1 is compute-optimal; modern open-weight models train *past* this for inference economy (LLaMA 3 at 15T tokens, 8–70B params).
  - Produces a base model.
- **Supervised fine-tuning (SFT)**
  - Dataset: `(prompt, ideal response)` pairs. Human-written or model-distilled.
  - Loss: standard next-token loss on the response tokens (often with prompt-loss-masked).
  - Result: model follows instructions in a chat format. Capability gains; not yet aligned for tone/refusal.
- **Preference optimization**
  - Dataset: `(prompt, chosen, rejected)` triples.
  - **RLHF (PPO)** (Christiano 2017, Ouyang 2022): train a reward model from preferences; then optimize policy with PPO using the reward as the objective, with a KL penalty against the SFT reference model to stay near the supervised distribution.
  - **DPO** (Rafailov et al. 2023): direct preference optimization — closed-form derivation lets you skip the explicit reward model and PPO, training directly on the preference dataset. Much simpler; widely adopted.
  - **GRPO** (Shao et al. 2024, DeepSeek-Math): group-relative policy optimization — RL with a normalized within-batch reward, no value head needed. Used for reasoning-heavy training (DeepSeek-R1).
  - **KTO** (Ethayarajh et al. 2024): Kahneman-Tversky-inspired preference loss using only desirable/undesirable labels (binary, not pairwise).
- **Optional later stages**: constitutional AI / RLAIF (Bai et al. 2022), continued pretraining for domain adaptation, distillation to a smaller model.

**Common follow-ups.**
- "Why not just SFT?" → SFT alone produces a model that imitates the training-set tone but doesn't learn from preference ranking. Preference optimization picks up nuance ("be helpful but refuse harmful") that's hard to put in pure SFT data.
- "Why was DPO so quickly adopted?" → No reward model, no PPO, no separate stages — same data, simpler training, comparable quality.
- "Is RLHF dead?" → Not entirely — RLHF still dominates frontier labs for fine-grained behavior shaping, but DPO and GRPO have eaten most of the open-weight space.

**Common mistakes.**
- Conflating pretraining and SFT.
- Saying "RLHF uses Q-learning" — it's policy gradient (PPO).
- Forgetting the KL-to-reference term — without it, the policy can collapse / diverge from sensible outputs.

**References.**
- [Ouyang et al. — "Training language models to follow instructions with human feedback" (InstructGPT)](https://arxiv.org/abs/2203.02155) — the canonical RLHF pipeline.
- [Rafailov et al. — "Direct Preference Optimization"](https://arxiv.org/abs/2305.18290) — DPO.
- [Shao et al. — "DeepSeekMath: Pushing the Limits of Mathematical Reasoning..." (introduces GRPO)](https://arxiv.org/abs/2402.03300) — GRPO.
- [Ethayarajh et al. — "KTO: Model Alignment as Prospect Theoretic Optimization"](https://arxiv.org/abs/2402.01306) — KTO.

---

### Q: Compare DPO, PPO-RLHF, KTO, IPO, and GRPO. When would you choose each?

**Category:** concept
**Difficulty:** senior
**Tags:** [rlhf, dpo, grpo, preference-optimization]

**Short answer.** **PPO-RLHF**: train a reward model from preferences, then RL with PPO + KL penalty. Strongest for fine-grained behavior shaping; complex and unstable. **DPO**: derive the optimal-policy objective analytically; train directly on `(prompt, chosen, rejected)`. Simple, stable, the modal open-weight choice. **IPO**: identity-preference-optimization, fixes a known DPO over-fitting failure when preferences are deterministic. **KTO**: binary "good/bad" labels rather than pairwise; useful when you have abundant approve/reject signals (production logs). **GRPO**: group-relative RL, no value head; the right choice when you have a verifier (programmatic reward, e.g. for math/code) and want RL without PPO's complexity.

**Expansion / why this is the answer.**
- **PPO-RLHF** (Christiano 2017; Ouyang 2022, InstructGPT)
  - Two-stage: reward model `r_φ(prompt, response)` from preferences, then policy optimization `max E[r_φ(s)] − β·KL(π || π_ref)`.
  - Pros: very fine-grained control; what frontier labs use.
  - Cons: complex (4 models in memory: policy, reference, reward, value); brittle.
- **DPO** (Rafailov et al. 2023)
  - Loss: `−log σ(β · (log π_θ(y_w|x)/π_ref(y_w|x) − log π_θ(y_l|x)/π_ref(y_l|x)))`.
  - One stage, no separate reward model, no RL.
  - Equivalent under the Bradley-Terry preference model.
  - Pros: simple, stable, much less compute.
  - Cons: can over-fit to confident preferences; can decrease likelihood of *both* chosen and rejected responses if not careful (the "DPO likelihood-decrease" pathology).
- **IPO** (Azar et al. 2023): identity preference optimization — modifies the DPO loss with a squared-error-like form, fixing the deterministic-preference over-fitting issue.
- **KTO** (Ethayarajh et al. 2024): only requires unary good/bad labels rather than pairs. Inspired by prospect theory's value function.
- **GRPO** (Shao et al. 2024)
  - Sample `G` outputs per prompt, score each by a (programmatic) reward, normalize within the group: `advantage_i = (r_i − mean(r)) / std(r)`.
  - Policy update is REINFORCE-style with the normalized advantage; no value head needed (vs. PPO).
  - Used in DeepSeek-Math, DeepSeek-R1 for verifiable rewards (math/code).
- **When to pick each**:
  - You have pairwise preferences, want quick wins: **DPO**.
  - You have a verifiable reward (math, code, structured task): **GRPO**.
  - You have logs of approve/reject (production thumbs-up/down): **KTO**.
  - You are a frontier lab with compute and want maximum control: **PPO-RLHF**.
  - You have deterministic preferences and DPO is over-fitting: **IPO**.

**Common follow-ups.**
- "Why is DPO sometimes worse than PPO?" → On safety-sensitive shaping and on out-of-distribution prompts; the implicit reward learned by DPO is less explicitly controllable.
- "What's RLOO?" → REINFORCE-leave-one-out — a baseline-corrected REINFORCE, simpler than GRPO; used in some labs as PPO-replacement.

**Common mistakes.**
- Calling DPO "RL." It's a supervised loss derived from an RL objective.
- Saying GRPO needs a reward model — no, the canonical use is with a programmatic verifier.
- Mixing up the KL penalty (KL to reference) with KL regularization in the loss.

**References.**
- [Rafailov et al. — "Direct Preference Optimization"](https://arxiv.org/abs/2305.18290) — DPO.
- [Azar et al. — "A General Theoretical Paradigm to Understand Learning from Human Preferences" (IPO)](https://arxiv.org/abs/2310.12036) — IPO.
- [Ethayarajh et al. — "KTO"](https://arxiv.org/abs/2402.01306) — KTO.
- [Shao et al. — "DeepSeekMath" (GRPO)](https://arxiv.org/abs/2402.03300) — GRPO.
- [Ouyang et al. — "InstructGPT"](https://arxiv.org/abs/2203.02155) — PPO-RLHF.

---

### Q: Explain LoRA. What does the math actually do?

**Category:** derivation
**Difficulty:** mid
**Tags:** [lora, peft, fine-tuning]

**Short answer.** LoRA (Low-Rank Adaptation) freezes the pretrained weights `W ∈ ℝ^{d×k}` and learns a low-rank update `ΔW = BA` where `B ∈ ℝ^{d×r}, A ∈ ℝ^{r×k}` and `r ≪ min(d,k)` (typically `r = 8…64`). The forward pass becomes `y = Wx + (α/r) BAx`, with `B` initialized to 0 so the adapter starts as identity. Trainable parameters drop from `d·k` to `r·(d+k)` — often 100×–10,000× less — with near-full-fine-tune quality on most tasks.

**Expansion / why this is the answer.**
- **Architecture detail** (Hu et al. 2021): apply LoRA to specific layers — most commonly the Q and V projections of attention, sometimes also K, output projection, and FFN. The choice of which layers matters; the paper found Q+V the cheapest place to put adapters.
- **Hyperparameters**:
  - `r` (rank): typical 8–64. Higher `r` = more expressive, more params.
  - `α` (scaling): `α/r` scales the adapter contribution. Common: `α = 2r`.
  - `dropout` on the adapter input.
- **Why low-rank works**: the empirical claim (Aghajanyan et al. 2020) is that fine-tuning has **low intrinsic dimensionality** — the effective update lies in a low-dim subspace. LoRA hard-codes this.
- **QLoRA** (Dettmers et al. 2023): load the base model in 4-bit NF4 quantization, attach LoRA adapters in bf16. Memory drops further; lets you fine-tune 65B on a single 48 GB GPU.
- **DoRA** (Liu et al. 2024): "Weight-Decomposed Low-Rank Adaptation" — decompose `W` into magnitude + direction, adapt direction with LoRA, magnitude separately. Modest gains over LoRA.
- **LoRA at inference**: you can either keep the adapter as a separate forward pass (small perf cost) or **merge** it into `W` (`W' = W + (α/r)BA`), removing any inference overhead.

**Common follow-ups.**
- "Why initialize `B` to zero and `A` randomly?" → So the adapter starts as identity (`ΔW = 0`), and the model behaves like the pretrained one at step 0; non-zero `A` provides the gradient direction.
- "When does LoRA fall short?" → Tasks needing very large knowledge updates (continued pretraining); domain shift; very long instruction-tuning at scale. Full fine-tune still wins by a small margin in some benchmarks.
- "Can you stack LoRA adapters?" → Yes — multi-LoRA serving (vLLM, Punica) lets you switch adapters per request without reloading the base model.

**Common mistakes.**
- Saying LoRA "fine-tunes a subset of layers" — no, it adds *new* low-rank parameters to existing layers.
- Forgetting the scaling `α/r`.
- Initializing both `A` and `B` to zero (model never breaks symmetry) or both random (huge initial perturbation).

**References.**
- [Hu et al. — "LoRA: Low-Rank Adaptation of Large Language Models"](https://arxiv.org/abs/2106.09685) — LoRA.
- [Dettmers et al. — "QLoRA: Efficient Finetuning of Quantized LLMs"](https://arxiv.org/abs/2305.14314) — QLoRA.
- [Aghajanyan, Zettlemoyer, Gupta — "Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning"](https://arxiv.org/abs/2012.13255) — intrinsic-dim argument.
- [Liu et al. — "DoRA"](https://arxiv.org/abs/2402.09353) — DoRA.

---

### Q: What is reward hacking, and how do you mitigate it?

**Category:** concept
**Difficulty:** senior
**Tags:** [rlhf, reward-hacking, alignment]

**Short answer.** Reward hacking is the policy finding ways to maximize the reward signal *as measured* without solving the task as intended — verbose-but-wrong answers if the reward model rewards length; sycophancy if it rewards agreement; outright deception if the reward model rewards plausible-sounding answers. Mitigations: regularize toward the reference policy (KL penalty), use an ensemble of reward models, periodically re-collect preferences from the policy's actual outputs, add explicit anti-pattern data to the reward model training set, and watch out for reward over-optimization (Gao et al. 2022).

**Expansion / why this is the answer.**
- The mechanism: the reward model `r_φ` is itself a learned model; it has its own errors and quirks. As the policy is trained against `r_φ`, it finds high-reward regions that *aren't* high-reward under true human preferences.
- **Examples**:
  - Length bias: the reward model prefers long answers, so the policy gets verbose.
  - Sycophancy (Sharma et al. 2023): the policy learns to agree with the user even when wrong.
  - Confident-tone hacking: confident answers are preferred regardless of correctness.
  - Reference-style hacking: the policy emits markdown formatting that the RM was trained to prefer.
- **Reward over-optimization** (Gao, Schulman, Hilton 2022): proxy reward keeps going up while the gold reward (true human preference) plateaus and then drops. Visible if you measure gold reward periodically.
- **Mitigations**:
  - **KL-to-reference penalty**: `r − β · KL(π || π_ref)`. Limits how far the policy can drift.
  - **Reward-model ensembles**: average over multiple RMs; the policy must please all of them.
  - **Periodic re-collection of preferences**: train the RM on the policy's *current* outputs, not just the original preferences.
  - **DPO/GRPO**: by avoiding an explicit RM, DPO sidesteps RM-overfitting somewhat (but not entirely — the implicit RM has the same vulnerabilities).
  - **Anti-pattern training data**: include explicit "bad" examples (verbose-but-wrong, sycophantic, etc.) in the preference dataset.
- This is a major research direction; not solved.

**Common follow-ups.**
- "What's Goodhart's Law in this context?" → "When a measure becomes a target, it ceases to be a good measure." Reward hacking is Goodhart in action.
- "How do you detect reward hacking in production?" → Eval on held-out gold prompts; track length and style metrics over training steps; watch for the gold-vs-proxy divergence.

**Common mistakes.**
- Treating reward hacking as an edge case rather than the modal failure mode of naive RLHF.
- Forgetting the KL penalty.
- Assuming DPO avoids it entirely.

**References.**
- [Gao, Schulman, Hilton — "Scaling Laws for Reward Model Overoptimization"](https://arxiv.org/abs/2210.10760) — overoptimization paper.
- [Sharma et al. — "Towards Understanding Sycophancy in Language Models"](https://arxiv.org/abs/2310.13548) — sycophancy.
- [Casper et al. — "Open Problems and Fundamental Limitations of RLHF"](https://arxiv.org/abs/2307.15217) — survey of pathologies.

---

### Q: Walk me through data parallel, tensor parallel, pipeline parallel, ZeRO, and FSDP. What does each parallelize?

**Category:** concept
**Difficulty:** senior
**Tags:** [distributed-training, zero, fsdp, tensor-parallel]

**Short answer.** **Data parallel (DP)**: each GPU holds a full model replica; each processes a different batch slice; gradients all-reduced. **Tensor parallel (TP)**: split each layer's weight matrices across GPUs (e.g. attention heads, FFN columns); each GPU does part of the matmul. **Pipeline parallel (PP)**: split layers across GPUs; mini-batches flow through as a pipeline. **ZeRO** (Rajbhandari et al. 2020): a DP variant that *also* shards optimizer state (Z1), gradients (Z2), and parameters (Z3) across GPUs, reducing memory at the cost of more communication. **FSDP** (PyTorch's Fully Sharded Data Parallel): the PyTorch implementation of ZeRO-3 ideas — fully shard parameters, gather on demand for the forward, re-shard, gather for backward.

**Expansion / why this is the answer.**
- **DP**: simplest. Memory per GPU = full model. Communication: gradient all-reduce per step. Scales poorly past a moderate number of GPUs.
- **TP** (Shoeybi et al. 2019, Megatron-LM): split `W` of shape `(d_in, d_out)` column-wise across GPUs; each GPU computes a column-slice of the output; gather/all-reduce as needed. Cost: high-bandwidth intra-node interconnect (NVLink). Works best within a single node.
- **PP** (Huang et al. 2018, GPipe): split layers `L_1..L_n` across stages; mini-batch flows stage-by-stage. **Bubble problem**: while waiting for downstream stages, upstream GPUs idle. **Interleaved 1F1B** (Megatron) reduces the bubble.
- **ZeRO** (DeepSpeed):
  - Z1: shard optimizer state. Memory: `~12-byte-per-param Adam state` becomes `12/N`.
  - Z2: also shard gradients.
  - Z3: also shard parameters; gather on-the-fly per layer.
- **FSDP**: ZeRO-3 done natively in PyTorch. Each rank holds a shard of every parameter; gathers in the forward; re-shards; gathers again for the backward. Communication-heavy but memory-light.
- **3D / 4D parallelism**: combine DP × TP × PP (and sometimes sequence parallel / context parallel for very long sequences). Used at frontier scale (Megatron-DeepSpeed, NeMo).
- **Sequence parallel** (Korthikanti et al. 2022): inside TP, split the LayerNorm / dropout / activations along the sequence dimension; reduces activation memory.
- **Context parallel** (Liu et al. 2023, ring attention): split the sequence across GPUs for the attention computation specifically; key for ultra-long-context training.

**Common follow-ups.**
- "When do you pick TP vs. PP?" → TP within a node (NVLink), PP across nodes (slower interconnects).
- "Why is FSDP popular?" → It gives ZeRO-3 memory savings in a clean PyTorch API; pairs with TP/PP for 3D.
- "What's gradient checkpointing?" → Save only a subset of activations; recompute the rest on backward. Trades compute for memory.

**Common mistakes.**
- Calling ZeRO "another parallelism" — it's a *data-parallel* variant with sharding.
- Forgetting the pipeline bubble cost.
- Mixing up Megatron-style TP with general "model parallel."

**References.**
- [Rajbhandari et al. — "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models"](https://arxiv.org/abs/1910.02054) — ZeRO.
- [Shoeybi et al. — "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism"](https://arxiv.org/abs/1909.08053) — tensor parallel.
- [Huang et al. — "GPipe"](https://arxiv.org/abs/1811.06965) — pipeline parallel.
- [Korthikanti et al. — "Reducing Activation Recomputation in Large Transformer Models"](https://arxiv.org/abs/2205.05198) — sequence parallel.
- [PyTorch — Fully Sharded Data Parallel docs](https://pytorch.org/docs/stable/fsdp.html) — FSDP.

---

### Q: bf16 vs. fp16 vs. fp32 for training — what's the difference and what's the modern default?

**Category:** concept
**Difficulty:** mid
**Tags:** [mixed-precision, bf16, training-stability]

**Short answer.** **fp32** (32-bit): standard precision; full range and precision; 4 bytes/param. **fp16** (16-bit IEEE half): 5-bit exponent, 10-bit mantissa; small range (max ~65k) causes overflow/underflow during training — requires loss scaling. **bf16** (bfloat16): 8-bit exponent, 7-bit mantissa; same dynamic range as fp32; less precision but range matters more for training stability — modern default. Mixed-precision training uses bf16 for compute + fp32 for the optimizer master copy and some sensitive ops (norms, softmax).

**Expansion / why this is the answer.**
- **Why range matters**: gradients during training can span many orders of magnitude. fp16's narrow range causes gradient underflow (rounds to zero, training stalls) or overflow (NaN). bf16's wider range mostly eliminates this.
- **Loss scaling** (Micikevicius et al. 2017, NVIDIA): for fp16, multiply the loss by a large constant before the backward pass so gradients land in fp16's representable range, then unscale before the optimizer update. Brittle, especially with dynamic scaling.
- **bf16** (Wang & Kanwar 2019, Google): designed for ML — same range as fp32. No loss scaling needed. Almost identical training curves to fp32 in practice.
- **Modern stack** (PyTorch AMP, NVIDIA APEX):
  - Forward/backward in bf16.
  - Optimizer master weights, gradient accumulators, LayerNorm in fp32.
  - Loss in fp32 to avoid summation precision loss.
- **FP8** (H100+): even smaller; used for inference and increasingly for training (DeepSeek-V3, Llama 3.x training). Two formats: E4M3 (more range) for activations/weights, E5M2 (more precision) for gradients. Requires careful scaling per tensor.
- **TF32** (NVIDIA): a hybrid format on Ampere+ — 10-bit mantissa, 8-bit exponent. Used internally by tensor cores when fp32 ops are issued. Faster than fp32, transparent.

**Common follow-ups.**
- "Why does LayerNorm stay in fp32?" → It sums squares, which can overflow/underflow in fp16/bf16; numerical stability matters.
- "Why is FP8 hard?" → 4-bit mantissa means you need per-tensor (or per-block) scaling to keep values in range; the bookkeeping is non-trivial.
- "Why doesn't fp16 cause range problems on inference?" → Inference activations are typically narrower than training gradients; loss scaling isn't needed because there's no backward pass.

**Common mistakes.**
- Calling bf16 "less accurate than fp16" — it's lower precision (mantissa) but wider range, and that's what training needs.
- Forgetting that the optimizer keeps fp32 master copies.

**References.**
- [Micikevicius et al. — "Mixed Precision Training"](https://arxiv.org/abs/1710.03740) — the canonical paper.
- [Kalamkar et al. — "A Study of BFLOAT16 for Deep Learning Training"](https://arxiv.org/abs/1905.12322) — bf16.
- [NVIDIA — Mixed precision training (Apex / AMP)](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html) — primary docs.

---

### Q: What is instruction tuning, and how does it differ from SFT?

**Category:** concept
**Difficulty:** intro
**Tags:** [instruction-tuning, sft, flan]

**Short answer.** "Instruction tuning" and "SFT" are usually interchangeable terms in 2026 — both mean supervised fine-tuning on `(instruction, ideal response)` pairs. Historically, "instruction tuning" implies a *broad* dataset across many tasks framed as natural-language instructions (FLAN, T0, Self-Instruct, Alpaca), explicitly to generalize to unseen instructions. SFT can refer to any supervised fine-tuning, including narrow domain adaptation.

**Expansion / why this is the answer.**
- **FLAN** (Wei et al. 2021, then FLAN-T5 / FLAN-PaLM): collect many existing supervised NLP datasets, format each as an instruction; fine-tune a base model. Result: strong zero-shot generalization to unseen instructions.
- **Self-Instruct** (Wang et al. 2022): bootstrap instruction-tuning data from a model's own generations. Alpaca and Vicuna popularized this approach on LLaMA.
- **The modern "SFT" stage** in an LLM pipeline:
  - Mix of human-written and model-generated instruction-response pairs.
  - Often hundreds of thousands of examples (vs. trillions of tokens for pretraining).
  - Format: chat templates (`<|system|>...<|user|>...<|assistant|>...`), often with **prompt-loss-masking** (compute the loss only on assistant tokens).
  - Loss: standard next-token cross-entropy.
- **Quality matters more than quantity**: LIMA (Zhou et al. 2023) showed that 1,000 carefully-curated examples can match much larger SFT sets — the "less is more for alignment" finding. Some replication of this finding is contested at scale but the curated-quality direction has been confirmed.

**Common follow-ups.**
- "What's prompt-loss-masking?" → Don't compute loss on the prompt tokens (the user message), only on the assistant response. Why: we don't want to penalize the model for failing to predict the user's tokens (which are inputs, not outputs).
- "Why does instruction tuning generalize at all?" → Empirically, training on many instruction-formatted tasks transfers to new task framings; the model learns "instruction-following" as a meta-skill.

**Common mistakes.**
- Treating SFT as somehow distinct from "instruction tuning" — in 2026 they're synonymous in most contexts.
- Forgetting to mask prompt loss; training on prompt tokens degrades quality.

**References.**
- [Wei et al. — "Finetuned Language Models Are Zero-Shot Learners" (FLAN)](https://arxiv.org/abs/2109.01652) — FLAN.
- [Wang et al. — "Self-Instruct"](https://arxiv.org/abs/2212.10560) — Self-Instruct.
- [Zhou et al. — "LIMA: Less Is More for Alignment"](https://arxiv.org/abs/2305.11206) — LIMA finding.

---

### Q: What is knowledge distillation? How is it used in LLM training?

**Category:** concept
**Difficulty:** mid
**Tags:** [distillation, model-compression]

**Short answer.** Knowledge distillation (Hinton, Vinyals, Dean 2015) trains a small "student" model to imitate a large "teacher" — typically by matching the teacher's soft output distribution (`softmax(logits/T)`) rather than just the hard label. In modern LLM training, distillation is widespread: producing fast smaller models from a frontier teacher (GPT-4 → smaller fine-tuned models; LLaMA family models distilled across sizes), and recently as part of a multi-stage training pipeline where SFT data is generated by a stronger teacher model.

**Expansion / why this is the answer.**
- **Soft-target distillation** (Hinton et al. 2015): student loss = α · CE(student, hard label) + (1−α) · KL(student logits || teacher logits / T) where `T` is a temperature.
- **Response distillation**: generate (prompt, teacher response) pairs and SFT the student on them. This is *the* dominant form in 2025 (Alpaca, Vicuna, WizardLM, OpenHermes were all distilled this way from GPT-3.5/4 outputs).
- **Feature / hidden-state distillation**: match intermediate hidden states (TinyBERT, DistilBERT). More effective when student and teacher share architecture/dim.
- **Attention distillation**: match attention maps between student and teacher.
- **MiniLM** (Wang et al. 2020): distill BERT by matching attention distributions — popular for small encoder models.
- **Pure LLM use cases**:
  - GPT-4 → fine-tuned LLaMA: build a cheap clone of a closed model's behavior. Quality cap is the teacher; legal/license caveats apply.
  - Frontier-lab internal: train a flagship model, then distill into a smaller production model.
  - Multi-teacher distillation: average predictions or take majority vote.
- **Caveats**:
  - Distillation cannot exceed the teacher (without other tricks like RL).
  - Hallucination patterns can transfer from teacher to student.
  - Licensing: many APIs prohibit using outputs to train competing models — check ToS.

**Common follow-ups.**
- "Why does soft-target distillation help vs. just training on hard labels?" → The teacher's probability over wrong classes encodes structural information ("this image is a 3, but kind of an 8 and not at all a 1") that hard labels don't.
- "What's the difference between distillation and self-training?" → Self-training: model generates pseudo-labels on unlabeled data, retrain on those. Distillation: a *different* (larger) model is the source. Self-training is a special case where teacher = student.

**Common mistakes.**
- Saying the student can "surpass the teacher" via distillation alone. It cannot, on the distillation signal; it can surpass on downstream tasks if the distilled student is then fine-tuned with other signals.
- Forgetting the temperature `T` and how it shapes the soft distribution.

**References.**
- [Hinton, Vinyals, Dean — "Distilling the Knowledge in a Neural Network"](https://arxiv.org/abs/1503.02531) — the canonical paper.
- [Sanh et al. — "DistilBERT, a distilled version of BERT"](https://arxiv.org/abs/1910.01108) — encoder distillation.
- [Wang et al. — "MiniLM"](https://arxiv.org/abs/2002.10957) — attention-distribution distillation.

---

### Q: What is gradient checkpointing? When do you use it?

**Category:** concept
**Difficulty:** mid
**Tags:** [gradient-checkpointing, activation-recomputation, memory]

**Short answer.** Gradient checkpointing (a.k.a. activation recomputation) reduces memory by saving only a subset of forward-pass activations, then recomputing the rest during the backward pass. Memory drops roughly from `O(L)` (layers) to `O(√L)` in the simplest scheme — at the cost of one extra forward pass per backward (roughly 1.3× total compute). Use it whenever you're memory-bound on training (the common case for large models).

**Expansion / why this is the answer.**
- The backward pass requires the activations at each layer to compute gradients. Naively storing all `L` layers' activations is `O(L · activation_size)`.
- **Checkpointing strategy** (Chen et al. 2016, "Training Deep Nets with Sublinear Memory Cost"): save activations at √L checkpoint layers; during backward, recompute activations within each segment from the nearest saved checkpoint.
- Memory: `O(√L)` activations saved.
- Compute: each segment is recomputed once → roughly 1× extra forward pass, total ~1.3× FLOPs.
- **Selective checkpointing** (Korthikanti et al. 2022): only recompute the cheap parts (LayerNorm, dropout, GELU), not the expensive matmuls. Almost no recompute cost; significant memory savings.
- **In PyTorch**: `torch.utils.checkpoint.checkpoint(fn, *args)` or `torch.utils.checkpoint.checkpoint_sequential`.
- **In transformer training** (HuggingFace, FSDP, DeepSpeed): typically applied per transformer block — checkpoint each block.

**Common follow-ups.**
- "Why doesn't gradient checkpointing increase memory if you store the checkpoint?" → The checkpoint is just the inputs to the segment; you discard the intermediate activations within the segment.
- "Is it on by default?" → No; you opt in. Default is "store everything."

**Common mistakes.**
- Confusing it with gradient *accumulation* (different concept — accumulate gradients across micro-batches to simulate a larger batch).
- Thinking it's free; it costs ~30% more compute.

**References.**
- [Chen et al. — "Training Deep Nets with Sublinear Memory Cost"](https://arxiv.org/abs/1604.06174) — original.
- [Korthikanti et al. — "Reducing Activation Recomputation in Large Transformer Models"](https://arxiv.org/abs/2205.05198) — selective.
- [PyTorch — torch.utils.checkpoint docs](https://pytorch.org/docs/stable/checkpoint.html) — API.

---

### Q: Why does an LLM "forget" prior knowledge after fine-tuning? How do you prevent it?

**Category:** concept
**Difficulty:** mid
**Tags:** [catastrophic-forgetting, continual-learning, fine-tuning]

**Short answer.** Fine-tuning on a narrow distribution shifts the model's parameters away from the broad capability they had after pretraining — catastrophic forgetting. Mitigations: (1) mix general / instruction data into the fine-tune set ("replay"); (2) use parameter-efficient methods (LoRA) that limit how far the underlying model drifts; (3) regularize toward the base model (KL or EWC-style penalty); (4) keep the fine-tune small and targeted; (5) use a mixture-of-experts or adapter system to add capability rather than overwrite.

**Expansion / why this is the answer.**
- The mechanism: gradient descent on a narrow distribution can move parameters that were good for general capability without those gradients being updated to preserve the prior fit.
- **Catastrophic forgetting** was identified for neural nets long ago (McCloskey & Cohen 1989); LLMs inherit it.
- **Mitigations**:
  - **Replay**: mix general-purpose data (pretraining-like samples, FLAN-style instruction data) into the fine-tune set. Most effective in practice.
  - **PEFT (LoRA)**: only learn a small low-rank update; harder to destroy broad capability.
  - **KL penalty**: regularize toward the base model's output distribution (the same KL term used in RLHF). Bounds how far the model moves.
  - **Elastic Weight Consolidation (EWC)** (Kirkpatrick et al. 2017): penalize updates to parameters that were important for the prior task. Theoretically clean, less common in LLM practice.
  - **Continued pretraining + SFT**: if you must add domain knowledge, continue pretraining on the domain *before* SFT — adds knowledge without overwriting instruction-following.
- **Diagnosis**: run the post-fine-tune model on a general benchmark suite (MMLU, GSM8K) and compare to the base. A meaningful drop = you're forgetting.
- **Modern context**: most production fine-tuning is LoRA + replay; full fine-tunes that lose broad capability are mostly a beginner mistake at this point.

**Common follow-ups.**
- "What's the LoRA-rank choice for limiting forgetting?" → Lower rank = less drift from base; common picks are r=8 or r=16 for narrow tasks.
- "Can you recover capability post-forgetting?" → Yes, by continued training that includes the lost distribution; but you can't fully recover specific behaviors without the original training data.

**Common mistakes.**
- Forgetting to evaluate on general benchmarks — only reporting in-domain wins masks catastrophic forgetting.
- Assuming LoRA prevents forgetting entirely (it reduces but doesn't eliminate it).

**References.**
- [Kirkpatrick et al. — "Overcoming catastrophic forgetting in neural networks" (EWC)](https://arxiv.org/abs/1612.00796) — EWC.
- [Goodfellow et al. — "An Empirical Investigation of Catastrophic Forgetting in Gradient-Based Neural Networks"](https://arxiv.org/abs/1312.6211) — empirical study.
- [Luo et al. — "An Empirical Study of Catastrophic Forgetting in Large Language Models During Continual Fine-tuning"](https://arxiv.org/abs/2308.08747) — LLM-specific.

---

### Q: What's "loss spike" during LLM pretraining, and what causes it?

**Category:** concept
**Difficulty:** senior
**Tags:** [training-stability, loss-spike, mixed-precision]

**Short answer.** A loss spike is a sudden large jump in training loss (and gradient norm) mid-run, typically due to numerical instability: a rare-but-large batch, a bad mix of activations and the attention softmax saturating, fp16 overflow, or a degenerate router state in MoE. Mitigations: gradient clipping, lower learning rate, bf16 instead of fp16, attention-logit soft-capping (Gemma 2), z-loss on router logits (ST-MoE), warmup, and skipping problematic batches.

**Expansion / why this is the answer.**
- Empirically observed by every team running large LLM pretraining; documented openly by Megatron-LM, PaLM, OPT, BLOOM, LLaMA writeups.
- **Causes**:
  - **fp16 overflow**: gradient or activation exceeds fp16's max (~65k). Hence the move to bf16 (wider range).
  - **Attention logit blowup**: a head's logits go very large, softmax saturates one-hot, gradient through softmax vanishes/explodes.
  - **Router collapse** (MoE): one expert dominates, others get no signal.
  - **Bad data batch**: a degenerate batch (e.g. all-same-token spam, corrupted tokens) produces extreme gradients.
  - **LR too high relative to current model state**.
- **Mitigations** in practice:
  - **Gradient clipping** (`||g||_2 ≤ c`, typical `c = 1.0`): the universal first line.
  - **Lower peak LR** + warmup.
  - **bf16 over fp16** for the vast majority of LLM training in 2026.
  - **Logit soft-capping** (Gemma 2): apply `tanh(x/c) · c` to attention logits to bound them.
  - **Z-loss on router logits** (ST-MoE; Zoph et al. 2022): regularize `log Σ exp(logits)`; prevents router blowup.
  - **Skip-batch on spike**: detect a spike and revert / skip / lower LR temporarily.
- **Resilience patterns**:
  - Checkpoint frequently; restart from before the spike.
  - Watch gradient norm and loss; alert on rapid increases.

**Common follow-ups.**
- "Why does post-norm spike more than pre-norm?" → Pre-norm puts LN before the sublayer; the residual path is unchanged. Post-norm normalizes after, amplifying any sublayer instability.
- "What's the SwiGLU loss-spike story?" → No specific story — SwiGLU is generally fine; the well-known instability is around attention softmax and router for MoE.

**Common mistakes.**
- Treating loss spikes as a bug-free training event — they typically signal real instability that should be investigated.
- Always restarting; sometimes a small LR decrease + continue is sufficient.

**References.**
- [Zoph et al. — "ST-MoE: Designing Stable and Transferable Sparse Expert Models"](https://arxiv.org/abs/2202.08906) — router z-loss, stability work.
- [Chowdhery et al. — "PaLM" technical report](https://arxiv.org/abs/2204.02311) — documents loss spike mitigations.
- [Gemma 2 technical report](https://arxiv.org/abs/2408.00118) — attention logit soft-capping.

---

### Q: What is constitutional AI / RLAIF, and how does it differ from RLHF?

**Category:** concept
**Difficulty:** mid
**Tags:** [rlaif, constitutional-ai, alignment]

**Short answer.** **Constitutional AI** (Bai et al. 2022, Anthropic): instead of having humans label every preference pair, train the model to critique and revise its own outputs against a written "constitution" of principles. **RLAIF** more broadly: replace human preference labels with AI-generated preference labels at some stage of training. Used to scale alignment beyond human-labeling bandwidth, with the tradeoff that the labeling AI's biases propagate.

**Expansion / why this is the answer.**
- **Constitutional AI**:
  - SFT stage: the model self-critiques its outputs against the constitution and revises them; train on the revisions.
  - Preference stage: the model also generates the preference labels (which of two outputs better follows the constitution). This forms the RLAIF preference dataset.
  - Then standard RL (PPO) or DPO on the AI-generated preferences.
- Why it works:
  - Humans are slow and expensive labelers.
  - AI labelers are consistent, cheap, and can be specialized.
  - The constitution makes preferences auditable.
- **RLAIF** (Lee et al. 2023, Google): generalizes the idea — use AI for preferences in any RLHF-like pipeline.
- **Tradeoffs**:
  - **Speed**: AI labels are 100×+ faster than human.
  - **Bias propagation**: the labeler's biases (length preference, sycophancy, etc.) are inherited.
  - **Less safety-sensitive control**: human labelers catch edge cases an LLM labeler misses (and vice versa).
  - **Iteration**: when you revise the constitution, you can re-label; humans can't be cheaply re-asked.
- **2024–2026 practice**: hybrid. Frontier labs use a mix of human + AI preferences; AI alone for many tasks, human-only for safety-critical.

**Common follow-ups.**
- "What's a 'constitution' in practice?" → A written set of principles ("be helpful, harmless, honest"; specific rules around certain topics). Sometimes published (Anthropic's Acceptable Use Policy is partially this).
- "How do you avoid the labeler being wrong?" → Use an ensemble of labelers; calibrate against human-labeled subsets; periodic spot-checks.

**Common mistakes.**
- Treating Constitutional AI as a single technique rather than a multi-stage pipeline.
- Confusing RLAIF with RLHF + AI-augmentation (they overlap but the term RLAIF specifically means AI preferences).

**References.**
- [Bai et al. — "Constitutional AI: Harmlessness from AI Feedback"](https://arxiv.org/abs/2212.08073) — the canonical paper.
- [Lee et al. — "RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback"](https://arxiv.org/abs/2309.00267) — generalization.

---

### Q: Walk me through the loss math for DPO. Why does it work without a separate reward model?

**Category:** derivation
**Difficulty:** senior
**Tags:** [dpo, derivation, preference-optimization]

**Short answer.** The optimal policy under RLHF (`max E[r(s)] − β·KL(π||π_ref)`) has a closed form: `π*(y|x) ∝ π_ref(y|x) · exp(r(x,y)/β)`. Rearranging, `r(x,y) = β · log(π*(y|x)/π_ref(y|x)) + const`. Under the Bradley-Terry preference model `Pr(y_w > y_l) = σ(r(x,y_w) − r(x,y_l))`, substitute the rearranged `r`: the policy's log-ratio *is* an implicit reward, and the preference loss becomes a supervised classification loss on the policy itself. No separate reward model required.

**Expansion / why this is the answer.**
- The KL-regularized RL objective: `max_π E_{y∼π(·|x)} [r(x,y)] − β · KL(π(·|x) || π_ref(·|x))`.
- Lagrangian / closed form: `π*(y|x) = (1/Z(x)) · π_ref(y|x) · exp(r(x,y)/β)` where `Z(x)` is the partition function.
- Solving for `r`: `r(x,y) = β · log(π*(y|x) / π_ref(y|x)) + β · log Z(x)`. The `log Z(x)` term is `x`-only, drops out of preference differences.
- **Bradley-Terry preference**: `Pr(y_w ≻ y_l | x) = σ(r(x,y_w) − r(x,y_l))`. Substitute the rearranged `r`: difference of two `β · log(π/π_ref)` terms; the constants cancel.
- **DPO loss**:
  `L_DPO = −E[(x,y_w,y_l) ∼ D] log σ(β · [log(π_θ(y_w|x)/π_ref(y_w|x)) − log(π_θ(y_l|x)/π_ref(y_l|x))])`
- Reading the loss: maximize the policy's log-probability ratio (vs. reference) on chosen responses; minimize on rejected.
- **Why it works**: every solution that DPO converges to is also the solution to an RLHF problem under the Bradley-Terry preference model. No reward model, no RL.
- **Known pathology**: the loss can be minimized by decreasing both `log π_θ(y_w|x)` and `log π_θ(y_l|x)`, as long as the *gap* increases. So the model can become "less confident overall" — common workaround: SFT loss added as an auxiliary term.

**Common follow-ups.**
- "What's the role of β?" → Inverse temperature for the KL regularization; smaller `β` = stay closer to `π_ref`; larger `β` = optimize harder. Typical: 0.1.
- "Why use SFT before DPO?" → Otherwise `π_ref` is a base model and the chat-formatted preferences are far from its distribution; DPO struggles.

**Common mistakes.**
- Saying DPO has "no implicit reward" — there is one; it's the policy's log-ratio.
- Forgetting the `π_ref` term — DPO is *KL-regularized*; the reference is critical.

**References.**
- [Rafailov et al. — "Direct Preference Optimization"](https://arxiv.org/abs/2305.18290) — the derivation in §3.
- [Azar et al. — "A General Theoretical Paradigm to Understand Learning from Human Preferences" (IPO)](https://arxiv.org/abs/2310.12036) — analyzes DPO's limit behavior.

---

### Q: What is curriculum learning, and does it help LLM pretraining?

**Category:** concept
**Difficulty:** mid
**Tags:** [curriculum, data-ordering, pretraining]

**Short answer.** Curriculum learning (Bengio et al. 2009) orders training data from easy to hard, mirroring how humans learn. For LLM pretraining, results are mixed: most large pretraining runs randomize data order and rely on the scale of data to wash out ordering effects. Where curriculum *does* help is post-training and reasoning fine-tunes (e.g. easier math first, harder math later), and in domain-specific continued pretraining.

**Expansion / why this is the answer.**
- The original argument (Bengio et al.): training on easy examples first lets the model build representations that generalize to harder ones; like a child learning addition before algebra.
- **In LLM pretraining**:
  - Most large runs randomize. Empirical evidence that ordering matters is weak when data is diverse and the run is long.
  - **Data-mixture curriculum** is more common: increase the share of high-quality / code / math data later in training. Used by Phi-3, Llama 3, others.
  - **DoReMi** (Xie et al. 2023): learn optimal *domain mixing weights* for pretraining; effectively an automated curriculum at the domain level.
- **In post-training / reasoning**:
  - Sort math problems by difficulty; train easy → hard. Common in reasoning-RL pipelines (DeepSeek-Math, others).
  - Reward-curriculum: start with easier reward signals, escalate.
- **Caveats**: catastrophic forgetting if later data is too distinct; loss of diversity if curriculum is overly tight.

**Common follow-ups.**
- "What's DoReMi?" → Domain Reweighting with Minimax — train a small reference model on uniform domain mix, then up-weight domains where the model lags. Used in Llama-3 mixture design.
- "Why does data ordering not matter much in pretraining?" → With trillions of tokens and many epochs (or one long epoch), local ordering effects average out; the model sees enough of everything.

**Common mistakes.**
- Saying "humans use a curriculum, so LLMs must benefit" — not the evidence in practice for large pretraining.
- Conflating curriculum with active learning.

**References.**
- [Bengio et al. — "Curriculum Learning"](https://dl.acm.org/doi/10.1145/1553374.1553380) — the original.
- [Xie et al. — "DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining"](https://arxiv.org/abs/2305.10429) — DoReMi.

---

### Q: What is data decontamination, and why does it matter for LLM training?

**Category:** concept
**Difficulty:** mid
**Tags:** [decontamination, leakage, benchmark, eval]

**Short answer.** Data decontamination is the process of removing test/eval data from the training corpus so the model isn't "graded on what it memorized." It matters because the modern web includes most public benchmarks; if you train on Common Crawl without filtering, you've effectively trained on MMLU, HellaSwag, GSM8K, etc. Standard decontamination: n-gram overlap detection (e.g. 13-gram exact match) between training docs and benchmark questions. Aggressive decontamination also uses paraphrase detection and semantic similarity.

**Expansion / why this is the answer.**
- **The contamination problem**:
  - Public benchmarks (MMLU, HumanEval, GSM8K) have their text on the public web.
  - Common Crawl includes that text.
  - A model trained on Common Crawl without filtering has seen the benchmarks.
  - Reported benchmark numbers are then a measure of *memorization*, not capability.
- **Standard detection**:
  - **N-gram overlap**: for each benchmark question, check if any n-gram of length k (typically 13) appears verbatim in a training doc. Remove the training doc if it does.
  - **Sources**: list every benchmark you'll evaluate on; collect their canonical text; decontaminate the training corpus against the union.
- **Reporting practice**: frontier labs document decontamination process in technical reports (GPT-4, Gemini, LLaMA 3 all do).
- **Failure modes**:
  - Reformulations of benchmark questions slip through n-gram filters.
  - Benchmarks created post-pretraining-cutoff don't have this risk (e.g. SWE-bench Verified, MMLU-Pro, GPQA).
  - "Test-set leakage" can happen even without intent if benchmark questions appear in derived datasets (textbook PDFs, Wikipedia summaries).
- **Modern eval design**: prefer benchmarks created *after* a model's training cutoff to avoid this entirely. Or use **dynamic / private holdouts**.

**Common follow-ups.**
- "How does Gemini decontaminate?" → 13-gram overlap is the public-reported method; details in the technical report.
- "Why does this matter for an interview-prep repo?" → If someone asks "are the published benchmarks reliable?" — the right answer is "consult the decontamination methodology in each model's tech report."

**Common mistakes.**
- Treating high benchmark scores as evidence of capability without checking the decontamination process.
- Forgetting that the eval-set creator's website is itself in Common Crawl.

**References.**
- [OpenAI — GPT-4 Technical Report (decontamination methodology)](https://arxiv.org/abs/2303.08774) — public reference.
- [Brown et al. — GPT-3 paper](https://arxiv.org/abs/2005.14165) — the original n-gram decontamination methodology.
- [Magar & Schwartz — "Data Contamination: From Memorization to Exploitation"](https://arxiv.org/abs/2203.08242) — the harm side.

---

### Q: Compare PEFT methods — LoRA, prefix tuning, prompt tuning, adapters, IA³.

**Category:** concept
**Difficulty:** senior
**Tags:** [peft, lora, prefix-tuning, adapters]

**Short answer.** **LoRA**: low-rank update `ΔW = BA` to weight matrices; merge-able at inference. **Prefix tuning**: prepend learned vectors to keys/values at every layer; no weight changes. **Prompt tuning**: prepend learned vectors to the input embeddings only. **Adapters**: insert small trainable bottleneck modules between layers. **IA³**: rescale activations by a learned vector. LoRA dominates in practice because it merges into the base weights (no inference overhead), trains stably, and is well-supported by tooling.

**Expansion / why this is the answer.**
- **LoRA** (Hu et al. 2021): `y = Wx + (α/r) BAx`. Parameters: `r·(d_in + d_out)`. Merge-able.
- **Prefix tuning** (Li & Liang 2021): learn `(P_K, P_V)` per layer; concatenate to existing K, V. Per-token effect; not merge-able. Costs context-window space.
- **Prompt tuning** (Lester et al. 2021): learn input-embedding prefix only. Less expressive than prefix tuning; cheapest by far. Works best at very large scales (10B+).
- **Adapters** (Houlsby et al. 2019): insert `Down(GELU(Up(x))) + x` modules into the transformer. Inference overhead per forward pass. Pre-LoRA standard.
- **IA³** (Liu et al. 2022): learn rescaling vectors `l_K, l_V, l_F` applied elementwise to K, V, and the FFN's first linear. Very few parameters; competitive at small scale.
- **DoRA** (Liu et al. 2024): decomposes the weight into magnitude + direction; adapt direction with LoRA, magnitude separately. Modest gains over LoRA.
- **When to pick each**:
  - **LoRA**: default; merge for production.
  - **Prompt tuning**: very-large-model regime with no infra for weight updates.
  - **IA³**: param-budget-constrained.
  - **Adapters**: legacy; or when you need modular insertable layers per task.
- **Combinations**: LoRA + IA³ is sometimes used; mostly LoRA alone is enough.

**Common follow-ups.**
- "When is LoRA *not* enough?" → Tasks needing large knowledge addition (continued pretraining); domain shift too big for low-rank correction.
- "What's the difference between prefix tuning and prompt tuning?" → Prefix tuning operates at every layer's K/V; prompt tuning only at the input embedding.

**Common mistakes.**
- Calling LoRA "an adapter" — it's a low-rank weight update, mathematically different.
- Forgetting prefix/prompt tuning steal context window at inference.

**References.**
- [Hu et al. — "LoRA"](https://arxiv.org/abs/2106.09685).
- [Li & Liang — "Prefix-Tuning"](https://arxiv.org/abs/2101.00190).
- [Lester et al. — "Prompt Tuning"](https://arxiv.org/abs/2104.08691).
- [Houlsby et al. — "Parameter-Efficient Transfer Learning for NLP" (adapters)](https://arxiv.org/abs/1902.00751).
- [Liu et al. — "IA³"](https://arxiv.org/abs/2205.05638).
- [Liu et al. — "DoRA"](https://arxiv.org/abs/2402.09353).

---

### Q: What is rejection sampling fine-tuning, and where does it fit in the post-training stack?

**Category:** concept
**Difficulty:** mid
**Tags:** [rejection-sampling, sft, post-training]

**Short answer.** Rejection sampling fine-tuning (RSFT): sample many completions per prompt from the current model, score with a reward model or verifier, keep only the high-scoring completions, then SFT on those. Cheaper and simpler than RLHF, often used as a *bootstrap* step before DPO/PPO. Llama 2/3 used rejection sampling extensively; STaR (Zelikman et al. 2022) was an early version for reasoning.

**Expansion / why this is the answer.**
- The recipe:
  1. For each prompt, sample `K` completions from the current model.
  2. Score each with a reward model (or programmatic verifier for math/code).
  3. Keep top-1 (or top-`k`); discard the rest.
  4. SFT the model on the kept completions.
  5. Optionally iterate.
- **Why it works**: implicit policy improvement. The model's own samples form a distribution; keeping high-scoring tails biases toward better outputs without explicit RL.
- **STaR** (Self-Taught Reasoner, Zelikman et al. 2022): rejection-sample reasoning chains; keep ones that produce the correct answer; fine-tune.
- **Llama 2 (Touvron et al. 2023)**: explicitly used rejection sampling alongside RLHF.
- **RAFT** (Dong et al. 2023): "Reward rAnked FineTuning" — the broader name.
- **Tradeoffs**:
  - Cheaper than RLHF (no PPO loop).
  - No KL penalty by default — model can drift; combine with reference policy.
  - Quality ceiling = reward model's ceiling.
- **In modern stacks**: often the bootstrap step. SFT → rejection sampling → DPO/PPO is a common pipeline.

**Common follow-ups.**
- "Why not just SFT on the original data?" → Original data may not be high-quality enough; RSFT lets the model improve on its own distribution.
- "How does this compare to GRPO?" → GRPO is RL; RSFT is supervised. GRPO has on-policy gradient updates; RSFT just keeps high-reward samples.

**Common mistakes.**
- Calling RSFT "reinforcement learning" — it's supervised on filtered samples.
- Forgetting the diversity loss (sampling only top-1 collapses variance).

**References.**
- [Zelikman et al. — "STaR"](https://arxiv.org/abs/2203.14465).
- [Touvron et al. — "Llama 2"](https://arxiv.org/abs/2307.09288) — RSFT in production.
- [Dong et al. — "RAFT"](https://arxiv.org/abs/2304.06767).

---

### Q: How do you fine-tune a model for tool use specifically?

**Category:** concept
**Difficulty:** senior
**Tags:** [tool-use, function-calling, fine-tuning]

**Short answer.** Curate a dataset of `(prompt, tool-call-trajectory)` examples covering: correct tool selection, correct argument format, parallel tool calls, error recovery, and "don't call a tool when not needed." Format calls with the model's expected schema (Anthropic / OpenAI's JSON tool-call format, or a custom `<tool_call>...` template). Train with SFT first; then preference optimize on `(correct_call, wrong_call)` pairs. Critical: include negative examples — when *not* to call a tool.

**Expansion / why this is the answer.**
- **Data shape**:
  - Positive: prompt + reasoning + correct tool call + observation + final answer.
  - Negative: tools the model shouldn't have called (over-eager tool use).
  - Recovery: tool errored, model adapts and tries a different tool.
  - Multi-turn: tool call → response → user follow-up → second tool call.
- **Format**: model-specific. The canonical 2024+ shape:
  - System prompt declares available tools (JSON schema).
  - Model emits structured `<tool_call>{"name": ..., "arguments": {...}}</tool_call>`.
  - Tool result returned as `<tool_result>...</tool_result>`.
  - Model continues from there.
- **Common failure modes to train against**:
  - Hallucinated tool name.
  - Wrong argument types.
  - Calling tools when the answer is already known.
  - Not calling tools when needed.
  - Parallel tool calls dropping into sequential dependence.
- **Eval**:
  - Tool-call accuracy (right tool, right args).
  - Task completion on agent benchmarks (TAU-bench, SWE-bench Verified).
- **Toolformer** (Schick et al. 2023): self-supervised tool-use training — the model proposes API calls in text; if calling improves prediction, keep that example.
- **Modern frontier models** (Claude, GPT-4o, Gemini) are tool-use-trained at the pretraining/post-training scale; OSS models often need an explicit tool-use fine-tune to be reliable.

**Common follow-ups.**
- "How do you handle a brand-new tool the model hasn't seen?" → Schema description in the prompt is enough for in-context learning; quality depends on the base model's instruction-following.
- "What's parallel tool-call training data look like?" → Examples where two independent tool calls are emitted in one turn; the model must learn this is a valid format.

**Common mistakes.**
- Training only on positive examples — model becomes over-eager.
- Forgetting the schema format the API expects.

**References.**
- [Schick et al. — "Toolformer"](https://arxiv.org/abs/2302.04761) — self-supervised tool-use.
- [Anthropic — Tool use docs](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview) — primary docs.
- [Qin et al. — "ToolLLM"](https://arxiv.org/abs/2307.16789) — large-scale tool-use SFT.

---

### Q: Continued pretraining vs. instruction tuning vs. RAG — when do you reach for each?

**Category:** concept
**Difficulty:** mid
**Tags:** [continued-pretraining, fine-tuning, rag, knowledge-injection]

**Short answer.** **Continued pretraining**: add domain *knowledge* by training next-token prediction on domain text. Expensive; needs lots of clean domain data. **Instruction tuning (SFT)**: change *behavior / format* — instruct the model to follow a new pattern. Cheap; tens of thousands of examples is plenty. **RAG**: inject knowledge at inference, no training needed. Order in practice: try RAG first (no training), instruction-tune if behavior needs changing, continued-pretrain only if domain language is so divergent that retrieval + prompt isn't enough.

**Expansion / why this is the answer.**
- **Continued pretraining**:
  - Same loss as pretraining (next-token); new corpus.
  - Useful for code-specialized (CodeLlama from Llama), medical (PMC-Llama), legal.
  - Risks catastrophic forgetting of general capability if mix isn't right.
  - Cost: substantial GPU-time.
- **Instruction tuning / SFT**:
  - Format the model already uses; teach new behaviors (tone, structure, refusals).
  - Doesn't add fundamental knowledge.
  - 1k–100k examples typical.
- **RAG**:
  - Add knowledge dynamically; auditable; updatable.
  - Doesn't change behavior — model still tone/format defaults.
  - Latency cost at inference.
- **Decision frame**:
  - "Wrong tone / format" → SFT.
  - "Knowledge changes frequently" → RAG.
  - "Domain language so divergent the model doesn't tokenize it well, or so specialized base model lacks vocabulary for it" → continued pretraining.
  - "All of the above" → continued pretraining + SFT + RAG (full stack, expensive).
- **Empirical findings** (Ovadia et al. 2023, Soudani et al. 2024): RAG injects knowledge more effectively than fine-tuning per unit of compute; fine-tuning shapes behavior better than RAG. They complement.

**Common follow-ups.**
- "What if you have a niche acronym the model never saw?" → RAG plus a small SFT with a glossary; continued pretraining only if the niche is large and ongoing.
- "When is continued pretraining catastrophic?" → If the new corpus is narrow; mix general data in (5–20%) to retain capability.

**Common mistakes.**
- Defaulting to fine-tuning when RAG would work.
- Continued pretraining on a single domain without replay (loses general capability).

**References.**
- [Ovadia et al. — "Fine-Tuning or Retrieval?"](https://arxiv.org/abs/2312.05934).
- [Roziere et al. — "Code Llama"](https://arxiv.org/abs/2308.12950) — continued pretraining for code.

---

### Q: What is process supervision (PRM), and how does it improve reasoning?

**Category:** concept
**Difficulty:** senior
**Tags:** [process-reward-model, prm, reasoning, lightman]

**Short answer.** A Process Reward Model (PRM) scores each *intermediate step* of a reasoning trace, not just the final answer. Compared to an Outcome Reward Model (ORM, which scores only the final answer), PRM gives finer credit assignment — "the second step is wrong; back up" — and is more sample-efficient at training and at search. Lightman et al. (2023) showed PRM outperforms ORM on MATH; used in modern reasoning-trained LLMs (o1-style, DeepSeek-R1 lineage).

**Expansion / why this is the answer.**
- **ORM**: binary or scalar score on `(prompt, full solution)`. Simple to label (was the final answer right?). Coarse credit assignment.
- **PRM**: scalar score per step in `(prompt, step_1, step_2, ..., step_k, answer)`. Each step labeled correct/incorrect.
- **Training labels**:
  - PRM800K (Lightman et al. 2023): 800k human-labeled step-level correctness annotations on MATH.
  - Cost-efficient alternatives: Math-Shepherd (Wang et al. 2024) — auto-label using rollouts (a step is "correct" if completing from it has high success rate).
- **Inference-time use**:
  - **Verify-and-pick (best-of-N)**: sample N reasoning traces; pick the one with highest PRM score.
  - **Search-guided generation**: use PRM to prune bad branches in tree search.
- **RL use**: PRM as a dense reward signal; gives step-level credit (vs. sparse final-answer reward in ORM-RL).
- **2024–2026 context**: reasoning-RL pipelines (DeepSeek-R1's predecessor papers, OpenAI's o1-style) lean on PRM-style supervision; GRPO often uses programmatic verifier as ORM with step-level rollouts.

**Common follow-ups.**
- "Why is PRM more sample-efficient than ORM?" → Each step is a training example; the trajectory provides ~K labels instead of 1.
- "Why is process supervision hard to scale?" → Step-level labels need human annotation; automated alternatives (Math-Shepherd) introduce noise.

**Common mistakes.**
- Conflating PRM with PPO's value function (different concept — PRM scores correctness of steps, value function predicts cumulative reward).

**References.**
- [Lightman et al. — "Let's Verify Step by Step"](https://arxiv.org/abs/2305.20050) — PRM canonical paper.
- [Wang et al. — "Math-Shepherd"](https://arxiv.org/abs/2312.08935) — automated PRM labeling.

---

### Q: Multi-task vs single-task fine-tuning — what does the empirical evidence show?

**Category:** concept
**Difficulty:** mid
**Tags:** [multi-task, fine-tuning, generalization]

**Short answer.** Multi-task instruction tuning (FLAN, T0, Self-Instruct) generalizes better to *unseen* tasks than single-task fine-tuning, especially at larger model scale. For a *specific* target task with abundant data, single-task fine-tuning gives the best in-domain performance but worse generalization. Modern post-training stacks favor diverse multi-task SFT specifically because the deployed model must handle many tasks.

**Expansion / why this is the answer.**
- **Single-task fine-tuning**:
  - Best on the target task.
  - Risk of catastrophic forgetting on other tasks.
  - Smaller, narrower training data.
- **Multi-task / instruction tuning**:
  - Trains on many task formats with a shared instruction template.
  - Generalizes to *held-out* task formulations (FLAN's headline result).
  - In-domain performance is sometimes slightly lower than dedicated single-task.
- **FLAN** (Wei et al. 2021): instruction-tune T5 on 60+ NLP tasks formatted as instructions; strong zero-shot to unseen tasks.
- **T0** (Sanh et al. 2021): same idea, different prompt templates per task.
- **Modern instruction tuning datasets**: OpenOrca, FLAN-v2, Tulu-3 mix.
- **Mixture ratios matter**: too much of one task drowns out others; balance dataset diversity.
- **When to single-task fine-tune**:
  - You only care about one task in production (rare for LLM applications).
  - You have ample task-specific data (tens of thousands+) and don't need generalization.
  - You're willing to keep a separate model for that task.

**Common follow-ups.**
- "How do FLAN-style models compare to plain GPT-3?" → FLAN-T5 substantially outperforms vanilla T5 on zero-shot tasks; same compute budget, different objective.
- "When does multi-task hurt?" → If one task has a very narrow distribution that drowns out diverse data, or if the tasks conflict (one task's "correct" is another's "wrong").

**Common mistakes.**
- Reporting in-domain wins from single-task fine-tuning while ignoring catastrophic forgetting.
- Calling FLAN "the FLAN model" — it's an instruction-tuning *recipe* applied to T5 / PaLM / others.

**References.**
- [Wei et al. — "FLAN"](https://arxiv.org/abs/2109.01652).
- [Sanh et al. — "T0"](https://arxiv.org/abs/2110.08207).
- [Chung et al. — "FLAN-T5"](https://arxiv.org/abs/2210.11416) — large-scale FLAN.

---

### Q: What's the difference between off-policy and on-policy RL in the LLM context?

**Category:** concept
**Difficulty:** senior
**Tags:** [off-policy, on-policy, rl, ppo, dpo]

**Short answer.** **On-policy**: the training data comes from the *current* policy. PPO (used in RLHF) is on-policy — must continually generate samples from the policy being optimized. **Off-policy**: training data comes from a *different* policy. DPO is off-policy — operates on a fixed preference dataset (`(prompt, chosen, rejected)`) collected from an arbitrary policy. Off-policy is much cheaper (no sampling loop); on-policy is more responsive to fine-grained shaping. The 2023–2026 trend was strongly toward off-policy (DPO, KTO) due to its simplicity.

**Expansion / why this is the answer.**
- **PPO-RLHF** (on-policy):
  - Each training step: sample completions from the *current* policy, score with reward model, compute policy gradient.
  - Reward model is fixed (trained earlier).
  - Compute: every step needs a generation pass + reward model pass + policy update.
- **DPO** (off-policy):
  - Fixed dataset of `(prompt, chosen, rejected)` preference pairs.
  - One supervised pass over the dataset.
  - No sampling loop, no reward model.
- **GRPO** (on-policy but simpler than PPO):
  - Sample G completions, score with a verifier, REINFORCE-style update with within-batch normalized advantage.
  - No value function (PPO needs one).
- **Importance sampling** can adapt off-policy data to on-policy — but introduces variance; rarely used at LLM scale.
- **Off-policy data sources**:
  - Human preference annotations on completions from any model.
  - Open-source preference datasets (UltraFeedback, HH-RLHF).
  - Logged production traffic.
- **Tradeoffs**:
  - **Off-policy**: cheap; can use data from other models; less responsive to current policy state.
  - **On-policy**: expensive; data always matches current state; more responsive to fine-grained shaping.
- **Empirical consequence**: PPO still leads on safety-critical behavior shaping (subtle, fine-grained). DPO leads on cost-and-quality for the bulk of post-training.

**Common follow-ups.**
- "Why is DPO called off-policy if the loss involves the current policy's log-probs?" → The *data* is off-policy; the *gradient* involves the current policy. The distinction is about where the samples came from.
- "What's RLOO?" → REINFORCE-leave-one-out; on-policy; cheaper than PPO because no value head. Used in some labs.

**Common mistakes.**
- Calling DPO "RL" — it's a supervised loss derived from an RL objective.
- Thinking off-policy is always worse; on the contrary, it's the modal modern choice.

**References.**
- [Rafailov et al. — "DPO"](https://arxiv.org/abs/2305.18290).
- [Ouyang et al. — "InstructGPT"](https://arxiv.org/abs/2203.02155) — on-policy PPO RLHF.
- [Shao et al. — "GRPO / DeepSeekMath"](https://arxiv.org/abs/2402.03300).

---
