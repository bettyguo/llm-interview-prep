# Research Discussion & Paper Deep-Dives — questions

Entries follow the [Q&A schema](../../CONTRIBUTING.md#the-qa-entry-schema).

---

### Q: Walk me through "Attention Is All You Need" as if you were giving a paper deep-dive.

**Category:** concept
**Difficulty:** mid
**Tags:** [transformer, vaswani-2017, paper-deep-dive]

**Short answer.** Contribution: introduced the **transformer**, a sequence model built entirely from attention (no recurrence, no convolution) — enabling massive parallelization and the modern LLM era. Method: encoder–decoder with multi-head scaled dot-product attention, position-wise FFN, sinusoidal positional encoding, residuals + post-norm. Evaluation: WMT 2014 EN-DE / EN-FR translation, beating prior SOTA at a fraction of training time. Limits: post-norm training instability at depth (fixed later by pre-norm); sinusoidal encoding doesn't extrapolate well (fixed by RoPE/ALiBi).

**Expansion / why this is the answer.**
- **The contribution**, in one sentence: replace RNN/CNN sequence encoders with self-attention; remove the sequential bottleneck.
- **Key components**:
  - Scaled dot-product attention with `/√d_k` scaling.
  - Multi-head: `h` parallel attention heads on a split feature dim; concat + project.
  - Position-wise FFN: 2-layer MLP with ReLU.
  - Sinusoidal positional encoding (added to embeddings).
  - Encoder + decoder with masked self-attention in the decoder + cross-attention from decoder to encoder.
  - Post-norm (`LN(x + Sublayer(x))`).
- **Evaluation**:
  - WMT'14 EN-DE BLEU 28.4 (base) / 26.4 (big variant); state-of-the-art at the time.
  - Trained on 8 GPUs for ~12 hours (base), 3.5 days (big) — vastly less than prior NMT systems.
- **Follow-up gains the paper enabled**:
  - BERT / GPT-2/3 family showed transformers scale.
  - Decoder-only LMs (GPT) became the dominant architecture.
  - Almost every 2018+ NLP advance traces here.
- **Limits / what's been revised since**:
  - Post-norm → pre-norm (Xiong et al. 2020).
  - Sinusoidal → RoPE / ALiBi.
  - MHA → GQA / MQA / MLA.
  - Standard FFN → SwiGLU.
  - Naive attention → FlashAttention.
- **30-second version**: "Vaswani et al. introduced the transformer in 2017; attention-only, fully parallel, beat RNN translation. Sets up everything since."

**Common follow-ups.**
- "Why `√d_k` scaling?" → See T2.
- "Why is post-norm worse?" → See T2 pre-norm question.

**Common mistakes.**
- Treating it as a "translation paper" rather than the foundational architecture paper.
- Citing modern details (RoPE, GQA) as in the original — they came later.

**References.**
- [Vaswani et al. — "Attention Is All You Need"](https://arxiv.org/abs/1706.03762) — primary.
- [Annotated Transformer (Harvard NLP)](https://nlp.seas.harvard.edu/2018/04/03/attention.html) — code-level walkthrough.

---

### Q: Tell me about a recent paper you read.

**Category:** behavioral
**Difficulty:** mid
**Tags:** [recent-paper, research-discussion]

**Short answer.** Don't summarize — *take a position*. Frame: (1) **the paper** (1 sentence: who, when, the claim); (2) **why it matters** (the bigger picture); (3) **how it works** (1–2 minutes of method); (4) **what's wrong with it / what surprised you** (your take — this is what the interviewer wants); (5) **what you'd do next**. Pick a paper you've read deeply enough to have a genuine take. Avoid only-the-most-famous-2017-paper (it's stale) and only-this-week's-arXiv (you haven't digested it).

**Expansion / why this is the answer.**
A good answer takes 3–5 minutes. Aim for one foundational paper, one recent paper (within 6 months), and one in your own line of work — rehearsed and ready. The bad answers are: a surface-level summary with no critique; a buzzword soup that doesn't demonstrate you actually understood the method; or "I read the GPT paper" (stale; everyone has read it). Practice all three papers out loud, recorded.

**Common follow-ups.**
- "What surprised you about this paper?" → Always have an honest answer; "nothing surprised me" reads as you didn't engage.
- "What experiment would you run next?" → Concrete and small-scope is better than ambitious. "I'd replicate their table 3 on a different model" is excellent.
- "What's the weakest part of the paper?" → Pick a real weakness, not a strawman.

**Common mistakes.**
- Picking a paper to seem impressive rather than one you understand.
- Summary with no take.
- Defending the paper from any critique — looks uncritical.

**Signal.**
This is a research-judgment probe. The interviewer wants to see: (a) you read papers actively, not passively; (b) you can critique without being unfair — a real take, not a Twitter dunk; (c) you connect the paper to broader questions in the field; (d) you'd know what experiment to run next.

---

### Q: Critique a paper in 60 seconds. Walk me through the structure.

**Category:** concept
**Difficulty:** mid
**Tags:** [paper-critique, structure]

**Short answer.** **Contribution** (1 sentence: what's new). **Method** (sketch + the trick that makes it work). **Evaluation** (datasets, metrics, baselines, headline result). **Threats to validity** (over-claim? cherry-picked baselines? eval-set contamination? confound? unrealistic assumption?). **Follow-ups** (what you'd do next — extend, replicate, falsify). Adapt depth to the audience and time budget; the threats-to-validity step is what separates strong from weak critiquers.

**Expansion / why this is the answer.**
- **Contribution**: be specific. "They show that X improves Y by Z%" beats "They study X."
- **Method**: don't recite the architecture; name the *key idea* (e.g. "they note that K and V can be projected from a small latent; this cuts the KV cache by 5×").
- **Evaluation**: ask whether the baselines are fair. Standard tricks: comparing against under-tuned baselines, omitting comparable simpler methods, picking benchmarks the method happens to win on.
- **Threats to validity** (the part interviewers love):
  - **Over-claim**: "we beat GPT-4 on benchmark X" with no significance test.
  - **Cherry-picked baselines**: ignored a known-stronger competitor.
  - **Contamination**: is the eval set in the training data?
  - **Confound**: more compute, more data, larger model — what's actually causing the gain?
  - **Realism**: does the experimental setting match real deployment?
- **Follow-ups**: what experiment proves the method's limits?

**Common follow-ups.**
- "What's a confound in an LLM paper?" → A factor co-varying with the studied factor (more data + new method → can't isolate).
- "How would you spot cherry-picked baselines?" → Read the appendix; check whether prior best on the benchmark was cited.

**Common mistakes.**
- Spending all 60 seconds on the method; no critique.
- Calling every paper "great" — interviewers want to see independent judgment.

**References.**
- [Sutton & Barto — *Reinforcement Learning: An Introduction* (preface)](http://incompleteideas.net/book/the-book-2nd.html) — example of clean exposition to model.
- [Bommasani et al. — "On the Opportunities and Risks of Foundation Models"](https://arxiv.org/abs/2108.07258) — example survey to model critique on.

---

### Q: How do you talk about your own work in 30 seconds, 5 minutes, and 30 minutes?

**Category:** behavioral
**Difficulty:** mid
**Tags:** [own-work, elevator-pitch, research-talk]

**Short answer.** **30 seconds**: problem + your one-sentence approach + headline result. **5 minutes**: + method intuition + key result + limitation. **30 minutes**: full talk — motivation, related work, method, experiments, ablations, future work. Practice all three. The 30-second version is the entry-point at every conversation; the 5-minute is the screening-call answer; the 30-minute is the formal talk / Ph.D. proposal version.

**Expansion / why this is the answer.**
The 30-second version is the highest-ROI thing to practice. Rehearse out loud, recorded, until it's natural. A failure mode at all three lengths is to start with the method instead of the problem — interviewers care first about *what you were trying to solve* and *why it matters*; the technical detail follows. For the 30-minute version, leave 5 minutes for Q&A; staffing yourself for the full 30 reads as inflexible.

**Common follow-ups.**
- "What's the one-sentence pitch?" → If you can't give it without throat-clearing, the 30-second isn't ready.
- "What's the weakest part of your own work?" → Have an honest answer; defensiveness reads poorly.
- "What did you learn from this project that you didn't expect?" → Strong probe; have a specific story.

**Common mistakes.**
- 30-second version that's just buzzwords ("We use deep learning for X").
- 5-minute version that's all method, no result.
- 30-minute version with no time for questions.

**Signal.**
The interviewer is grading: (a) can you communicate research clearly — big signal for senior+ roles; (b) do you know what's important — the 30-second version is hard *because* you must pick the one sentence that matters; (c) are you intellectually honest about limits; (d) can you connect your work to a broader research direction.

---

### Q: What benchmarks should you cite when claiming a model is "good"?

**Category:** concept
**Difficulty:** mid
**Tags:** [benchmarks, evaluation, citing]

**Short answer.** Don't cite a single benchmark. A defensible "this model is good" claim covers: **knowledge** (MMLU-Pro, GPQA), **reasoning** (MATH, GPQA-Diamond, ARC-AGI-2 if relevant), **coding** (SWE-bench Verified, LiveCodeBench), **multimodal** (if applicable: MMMU, MathVista), **instruction-following** (IFEval, MT-Bench), and **user preference** (Chatbot Arena Elo). Mention the methodology when citing (framework, prompt, decoding params).

**Expansion / why this is the answer.**
- A single benchmark = a single failure mode you could be optimizing against. A diversified scorecard is what frontier-lab tech reports do.
- **Required by 2026** for a credible claim:
  - **Reasoning**: MATH, GPQA-Diamond.
  - **Coding**: SWE-bench Verified, LiveCodeBench.
  - **Knowledge**: MMLU-Pro (MMLU saturated).
  - **Instruction-following**: IFEval.
  - **Preference**: Chatbot Arena Elo (Hard).
- **Add for specific capabilities**: agent (TAU-bench, GAIA), multimodal (MMMU, MMSTAR), long-context (RULER, Needle-in-a-Haystack).
- **What to skip**: GSM8K alone (saturated, contaminated), MMLU alone (same), HumanEval alone (saturated, contaminated by HumanEval+).
- **Reporting hygiene**: framework + version + prompt + decoding params + if any temperature.

**Common follow-ups.**
- "Why is MMLU-Pro a meaningful upgrade?" → Harder questions, 10 choices, post-MMLU-cutoff, more contamination-resistant.
- "Why include Arena Elo?" → Human user preference; resistant to benchmark gaming; reflects real use.

**Common mistakes.**
- Citing a single number ("we got 92 on MMLU") as proof of capability.
- Missing contamination caveats.

**References.**
- [Open LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) — multi-benchmark.
- [Chatbot Arena](https://lmarena.ai/) — Arena Elo.

---

### Q: How would you read an unfamiliar paper in 15 minutes during an interview?

**Category:** concept
**Difficulty:** mid
**Tags:** [paper-reading, interview-skill]

**Short answer.** Title + abstract + intro (3 min). Look at all figures and tables — including in the appendix (4 min). Read the method section's first paragraph and the key equation(s) (5 min). Read the conclusion and limitations (2 min). Form a take. Then answer the interviewer's questions, flagging "I'd want to check X more carefully" when you'd actually want to. Authenticity beats false fluency.

**Expansion / why this is the answer.**
- **What to look for in each phase**:
  - **Abstract + intro**: the claim and the contrast (what was wrong before).
  - **Figures/tables first**: the result tells you what the paper actually shows; the prose often over-claims.
  - **Method's key idea**: one or two sentences usually carry the technical novelty.
  - **Conclusion + limits**: what *don't* they handle?
- **What you say in the interview**:
  - "I haven't fully digested X but my read is Y."
  - "The headline result is Z; but I notice they didn't compare against W, which would be the natural baseline."
  - "I'd want to look at the appendix for Q before fully trusting this."
- **What not to do**:
  - Pretend to know more than you do.
  - Restate the abstract.
  - Take everything at face value (especially the strongest claims).

**Common follow-ups.**
- "What if you can't follow the math?" → Be honest. "I can follow the intuition but I'd need to read the appendix carefully on the proof of X." Honesty signals research maturity.

**Common mistakes.**
- Trying to memorize details; better to grasp the structure.
- Skipping the appendix; it often contains the inconvenient ablation.

**References.**
- [Keshav — "How to Read a Paper"](https://web.stanford.edu/class/ee384m/Handouts/HowtoReadPaper.pdf) — the canonical 3-pass method.

---

### Q: Walk me through a paper deep-dive on RAG (Lewis et al. 2020).

**Category:** concept
**Difficulty:** mid
**Tags:** [rag, paper-deep-dive, lewis-2020]

**Short answer.** **Contribution**: a parametric+non-parametric hybrid — a generator (BART) conditioned on retrieved passages from a dense retriever (DPR), trained end-to-end. **Method**: at each generation step, retrieve top-k docs; marginalize the generator's distribution over them (RAG-Sequence) or marginalize per-token (RAG-Token). **Evaluation**: open-domain QA (NaturalQuestions, TriviaQA, WebQ), abstractive QA, fact verification — beats prior parametric and non-parametric baselines. **Why it matters**: founded the RAG paradigm; modern systems are descendants.

**Expansion / why this is the answer.**
- **Method**:
  - Retriever: DPR — dual-encoder BERT; produces a top-k of document vectors per query.
  - Generator: BART, conditioned on `[query; retrieved doc]`.
  - **RAG-Sequence**: pick same doc for the whole sequence; marginalize the sequence likelihood over docs.
  - **RAG-Token**: different doc per token; marginalize at each step.
- **Trained end-to-end**: gradients flow through the generator with respect to retrieval (retriever is frozen in their main experiments but can be tuned).
- **Eval**:
  - Open-domain QA: substantial gains over closed-book and prior open-book baselines.
  - Highlight: model could cite which document it used.
- **Why modern RAG diverges**:
  - Modern systems use much larger LMs (no end-to-end training of the generator with the retriever).
  - Cross-encoder reranking instead of pure DPR.
  - Hybrid (BM25 + dense) instead of dense-only.
  - Long-context decoder-only LMs make the model's "use of context" much stronger.
- **30-second version**: "Lewis et al. 2020: introduced RAG — generator conditioned on retrieved docs, end-to-end trainable. Founded the modern RAG paradigm; production systems have moved beyond the exact architecture but the framing is foundational."

**Common follow-ups.**
- "Why isn't end-to-end RAG done at scale today?" → Cost; generators are too large to backprop through retrieval economically. Most production RAG uses a frozen LLM and a separately-tuned retriever.
- "What's DPR?" → Dense Passage Retrieval (Karpukhin et al. 2020) — dual-encoder BERT retriever, predecessor to modern embedding models.

**Common mistakes.**
- Conflating "RAG" the paper with "RAG" the modern paradigm — the paper's specific architecture is rare in production.

**References.**
- [Lewis et al. — "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"](https://arxiv.org/abs/2005.11401) — RAG.
- [Karpukhin et al. — "Dense Passage Retrieval"](https://arxiv.org/abs/2004.04906) — DPR.

---

### Q: What papers should every AI engineer interview candidate be ready to discuss?

**Category:** concept
**Difficulty:** mid
**Tags:** [must-know-papers, foundational]

**Short answer.** The minimum: **Attention Is All You Need** (Vaswani 2017), **BERT** (Devlin 2018), **GPT-3** (Brown 2020), **Chinchilla** (Hoffmann 2022), **InstructGPT** (Ouyang 2022), **LoRA** (Hu 2021), **RLHF** (Christiano 2017), **DPO** (Rafailov 2023), **RAG** (Lewis 2020), **Chain-of-Thought** (Wei 2022), **FlashAttention** (Dao 2022), **Mixtral / DeepSeek-V3** (recent MoE), **ReAct** (Yao 2022). Pick 3 of these for the "tell me about a paper" question — one foundational, one recent, one in your own line of work.

**Expansion / why this is the answer.**
- These cover the field: architecture (Vaswani, BERT, GPT-3), scaling (Chinchilla), training (RLHF, InstructGPT, DPO, LoRA), inference (FlashAttention), modern model designs (Mixtral, DeepSeek-V3), application paradigms (RAG, ReAct, CoT).
- An interviewer can ask "tell me about X" and you should have a 30-second + 5-minute answer.
- For research / staff roles, add: **Anthropic Mechanistic Interpretability** (Olsson et al. 2022 on induction heads), **Reflexion** (Shinn 2023), **GRPO** (Shao 2024, DeepSeek-Math), **Q-learning style / o1** (OpenAI / DeepSeek-R1 reasoning RL).
- For multimodal roles: **CLIP** (Radford 2021), **DALL-E**, **diffusion** papers.

**Common follow-ups.**
- "Why not LLaMA?" → It's a model release with a tech report; the *technical contribution* you're asked about is usually from the underlying papers (RoPE, GQA, etc.). Cite the tech report when discussing modern open-source LLMs.

**Common mistakes.**
- Treating recent-but-unread papers as well-understood; an interviewer will probe.
- Preparing one paper deeply and nothing else — interviewers ask specifically.

**References.**
- [Vaswani et al. — "Attention Is All You Need"](https://arxiv.org/abs/1706.03762) — foundational.
- [Brown et al. — GPT-3](https://arxiv.org/abs/2005.14165) — in-context learning.
- [Hoffmann et al. — Chinchilla](https://arxiv.org/abs/2203.15556) — modern scaling.
- [Rafailov et al. — DPO](https://arxiv.org/abs/2305.18290) — modern preference optimization.
- [Lewis et al. — RAG](https://arxiv.org/abs/2005.11401) — the RAG paradigm.
- [Hu et al. — LoRA](https://arxiv.org/abs/2106.09685) — modern fine-tuning.

---

### Q: How do you stay current with AI research without drowning?

**Category:** behavioral
**Difficulty:** intro
**Tags:** [research-process, reading-habit]

**Short answer.** A sustainable practice: subscribe to a few high-signal newsletters (e.g. Jack Clark's Import AI, Sebastian Raschka's Substack, AlphaSignal), read 1–2 papers a week deeply rather than 20 superficially, maintain a reading log (one paragraph per paper — title, claim, why-it-matters, your take), and follow 5–10 trusted researchers on Twitter/X for the social signal of what to read next. Don't try to read everything; cultivate taste.

**Expansion / why this is the answer.**
The volume of arXiv posts in 2026 is impossible to fully track; the goal is *not* coverage but *signal*. A small reading log compounds over months into a useful personal index. Pair each paper with a "what would change if this is true" note — forces you to engage rather than passively consume.

**Common follow-ups.**
- "Who do you follow?" → Have a real list.
- "What's the last thing you wrote a take on?" → Have an answer.

**Common mistakes.**
- Trying to read everything (burnout).
- Reading only what's trending.
- No note-taking; can't recall what you read 3 months ago.

**Signal.**
The interviewer wants to know you have an actual research-reading habit, not that you frantically catch up the night before. A confident answer that says "I read deeply over breadth" beats one that name-drops 50 papers. What works well: a specific current example ("I read X last week; my take is Y"); a named newsletter; a reading log. What works poorly: "I read everything on arXiv"; name-dropping without depth.

---

### Q: Walk me through the DeepSeek-V3 tech report.

**Category:** concept
**Difficulty:** senior
**Tags:** [deepseek-v3, moe, paper-deep-dive]

**Short answer.** **Contribution**: 671B-total / 37B-active MoE LLM trained in FP8 on 14.8T tokens; competitive with frontier closed models at ~$5.5M training cost. **Method**: multi-head latent attention (MLA, smaller KV cache than GQA), fine-grained MoE (256 routed + 1 shared expert, top-8), multi-token prediction (MTP) training, FP8 training stack. **Eval**: matches/beats GPT-4o and Claude 3.5 Sonnet on many reasoning + code benchmarks at a fraction of training cost. **Why it matters**: established that open-weight frontier-class models are economically feasible; FP8 training viability; MLA + fine-grained MoE as a serving-economic shape.

**Expansion / why this is the answer.**
- **Architecture**:
  - **MLA** (multi-head latent attention): K and V derived from a small shared latent vector; KV cache an order of magnitude smaller than GQA.
  - **Fine-grained MoE**: 256 routed experts (vs. Mixtral's 8), 1 shared expert always active. 8 routed + 1 shared per token = 37B active params.
  - **MTP heads** during training: dense supervision; also doubles as speculative-decoding draft heads at inference.
- **Training**:
  - **FP8 throughout**: first production-scale model trained primarily in FP8.
  - **14.8T tokens**: substantial corpus.
  - **Auxiliary-loss-free load balancing** for MoE: a bias adjustment that gets load-balancing without the auxiliary loss's distortion.
- **Eval**: strong on MATH, MMLU-Pro, GPQA, LiveCodeBench, SWE-bench Verified — competitive with the strongest closed models at the time.
- **Why this is a notable paper**:
  - First open-weight model that's plausibly frontier-class.
  - The serving-economic architecture (MLA + fine-grained MoE) made it feasible to deploy at a per-token cost competitive with closed models.
  - FP8 training at this scale was previously unproven in public.
- **30-second version**: "DeepSeek-V3 is a 671B-total / 37B-active MoE trained in FP8 on 14.8T tokens. Architectural innovations: MLA (KV cache reduction), fine-grained MoE (256 experts), MTP training objective. Demonstrates that open-weight frontier-class models are economically feasible."

**Common follow-ups.**
- "How does MLA compare to GQA?" → MLA factorizes through a small latent; KV cache is even smaller. Quality slightly different; DeepSeek-V3's results suggest competitive.
- "Why no auxiliary balancing loss?" → They report auxiliary-loss-free balancing works better — adjusts expert biases dynamically instead.

**Common mistakes.**
- Treating it as "yet another MoE model" — the contributions are architectural + training-stack.

**References.**
- [DeepSeek-AI — "DeepSeek-V3 Technical Report"](https://arxiv.org/abs/2412.19437) — primary.
- [DeepSeek-V2 paper](https://arxiv.org/abs/2405.04434) — MLA predecessor.

---

### Q: Walk me through the FlashAttention paper.

**Category:** concept
**Difficulty:** senior
**Tags:** [flashattention, paper-deep-dive, kernel]

**Short answer.** **Contribution**: an IO-aware exact attention kernel that tiles the computation so the `n × n` attention matrix never lives in HBM. Tiles of Q, K, V are loaded into on-chip SRAM; partial softmax statistics maintained online; output accumulated tile by tile. **Result**: same numerical output as standard attention, 2–4× wall-clock speedup on long sequences, lower memory. **Why it matters**: enabled longer-context training and inference at the same cost; became the de facto default attention kernel (PyTorch SDPA's "flash" backend).

**Expansion / why this is the answer.**
- **The bottleneck FlashAttention attacks**: naive attention materializes the `n × n` softmax matrix in HBM. For seq length 8k that's 64M entries per head per batch; HBM bandwidth dominates wall time.
- **The trick**:
  - Tile Q, K, V into blocks that fit in SRAM (the GPU's fast on-chip memory).
  - For each Q-tile, iterate over K-tile and V-tile blocks; compute partial logits, accumulate the output with an online-softmax algorithm that maintains the running max and sum.
  - The `n × n` matrix is never fully materialized.
- **Numerically exact**: the online-softmax (Milakov & Gimelshein 2018) is provably equivalent to the standard softmax; FlashAttention is *not* approximate.
- **Backward pass**: similar tiling; recomputes softmax from the saved statistics (no need to store the attention matrix).
- **FlashAttention-2** (Dao 2023): better parallelization across heads and sequence positions; ~2× faster than v1.
- **FlashAttention-3** (Shah et al. 2024): Hopper-specific (FP8, async warp specialization); leverages H100 features.
- **What it doesn't change**:
  - Doesn't reduce FLOPs.
  - Doesn't change model behavior.
- **Why this is on every paper-deep-dive list**:
  - Practical impact: every modern training stack uses it.
  - Method elegance: the IO-aware framing was novel.
  - Made long-context training affordable.

**Common follow-ups.**
- "Is FlashAttention sparse?" → No; it's dense (computes the full attention matrix's effect; just doesn't materialize it in HBM).
- "How does it compare to sparse attention (Longformer, Big Bird)?" → Sparse attention computes *fewer* entries (reduces FLOPs); FlashAttention computes the *same* entries but with better memory layout (reduces HBM traffic).

**Common mistakes.**
- Calling FlashAttention an approximation.
- Confusing it with linear-attention methods (Performer, Linformer).

**References.**
- [Dao et al. — "FlashAttention"](https://arxiv.org/abs/2205.14135).
- [Dao — "FlashAttention-2"](https://arxiv.org/abs/2307.08691).
- [Shah et al. — "FlashAttention-3"](https://arxiv.org/abs/2407.08608).

---

### Q: Walk me through the Mixtral / MoE paper.

**Category:** concept
**Difficulty:** mid
**Tags:** [mixtral, moe, paper-deep-dive]

**Short answer.** **Contribution**: Mixtral 8x7B (47B total, 13B active) — first open-weight MoE that matched GPT-3.5-class quality. **Method**: 8 experts per FFN sublayer, top-2 routing, GQA attention, otherwise standard transformer. **Eval**: matches or beats Llama 2 70B at ~5× lower per-token compute. **Why it matters**: proved MoE viable for open-weight LLMs; established the "decouple total params from active compute" pattern that DeepSeek-V3 later extended.

**Expansion / why this is the answer.**
- **Architecture**:
  - Each transformer block has 8 FFN experts.
  - Per token, top-2 experts active; routing via a small gating network.
  - Active params per token: ~13B; total: 47B.
- **GQA attention**: shared with the Mistral 7B base.
- **Load balancing**: standard MoE auxiliary loss.
- **Eval**:
  - Matches GPT-3.5 on most benchmarks.
  - Beats Llama 2 70B on most reasoning + code benchmarks.
  - Inference cost approximately matches 13B dense model.
- **Mixtral 8x22B** (April 2024): scaled-up version; 141B total, 39B active.
- **Why this is a paper-deep-dive question**:
  - Established MoE viability at the open-weight scale.
  - The "active vs. total" economic decoupling is now standard.
  - Predecessor to DeepSeek-V3's fine-grained MoE.

**Common follow-ups.**
- "Why top-2 instead of top-1?" → Top-1 (Switch Transformer) is faster; top-2 gives more parameter use per token; quality typically better.
- "How does Mixtral compare to DeepSeek-V3's fine-grained MoE?" → Mixtral: few coarse experts (8). DeepSeek: many fine-grained (256). Fine-grained empirically does better per active-param.

**Common mistakes.**
- Calling Mixtral "8 separate 7B models" — only the FFN is partitioned across experts; attention is shared.
- Confusing 47B total params with 13B active (the cost is the latter for compute, the former for memory).

**References.**
- [Jiang et al. — "Mixtral of Experts"](https://arxiv.org/abs/2401.04088).
- [Fedus et al. — "Switch Transformer"](https://arxiv.org/abs/2101.03961) — predecessor.

---

### Q: How would you structure a 30-minute job-talk on your own research?

**Category:** behavioral
**Difficulty:** senior
**Tags:** [research-talk, presentation, job-talk]

**Short answer.** Structure: (1) **The big problem** (2 minutes — the field-level motivation). (2) **My thesis / contribution** (1 minute — the one-sentence claim). (3) **Background** (5 minutes — minimum needed for the audience). (4) **Method** (10 minutes — the core technical content). (5) **Results** (5 minutes — headline + key ablation). (6) **Limitations + future work** (3 minutes — credibility-building). (7) **Q&A** (4 minutes). Leave 5+ minutes for questions; talks always run long.

**Expansion / why this is the answer.**
The structure interviewers expect for a research talk:
- **Open with the big problem**, not the method. "Why should the audience care?"
- **State the contribution upfront** — don't bury the lede.
- **Background**: just enough; trust the audience.
- **Method**: visual diagrams; one equation per slide max; build intuition before formality.
- **Results**: the headline (your strongest table/figure) + one ablation that shows the method is doing what you claim.
- **Limitations**: critical for credibility. Audiences trust speakers who acknowledge what doesn't work.
- **Future work**: 1–2 concrete directions; signals what's next, not "and they all lived happily ever after."
- **Q&A**: never the rushed bit at the end. Time-budget for it.

Common job-talk failure modes:
- **Too much method, no motivation**: audience checks out in minute 4.
- **Too much background**: nothing original in 15 minutes.
- **No headline result**: audience leaves not knowing what was achieved.
- **No limitations**: signals over-confidence.
- **Run over time**: cardinal sin; cut content, don't rush.

**Common follow-ups.**
- "How do you handle a hostile question?" → Acknowledge, agree where you can, disagree on substance. Never get defensive.
- "What if you don't know the answer?" → "I don't know — let me think... my best guess is X but I'd want to verify."

**Common mistakes.**
- Spending 20 minutes on method, 5 on results.
- Not rehearsing; live-talking yields chaos.
- Going over time.

**Signal.**
A research talk is a hiring signal at researcher / staff levels: communication, taste in what's important, ownership of limitations, clarity under pressure. The structure above is the recipe; rehearsal makes it work.

---

### Q: How do you find the "right" paper to read on a new topic?

**Category:** behavioral
**Difficulty:** intro
**Tags:** [paper-discovery, research-process]

**Short answer.** Start with the canonical / most-cited paper (Google Scholar, Semantic Scholar). Read its abstract + figures + related work. Identify the 2–3 sub-fields. For each: find the most-recent strong paper. Iterate. Within a week, you have a clean map. Avoid: reading 20 papers superficially before knowing which matter.

**Expansion / why this is the answer.**
Effective paper discovery: depth-first on the most-relevant trunk; broad reading comes after you know the structure.

**Common follow-ups.**
- "What if I can't tell what's canonical?" → Survey papers; ask a domain expert; the most-cited recent paper in a venue is often a good start.

**Common mistakes.**
- Reading the latest paper on a topic before reading the foundational one — you don't understand the context.

**Signal.**
Hiring signal at researcher levels: do you have effective research-discovery habits?

---

### Q: How do you keep a research log?

**Category:** behavioral
**Difficulty:** intro
**Tags:** [research-log, note-taking]

**Short answer.** Per paper: title, one-paragraph summary, why-it-matters, your take. Per experiment: hypothesis, setup, result, conclusion, next-step. Markdown notes in a personal repo / Obsidian / Notion / Logseq. The log compounds: in 6 months you have a personal map of what you know and what's open.

**Expansion / why this is the answer.**
A research log is the cheapest, highest-ROI research tool. Without it, papers blur; with it, knowledge compounds.

**Common follow-ups.**
- "What template do you use?" → Whatever you'll actually maintain.

**Common mistakes.**
- No log; you can't recall what you read 3 months ago.

**Signal.**
Discipline + research process maturity.

---

### Q: What's a "research taste" question an interviewer might ask?

**Category:** behavioral
**Difficulty:** senior
**Tags:** [research-taste, judgment]

**Short answer.** Examples: "Pick a paper you think is over-cited but actually weak." "What's a problem the field is working on that you think is mis-framed?" "What's a paper you read last year that you've changed your mind about?" These probe research judgment — can you have opinions on the field, defend them, and update?

**Expansion / why this is the answer.**
What lands:
- A specific opinion, defended with reasons.
- Updating your view when challenged.
- Distinguishing "I dislike the paper" from "the paper is wrong."

What doesn't:
- "All papers in field X are interesting!" — no taste.
- Disparaging without substance.

**Common follow-ups.**
- "Can you justify your view?" → Have specific points.

**Common mistakes.**
- No prepared opinions; freezes in the interview.

**Signal.**
Senior+ research role probe: do you have independent judgment?

---

### Q: Walk through the Llama 3 paper.

**Category:** concept
**Difficulty:** senior
**Tags:** [llama-3, paper-deep-dive]

**Short answer.** **Contribution**: 8B/70B/405B family trained on ~15T tokens; substantial improvements over Llama 2 via data quality + scale + extensive post-training. **Architecture**: standard dense decoder-only with GQA + RoPE + SwiGLU + RMSNorm. **Training**: Chinchilla-scaling-violating (well past compute-optimal); aggressive data filtering; iterative DPO post-training across 5+ rounds. **Eval**: competitive with GPT-4 / Claude 3.x at 405B scale.

**Expansion / why this is the answer.**
- Architecture: nothing radical; rely on scale and data.
- Pretraining data: 15.6T tokens; substantial code + multilingual.
- Vocab: 128k tokens (vs Llama 2's 32k); better multilingual + code.
- Post-training: SFT → DPO iterations; explicit rejection-sampling-FT before DPO.
- Long-context: 8k → 32k → 128k via continued pretraining stages.

**Common follow-ups.**
- "What's notable about Llama 3 vs 2 architectures?" → GQA throughout (was 70B-only in Llama 2); larger vocab; FFN sizing changes.
- "Why over-train past Chinchilla-optimal?" → Inference economy: smaller model, fully baked.

**Common mistakes.**
- Citing Llama 3 architecture as innovative; the contribution is scale + data + training.

**References.**
- [Llama Team — "The Llama 3 Herd of Models"](https://arxiv.org/abs/2407.21783).

---

### Q: Walk through "Direct Preference Optimization" (Rafailov et al.).

**Category:** concept
**Difficulty:** senior
**Tags:** [dpo, paper-deep-dive]

**Short answer.** **Contribution**: derive a closed-form supervised loss equivalent to KL-regularized RLHF; skip the reward model and PPO. **Method**: under Bradley-Terry preference + KL-regularized policy objective, the optimal policy implicitly defines a reward; substitute and the BCE-style loss falls out. **Eval**: matches or beats PPO-RLHF at much lower compute. **Why it matters**: changed the entire post-training landscape — most open-weight models post-2023 use DPO over PPO.

**Expansion / why this is the answer.**
See T3 "DPO loss math" entry for the derivation. The paper deep-dive structure:
- **Method**: derive the equivalence; loss is `−log σ(β [log π_θ(y_w|x)/π_ref(y_w|x) − log π_θ(y_l|x)/π_ref(y_l|x)])`.
- **Theoretical equivalence**: optimal under BT + KL-regularized RL.
- **Eval**: better than PPO on TL;DR summarization and other tasks; much cheaper to train.

**Common follow-ups.**
- "Limitations?" → Likelihood-decrease pathology; less responsive than PPO to fine-grained reward shaping.

**Common mistakes.**
- Calling DPO "RL"; it's supervised.

**References.**
- [Rafailov et al. — "DPO"](https://arxiv.org/abs/2305.18290).

---

### Q: Walk me through the "Tools and Agents in LLMs" landscape circa 2026.

**Category:** concept
**Difficulty:** senior
**Tags:** [agents, landscape, 2026]

**Short answer.** State as of 2026: (a) **frontier models** (Claude 4.x, GPT-5, Gemini 2.x) are highly capable single-agent operators with strong tool use; (b) **agentic benchmarks** (SWE-bench Verified, TAU-bench, GAIA) show progressively-better performance; (c) **production patterns**: ReAct single-agent + tools dominate; multi-agent for specific cases (research swarms); (d) **MCP** is the emerging tool-protocol standard. Compared to 2024: more reliable, longer-horizon, but long-horizon coherence remains a research frontier.

**Expansion / why this is the answer.**
- 2023: agent demos; brittle.
- 2024: SWE-bench started moving (5% → 20%+ in a year); first production deployments (Devin, Cursor).
- 2025: SWE-bench Verified plateaued near 70%+; computer-use agents launched.
- 2026: long-horizon agents reaching minutes-to-hours of useful work; reliability still gap to humans on complex tasks.

**Common follow-ups.**
- "What's the next frontier?" → Long-horizon coherence; multimodal world-modeling; agent-agent collaboration.

**Common mistakes.**
- Treating agents as solved or treating them as completely broken; reality is in between.

**References.**
- [Anthropic — "Building effective agents"](https://www.anthropic.com/research/building-effective-agents).
- [Jimenez et al. — "SWE-bench"](https://arxiv.org/abs/2310.06770).

---

### Q: What's a recent paper / approach you'd want to learn more about?

**Category:** behavioral
**Difficulty:** mid
**Tags:** [research-curiosity]

**Short answer.** Pick a specific paper or approach — `DeepSeek-V3`'s MTP, `Mamba-2`, `Mixture-of-Depths`, `EAGLE-3`, GRPO+verifier-RL pipelines, sparse autoencoders for interpretability. State why you find it interesting (a specific question or claim). Avoid generic "AGI / superintelligence" framings — too broad.

**Expansion / why this is the answer.**
Signal: do you have an active reading life? Can you explain why you're curious?

What works:
- "I want to understand why MoE fine-grained experts (DeepSeek-V3) beat coarse (Mixtral) at the same active-params."
- "I want to follow up on the lost-in-the-middle problem at 1M context."

What doesn't:
- "I want to understand AGI." — too vague.

**Common follow-ups.**
- "Why specifically?" → Have a real reason.

**Common mistakes.**
- Saying "I want to read more papers in general."

**Signal.**
Specific curiosity = research maturity.

---

### Q: How do you balance reading papers with building?

**Category:** behavioral
**Difficulty:** mid
**Tags:** [reading-vs-building, balance]

**Short answer.** Both are essential and reinforce each other: papers without building → shallow understanding; building without reading → reinventing wheels. Concrete balance: 60–70% building, 20–30% reading + writing, occasional deep dives. The exact ratio shifts by phase: early-career → more reading; senior IC → more building; research role → more both.

**Expansion / why this is the answer.**
Reading-only failure mode: armchair expert who can't ship.
Building-only failure mode: re-invents-the-wheel and stays at one level.

**Common follow-ups.**
- "How do you decide what to read?" → Driven by current building; read what unblocks the next step.

**Common mistakes.**
- Treating reading as a separate "research time" from building work.

**Signal.**
Pragmatic balance; not pedantically academic.

---

### Q: Walk through "Constitutional AI" by Anthropic.

**Category:** concept
**Difficulty:** senior
**Tags:** [constitutional-ai, anthropic, paper-deep-dive]

**Short answer.** **Contribution**: replace human preference labeling with AI self-critique against a written constitution. **Method**: SFT on self-critique-then-revise outputs; RLAIF with AI-generated preference labels based on constitution adherence. **Eval**: better helpful-harmless tradeoff; lower over-refusal rate. **Why it matters**: scaled alignment beyond human-labeling bandwidth; established RLAIF as a viable approach.

**Expansion / why this is the answer.**
- The pipeline:
  - Stage 1: SFT on self-revised outputs (model critiques + revises against constitution).
  - Stage 2: model judges which of two outputs better follows constitution; produces preference dataset.
  - Stage 3: RL (PPO) on the AI-labeled preferences.
- Constitution: principles + few-shot examples for self-critique.
- Modern Claude uses CAI-style training.

**Common follow-ups.**
- "Risks?" → AI-labeler biases propagate; need diverse labelers.

**Common mistakes.**
- Treating CAI as a single technique vs. a multi-stage pipeline.

**References.**
- [Bai et al. — "Constitutional AI"](https://arxiv.org/abs/2212.08073).

---

### Q: Walk through "Chain-of-Thought Prompting" (Wei et al. 2022).

**Category:** concept
**Difficulty:** mid
**Tags:** [cot, paper-deep-dive]

**Short answer.** **Contribution**: showed that prompting LLMs with intermediate reasoning examples dramatically improves multi-step task performance. **Method**: few-shot examples include `(question, reasoning chain, answer)` triples; LLM imitates the pattern. **Eval**: order-of-magnitude gains on GSM8K, math benchmarks. **Why it matters**: established a new prompting paradigm; precursor to modern reasoning models (o1, R1).

**Expansion / why this is the answer.**
- Earlier prompting: question → answer.
- CoT: question → step-by-step reasoning → answer.
- Emergent behavior: only large models benefit; small models can even regress.
- Subsequent: zero-shot CoT (Kojima et al.), self-consistency (Wang et al.), o1-style learned reasoning.

**Common follow-ups.**
- "Does CoT actually represent the model's reasoning?" → Not necessarily faithful (Turpin et al. 2023).

**Common mistakes.**
- Treating CoT as universally beneficial; small models regress.

**References.**
- [Wei et al. — "Chain-of-Thought Prompting"](https://arxiv.org/abs/2201.11903).

---
