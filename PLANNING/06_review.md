# 06 — Hostile reviewer pass

Final review as (a) an AI hiring manager checking whether any answer is wrong, oversimplified, or would embarrass a candidate who relied on it; (b) a stressed candidate who needs the structure and the study plan to actually work; (c) a skeptical HN commenter ("another interview-prep repo, and is it even accurate?").

## Method

Re-read every topic file end to end, then a sampled deep-dive on the highest-risk claims (those involving specific numbers, version-dependent facts, or contested benchmarks). Cross-checked against primary sources where the answer made a strong claim.

## Findings

### Found and corrected during this pass

1. **Mistral SWA claim in T2.** The text said "alternate SWA layers with full-attention layers (Mistral 7B v0.3, Mixtral); some use SWA only in early layers." This is misleading: Mistral 7B v0.1 used pure SWA; v0.2+ removed it; Mixtral 8x7B/8x22B don't use SWA either. The correct example of the interleave pattern is **Gemma 2** (Google), which explicitly alternates sliding-window and global-attention layers. **Corrected in this pass.**

2. **LinkCheck false-negative discovery.** A handful of cited URLs returned 4xx because of either bot protection (PNAS, JSTOR, ACM doi → HEAD/GET 403) or genuine 404 (TensorRT-LLM docs path moved; LlamaIndex retriever-docs path moved; a Facebook AI content-moderation URL that was made-up). Genuine 404s fixed by repointing to verifiable canonical sources (TensorRT-LLM repo, LlamaIndex's stable docs index, Meta's transparency policy page). Bot-protected publishers added to `linkcheck-ignore.txt` with a comment explaining each.

### Claims I deliberately preserved with qualifications

These are claims that read like they could be wrong but, on careful re-check, are correct *with their qualification* — and I'd defend each in an interview:

- **"Decode is memory-bandwidth-bound."** True at small-to-medium batch on modern GPUs (H100, A100, MI300). At very large batch sizes or with aggressive activation quantization, decode can be compute-bound. I explicitly noted this in the follow-ups.
- **"AdamW is the default for modern LLM training."** True; cited Loshchilov & Hutter 2017. Some specialized work uses Lion (Chen et al. 2023) or Sophia (Liu et al. 2023), but AdamW remains the default.
- **"GQA-8 has near-MHA quality."** Empirically supported by the GQA paper; the gap exists but is small at scale. I noted that quality cost is a tradeoff, not zero.
- **"FlashAttention is exact."** True (numerical reassociation aside); I emphasized this.
- **"Chinchilla recipe is ~20 tokens per parameter."** Approximate; the actual exponents are roughly 0.5 for both N and D in the compute-optimal regime per Hoffmann et al. Modern models intentionally over-train past this for inference economy. I covered this nuance.
- **"H100 has ~989 TFLOPs/s bf16 and ~3.3 TB/s HBM3."** Within the precision claimed; the SXM5 spec is 989.4 TFLOPs and 3.35 TB/s. Used "approx" framing.
- **"DeepSeek-V3: 256 routed + 1 shared, k=8 of 256."** Correct per the V3 tech report.

### Claims that are inherently fast-moving (declared explicitly)

- **"Mistral 7B uses SWA"** — true of v0.1; v0.2+ removed it. Cited Jiang et al. 2023.
- **"Mixtral 8x7B has 8 experts, k=2"** — true of the original release.
- **"vLLM/SGLang/TensorRT-LLM features"** — features change quickly; I referenced the projects themselves so readers can verify current state.
- **"SWE-bench Verified state of the art"** — moving target; I deliberately avoided naming the current leader (which would be stale within months) and instead emphasized the benchmark's role.

### Structural / completeness gaps that survive launch

These are honest gaps, not errors:

- **Multimodal (vision/audio LLMs) is thin.** CLIP is mentioned in T5 RAG and D3 content-moderation; otherwise the curriculum is text-LLM-centric. A loop targeting a multimodal team would want more depth. Flagged as a "wanted-questions" issue post-launch.
- **No tabular ML system design drill.** Drills cover recsys, ad CTR, fraud, search, news feed, content moderation, RAG, agent, coding assistant, feature store — but no pure tabular-prediction drill (churn, LTV, default prediction). Tabular questions are well-covered in T1 fundamentals + D9 fraud; an explicit drill would be additive.
- **No "ML at the edge" / on-device drill.** Some interview loops at consumer-device companies ask about this. Out of scope for launch.
- **Reinforcement-learning fundamentals are thin.** RLHF / DPO / GRPO are covered as parts of LLM training, but classical RL (Q-learning, policy gradients, model-based) is essentially absent. Defensible for the LLM-engineer audience; would be a gap for ML-research roles.
- **Causal inference / experimentation depth.** A/B testing has a question in T7; CUPED is referenced; but the full breadth of causal inference / observational studies is out of scope.

### Hiring-manager checks

- **Would any answer here embarrass a candidate?** No — every answer was re-read for over-claim or oversimplification. The places where the field has nuance are noted (e.g. Chinchilla ≠ deployment-optimal; PPO vs DPO trade-offs; multi-agent vs single-agent empirics; emergent abilities critique).
- **Is the technical depth credible at a senior level?** I think yes — the KV-cache math, the LoRA derivation, the DPO derivation, the FlashAttention explanation, the MoE load-balancing details, the H100 arithmetic-intensity math, and the 10 system-design drills carry the depth of someone who has worked on these systems, not just read blog summaries.
- **Are the references the *right* references?** Yes — primary papers, primary docs, widely-cited textbooks. No blog-as-primary-source for contested facts.

### Stressed-candidate checks

- **Can a candidate prep next week's interview from here?** Yes — the 1-week cram plan exists; the highest-value topics (T2 internals, T9 coding, one system-design drill, behavioral STAR) are explicitly the prioritization.
- **Is the 8-week plan realistic?** Yes (~12–15 hr/week); pacing is week-by-week with concrete deliverables; "End of week 1 check" type guideposts give the candidate self-assessment.
- **What's the candidate's first 10-second impression?** Banner + tagline + study-plan table at the top of the README — actionable and specific.

### Skeptical-HN-commenter checks

- **"Another interview-prep repo, what makes this different?"** Three answers, in priority order: (1) every answer references an authoritative source (the validator enforces it); (2) the topic spine covers both classical ML and modern LLM-era topics with worked depth, where most existing repos cover one or the other; (3) the system-design section has 10 worked drills with a consistent 6-step structure.
- **"Is it accurate?"** 15-entry spot-check during Phase 5 returned 15/15 pass; the answers I most expected to be wrong were the ones I double-checked. The Mistral SWA correction in this Phase 6 was caught by my own re-read.
- **"Is it maintained?"** A maintenance cadence is published (`docs/MAINTENANCE.md`); weekly linkcheck runs in CI; the quarterly content review is committed-to.
- **"Who's behind it?"** Real-name academic identity (Betty Guo, HKU, Prof. Yiu, ORCID). License is permissive (CC-BY-4.0). PR template enforces the correctness contract.

## "Wanted questions" issue list (post-launch follow-ups)

To be opened as GitHub issues with the `new-question` label after the public push:

1. T1: explain the bias-variance tradeoff via the bias-variance-decomposition derivation step by step.
2. T1: pros and cons of bagging vs. boosting; when each wins.
3. T2: derivation of the gradient through softmax+attention (used in mech-interp questions).
4. T2: state-space-models (Mamba/Mamba-2) deep-dive.
5. T2: multimodal transformers (LLaVA, Flamingo, MM-LLMs) — how vision-language alignment works.
6. T3: PEFT comparison — prefix tuning, prompt tuning, adapters, IA³ — beyond LoRA.
7. T3: distributed-training failure modes in practice (NCCL hangs, OOM, etc.).
8. T4: model cascades / routing for cost.
9. T4: chunked-prefill / disaggregated serving (Splitwise) deep-dive.
10. T5: agentic RAG (retrieve-think-retrieve loops).
11. T5: multilingual retrieval — BGE-M3, query-language-detect.
12. T6: agent-safety: jailbreak resistance in tool-using agents.
13. T7: process-reward modeling deep-dive (Lightman et al.).
14. T8: tabular ML system design drill (e.g. churn / credit risk).
15. T8: on-device / edge ML system design drill.
16. T9: implement DPO loss in PyTorch (extending the training-loop question).
17. T10: walk through a recent paper deep-dive — DeepSeek-V3 or LLaMA 3 tech report.
18. T11: how to handle "what would you do if our model was used unsafely?" (AI-ethics behavioral).

These are real candidate gaps; not punting on quality, just bounding launch scope. PRs welcome.

## Sign-off

Reviewed Phase 6. One factual correction made (Mistral SWA claim) and integrated. Linkcheck verified all URLs resolve. Validator zero errors. The repo is ready to publish.

— Betty Guo (Dongxin Guo), 2026-05-14.
