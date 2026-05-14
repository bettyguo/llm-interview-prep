# 03 — Verification log

The answer-correctness protocol (Phase 0) requires every non-trivial answer to carry an authoritative reference, verified at write time.

## Status

At launch (2026-05-14):

- **Total entries**: 147 across 12 topic files + 10 system-design drill files = 22 files containing Q&A.
- **Entries requiring references** (category ∈ {concept, derivation, system-design}): 121.
- **Entries with reference**: 121 / 121 (100%). Enforced by `tools/validate_entries.py` — CI rejects unsourced entries.
- **Entries requiring implementation** (category = coding): 14. All carry working Implementation blocks; runnable variants tested locally.
- **Entries requiring signal block** (category = behavioral): 12. All carry the Signal block per schema.
- **CI status**: `python tools/validate_entries.py` returns 0; `python tools/linkcheck.py` to be run in scheduled GHA.
- **Unverified entries**: 0.

## Reference source breakdown (rough)

- Original research papers (arXiv, conference): ~70%
- Primary framework / vendor documentation (PyTorch, HuggingFace, Anthropic, OpenAI, vLLM, etc.): ~18%
- Textbooks (Bishop, Hastie, Goodfellow, Murphy, Jurafsky & Martin): ~7%
- High-quality surveys / production-blog case studies (Anthropic, OpenAI, Google research): ~5%

No blog-only sources for contested facts. All claims involving exponents, benchmark numbers, default hyperparameters, or version-dependent behavior carry primary-source citations.

## Spot-check protocol (Phase 5)

Sampled 15 entries deterministically (10% of 147 ≈ 15) for re-verification against the cited source. Procedure:

1. Re-read the cited source.
2. Confirm the entry's claim is supported by the source.
3. Mark `pass` or `fail` below.

Sampling list:

| # | topic | question-slug | reference-used | spot-check |
|---|-------|---------------|----------------|------------|
| 1 | 01-fundamentals | Bias-variance tradeoff | Bishop §3.2 + Belkin et al. PNAS 2019 + Nakkiran et al. 2020 | pass |
| 2 | 01-fundamentals | L1 vs. L2 regularization | ESL §3.4 + Loshchilov AdamW + Tibshirani Lasso | pass |
| 3 | 02-transformers | Scaled dot-product attention derivation | Vaswani et al. 2017 §3.2.1 | pass |
| 4 | 02-transformers | GQA vs MHA vs MQA | Shazeer MQA + Ainslie GQA + DeepSeek-V2 MLA | pass |
| 5 | 02-transformers | Chinchilla vs Kaplan scaling | Kaplan 2020 + Hoffmann Chinchilla 2022 + Wei 2022 + Schaeffer 2023 | pass |
| 6 | 03-training | DPO loss derivation | Rafailov et al. 2023 §3 | pass |
| 7 | 03-training | LoRA architecture | Hu et al. LoRA + Dettmers QLoRA + Aghajanyan intrinsic-dim | pass |
| 8 | 04-inference | Paged attention | Kwon et al. 2023 (vLLM) | pass |
| 9 | 04-inference | KV cache memory math | Pope et al. 2022 + Kwon et al. 2023 | pass |
| 10 | 05-rag | Reciprocal Rank Fusion | Cormack et al. 2009 | pass |
| 11 | 05-rag | Lost-in-the-middle | Liu et al. 2023 | pass |
| 12 | 06-agents | ReAct vs Plan-and-Execute vs Reflexion | Yao ReAct + Shinn Reflexion + Anthropic "Building effective agents" | pass |
| 13 | 06-agents | SWE-bench / TAU-bench / GAIA | Jimenez SWE-bench + Yao TAU-bench + Mialon GAIA | pass |
| 14 | 07-eval | LLM-as-judge biases | Zheng MT-Bench + Panickssery self-preference | pass |
| 15 | 08-system-design | D1 video recommendation | Covington YouTube + Naumov DLRM + Chen off-policy | pass |

**Result**: 15 / 15 pass. No corrections required.

## Known limitations (declared honestly)

- A handful of entries reference an *organization's* primary docs (vLLM, TensorRT-LLM, Tecton, Feast). These can change without notice; they are the right primary source today but the linkcheck cadence catches dead links.
- A few entries reference vendor blog announcements (Anthropic, OpenAI engineering posts). These are first-party but informal; treated as supplementary alongside papers where applicable.
- Some referenced 2024–2025 papers are still being actively cited / followed up; their findings may be refined. I track this via the quarterly content review (see `docs/MAINTENANCE.md`).

## Maintenance signal

This log is updated on every quarterly content review. New entries added by contributors via PR must include a reference; the validator enforces this; the reviewer spot-checks the reference per the PR template's reviewer-guide section.

## Sign-off

**Phase 5 verification result**: zero unverified entries; spot-check sample (15 entries, 10%) returned 15/15 pass.

Reviewed and signed off by Betty Guo (Dongxin Guo), 2026-05-14.
