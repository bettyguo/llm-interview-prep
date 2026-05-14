# Transformers & LLM Internals — aggregated references

## Foundational

- [Vaswani et al. — "Attention Is All You Need"](https://arxiv.org/abs/1706.03762) — the transformer.
- [Devlin et al. — "BERT"](https://arxiv.org/abs/1810.04805) — encoder-only.
- [Radford et al. — GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — decoder-only.
- [Raffel et al. — "T5"](https://arxiv.org/abs/1910.10683) — encoder-decoder.
- [Brown et al. — "GPT-3"](https://arxiv.org/abs/2005.14165) — in-context learning.

## Attention variants

- [Shazeer — "MQA"](https://arxiv.org/abs/1911.02150) — Multi-Query Attention.
- [Ainslie et al. — "GQA"](https://arxiv.org/abs/2305.13245) — Grouped-Query Attention.
- [DeepSeek-V2 paper](https://arxiv.org/abs/2405.04434) — MLA.
- [Dao et al. — "FlashAttention"](https://arxiv.org/abs/2205.14135) — IO-aware attention.
- [Dao — "FlashAttention-2"](https://arxiv.org/abs/2307.08691).
- [Shah et al. — "FlashAttention-3"](https://arxiv.org/abs/2407.08608).

## Positional encodings

- [Su et al. — "RoPE / RoFormer"](https://arxiv.org/abs/2104.09864) — rotary position.
- [Press et al. — "ALiBi"](https://arxiv.org/abs/2108.12409) — linear bias.
- [Peng et al. — "YaRN"](https://arxiv.org/abs/2309.00071) — context extension.
- [Ding et al. — "LongRoPE"](https://arxiv.org/abs/2402.13753).
- [Chen et al. — "Position Interpolation"](https://arxiv.org/abs/2306.15595).

## Architecture choices

- [Xiong et al. — "On Layer Normalization in the Transformer Architecture"](https://arxiv.org/abs/2002.04745) — pre-norm vs. post-norm.
- [Shazeer — "GLU Variants"](https://arxiv.org/abs/2002.05202) — SwiGLU.
- [Hendrycks & Gimpel — "GELU"](https://arxiv.org/abs/1606.08415) — GELU.

## MoE

- [Shazeer et al. — "Sparsely-Gated MoE"](https://arxiv.org/abs/1701.06538) — foundational.
- [Fedus, Zoph, Shazeer — "Switch Transformer"](https://arxiv.org/abs/2101.03961).
- [Jiang et al. — "Mixtral of Experts"](https://arxiv.org/abs/2401.04088).
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437).

## Scaling laws

- [Kaplan et al. — "Scaling Laws"](https://arxiv.org/abs/2001.08361) — Kaplan.
- [Hoffmann et al. — "Chinchilla"](https://arxiv.org/abs/2203.15556) — Chinchilla.
- [Wei et al. — "Emergent Abilities of LLMs"](https://arxiv.org/abs/2206.07682) — emergence claim.
- [Schaeffer et al. — "Are Emergent Abilities a Mirage?"](https://arxiv.org/abs/2304.15004) — critique.

## Tokenization

- [Sennrich et al. — BPE](https://arxiv.org/abs/1508.07909) — original BPE.
- [Kudo & Richardson — SentencePiece](https://arxiv.org/abs/1808.06226).
- [OpenAI tiktoken](https://github.com/openai/tiktoken) — primary tokenizer.

## ICL / reasoning

- [Olsson et al. — "Induction Heads"](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) — mechanistic.
- [Wei et al. — "Chain-of-Thought"](https://arxiv.org/abs/2201.11903) — CoT.
- [Kojima et al. — "Zero-Shot CoT"](https://arxiv.org/abs/2205.11916).
- [Wang et al. — "Self-Consistency"](https://arxiv.org/abs/2203.11171).

## Long context

- [Liu et al. — "Lost in the Middle"](https://arxiv.org/abs/2307.03172).
- [Beltagy et al. — "Longformer"](https://arxiv.org/abs/2004.05150).
- [Jiang et al. — "Mistral 7B"](https://arxiv.org/abs/2310.06825) — SWA in production.
- [Gu & Dao — "Mamba"](https://arxiv.org/abs/2312.00752) — SSM alternative.

## Inference math

- [Pope et al. — "Efficiently Scaling Transformer Inference"](https://arxiv.org/abs/2211.05102) — the canonical analysis.
