# Training & Fine-Tuning — aggregated references

## RLHF / preference optimization

- [Ouyang et al. — "InstructGPT"](https://arxiv.org/abs/2203.02155) — RLHF pipeline.
- [Christiano et al. — "Deep Reinforcement Learning from Human Preferences"](https://arxiv.org/abs/1706.03741) — original RLHF.
- [Rafailov et al. — "DPO"](https://arxiv.org/abs/2305.18290) — Direct Preference Optimization.
- [Azar et al. — "IPO"](https://arxiv.org/abs/2310.12036) — Identity Preference Optimization.
- [Ethayarajh et al. — "KTO"](https://arxiv.org/abs/2402.01306) — Kahneman-Tversky Optimization.
- [Shao et al. — "DeepSeekMath / GRPO"](https://arxiv.org/abs/2402.03300) — GRPO.
- [Bai et al. — "Constitutional AI"](https://arxiv.org/abs/2212.08073) — CAI / RLAIF.
- [Lee et al. — "RLAIF"](https://arxiv.org/abs/2309.00267).

## PEFT / Fine-tuning

- [Hu et al. — "LoRA"](https://arxiv.org/abs/2106.09685) — LoRA.
- [Dettmers et al. — "QLoRA"](https://arxiv.org/abs/2305.14314) — QLoRA.
- [Liu et al. — "DoRA"](https://arxiv.org/abs/2402.09353) — DoRA.
- [Aghajanyan et al. — "Intrinsic Dimensionality"](https://arxiv.org/abs/2012.13255) — why low-rank works.

## Instruction tuning

- [Wei et al. — "FLAN"](https://arxiv.org/abs/2109.01652) — instruction tuning.
- [Wang et al. — "Self-Instruct"](https://arxiv.org/abs/2212.10560).
- [Zhou et al. — "LIMA"](https://arxiv.org/abs/2305.11206) — quality > quantity.

## Distributed training

- [Rajbhandari et al. — "ZeRO"](https://arxiv.org/abs/1910.02054) — ZeRO.
- [Shoeybi et al. — "Megatron-LM"](https://arxiv.org/abs/1909.08053) — tensor parallel.
- [Huang et al. — "GPipe"](https://arxiv.org/abs/1811.06965) — pipeline parallel.
- [Korthikanti et al. — "Reducing Activation Recomputation"](https://arxiv.org/abs/2205.05198) — sequence parallel + selective recompute.
- [PyTorch FSDP docs](https://pytorch.org/docs/stable/fsdp.html).

## Mixed precision

- [Micikevicius et al. — "Mixed Precision Training"](https://arxiv.org/abs/1710.03740).
- [Kalamkar et al. — "bf16 Study"](https://arxiv.org/abs/1905.12322).
- [Micikevicius et al. — "FP8 Formats"](https://arxiv.org/abs/2209.05433).

## Distillation

- [Hinton, Vinyals, Dean — "Distilling the Knowledge"](https://arxiv.org/abs/1503.02531).
- [Sanh et al. — "DistilBERT"](https://arxiv.org/abs/1910.01108).
- [Wang et al. — "MiniLM"](https://arxiv.org/abs/2002.10957).

## Reward hacking / alignment problems

- [Gao, Schulman, Hilton — "Reward Model Overoptimization"](https://arxiv.org/abs/2210.10760).
- [Sharma et al. — "Sycophancy"](https://arxiv.org/abs/2310.13548).
- [Casper et al. — "Open Problems and Fundamental Limitations of RLHF"](https://arxiv.org/abs/2307.15217).

## Catastrophic forgetting

- [Kirkpatrick et al. — "EWC"](https://arxiv.org/abs/1612.00796).
- [Luo et al. — "Catastrophic Forgetting in LLM Continual Fine-tuning"](https://arxiv.org/abs/2308.08747).

## Training stability

- [Zoph et al. — "ST-MoE"](https://arxiv.org/abs/2202.08906) — stability for MoE.
- [Chowdhery et al. — PaLM Technical Report](https://arxiv.org/abs/2204.02311) — loss-spike documentation.
- [Gemma 2 Technical Report](https://arxiv.org/abs/2408.00118) — soft-capping.

## Memory & checkpointing

- [Chen et al. — "Training Deep Nets with Sublinear Memory Cost"](https://arxiv.org/abs/1604.06174) — gradient checkpointing.

## Data curation / decontamination

- [Magar & Schwartz — "Data Contamination: From Memorization to Exploitation"](https://arxiv.org/abs/2203.08242).
- [Xie et al. — "DoReMi"](https://arxiv.org/abs/2305.10429).
- [Bengio et al. — "Curriculum Learning"](https://dl.acm.org/doi/10.1145/1553374.1553380).
