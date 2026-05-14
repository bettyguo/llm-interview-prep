# 03 — Training & Fine-Tuning

The "have you actually trained one" topic. Interviewers care whether you can reason about pretraining objectives, instruction tuning, preference optimization (RLHF/DPO/GRPO), and parameter-efficient fine-tuning (LoRA/QLoRA) — and whether you can debug training-time pathologies (loss spikes, reward hacking, mode collapse).

## What you should walk in able to do

- Contrast **next-token prediction, masked LM, span corruption** pretraining objectives.
- Describe what makes a good **SFT dataset** and how it differs from pretraining data.
- Compare **RLHF (PPO), DPO, IPO, KTO, GRPO** — what each optimizes, when each wins, what each fails at.
- Explain **reward hacking** and **reward over-optimization**, and the mitigations (KL penalty, reward-model ensembles, periodic re-collection).
- Explain **LoRA, QLoRA, DoRA** — the math (low-rank update ΔW = BA), the rank choice, the parameter-count savings.
- Reason about **distributed training**: data parallel, ZeRO 1/2/3, tensor parallel, pipeline parallel, FSDP — what each parallelizes and what each costs in communication.
- Reason about **bf16 vs. fp16** numerical stability for LLM training.
- Diagnose loss spikes, gradient explosions, and divergence patterns.

## Questions

See [`questions.md`](questions.md).

## References (aggregated)

See [`references.md`](references.md).
