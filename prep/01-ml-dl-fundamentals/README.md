# 01 — ML & DL Fundamentals

The fundamentals layer. Every AI/ML/LLM interview loop assumes you can answer these — they show up as warmup questions, in coding interviews disguised as "implement this," and as the substrate of every system-design and modeling discussion.

## What you should walk in able to do

- Define and contrast: bias and variance, overfitting and underfitting, regularization variants, supervised vs. self-supervised, classification vs. regression, generative vs. discriminative.
- Pick the right **loss function** for a problem and defend the choice (MSE vs. cross-entropy vs. hinge vs. focal vs. KL).
- Walk through how **SGD, momentum, Adam, AdamW, and Lion** differ — and why AdamW became the default for LLM training.
- Pick the right **normalization** (BatchNorm vs. LayerNorm vs. RMSNorm vs. GroupNorm) for a given architecture and justify it.
- Pick the right **initialization** (Xavier vs. He) given the activation function.
- Pick between classical algorithms (linear/logistic regression, GBMs, SVMs, kNN, naive Bayes, k-means, PCA) and defend the choice.
- Use evaluation metrics correctly: precision/recall vs. accuracy under class imbalance, ROC-AUC vs. PR-AUC, calibration.
- Recognize and avoid **data leakage**, common in interview design exercises.

## Self-assessment quiz

Skim the questions below. If you can answer ten in a row out loud without checking, fast-track this topic. Otherwise plan for one focused day.

## Questions

See [`questions.md`](questions.md). Each entry follows the [Q&A schema](../../CONTRIBUTING.md#the-qa-entry-schema).

## References (aggregated)

See [`references.md`](references.md).
