# 09 — ML / AI Coding Questions

The coding interviews specific to AI/ML roles. Not LeetCode — those are owned by `coding-interview-university`. These are the implementation challenges that AI loops actually use: implement attention, write a stable softmax, code a training loop, implement a sampler.

## What you should walk in able to do

- Implement **softmax with numerical stability** (subtract the max).
- Implement **scaled dot-product attention** in NumPy and PyTorch.
- Implement **LayerNorm** and **RMSNorm**.
- Write a minimal **transformer block** (attention + FFN + residuals + norm).
- Write a minimal **training loop** with gradient accumulation and mixed precision.
- Implement **k-means**, **kNN**, **logistic regression with gradient descent** from scratch.
- Implement **nucleus (top-p), top-k, and temperature sampling**.
- Implement **BPE tokenizer training** on a toy corpus.
- Implement **cosine similarity top-k retrieval** over an embedding matrix.

## Format

Each entry in [`questions.md`](questions.md) carries an `**Implementation.**` block with a working code snippet (verified to run). Runnable versions live under [`snippets/`](snippets/) so you can `python snippets/attention.py` and verify the output.

## Common bugs interviewers watch for

- Using `argmax` where you wanted `softmax`.
- In-place ops (`x += y`) that break autograd.
- Broadcasting traps (e.g. `(batch, 1) * (n,)` instead of `(batch, n)`).
- Forgetting the √d_k scale in attention.
- Forgetting the causal mask in decoder attention.
- Forgetting to detach the value when computing the loss baseline.
