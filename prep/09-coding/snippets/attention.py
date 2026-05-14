"""Scaled dot-product attention from scratch in NumPy.

Companion to the question "Implement scaled dot-product attention from scratch in NumPy"
in prep/09-coding/questions.md.

Run:
    python attention.py
"""
from __future__ import annotations

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / np.sum(e, axis=axis, keepdims=True)


def attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, causal: bool = True) -> np.ndarray:
    """Scaled dot-product attention.

    Q, K, V have shape (B, H, T, D). Returns shape (B, H, T, D).
    """
    B, H, T, D = Q.shape
    scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(D)
    if causal:
        mask = np.triu(np.ones((T, T), dtype=bool), k=1)
        scores = np.where(mask, -np.inf, scores)
    attn = softmax(scores, axis=-1)
    return attn @ V


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    B, H, T, D = 2, 4, 8, 16
    Q, K, V = (rng.normal(size=(B, H, T, D)) for _ in range(3))
    out = attention(Q, K, V, causal=True)
    assert out.shape == (B, H, T, D)
    print("attention output shape:", out.shape)
    # Sanity check: with causal mask, attention is lower-triangular in T x T attn weights
    # (computed inside; not returned). The output is well-defined and finite.
    assert np.all(np.isfinite(out))
    print("attention OK")
