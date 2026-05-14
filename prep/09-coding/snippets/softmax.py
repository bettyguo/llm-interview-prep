"""Numerically stable softmax in NumPy.

Companion to the question "Implement a numerically stable softmax in NumPy"
in prep/09-coding/questions.md.

Run:
    python softmax.py
"""
from __future__ import annotations

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / np.sum(e, axis=axis, keepdims=True)


def log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_max = np.max(x, axis=axis, keepdims=True)
    log_sum_exp = np.log(np.exp(x - x_max).sum(axis=axis, keepdims=True)) + x_max
    return x - log_sum_exp


if __name__ == "__main__":
    # Row 1: ordinary values; row 2: large values that would overflow naive softmax.
    x = np.array([[1.0, 2.0, 3.0], [1000.0, 1001.0, 1002.0]])
    p = softmax(x, axis=-1)
    assert np.allclose(p.sum(axis=-1), 1.0)
    assert np.all(np.isfinite(p))
    print("softmax:\n", p)

    lp = log_softmax(x, axis=-1)
    # log_softmax should satisfy log(sum(exp(log_softmax))) == 0
    assert np.allclose(np.log(np.exp(lp).sum(axis=-1)), 0.0, atol=1e-6)
    print("log_softmax OK")
