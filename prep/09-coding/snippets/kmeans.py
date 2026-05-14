"""K-means clustering from scratch in NumPy (Lloyd's algorithm).

Companion to the question "Implement k-means clustering from scratch" in
prep/09-coding/questions.md.

Run:
    python kmeans.py
"""
from __future__ import annotations

import numpy as np


def kmeans(X: np.ndarray, k: int, max_iters: int = 100, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    N, D = X.shape
    centroids = X[rng.choice(N, size=k, replace=False)].copy()
    labels = np.zeros(N, dtype=int)
    for _ in range(max_iters):
        d2 = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(-1)
        new_labels = d2.argmin(axis=1)
        if np.all(new_labels == labels):
            break
        labels = new_labels
        for j in range(k):
            mask = labels == j
            if mask.any():
                centroids[j] = X[mask].mean(axis=0)
            else:
                centroids[j] = X[rng.integers(N)]
    return labels, centroids


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X = np.vstack([
        rng.normal(loc=[0, 0], size=(100, 2)),
        rng.normal(loc=[5, 5], size=(100, 2)),
        rng.normal(loc=[-3, 4], size=(100, 2)),
    ])
    labels, centroids = kmeans(X, k=3, seed=1)
    print("centroids:")
    print(centroids)
    # Sanity check: 3 unique labels, centroids roughly at (0,0), (5,5), (-3,4)
    assert len(np.unique(labels)) == 3
    print("kmeans OK")
