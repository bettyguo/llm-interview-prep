# ML / AI Coding Questions

Coding questions specific to AI/ML interview loops. Each entry has an `Implementation` block with a working code snippet. Runnable versions live under [`snippets/`](snippets/).

---

### Q: Implement a numerically stable softmax in NumPy.

**Category:** coding
**Difficulty:** intro
**Tags:** [softmax, numerical-stability, numpy]

**Short answer.** Subtract the row-wise max before exponentiating: `exp(x − max(x)) / sum(exp(x − max(x)))`. This shifts logits into a range where `exp` doesn't overflow (max becomes 0), without changing the result mathematically — the constant factor cancels in the ratio.

**Expansion / why this is the answer.**
- Naive `exp(x) / sum(exp(x))` overflows for large `x` (e.g. `x = 1000` → `inf`).
- Mathematically: `softmax(x − c) = softmax(x)` for any constant `c` (the `exp(−c)` factor in numerator and denominator cancels). Pick `c = max(x)` so the largest exponent is 0.
- Implementation:
  - 1-D: `x_shift = x - max(x); return exp(x_shift) / exp(x_shift).sum()`.
  - 2-D row-wise: subtract per-row max with `keepdims=True`.
- Same trick used inside FlashAttention's online softmax.

**Common follow-ups.**
- "What's `logsumexp`?" → `log(Σ exp(x_i)) = max(x) + log(Σ exp(x_i − max(x)))`. Same trick, gives the log-partition.
- "Numerical stability of cross-entropy?" → Use `log_softmax(x)` and then index, never `log(softmax(x))`.

**Common mistakes.**
- Forgetting `keepdims=True` on the 2-D version.
- Subtracting global max instead of per-row max (correct but unusual).

**Implementation.**
```python
import numpy as np

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax along `axis`."""
    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / np.sum(e, axis=axis, keepdims=True)


if __name__ == "__main__":
    x = np.array([[1.0, 2.0, 3.0], [1000.0, 1001.0, 1002.0]])
    p = softmax(x, axis=-1)
    # Row sums to 1; row 2 doesn't overflow.
    assert np.allclose(p.sum(axis=-1), 1.0)
    assert np.all(np.isfinite(p))
    print(p)
```

---

### Q: Implement scaled dot-product attention from scratch in NumPy.

**Category:** coding
**Difficulty:** mid
**Tags:** [attention, numpy, transformer]

**Short answer.** Take `Q, K, V` of shape `(B, H, T, D)`. Compute `scores = Q @ Kᵀ / √D`, apply causal mask (set future positions to `-inf`), softmax, multiply by `V`.

**Expansion / why this is the answer.**
- Match the canonical formula: `Attention(Q,K,V) = softmax(QKᵀ/√d_k) V`.
- Causal mask: a lower-triangular `(T, T)` matrix; set upper-triangle scores to `-inf` before softmax.
- Broadcasting handles the batch and head dims.

**Common follow-ups.**
- "Where would you use this kernel in practice?" → Toy demo only. Real systems call `scaled_dot_product_attention` which dispatches to FlashAttention.
- "Multi-head — what changes?" → Just an extra reshape: project to `(B, T, H*D)`, view as `(B, T, H, D)`, transpose to `(B, H, T, D)`, run this, undo.

**Common mistakes.**
- Forgetting the `/ √D` scaling.
- Causal mask off-by-one — `triu(..., k=1)` is correct (mask strictly above diagonal).
- Computing softmax over the wrong axis.

**Implementation.**
```python
import numpy as np

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / np.sum(e, axis=axis, keepdims=True)


def attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, causal: bool = True) -> np.ndarray:
    """Scaled dot-product attention.

    Q, K, V have shape (B, H, T, D). Returns shape (B, H, T, D).
    """
    B, H, T, D = Q.shape
    scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(D)  # (B, H, T, T)
    if causal:
        mask = np.triu(np.ones((T, T), dtype=bool), k=1)  # strictly above diagonal
        scores = np.where(mask, -np.inf, scores)
    attn = softmax(scores, axis=-1)
    return attn @ V  # (B, H, T, D)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    B, H, T, D = 2, 4, 8, 16
    Q, K, V = (rng.normal(size=(B, H, T, D)) for _ in range(3))
    out = attention(Q, K, V, causal=True)
    assert out.shape == (B, H, T, D)
    print(out.shape)
```

---

### Q: Implement LayerNorm in NumPy.

**Category:** coding
**Difficulty:** intro
**Tags:** [layernorm, normalization, numpy]

**Short answer.** Compute per-sample mean and variance over the feature dimension, normalize, then scale and shift: `y = γ · (x − μ) / √(σ² + ε) + β`.

**Expansion / why this is the answer.**
- Per-sample, per-position normalization across the feature dim — independent of batch size.
- `ε` (typically `1e-5`) prevents division by zero on near-constant features.
- Learnable `γ` (gain) and `β` (bias) restore representational capacity.

**Common follow-ups.**
- "RMSNorm difference?" → Skip the mean subtraction; divide by RMS only.
- "Why is the `ε` inside the sqrt?" → Numerical stability — sqrt(0) is fine but division by 0 is not.

**Common mistakes.**
- Computing var across batch (that's BatchNorm).
- Forgetting `keepdims` on the mean / var.

**Implementation.**
```python
import numpy as np

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """LayerNorm over the last axis. x: (..., D), gamma & beta: (D,)."""
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    x_hat = (x - mu) / np.sqrt(var + eps)
    return gamma * x_hat + beta


def rms_norm(x: np.ndarray, gamma: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """RMSNorm — LayerNorm without mean-subtraction."""
    rms = np.sqrt((x * x).mean(axis=-1, keepdims=True) + eps)
    return gamma * x / rms


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2, 4, 8))
    gamma = np.ones(8)
    beta = np.zeros(8)
    y = layer_norm(x, gamma, beta)
    # After LayerNorm with γ=1, β=0, the last-axis mean ≈ 0 and var ≈ 1.
    assert np.allclose(y.mean(axis=-1), 0.0, atol=1e-6)
    assert np.allclose(y.var(axis=-1), 1.0, atol=1e-4)
    print("LayerNorm OK")
```

---

### Q: Implement nucleus (top-p) sampling.

**Category:** coding
**Difficulty:** mid
**Tags:** [sampling, top-p, decoding]

**Short answer.** Sort the probability vector descending; take the cumulative sum; find the smallest prefix whose cumulative probability ≥ `p`; zero out everything after; renormalize; sample.

**Expansion / why this is the answer.**
- Steps:
  1. Optionally apply temperature: `logits /= T`.
  2. Compute `probs = softmax(logits)`.
  3. Sort descending; track original indices.
  4. Cumulative sum.
  5. Mask out positions where cumulative > `p` (but keep the first one that crosses).
  6. Renormalize the remaining; sample.
- Edge case: include the first token whose cumulative exceeds `p` (so we always have at least one token to sample).

**Common follow-ups.**
- "Combine with top-k?" → Apply top-k first (truncate to top-k), then top-p on the truncated. Belt-and-suspenders.
- "Temperature placement?" → Before softmax (divides logits).

**Common mistakes.**
- Excluding the first token that crosses `p` — leaves you with `< p` mass; can be empty for very peaked distributions.
- Sampling from sorted indices without inverting the sort.

**Implementation.**
```python
import numpy as np

def top_p_sample(logits: np.ndarray, p: float = 0.9, temperature: float = 1.0, rng: np.random.Generator | None = None) -> int:
    """Nucleus (top-p) sample one token from a 1-D logits vector."""
    rng = rng or np.random.default_rng()
    logits = logits / max(temperature, 1e-6)
    # Stable softmax
    probs = np.exp(logits - logits.max())
    probs = probs / probs.sum()
    # Sort descending; cumulative sum
    sorted_idx = np.argsort(-probs)
    sorted_probs = probs[sorted_idx]
    cum = np.cumsum(sorted_probs)
    # Cut off at the first index that exceeds p (inclusive)
    cutoff = np.searchsorted(cum, p) + 1
    cutoff = min(cutoff, len(probs))
    nucleus_idx = sorted_idx[:cutoff]
    nucleus_probs = sorted_probs[:cutoff]
    nucleus_probs = nucleus_probs / nucleus_probs.sum()
    return int(rng.choice(nucleus_idx, p=nucleus_probs))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    logits = rng.normal(size=10)
    tok = top_p_sample(logits, p=0.9, temperature=0.7, rng=rng)
    print("sampled token id:", tok)
```

---

### Q: Implement k-means clustering from scratch.

**Category:** coding
**Difficulty:** mid
**Tags:** [k-means, clustering, numpy]

**Short answer.** Initialize `k` centroids (e.g. random points from the data). Iterate: assign each point to its nearest centroid; recompute centroids as the mean of their assigned points; stop when assignments stop changing (or max-iters).

**Expansion / why this is the answer.**
- Lloyd's algorithm — alternating between assignment and update.
- `k-means++` initialization (Arthur & Vassilvitskii 2007) usually beats random; picks centroids spread apart with probability ∝ squared distance.
- Convergence: local minimum of `Σ ||x_i − μ_{c_i}||²` (the within-cluster sum of squares).

**Common follow-ups.**
- "k-means++?" → Bias initial centroids to be spread out.
- "When does k-means fail?" → Non-globular clusters, very different cluster densities, picking wrong `k`.

**Common mistakes.**
- Computing distances in a loop (slow); broadcast `(N, 1, D) − (1, K, D)`.
- Not handling empty clusters (centroid receives no points).

**Implementation.**
```python
import numpy as np

def kmeans(X: np.ndarray, k: int, max_iters: int = 100, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Lloyd's algorithm. X: (N, D). Returns (labels, centroids)."""
    rng = np.random.default_rng(seed)
    N, D = X.shape
    centroids = X[rng.choice(N, size=k, replace=False)].copy()
    labels = np.zeros(N, dtype=int)
    for _ in range(max_iters):
        # Assign: compute squared distance to each centroid
        d2 = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(-1)  # (N, k)
        new_labels = d2.argmin(axis=1)
        if np.all(new_labels == labels):
            break
        labels = new_labels
        # Update: mean of each cluster
        for j in range(k):
            mask = labels == j
            if mask.any():
                centroids[j] = X[mask].mean(axis=0)
            # Empty cluster → re-seed at a random point
            else:
                centroids[j] = X[rng.integers(N)]
    return labels, centroids


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(loc=[0, 0], size=(100, 2)),
                   rng.normal(loc=[5, 5], size=(100, 2)),
                   rng.normal(loc=[-3, 4], size=(100, 2))])
    labels, centroids = kmeans(X, k=3, seed=1)
    print("centroids:\n", centroids)
```

---

### Q: Implement cosine similarity top-k retrieval over an embedding matrix.

**Category:** coding
**Difficulty:** intro
**Tags:** [retrieval, cosine-similarity, top-k]

**Short answer.** Normalize embeddings to unit length; the dot product *is* cosine similarity. For a query vector and a matrix of `(N, D)` doc embeddings, compute `scores = docs @ query`; take the top-k via `argpartition` (faster than full sort).

**Expansion / why this is the answer.**
- Normalization: `x / ||x||`. After normalization, `cos(a, b) = a · b`.
- Top-k via `argpartition` is `O(N)`; `argsort` is `O(N log N)`.
- For very large N, switch to an ANN index (HNSW/IVF) — this is the brute-force baseline.

**Common follow-ups.**
- "When use brute force?" → <1M vectors, latency tolerable, or as ground-truth eval against an ANN index.
- "Batch queries?" → Compute `(Q · D.T)` for a query matrix; broadcast.

**Common mistakes.**
- Forgetting to normalize; dot product ≠ cosine without normalization.
- Using `argsort` then slicing (wasteful at large N).

**Implementation.**
```python
import numpy as np

def normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(X, axis=-1, keepdims=True)
    return X / np.maximum(n, eps)


def topk_cosine(query: np.ndarray, docs: np.ndarray, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Return (top_k_indices, top_k_scores) for one query vector over an (N, D) doc matrix."""
    q = normalize(query)
    d = normalize(docs)
    scores = d @ q  # (N,)
    if k >= len(scores):
        order = np.argsort(-scores)
        return order, scores[order]
    # argpartition is O(N); then sort just the top-k
    part = np.argpartition(-scores, kth=k)[:k]
    order = part[np.argsort(-scores[part])]
    return order, scores[order]


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    docs = rng.normal(size=(1000, 64))
    query = rng.normal(size=64)
    idx, scores = topk_cosine(query, docs, k=5)
    print("top-5 idx:", idx, "scores:", scores)
```

---

### Q: Implement a single transformer block in PyTorch.

**Category:** coding
**Difficulty:** mid
**Tags:** [transformer, pytorch, block]

**Short answer.** Pre-norm: `x = x + attn(LN(x))`; `x = x + ffn(LN(x))`. Use PyTorch's `scaled_dot_product_attention` (FlashAttention-fused when available); FFN is a simple 2-layer MLP with GELU.

**Expansion / why this is the answer.**
- Modern pre-norm structure (LN → sublayer → residual).
- `F.scaled_dot_product_attention` calls into FlashAttention when conditions are met.
- The attention's QKV projections + output projection account for ⅓ of params; FFN for ⅔ (when `d_ff = 4·d_model`).

**Common follow-ups.**
- "Add causal mask?" → Pass `is_causal=True` to `scaled_dot_product_attention`.
- "Use RMSNorm?" → Swap `LayerNorm` for an `RMSNorm` module.

**Common mistakes.**
- Post-norm structure (works, but trains less stably at depth).
- Forgetting bias on the FFN projections (typically present in pre-LLaMA; omitted in LLaMA-style).

**Implementation.**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int = 512, n_heads: int = 8, d_ff: int | None = None, dropout: float = 0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.ln1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.attn_out = nn.Linear(d_model, d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

    def _attn(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # (B, H, T, Dh)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # FlashAttention when available
        y = y.transpose(1, 2).contiguous().view(B, T, D)
        return self.attn_out(y)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self._attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


if __name__ == "__main__":
    block = TransformerBlock(d_model=128, n_heads=8)
    x = torch.randn(2, 16, 128)
    y = block(x)
    assert y.shape == x.shape
    print("OK", y.shape)
```

---

### Q: Implement a minimal training loop with mixed precision and gradient accumulation.

**Category:** coding
**Difficulty:** mid
**Tags:** [training-loop, mixed-precision, gradient-accumulation]

**Short answer.** Use `torch.amp.autocast` to run forward in bf16/fp16; scale loss by `1/accum_steps`; backward each micro-batch; step the optimizer every `accum_steps` micro-batches. With `bf16`, no `GradScaler` is needed (fp16 needs one).

**Expansion / why this is the answer.**
- **Mixed precision**: forward/backward in bf16, optimizer state in fp32 (PyTorch's `torch.amp` handles this).
- **Gradient accumulation**: simulate a larger effective batch when the GPU can't hold the full batch.
- **Scaling**: divide the loss by `accum_steps` so accumulated gradients average rather than sum.
- **Clipping**: `clip_grad_norm_` after accumulation, before optimizer step.

**Common follow-ups.**
- "Why not fp16?" → bf16 has wider dynamic range; no loss scaling needed. Use it on bf16-capable GPUs (Ampere+).
- "When does accumulation help vs. hurt?" → Helps when GPU memory binds batch size. Hurts perf slightly because forward+backward count more per optimizer step.

**Common mistakes.**
- Forgetting to zero grads after the step.
- Stepping every micro-batch instead of every `accum_steps`.
- Using fp16 without `GradScaler` (gradients underflow).

**Implementation.**
```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def train_step(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, *,
               accum_steps: int = 4, max_grad_norm: float = 1.0, device: str = "cuda"):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for step, (x, y) in enumerate(loader, start=1):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(x)
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            loss = loss / accum_steps
        loss.backward()
        if step % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
```

---

### Q: Implement causal mask for attention.

**Category:** coding
**Difficulty:** intro
**Tags:** [causal-mask, attention]

**Short answer.** A lower-triangular mask: `mask = tril(ones(T, T))`; set `mask == 0` positions in the score matrix to `-inf` before softmax. After softmax, those positions are 0, so each position only attends to its past and itself.

**Expansion / why this is the answer.**
- During decoder training, position `i` must not see positions `j > i` (or the LM cheats).
- `numpy.tril` / `torch.tril` returns the lower-triangular part (including the diagonal); `triu(..., k=1)` returns the strictly-upper part (positions to mask).
- PyTorch's `scaled_dot_product_attention(..., is_causal=True)` does this without materializing the mask.

**Common follow-ups.**
- "What about padding masks?" → Combine with the causal mask: any padded key position is also `-inf`.
- "Why `-inf` and not `0`?" → After softmax, `exp(-inf) = 0`; using `0` would still let the position contribute.

**Common mistakes.**
- `triu` vs. `tril` confusion — `triu(..., k=1)` is what to mask out (strictly above diag).
- Setting mask to `0.0` rather than `-inf` (or, in PyTorch, `float('-inf')`).

**Implementation.**
```python
import torch

def causal_mask(T: int, device: str = "cpu") -> torch.Tensor:
    """Returns (T, T) tensor with 0s on/below diagonal and -inf strictly above."""
    return torch.where(
        torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1),
        torch.tensor(float("-inf"), device=device),
        torch.tensor(0.0, device=device),
    )


if __name__ == "__main__":
    m = causal_mask(5)
    print(m)
```

---

### Q: Implement a toy BPE tokenizer trainer.

**Category:** coding
**Difficulty:** senior
**Tags:** [bpe, tokenizer, training]

**Short answer.** Start with the byte-level (or character) alphabet. Count pair frequencies in the corpus. Merge the most-frequent pair into a new token; repeat. Track merges in order; that's the tokenizer.

**Expansion / why this is the answer.**
- The core algorithm in 30 lines; the production implementations (tiktoken, sentencepiece) are pickier about Unicode normalization, regex pre-tokenization, etc.
- Output: an ordered list of `(a, b) → new_token` merges that can be applied greedily to new text.

**Common follow-ups.**
- "Why pre-split into words first?" → Prevents merges that cross word boundaries; common in BPE for English.
- "Encoding new text?" → Apply merges in their training order, greedily.

**Common mistakes.**
- Not tracking which pair counts to update after a merge (only the pairs involving the new token's neighbors need recount).
- Merging on byte boundaries when the source is Unicode (use byte-level).

**Implementation.**
```python
from collections import Counter

def get_pair_counts(words: dict[tuple[str, ...], int]) -> Counter:
    counts: Counter[tuple[str, str]] = Counter()
    for word, freq in words.items():
        for i in range(len(word) - 1):
            counts[(word[i], word[i + 1])] += freq
    return counts


def merge_pair(words: dict[tuple[str, ...], int], pair: tuple[str, str]) -> dict[tuple[str, ...], int]:
    merged_token = "".join(pair)
    new: dict[tuple[str, ...], int] = {}
    for word, freq in words.items():
        out: list[str] = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
                out.append(merged_token)
                i += 2
            else:
                out.append(word[i])
                i += 1
        new[tuple(out)] = new.get(tuple(out), 0) + freq
    return new


def train_bpe(corpus: list[str], n_merges: int = 100) -> tuple[list[tuple[str, str]], set[str]]:
    """Train a toy character-level BPE on a corpus of strings."""
    # Initial: each word becomes a tuple of single chars, with a special end-of-word marker
    words: dict[tuple[str, ...], int] = Counter(tuple(w) + ("</w>",) for w in corpus)
    merges: list[tuple[str, str]] = []
    for _ in range(n_merges):
        pairs = get_pair_counts(words)
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        merges.append(best)
        words = merge_pair(words, best)
    vocab = {tok for w in words for tok in w}
    return merges, vocab


if __name__ == "__main__":
    corpus = ["hello world hello world", "hello there", "world peace"]
    tokens = []
    for line in corpus:
        tokens.extend(line.split())
    merges, vocab = train_bpe(tokens, n_merges=20)
    print("merges:", merges[:5], "...")
    print("vocab size:", len(vocab))
```

---

### Q: Implement logistic regression with gradient descent in NumPy.

**Category:** coding
**Difficulty:** intro
**Tags:** [logistic-regression, gradient-descent, numpy]

**Short answer.** Forward: `p = σ(Xw + b)`. Loss: binary cross-entropy `-Σ[y log p + (1-y) log(1-p)] / N`. Gradient: `∂L/∂w = Xᵀ(p − y) / N`, `∂L/∂b = mean(p − y)`. Update: `w -= η · grad`.

**Expansion / why this is the answer.**
- The gradient `Xᵀ(p − y) / N` is famously clean — falls out of softmax+cross-entropy by analogy.
- For numerical stability, use log-sum-exp / log_softmax forms when computing the loss directly.
- This is the simplest model in DL: one linear layer + sigmoid.

**Common follow-ups.**
- "Derive the gradient." → Show: `L = -[y log σ(z) + (1-y) log(1-σ(z))]`. `∂L/∂z = σ(z) - y`. Chain to `w` and `b`.
- "Regularization?" → Add `+ λ ||w||²` to loss; gradient gains `+ 2λ w`.

**Common mistakes.**
- Forgetting `/ N`.
- Forgetting numerical stability when computing the loss.

**Implementation.**
```python
import numpy as np

def sigmoid(z: np.ndarray) -> np.ndarray:
    # Stable sigmoid
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    out[~pos] = np.exp(z[~pos]) / (1.0 + np.exp(z[~pos]))
    return out


def logreg_fit(X: np.ndarray, y: np.ndarray, lr: float = 0.1, n_iter: int = 1000, l2: float = 0.0):
    N, D = X.shape
    w = np.zeros(D)
    b = 0.0
    for _ in range(n_iter):
        p = sigmoid(X @ w + b)
        err = p - y
        gw = X.T @ err / N + 2 * l2 * w
        gb = err.mean()
        w -= lr * gw
        b -= lr * gb
    return w, b


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    N = 500
    X = rng.normal(size=(N, 3))
    true_w = np.array([1.5, -2.0, 0.5])
    y = (sigmoid(X @ true_w) > 0.5).astype(float)
    w, b = logreg_fit(X, y, lr=0.5, n_iter=2000)
    pred = (sigmoid(X @ w + b) > 0.5).astype(float)
    acc = (pred == y).mean()
    print(f"learned w = {w}, b = {b:.3f}, train acc = {acc:.3f}")
```

---

### Q: Compute Recall@k and MRR for a retrieval result.

**Category:** coding
**Difficulty:** intro
**Tags:** [retrieval-metrics, recall, mrr]

**Short answer.** `Recall@k`: fraction of queries whose top-k retrieved set contains *any* relevant doc. `MRR`: mean of `1/rank_of_first_relevant`, with reciprocal = 0 if no relevant in top-k.

**Expansion / why this is the answer.**
- Inputs: for each query, a ranked list of retrieved doc IDs + a set of relevant doc IDs.
- Recall@k is a binary signal per query — was anything relevant in the top-k?
- MRR captures rank position; first hit at rank 1 = 1.0, rank 2 = 0.5, ...

**Common follow-ups.**
- "When is MRR misleading?" → It ignores recall beyond rank 1; if most queries need to find 5 relevant docs, use nDCG.
- "What's nDCG?" → Discounted gain weighting; harder to implement (needs a normalizer per query).

**Common mistakes.**
- Off-by-one on ranks (rank starts at 1 conventionally).
- Including the same doc twice in the ranking.

**Implementation.**
```python
import numpy as np

def recall_at_k(retrieved: list[list[int]], relevant: list[set[int]], k: int = 10) -> float:
    """retrieved[i] = ranked list of doc IDs for query i. relevant[i] = set of relevant doc IDs."""
    hits = 0
    for r, rel in zip(retrieved, relevant):
        if any(d in rel for d in r[:k]):
            hits += 1
    return hits / len(retrieved)


def mrr(retrieved: list[list[int]], relevant: list[set[int]], k: int | None = None) -> float:
    """Mean reciprocal rank (within first k; if k=None, full list)."""
    total = 0.0
    for r, rel in zip(retrieved, relevant):
        sl = r if k is None else r[:k]
        rr = 0.0
        for rank, d in enumerate(sl, start=1):
            if d in rel:
                rr = 1.0 / rank
                break
        total += rr
    return total / len(retrieved)


if __name__ == "__main__":
    retrieved = [[3, 1, 2, 5, 9], [4, 2, 1, 7, 8]]
    relevant = [{1}, {7}]
    print("recall@5:", recall_at_k(retrieved, relevant, k=5))
    print("MRR:", mrr(retrieved, relevant, k=5))
```

---

### Q: Implement a sliding-window text chunker that respects token boundaries.

**Category:** coding
**Difficulty:** mid
**Tags:** [chunking, rag, tokenization]

**Short answer.** Tokenize the document; slide a window of `chunk_size` tokens with `chunk_overlap` overlap; decode each window back to text. The chunker's contract: never split mid-token.

**Expansion / why this is the answer.**
- Token boundaries matter: a chunker that splits on characters produces inconsistent token counts across chunks.
- Overlap of 10–20% protects against sentences split at chunk boundaries.
- For long documents, this is the simplest correct chunker.

**Common follow-ups.**
- "How to respect paragraph boundaries?" → Split on `\n\n` first; then apply window chunking inside each long paragraph.
- "Why overlap?" → If a sentence ends at chunk boundary, it's split between two chunks; overlap ensures the sentence appears whole in at least one.

**Common mistakes.**
- Chunking on character indices (token counts vary).
- No overlap; sentences at boundaries lost.

**Implementation.**
```python
def chunk_tokens(token_ids: list[int], chunk_size: int = 512, chunk_overlap: int = 64) -> list[list[int]]:
    """Sliding-window chunking on a token-ID stream. Returns list of token-ID chunks."""
    assert chunk_overlap < chunk_size, "overlap must be < chunk_size"
    chunks: list[list[int]] = []
    step = chunk_size - chunk_overlap
    i = 0
    while i < len(token_ids):
        chunks.append(token_ids[i : i + chunk_size])
        if i + chunk_size >= len(token_ids):
            break
        i += step
    return chunks


if __name__ == "__main__":
    fake_tokens = list(range(2000))
    chunks = chunk_tokens(fake_tokens, chunk_size=500, chunk_overlap=50)
    print("num chunks:", len(chunks), "first chunk len:", len(chunks[0]))
```

---

### Q: Implement Reciprocal Rank Fusion (RRF) for hybrid retrieval.

**Category:** coding
**Difficulty:** mid
**Tags:** [rrf, hybrid-retrieval, fusion]

**Short answer.** Given multiple ranked lists, score each document as `Σ_lists 1/(k + rank_in_list(d))`, then sort by total. `k=60` is the standard constant. Documents not in a list contribute 0 from that list.

**Expansion / why this is the answer.**
- Robust to score-scale mismatches (BM25 scores aren't comparable to cosine scores).
- The constant `k` prevents rank-1 documents from dominating; standard `k=60` (Cormack et al. 2009).
- Used in production to fuse dense + sparse retrieval.

**Common follow-ups.**
- "What about weighted RRF?" → Add per-list weights: `Σ w_i / (k + rank)`.
- "When does it fail?" → If one ranker is much better than the other on the query mix; consider learned fusion.

**Common mistakes.**
- Using rank 0 instead of rank 1 (off-by-one breaks the constant's effect).
- Not handling docs absent from one list.

**Implementation.**
```python
from collections import defaultdict

def rrf(rankings: list[list[int]], k: int = 60, top_k: int = 10) -> list[tuple[int, float]]:
    """rankings: list of ranked doc-ID lists (each list is best-first).
    Returns top_k (doc_id, score) pairs by RRF score, descending.
    """
    scores: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: -kv[1])[:top_k]


if __name__ == "__main__":
    dense = [3, 7, 1, 4, 5]
    bm25 = [1, 2, 4, 7, 9]
    fused = rrf([dense, bm25], k=60, top_k=5)
    print(fused)
```

---
