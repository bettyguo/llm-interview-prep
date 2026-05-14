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

### Q: Implement the DPO loss in PyTorch.

**Category:** coding
**Difficulty:** senior
**Tags:** [dpo, preference-optimization, pytorch]

**Short answer.** For each preference triple `(x, y_w, y_l)`, compute the log-prob of `y_w` and `y_l` under both the policy and the reference; take the difference of log-ratios; apply `−log σ(β · (Δ_w − Δ_l))`. Standard PyTorch with `gather` to pick the correct token-id log-probs.

**Expansion / why this is the answer.**
- The math: `L_DPO = -log σ(β · [log π_θ(y_w|x)/π_ref(y_w|x) − log π_θ(y_l|x)/π_ref(y_l|x)])`.
- Token-level: sum log-probs over the response tokens (mask the prompt tokens).
- The reference model is frozen.

**Common follow-ups.**
- "Why mask the prompt log-probs?" → Identical between policy and ref for the prompt; only the response carries signal.
- "How do you avoid the 'decreased likelihood' DPO pathology?" → Add an SFT term: `L = L_DPO + α · L_SFT(y_w)`.

**Common mistakes.**
- Including the prompt tokens in the log-prob sum.
- Detaching the policy log-probs (gradients won't flow).
- Forgetting `torch.no_grad()` on the reference forward pass.

**Implementation.**
```python
import torch
import torch.nn.functional as F


def gather_logp(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-sample sum of log-probs over masked tokens.

    logits: (B, T, V), labels: (B, T), mask: (B, T) — 1 for response tokens, 0 elsewhere.
    Assumes logits[t] predicts labels[t] (already shifted).
    """
    logp = F.log_softmax(logits, dim=-1)
    per_token = logp.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)  # (B, T)
    return (per_token * mask).sum(dim=-1)  # (B,)


def dpo_loss(
    policy_logits_w: torch.Tensor, policy_logits_l: torch.Tensor,
    ref_logits_w: torch.Tensor, ref_logits_l: torch.Tensor,
    labels_w: torch.Tensor, labels_l: torch.Tensor,
    mask_w: torch.Tensor, mask_l: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    """Standard DPO loss (Rafailov et al. 2023).

    Each *_logits is shape (B, T, V); labels (B, T); mask (B, T) marks response tokens.
    Returns scalar loss.
    """
    pol_w = gather_logp(policy_logits_w, labels_w, mask_w)
    pol_l = gather_logp(policy_logits_l, labels_l, mask_l)
    with torch.no_grad():
        ref_w = gather_logp(ref_logits_w, labels_w, mask_w)
        ref_l = gather_logp(ref_logits_l, labels_l, mask_l)
    log_ratio_w = pol_w - ref_w
    log_ratio_l = pol_l - ref_l
    margin = beta * (log_ratio_w - log_ratio_l)
    return -F.logsigmoid(margin).mean()


if __name__ == "__main__":
    B, T, V = 2, 6, 100
    torch.manual_seed(0)
    pol_w = torch.randn(B, T, V, requires_grad=True)
    pol_l = torch.randn(B, T, V, requires_grad=True)
    ref_w = torch.randn(B, T, V)
    ref_l = torch.randn(B, T, V)
    labels_w = torch.randint(0, V, (B, T))
    labels_l = torch.randint(0, V, (B, T))
    mask_w = torch.ones(B, T)
    mask_l = torch.ones(B, T)
    loss = dpo_loss(pol_w, pol_l, ref_w, ref_l, labels_w, labels_l, mask_w, mask_l, beta=0.1)
    loss.backward()
    print(f"DPO loss = {loss.item():.4f}")
```

---

### Q: Implement a simple beam search.

**Category:** coding
**Difficulty:** mid
**Tags:** [beam-search, decoding]

**Short answer.** Maintain a min-heap of `(score, sequence)` for the top-K beams. At each step, expand every beam by every token, compute new cumulative log-prob, keep the top-K. Stop when all beams emit EOS or hit max length. Use length-normalized scoring to avoid the "shorter sequence wins" bias.

**Expansion / why this is the answer.**
- The standard beam-search algorithm; not used for open-ended LLM generation (favors generic continuations) but useful for translation / constrained decoding.
- Length normalization: divide cumulative log-prob by `(length^α)` with α ∈ [0.5, 1.0].

**Common follow-ups.**
- "Why is beam search worse than sampling for open text?" → Mode-seeking; produces "safe" generic output (Holtzman et al. 2019).
- "What's diverse beam search?" → Partition beams into groups; penalize within-group similarity.

**Common mistakes.**
- Not length-normalizing — short beams dominate.
- Sorting beams by raw score instead of normalized score.

**Implementation.**
```python
import math
import heapq
from typing import Callable


def beam_search(
    next_token_logprobs: Callable[[list[int]], list[float]],
    initial_token: int,
    eos_token: int,
    beam_size: int = 4,
    max_len: int = 50,
    length_penalty: float = 0.7,
) -> list[int]:
    """Toy beam search.

    next_token_logprobs(prefix) -> list[float] of log-probs over the vocab.
    """
    # Beams: list of (normalized_score, raw_logprob_sum, sequence, finished)
    beams: list[tuple[float, float, list[int], bool]] = [(0.0, 0.0, [initial_token], False)]
    for step in range(max_len):
        candidates: list[tuple[float, float, list[int], bool]] = []
        for norm_score, raw_score, seq, finished in beams:
            if finished:
                candidates.append((norm_score, raw_score, seq, finished))
                continue
            logprobs = next_token_logprobs(seq)
            # Top beam_size next tokens for this beam
            top = sorted(enumerate(logprobs), key=lambda kv: -kv[1])[:beam_size]
            for tok, lp in top:
                new_seq = seq + [tok]
                new_raw = raw_score + lp
                new_norm = new_raw / (len(new_seq) ** length_penalty)
                done = (tok == eos_token)
                candidates.append((new_norm, new_raw, new_seq, done))
        # Keep top beam_size by normalized score
        beams = sorted(candidates, key=lambda b: -b[0])[:beam_size]
        if all(b[3] for b in beams):
            break
    return beams[0][2]


if __name__ == "__main__":
    # Toy: simulate a model that always likes token 1, then 2, then 3, then EOS (=0).
    import random
    rng = random.Random(0)

    def fake_logprobs(prefix: list[int]) -> list[float]:
        VOCAB = 5
        preferred = (sum(prefix) + 1) % VOCAB  # deterministic but non-trivial
        out = [-5.0] * VOCAB
        out[preferred] = -0.1
        return out

    seq = beam_search(fake_logprobs, initial_token=4, eos_token=0, beam_size=3, max_len=10)
    print("beam result:", seq)
```

---

### Q: Implement min-p sampling.

**Category:** coding
**Difficulty:** mid
**Tags:** [sampling, min-p, decoding]

**Short answer.** Min-p sampling (Nguyen et al. 2024) keeps tokens with probability `≥ min_p · p_top`, where `p_top` is the largest probability. Adaptive: a peaked distribution keeps few tokens, a flat one keeps many. Less sensitive than top-p to long-tail noise.

**Expansion / why this is the answer.**
- Idea: instead of cumulative-probability threshold (top-p), use a relative-to-top threshold.
- More robust to distributions with a heavy tail of low-probability noise.
- Common params: `min_p = 0.05–0.1`; `T = 0.7–1.0`.

**Common follow-ups.**
- "When does min-p beat top-p?" → Long-tailed distributions where top-p includes many implausible tokens; min-p prunes them.

**Common mistakes.**
- Confusing `min_p` (relative threshold) with a fixed minimum probability.
- Forgetting to renormalize after pruning.

**Implementation.**
```python
import numpy as np


def min_p_sample(logits: np.ndarray, min_p: float = 0.05, temperature: float = 1.0, rng: np.random.Generator | None = None) -> int:
    rng = rng or np.random.default_rng()
    logits = logits / max(temperature, 1e-6)
    probs = np.exp(logits - logits.max())
    probs = probs / probs.sum()
    p_top = probs.max()
    threshold = min_p * p_top
    mask = probs >= threshold
    probs_kept = probs * mask
    probs_kept = probs_kept / probs_kept.sum()
    return int(rng.choice(len(probs), p=probs_kept))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    logits = rng.normal(size=20)
    tok = min_p_sample(logits, min_p=0.1, temperature=0.7, rng=rng)
    print("sampled token id:", tok)
```

---

### Q: Implement greedy and stochastic KV-cache update for autoregressive decode.

**Category:** coding
**Difficulty:** senior
**Tags:** [kv-cache, decode-step, autoregressive]

**Short answer.** Maintain `K_cache, V_cache` of shape `(B, H, T_cache, D)`. For each decode step: compute `Q, K_new, V_new` for the single new token; concatenate `K_new, V_new` into the cache; compute attention `softmax(Q · K_cacheᵀ / √D) · V_cache`; feed through the rest of the transformer; sample / argmax the next token. The cache grows by one position per step; never re-encode the prompt.

**Expansion / why this is the answer.**
- Key insight: at decode step `t`, only `Q` is recomputed (for the new position); `K_cache` and `V_cache` are reused.
- Memory cost: linear in T.
- Compute cost per step: `O(T · d)` for attention (vs. `O(T² · d)` if we re-attended).

**Common follow-ups.**
- "How does paged attention change this?" → Cache stored in pages; lookup via a block table; otherwise the same algorithm.
- "What's the GQA effect?" → `K_cache` and `V_cache` have shape `(B, n_kv_heads, T, d_head)` where `n_kv_heads < n_heads`.

**Common mistakes.**
- Re-encoding the full prompt every step.
- Forgetting to use causal mask (here implicit: there's only one query position, so no future to mask).

**Implementation.**
```python
import numpy as np


def attention_decode_step(
    q: np.ndarray,         # (B, H, 1, D)
    k_cache: np.ndarray,   # (B, H, T, D)
    v_cache: np.ndarray,   # (B, H, T, D)
    k_new: np.ndarray,     # (B, H, 1, D)
    v_new: np.ndarray,     # (B, H, 1, D)
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One decode step: append k_new/v_new to cache, attend over the full cache.

    Returns (output, updated_k_cache, updated_v_cache).
    """
    k_cache = np.concatenate([k_cache, k_new], axis=2)
    v_cache = np.concatenate([v_cache, v_new], axis=2)
    D = q.shape[-1]
    scores = q @ k_cache.transpose(0, 1, 3, 2) / np.sqrt(D)  # (B, H, 1, T+1)
    # No causal mask: only one query position, attending to past + itself.
    scores_max = scores.max(axis=-1, keepdims=True)
    attn = np.exp(scores - scores_max)
    attn = attn / attn.sum(axis=-1, keepdims=True)
    out = attn @ v_cache  # (B, H, 1, D)
    return out, k_cache, v_cache


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    B, H, T, D = 1, 4, 5, 16
    # Empty cache initially
    k_cache = np.zeros((B, H, 0, D))
    v_cache = np.zeros((B, H, 0, D))
    for step in range(T):
        q = rng.normal(size=(B, H, 1, D))
        k_new = rng.normal(size=(B, H, 1, D))
        v_new = rng.normal(size=(B, H, 1, D))
        out, k_cache, v_cache = attention_decode_step(q, k_cache, v_cache, k_new, v_new)
    print("after", T, "steps; k_cache shape:", k_cache.shape)
    assert k_cache.shape == (B, H, T, D)
    print("KV cache OK")
```

---

### Q: Implement gradient accumulation manually (without `torch.amp` magic).

**Category:** coding
**Difficulty:** mid
**Tags:** [gradient-accumulation, training-loop, pytorch]

**Short answer.** Divide the loss by `accum_steps`; backward each micro-batch (gradients accumulate in `.grad`); call `optimizer.step()` once every `accum_steps`; zero gradients after each step. Effective batch = `micro_batch × accum_steps × num_GPUs`.

**Expansion / why this is the answer.**
- PyTorch accumulates gradients across backward calls by default; this is what makes the technique simple.
- Why divide loss by accum_steps: to keep the gradient magnitude equivalent to one big-batch forward pass.

**Common follow-ups.**
- "How does this differ from DDP-style data parallel?" → Accumulation simulates a bigger batch on one GPU; DDP runs the bigger batch across GPUs with gradient all-reduce per step.

**Common mistakes.**
- Not dividing loss by accum_steps (effective gradient is `accum_steps`× too large).
- Stepping every micro-batch instead of every `accum_steps`.

**Implementation.**
```python
import torch
import torch.nn as nn


def train_one_epoch(model: nn.Module, loader, optimizer, *, accum_steps: int = 4, device: str = "cuda"):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for step, (x, y) in enumerate(loader, start=1):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        # Scale so accumulated gradients average instead of summing
        (loss / accum_steps).backward()
        if step % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
```

---

### Q: Implement a sliding-window attention from scratch.

**Category:** coding
**Difficulty:** senior
**Tags:** [sliding-window, attention, mistral]

**Short answer.** Same scaled dot-product attention but mask out scores where `|i - j| > window` (or for causal SWA, `i - j > window` or `j > i`). At inference, the KV cache can be capped at `window` tokens.

**Expansion / why this is the answer.**
- Mistral 7B's SWA: causal + window. Mask anything where `j > i` (future) or `j < i - window + 1` (too far in the past).
- Memory: KV cache truncated to `window`.

**Common follow-ups.**
- "What about the KV-cache implication?" → Cache size capped at `window`; older entries evicted.
- "How does this interact with RoPE?" → RoPE positions are absolute; SWA caps the attention range, not the position encoding.

**Common mistakes.**
- Off-by-one in window boundary.
- Forgetting the causal mask (this is *causal* SWA).

**Implementation.**
```python
import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / np.sum(e, axis=axis, keepdims=True)


def swa_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, window: int) -> np.ndarray:
    """Causal sliding-window attention. Q, K, V: (B, H, T, D)."""
    B, H, T, D = Q.shape
    scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(D)  # (B, H, T, T)
    # Build mask: position i can attend to j iff i - window + 1 <= j <= i.
    i = np.arange(T)[:, None]
    j = np.arange(T)[None, :]
    allowed = (j <= i) & (j > i - window)
    mask = ~allowed  # True where we mask out
    scores = np.where(mask, -np.inf, scores)
    attn = softmax(scores, axis=-1)
    return attn @ V


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    B, H, T, D = 1, 2, 8, 16
    Q, K, V = (rng.normal(size=(B, H, T, D)) for _ in range(3))
    out = swa_attention(Q, K, V, window=3)
    assert out.shape == (B, H, T, D)
    print("SWA output shape:", out.shape, "(window=3)")
```

---

### Q: Implement RoPE (Rotary Position Embedding) from scratch.

**Category:** coding
**Difficulty:** senior
**Tags:** [rope, positional-encoding, numpy]

**Short answer.** For each pair of dimensions `(2i, 2i+1)` in Q and K, rotate by an angle `m · θ_i` where `m` is the position and `θ_i = 10000^(-2i/d)`. Apply before the attention dot product.

**Expansion / why this is the answer.**
- Compute `θ_i` once per dimension; multiply by positions for the angle matrix.
- Reshape Q into `(..., d/2, 2)` complex-like pairs; apply rotation.

**Common follow-ups.**
- "Why pairs?" → 2D rotations; each pair is rotated by the position-dependent angle.

**Common mistakes.**
- Forgetting to rotate K as well as Q.

**Implementation.**
```python
import numpy as np

def apply_rope(x: np.ndarray, base: float = 10000.0) -> np.ndarray:
    """Apply RoPE to x of shape (..., T, D). D must be even."""
    *prefix, T, D = x.shape
    assert D % 2 == 0
    half = D // 2
    theta = base ** (-np.arange(0, D, 2) / D)  # (half,)
    positions = np.arange(T)  # (T,)
    angles = np.outer(positions, theta)  # (T, half)
    cos = np.cos(angles)
    sin = np.sin(angles)
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    rot1 = x1 * cos - x2 * sin
    rot2 = x1 * sin + x2 * cos
    out = np.empty_like(x)
    out[..., 0::2] = rot1
    out[..., 1::2] = rot2
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    q = rng.normal(size=(2, 4, 8, 16))
    q_rot = apply_rope(q)
    assert q_rot.shape == q.shape
    print("RoPE OK")
```

---

### Q: Implement BPE encoding (apply pretrained merges to text).

**Category:** coding
**Difficulty:** mid
**Tags:** [bpe, tokenization, encoding]

**Short answer.** Split text into characters; iteratively apply learned merges in their training order; stop when no more merges apply. The order matters — apply highest-priority (earliest-learned) merges first.

**Expansion / why this is the answer.**
- Tokenizer state: list of merges in order.
- Algorithm: for each word, repeatedly find the highest-priority adjacent pair that's in the merges dict; merge.

**Common follow-ups.**
- "Why is encoding order critical?" → Merging "a" + "b" before "ab" + "c" yields different results.

**Common mistakes.**
- Apply merges by frequency at encode time (use training order instead).

**Implementation.**
```python
def encode_bpe(text: str, merges: list[tuple[str, str]]) -> list[str]:
    merge_rank = {pair: i for i, pair in enumerate(merges)}
    tokens = list(text) + ["</w>"]
    while True:
        # Find best (lowest-rank) adjacent pair
        best_rank = None
        best_idx = None
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            if pair in merge_rank:
                if best_rank is None or merge_rank[pair] < best_rank:
                    best_rank = merge_rank[pair]
                    best_idx = i
        if best_idx is None:
            break
        # Merge
        tokens = tokens[:best_idx] + [tokens[best_idx] + tokens[best_idx + 1]] + tokens[best_idx + 2:]
    return tokens


if __name__ == "__main__":
    merges = [("h", "e"), ("he", "l"), ("hel", "l"), ("hell", "o")]
    print(encode_bpe("hello", merges))  # ['hello', '</w>']
```

---

### Q: Implement a small autoregressive sampling loop in PyTorch.

**Category:** coding
**Difficulty:** mid
**Tags:** [autoregressive, generation, pytorch]

**Short answer.** Initialize with prompt tokens; loop: forward pass on current sequence; take last-position logits; apply temperature + top-p / top-k; sample; append; stop on EOS or max length.

**Expansion / why this is the answer.**
- Use KV cache in real implementations; for clarity here, re-encode each step.

**Common follow-ups.**
- "How to use KV cache?" → Pass past_key_values; only encode the new token.

**Common mistakes.**
- Forgetting to detach / no_grad during inference.

**Implementation.**
```python
import torch
import torch.nn.functional as F


def generate(model, input_ids: torch.Tensor, max_new_tokens: int = 50, temperature: float = 0.7, top_p: float = 0.9, eos_id: int = 0) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_ids)[:, -1, :] / max(temperature, 1e-6)
            probs = F.softmax(logits, dim=-1)
            # Top-p truncation
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cum = torch.cumsum(sorted_probs, dim=-1)
            cutoff = (cum > top_p).float().argmax(dim=-1, keepdim=True) + 1
            mask = torch.arange(sorted_probs.size(-1)).unsqueeze(0) < cutoff
            sorted_probs = sorted_probs * mask.float()
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
            # Sample
            sampled_sorted = torch.multinomial(sorted_probs, num_samples=1)
            sampled = sorted_idx.gather(-1, sampled_sorted)
            input_ids = torch.cat([input_ids, sampled], dim=-1)
            if sampled.item() == eos_id:
                break
    return input_ids
```

---

### Q: Implement chunking with overlap that respects sentence boundaries.

**Category:** coding
**Difficulty:** mid
**Tags:** [chunking, sentence-boundary]

**Short answer.** Split text on sentence boundaries (e.g. `. `, `! `, `? `); pack sentences into chunks until target token budget reached; overlap by including the last K tokens of the previous chunk in the next.

**Common follow-ups.**
- "Sentence boundary detection?" → Use nltk / spacy for robust detection; regex is brittle.

**Common mistakes.**
- Splitting mid-sentence; embedding loses local context.

**Implementation.**
```python
import re


def split_sentences(text: str) -> list[str]:
    # Naive; for production use nltk/spacy.
    return re.split(r"(?<=[.!?])\s+", text.strip())


def chunk_text_sentences(text: str, max_chars: int = 1000, overlap_chars: int = 100) -> list[str]:
    sentences = split_sentences(text)
    chunks: list[str] = []
    current = ""
    for sent in sentences:
        if len(current) + len(sent) + 1 > max_chars and current:
            chunks.append(current.strip())
            # Overlap: keep last `overlap_chars`
            current = current[-overlap_chars:] + " " + sent
        else:
            current = (current + " " + sent).strip()
    if current:
        chunks.append(current.strip())
    return chunks


if __name__ == "__main__":
    text = "First sentence. Second one. Third! Fourth? Fifth and final."
    chunks = chunk_text_sentences(text, max_chars=30, overlap_chars=5)
    for c in chunks:
        print(repr(c))
```

---

### Q: Implement reciprocal-rank-fusion (RRF) for >2 rankings.

**Category:** coding
**Difficulty:** intro
**Tags:** [rrf, hybrid-retrieval]

**Short answer.** Already covered above in `rrf.py` snippet; this entry generalizes for arbitrary number of rankings with weighted variant.

**Common follow-ups.**
- "Weighted RRF?" → Each ranking has a weight; sum `w_i / (k + rank)`.

**Common mistakes.**
- Off-by-one on rank.

**Implementation.**
```python
from collections import defaultdict


def weighted_rrf(rankings_with_weights: list[tuple[list[int], float]], k: int = 60, top_k: int = 10) -> list[tuple[int, float]]:
    scores: dict[int, float] = defaultdict(float)
    for ranking, weight in rankings_with_weights:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] += weight / (k + rank)
    return sorted(scores.items(), key=lambda kv: -kv[1])[:top_k]


if __name__ == "__main__":
    dense = ([3, 7, 1, 4, 5], 1.0)
    bm25 = ([1, 2, 4, 7, 9], 1.5)  # bm25 weighted higher
    fused = weighted_rrf([dense, bm25], k=60, top_k=5)
    print(fused)
```

---

### Q: Implement a simple Bradley-Terry rating fit from pairwise preferences.

**Category:** coding
**Difficulty:** senior
**Tags:** [bradley-terry, elo, pairwise]

**Short answer.** Given pairs `(winner, loser)`, maximum-likelihood fit Bradley-Terry ratings: `P(i beats j) = σ(r_i − r_j)`. Use gradient descent or scipy's optimizer.

**Common follow-ups.**
- "How does this relate to Elo?" → Logistic Elo is mathematically Bradley-Terry.

**Common mistakes.**
- Forgetting to fix one rating as the gauge (otherwise solution is identifiable only up to a constant).

**Implementation.**
```python
import numpy as np
from scipy.optimize import minimize


def fit_bt(pairs: list[tuple[int, int]], n_items: int) -> np.ndarray:
    """pairs: list of (winner, loser) item IDs. Returns ratings vector."""
    def neg_log_lik(r: np.ndarray) -> float:
        nll = 0.0
        for w, l in pairs:
            diff = r[w] - r[l]
            nll -= np.log(1.0 / (1.0 + np.exp(-diff)))
        return nll

    # Gauge: rating[0] = 0
    x0 = np.zeros(n_items - 1)

    def wrapped(x):
        r = np.concatenate([[0.0], x])
        return neg_log_lik(r)

    res = minimize(wrapped, x0, method="L-BFGS-B")
    return np.concatenate([[0.0], res.x])


if __name__ == "__main__":
    # Item 0 beats 1; item 1 beats 2; etc.
    pairs = [(0, 1), (0, 1), (1, 2), (1, 2), (0, 2)]
    r = fit_bt(pairs, n_items=3)
    print("ratings:", r)
```

---

### Q: Implement Cohen's kappa for inter-rater agreement.

**Category:** coding
**Difficulty:** mid
**Tags:** [kappa, agreement, evaluation]

**Short answer.** Cohen's κ = `(p_o − p_e) / (1 − p_e)` where `p_o` is observed agreement and `p_e` is chance agreement. > 0.6 typically considered substantial.

**Common follow-ups.**
- "Cohen's vs Fleiss' kappa?" → Cohen's: two raters. Fleiss': N raters.

**Common mistakes.**
- Computing simple agreement (`p_o`) and calling it κ.

**Implementation.**
```python
from collections import Counter


def cohens_kappa(ratings_a: list, ratings_b: list) -> float:
    assert len(ratings_a) == len(ratings_b)
    n = len(ratings_a)
    p_o = sum(1 for a, b in zip(ratings_a, ratings_b) if a == b) / n
    cat_a = Counter(ratings_a)
    cat_b = Counter(ratings_b)
    categories = set(cat_a) | set(cat_b)
    p_e = sum((cat_a[c] / n) * (cat_b[c] / n) for c in categories)
    if p_e == 1.0:
        return 1.0 if p_o == 1.0 else 0.0
    return (p_o - p_e) / (1 - p_e)


if __name__ == "__main__":
    a = [1, 0, 1, 1, 0, 1, 0]
    b = [1, 0, 0, 1, 0, 1, 1]
    print("κ =", cohens_kappa(a, b))
```

---

### Q: Implement nDCG (normalized discounted cumulative gain) for retrieval eval.

**Category:** coding
**Difficulty:** mid
**Tags:** [ndcg, retrieval-metrics]

**Short answer.** DCG = Σ `relevance_i / log2(rank_i + 1)`; nDCG = DCG / ideal-DCG. Captures position-weighted relevance — higher-ranked hits worth more.

**Common follow-ups.**
- "Why log2 discount?" → Empirically chosen; smooth decay reflecting user attention drop with rank.

**Common mistakes.**
- Forgetting to normalize by ideal-DCG (so nDCG bounded in [0, 1]).

**Implementation.**
```python
import math


def dcg(relevances: list[float]) -> float:
    return sum(r / math.log2(i + 2) for i, r in enumerate(relevances))


def ndcg(retrieved_relevances: list[float], ideal_relevances: list[float]) -> float:
    return dcg(retrieved_relevances) / dcg(sorted(ideal_relevances, reverse=True))


if __name__ == "__main__":
    # Retrieved: ranking-1 has rel 3, ranking-2 has rel 1, ranking-3 has rel 2
    retrieved = [3, 1, 2]
    ideal = [3, 2, 1]
    print("nDCG =", ndcg(retrieved, ideal))
```

---

### Q: Implement a simple KV-cache for a small transformer in PyTorch.

**Category:** coding
**Difficulty:** senior
**Tags:** [kv-cache, pytorch, decode]

**Short answer.** Maintain per-layer K/V tensors; append at each decode step; use the full cache for attention against new query.

**Common follow-ups.**
- "Memory?" → Grows with sequence length; cap or prune for long contexts.

**Common mistakes.**
- Concatenating per step in place; creates new tensor copies.

**Implementation.**
```python
import torch
import torch.nn.functional as F


class CachedAttention(torch.nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.qkv = torch.nn.Linear(d_model, 3 * d_model)
        self.out = torch.nn.Linear(d_model, d_model)
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

    def forward(self, x: torch.Tensor, kv_cache: dict | None = None):
        B, T, D = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # (B, H, T, Dh)
        # Append to cache
        if kv_cache is not None:
            if "k" in kv_cache:
                k = torch.cat([kv_cache["k"], k], dim=2)
                v = torch.cat([kv_cache["v"], v], dim=2)
            kv_cache["k"] = k
            kv_cache["v"] = v
        # Causal attention
        y = F.scaled_dot_product_attention(q, k, v, is_causal=(T > 1))
        y = y.transpose(1, 2).contiguous().view(B, T, D)
        return self.out(y)


if __name__ == "__main__":
    layer = CachedAttention(d_model=128, n_heads=8)
    cache = {}
    # First forward (prefill)
    x1 = torch.randn(1, 5, 128)
    y1 = layer(x1, cache)
    # Decode step
    x2 = torch.randn(1, 1, 128)
    y2 = layer(x2, cache)
    print("after step, cache K shape:", cache["k"].shape)
```

---

### Q: Implement a basic ReAct loop in Python.

**Category:** coding
**Difficulty:** mid
**Tags:** [react, agent-loop, python]

**Short answer.** Loop: send context to LLM; parse output for Thought / Action / final answer; execute action via tool; append observation; continue.

**Common follow-ups.**
- "How to detect termination?" → Look for a `Final Answer:` line.

**Common mistakes.**
- Trusting model-written observations (use structured tool calls in production).

**Implementation.**
```python
import re
from typing import Callable


def react_loop(llm: Callable[[str], str], tools: dict[str, Callable], initial_prompt: str, max_steps: int = 10) -> str:
    context = initial_prompt
    for step in range(max_steps):
        output = llm(context)
        if "Final Answer:" in output:
            return output.split("Final Answer:", 1)[1].strip()
        m = re.search(r"Action:\s*(\w+)\((.*?)\)", output)
        if not m:
            return f"Failed (no action found): {output}"
        action_name, action_args = m.group(1), m.group(2)
        if action_name not in tools:
            obs = f"unknown tool {action_name}"
        else:
            try:
                obs = tools[action_name](action_args)
            except Exception as e:
                obs = f"error: {e}"
        context += f"\n{output}\nObservation: {obs}\n"
    return "Failed (max steps reached)"
```

---

### Q: Implement gradient clipping in PyTorch (by global norm).

**Category:** coding
**Difficulty:** intro
**Tags:** [gradient-clipping, training-stability]

**Short answer.** Use `torch.nn.utils.clip_grad_norm_(params, max_norm)` after backward but before optimizer step. Standard `max_norm = 1.0` for LLM training.

**Common follow-ups.**
- "Norm vs value clipping?" → Norm is dominant; preserves gradient direction.

**Common mistakes.**
- Calling before backward; gradients not populated.

**Implementation.**
```python
import torch


def train_step_with_clip(model: torch.nn.Module, loss: torch.Tensor, optimizer: torch.optim.Optimizer, max_norm: float = 1.0):
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
    optimizer.step()
```

---

### Q: Implement self-consistency voting for chain-of-thought.

**Category:** coding
**Difficulty:** mid
**Tags:** [self-consistency, cot, voting]

**Short answer.** Sample N chain-of-thought traces; extract the answer from each; majority-vote. Robust to noisy single traces.

**Common follow-ups.**
- "What N?" → 5–40 typical; diminishing returns past 20.

**Common mistakes.**
- Answer-extraction failures (string match brittle).

**Implementation.**
```python
from collections import Counter
import re
from typing import Callable


def self_consistency_solve(llm: Callable[[str], str], prompt: str, n_samples: int = 5) -> str:
    answers = []
    for _ in range(n_samples):
        trace = llm(prompt)
        m = re.search(r"Answer:\s*([^\n]+)", trace)
        if m:
            answers.append(m.group(1).strip())
    if not answers:
        return "(no parseable answer)"
    counter = Counter(answers)
    return counter.most_common(1)[0][0]
```

---

### Q: Implement temperature scaling for calibration.

**Category:** coding
**Difficulty:** mid
**Tags:** [temperature-scaling, calibration]

**Short answer.** After training, learn a single scalar T to apply to logits, minimizing NLL on a held-out validation set. Preserves accuracy; improves calibration (ECE).

**Common follow-ups.**
- "Doesn't change accuracy?" → Argmax unchanged; only confidence calibration.

**Common mistakes.**
- Using training set for calibration (overfit).

**Implementation.**
```python
import torch
import torch.nn.functional as F


def fit_temperature(val_logits: torch.Tensor, val_labels: torch.Tensor, max_iter: int = 100) -> float:
    """Fit a single temperature scalar T via LBFGS on validation NLL."""
    T = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.LBFGS([T], lr=0.1, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(val_logits / T, val_labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return T.item()


if __name__ == "__main__":
    torch.manual_seed(0)
    logits = torch.randn(100, 5)
    labels = torch.randint(0, 5, (100,))
    T = fit_temperature(logits, labels)
    print(f"learned T = {T:.3f}")
```

---

### Q: Implement embedding L2 normalization, batch-wise.

**Category:** coding
**Difficulty:** intro
**Tags:** [embeddings, normalization]

**Short answer.** Divide each embedding by its L2 norm; the cosine similarity then equals dot product. One-liner with NumPy or PyTorch broadcasting.

**Common mistakes.**
- Dividing by zero on degenerate inputs (zero embedding); add epsilon.

**Implementation.**
```python
import numpy as np


def l2_normalize(X: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(X, axis=axis, keepdims=True)
    return X / np.maximum(norm, eps)


if __name__ == "__main__":
    X = np.random.randn(5, 3)
    X_n = l2_normalize(X)
    print("norms:", np.linalg.norm(X_n, axis=-1))  # all ~1.0
```

---

### Q: Implement multi-head attention (separate heads).

**Category:** coding
**Difficulty:** mid
**Tags:** [mha, multi-head, attention]

**Short answer.** Project Q, K, V to `(B, T, H * D)`; reshape to `(B, H, T, D)`; compute per-head attention; concatenate; project to output dim.

**Common mistakes.**
- Forgetting the output projection.

**Implementation.**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class MHA(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, D = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        # F.scaled_dot_product_attention handles causal mask if mask is None
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        y = y.transpose(1, 2).contiguous().view(B, T, D)
        return self.proj(y)
```

---

### Q: Implement gradient accumulation correctly with mixed-precision.

**Category:** coding
**Difficulty:** mid
**Tags:** [gradient-accumulation, mixed-precision]

**Short answer.** Use `torch.autocast(bf16)` for forward; divide loss by `accum_steps`; backward each micro-batch; clip and step every `accum_steps`. With bf16, no GradScaler needed; with fp16, use `torch.cuda.amp.GradScaler`.

**Implementation.**
```python
import torch
import torch.nn as nn


def train_with_accum_bf16(model: nn.Module, loader, optimizer, *, accum_steps: int = 4):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(loader, start=1):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = compute_loss(model, batch)
            loss = loss / accum_steps
        loss.backward()
        if step % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)


def compute_loss(model, batch):
    # Stub
    x, y = batch
    logits = model(x)
    return torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
```

---

### Q: Implement perplexity computation given a model's per-token log-probs.

**Category:** coding
**Difficulty:** intro
**Tags:** [perplexity, ce, evaluation]

**Short answer.** Perplexity = `exp(mean(-log p_i))` = `exp(cross_entropy)`. One-liner over a sequence of log-probs.

**Implementation.**
```python
import numpy as np


def perplexity(neg_log_probs: np.ndarray) -> float:
    """neg_log_probs: per-token negative log-likelihood (cross-entropy)."""
    return float(np.exp(neg_log_probs.mean()))


if __name__ == "__main__":
    nll = np.array([2.3, 1.8, 2.5, 2.0])
    print(f"perplexity = {perplexity(nll):.2f}")
```

---

### Q: Implement IoU (intersection over union) for evaluating retrieval set overlap.

**Category:** coding
**Difficulty:** intro
**Tags:** [iou, set-overlap]

**Short answer.** `|A ∩ B| / |A ∪ B|`. Useful for measuring consistency across different retrieval runs.

**Implementation.**
```python
def iou(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


if __name__ == "__main__":
    a = {1, 2, 3, 4}
    b = {3, 4, 5, 6}
    print(f"IoU = {iou(a, b):.3f}")
```

---

### Q: Implement a function to compute the expected calibration error (ECE).

**Category:** coding
**Difficulty:** mid
**Tags:** [ece, calibration, eval]

**Short answer.** Bin predictions by confidence; per bin, compute (accuracy, confidence) and weighted gap; sum weighted gaps.

**Implementation.**
```python
import numpy as np


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """probs: predicted probability of the predicted class (max prob per example).
    labels: 1 if prediction correct, 0 else.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    N = len(probs)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        acc = labels[mask].mean()
        conf = probs[mask].mean()
        weight = mask.sum() / N
        ece += weight * abs(acc - conf)
    return ece


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    probs = rng.uniform(0.5, 1.0, size=1000)
    labels = (rng.uniform(size=1000) < probs).astype(int)  # well-calibrated
    print(f"ECE = {expected_calibration_error(probs, labels):.4f}")
```

---

### Q: Implement a sliding-window summarizer for compacting agent context.

**Category:** coding
**Difficulty:** mid
**Tags:** [compaction, agent-context]

**Short answer.** Keep the latest N tokens verbatim; replace the older portion with a summary from a cheap LLM. Trigger when context exceeds a threshold.

**Implementation.**
```python
from typing import Callable


def compact_context(messages: list[dict], llm_summarize: Callable[[str], str], keep_recent: int = 5) -> list[dict]:
    if len(messages) <= keep_recent + 1:
        return messages
    # Keep system + recent; summarize the rest
    system = messages[0] if messages[0].get("role") == "system" else None
    older = messages[1:-keep_recent] if system else messages[:-keep_recent]
    recent = messages[-keep_recent:]
    older_text = "\n".join(f"{m['role']}: {m['content']}" for m in older)
    summary = llm_summarize(older_text)
    summary_msg = {"role": "system", "content": f"[Summary of earlier turns: {summary}]"}
    return ([system] if system else []) + [summary_msg] + recent
```

---

### Q: Implement byte-pair-encoding decoder (apply encoded tokens back to text).

**Category:** coding
**Difficulty:** intro
**Tags:** [bpe, decoding]

**Short answer.** Join the tokens; strip the end-of-word marker; return.

**Implementation.**
```python
def decode_bpe(tokens: list[str], eos_marker: str = "</w>") -> str:
    text = "".join(tokens)
    # Replace eos_marker with space
    text = text.replace(eos_marker, " ")
    return text.strip()


if __name__ == "__main__":
    tokens = ["hel", "lo", "</w>", "wor", "ld", "</w>"]
    print(repr(decode_bpe(tokens)))  # 'hello world'
```

---

### Q: Implement a check for shape compatibility (broadcasting) given two shapes.

**Category:** coding
**Difficulty:** intro
**Tags:** [broadcasting, numpy]

**Short answer.** Right-align the shapes; for each aligned dim, ensure both are equal or one is 1. Trivial helper but useful in debugging.

**Implementation.**
```python
def can_broadcast(s1: tuple, s2: tuple) -> bool:
    for a, b in zip(reversed(s1), reversed(s2)):
        if a == 1 or b == 1 or a == b:
            continue
        return False
    return True


if __name__ == "__main__":
    assert can_broadcast((3, 1), (1, 4))
    assert can_broadcast((3, 4), (4,))
    assert not can_broadcast((3, 4), (5,))
    print("OK")
```

---
