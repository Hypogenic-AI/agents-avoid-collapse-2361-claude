"""Diversity and collapse metrics shared by the fine-tuning and RAG ecosystems.

Metrics implemented (all defined on a list of generated text strings, optionally
grouped by agent so we can also report inter-agent quantities):

* `distinct_n` — share of unique n-grams over total n-grams (Li et al. 2016).
* `pairwise_cosine_distance_matrix` — D[i,j] = 1 - cos(emb_i, emb_j).
* `mean_pairwise_distance` — average off-diagonal entry of D.
* `frobenius_distance` — Frobenius norm of D (Wang et al. 2025 diagnostic).
* `hill_shannon_diversity` — exp(Shannon entropy of soft cluster assignments
  computed via cosine similarity) — an "effective number of distinct outputs"
  in the embedding space (cf. Hodel & West 2026).
* `perplexity_eval` — perplexity of a text corpus under a fixed evaluation
  language model (used to track quality drift).

The embedding model is loaded lazily and cached.
"""

from __future__ import annotations

import math
from collections import Counter
from functools import lru_cache
from typing import List, Sequence

import numpy as np


# --------------------------------------------------------------------------- #
# Embedding utilities
# --------------------------------------------------------------------------- #

_EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _embed_model():
    from sentence_transformers import SentenceTransformer
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(_EMBED_MODEL_NAME, device=device)


def embed_texts(texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
    """Return L2-normalised embeddings for `texts`, shape (N, d)."""
    model = _embed_model()
    emb = model.encode(
        list(texts),
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return emb.astype(np.float32)


# --------------------------------------------------------------------------- #
# Lexical diversity
# --------------------------------------------------------------------------- #

def _tokens(text: str) -> List[str]:
    return text.lower().split()


def distinct_n(texts: Sequence[str], n: int = 2) -> float:
    """Fraction of unique n-grams across the concatenation of all texts."""
    counter: Counter = Counter()
    total = 0
    for text in texts:
        toks = _tokens(text)
        if len(toks) < n:
            continue
        for i in range(len(toks) - n + 1):
            counter[tuple(toks[i : i + n])] += 1
            total += 1
    if total == 0:
        return 0.0
    return len(counter) / total


# --------------------------------------------------------------------------- #
# Embedding-space diversity
# --------------------------------------------------------------------------- #

def pairwise_cosine_distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    """D[i,j] = 1 - cos(e_i, e_j) for L2-normalised embeddings."""
    sim = embeddings @ embeddings.T
    sim = np.clip(sim, -1.0, 1.0)
    return 1.0 - sim


def mean_pairwise_distance(embeddings: np.ndarray) -> float:
    """Mean of off-diagonal entries of the distance matrix."""
    if len(embeddings) < 2:
        return 0.0
    d = pairwise_cosine_distance_matrix(embeddings)
    n = d.shape[0]
    return float((d.sum() - np.trace(d)) / (n * (n - 1)))


def frobenius_distance(embeddings: np.ndarray) -> float:
    """Frobenius norm of the pairwise distance matrix (Wang et al. 2025)."""
    if len(embeddings) < 2:
        return 0.0
    d = pairwise_cosine_distance_matrix(embeddings)
    return float(np.linalg.norm(d, ord="fro"))


def hill_shannon_diversity(embeddings: np.ndarray) -> float:
    """Effective number of distinct items, via the Vendi Score.

    Friedman & Dieng (2023) define the order-1 Vendi Score as
    exp(-trace(K/n log(K/n))) where K is the n×n kernel matrix and n is the
    sample size. With cosine similarity (L2-normalised embeddings), K_ii = 1
    so trace(K) = n; we use λ_i / n where λ_i are the eigenvalues of K.

    Limits:
    * All items identical → K is rank-1 → λ = (n, 0, ..., 0) → VS = 1.
    * All items orthogonal → K = I → λ = (1, ..., 1) → VS = n.

    This is the principled multivariate analogue of the Hill–Shannon index used
    in Hodel & West (2026).
    """
    n = embeddings.shape[0]
    if n < 2:
        return 1.0
    K = embeddings @ embeddings.T  # n×n PSD kernel
    K = (K + K.T) / 2.0  # numerical symmetry
    eigvals = np.linalg.eigvalsh(K)
    # Probability distribution over eigenvalues:
    p = eigvals / max(n, 1e-12)
    p = p[p > 1e-12]
    if p.size == 0:
        return 1.0
    h = -np.sum(p * np.log(p))
    return float(math.exp(h))


# --------------------------------------------------------------------------- #
# Perplexity (used by E1, optional in E2)
# --------------------------------------------------------------------------- #

def perplexity_eval(
    texts: Sequence[str],
    model_name: str = "gpt2",
    device: str = "cuda",
    max_length: int = 256,
    stride: int = 128,
) -> float:
    """Compute perplexity of `texts` under a fixed pretrained LM (default GPT-2).

    Uses the standard sliding-window evaluation. We treat `texts` as one
    concatenation. Returns +inf on empty input.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not texts:
        return float("inf")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    encodings = tokenizer("\n\n".join(texts), return_tensors="pt").input_ids.to(device)
    seq_len = encodings.size(1)
    if seq_len < 2:
        return float("inf")

    nlls = []
    prev_end = 0
    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        target_len = end - prev_end
        input_ids = encodings[:, begin:end]
        target_ids = input_ids.clone()
        target_ids[:, :-target_len] = -100
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
        # outputs.loss is mean negative log-likelihood per token; multiply by
        # number of *predicted* tokens (target_len) to get summed NLL.
        nlls.append(outputs.loss * target_len)
        prev_end = end
        if end == seq_len:
            break
    nll = torch.stack(nlls).sum() / seq_len
    return float(torch.exp(nll))
