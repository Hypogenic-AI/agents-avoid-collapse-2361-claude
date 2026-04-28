"""E2 — RAG-style ecosystem experiment (Wang et al. 2025-style).

A population of N LLM agents share a common growing text pool ("the
internet"). At each iteration:

  1. Each agent retrieves k = ceil(beta * |pool|) random posts.
  2. Each agent writes a new post given those posts as context.
  3. New posts are appended to the pool.

We measure ecosystem-level diversity over the *new posts produced this
iteration*: lexical (distinct-2), semantic (mean pairwise distance,
Frobenius norm of pairwise distance matrix), and Hill–Shannon Diversity
(Vendi Score) over output embeddings.

Population sizes N are swept; each N is run with several random seeds.

This is `Experiment 2` in the planning doc. Cheap (API-only, no training)
so we can sweep N up to 8 in reasonable time.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List

import numpy as np
from openai import OpenAI

sys.path.insert(0, os.path.dirname(__file__))
from metrics import (
    distinct_n,
    embed_texts,
    frobenius_distance,
    hill_shannon_diversity,
    mean_pairwise_distance,
)


# --------------------------------------------------------------------------- #
# Model pool. Routed via OpenRouter; mostly small, cheap instruct models.
# --------------------------------------------------------------------------- #

MODEL_POOL: List[str] = [
    "meta-llama/llama-3.1-8b-instruct",
    "mistralai/mistral-7b-instruct",
    "qwen/qwen-2.5-7b-instruct",
    "google/gemma-2-9b-it",
    "openai/gpt-4o-mini",
    "deepseek/deepseek-chat-v3-0324",
    "nousresearch/hermes-3-llama-3.1-8b",
    "microsoft/wizardlm-2-8x22b",
]


def build_agents(n: int, mode: str = "model_family", base_model: str | None = None) -> List[Dict]:
    """Construct N agents.

    `mode`:
      * "model_family": cycle through MODEL_POOL → diversity via pretraining.
      * "persona":      single base model + N distinct system prompts.
      * "single":       single base model, single prompt (control: differs only
                        by sampling-time stochasticity).
      * "data_segment": single base model, single prompt; each agent is later
                        restricted to its own slice of the seed corpus (the
                        slicing is done in `run_ecosystem` since it touches the
                        pool, not the agent definition).
    """
    if mode == "model_family":
        return [
            {"id": f"agent_{i}", "model": MODEL_POOL[i % len(MODEL_POOL)], "system": DEFAULT_SYSTEM}
            for i in range(n)
        ]
    if mode == "persona":
        bm = base_model or MODEL_POOL[0]
        personas = PERSONAS[:n] if n <= len(PERSONAS) else (PERSONAS * ((n // len(PERSONAS)) + 1))[:n]
        return [
            {"id": f"agent_{i}", "model": bm, "system": p}
            for i, p in enumerate(personas)
        ]
    if mode in ("single", "data_segment"):
        bm = base_model or MODEL_POOL[0]
        return [{"id": f"agent_{i}", "model": bm, "system": DEFAULT_SYSTEM} for i in range(n)]
    raise ValueError(f"unknown mode {mode}")


DEFAULT_SYSTEM = (
    "You are an expert author. Continue the conversation by writing exactly one "
    "new short paragraph (2-4 sentences) on the topic suggested by the context. "
    "Do not preface or explain — output only the paragraph."
)

PERSONAS = [
    "You are a curious physicist. " + DEFAULT_SYSTEM,
    "You are a detective novelist. " + DEFAULT_SYSTEM,
    "You are a children's storyteller. " + DEFAULT_SYSTEM,
    "You are a tech journalist. " + DEFAULT_SYSTEM,
    "You are a medieval historian. " + DEFAULT_SYSTEM,
    "You are a culinary critic. " + DEFAULT_SYSTEM,
    "You are an avant-garde poet. " + DEFAULT_SYSTEM,
    "You are a courtroom lawyer. " + DEFAULT_SYSTEM,
]


# --------------------------------------------------------------------------- #
# OpenRouter API wrapper
# --------------------------------------------------------------------------- #

def make_client() -> OpenAI:
    return OpenAI(
        api_key=os.environ["OPENROUTER_KEY"],
        base_url="https://openrouter.ai/api/v1",
    )


def call_llm(
    client: OpenAI,
    model: str,
    system: str,
    user: str,
    max_tokens: int = 120,
    temperature: float = 1.0,
    retries: int = 3,
) -> str:
    """Single chat completion with retry-on-failure."""
    last_err = None
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.95,
            )
            text = (resp.choices[0].message.content or "").strip()
            if text:
                return text
            last_err = "empty response"
        except Exception as e:  # noqa: BLE001
            last_err = repr(e)
            time.sleep(2 + attempt * 3)
    return f"[FAILED:{last_err}]"


# --------------------------------------------------------------------------- #
# Seed corpus
# --------------------------------------------------------------------------- #

def load_seed_corpus(n_seeds: int = 30, min_chars: int = 200) -> List[str]:
    """Sample short factual paragraphs from WikiText-2 to seed the pool."""
    from datasets import load_from_disk

    ds = load_from_disk("datasets/wikitext2/wikitext-2-raw-v1")["train"]
    texts = []
    for row in ds:
        t = row["text"].strip()
        # Skip empty and section headers
        if (
            len(t) >= min_chars
            and not t.startswith("=")
            and "\n" not in t.strip()
        ):
            texts.append(t)
            if len(texts) >= n_seeds:
                break
    return texts


# --------------------------------------------------------------------------- #
# Single ecosystem run
# --------------------------------------------------------------------------- #

@dataclass
class RAGRunConfig:
    n: int               # population size
    iters: int           # number of iterations
    beta: float          # retrieval ratio
    seed: int            # RNG seed
    mode: str            # "model_family" | "persona" | "single"
    seed_corpus: List[str]
    query: str = "Continue the discussion by writing one new informative paragraph that builds on the topics raised."
    max_tokens: int = 120


def run_ecosystem(cfg: RAGRunConfig, log_path: Path | None = None) -> Dict:
    rng = random.Random(cfg.seed)
    np.random.seed(cfg.seed)
    client = make_client()
    agents = build_agents(cfg.n, mode=cfg.mode)
    # In "data_segment" mode, partition the seed corpus into N non-overlapping
    # slices; each agent maintains its own pool that grows only with its own
    # outputs and (optionally) its peers' outputs at retrieval time. To keep
    # things tractable and comparable to other modes, we still grow a SHARED
    # pool of all outputs, but each agent retrieves only from a subset of
    # current-pool indices that it "owns" — set up with a deterministic mask.
    seed_mask: List[List[int]] | None = None
    if cfg.mode == "data_segment":
        idxs = list(range(len(cfg.seed_corpus)))
        rng_init = random.Random(cfg.seed * 9973 + 1)
        rng_init.shuffle(idxs)
        # split as evenly as possible
        slices = [idxs[i :: cfg.n] for i in range(cfg.n)]
        seed_mask = slices
    pool: List[str] = list(cfg.seed_corpus)
    history: List[Dict] = []

    for t in range(1, cfg.iters + 1):
        k = max(1, int(np.ceil(cfg.beta * len(pool))))
        new_posts: List[str] = []
        for ai, agent in enumerate(agents):
            if seed_mask is not None:
                # Agent ai retrieves from its mask + its own posts
                allowed = seed_mask[ai]
                sample_pool = [pool[idx] for idx in allowed if idx < len(pool)]
                if not sample_pool:
                    sample_pool = pool
                sample = rng.sample(sample_pool, min(k, len(sample_pool)))
            else:
                sample = rng.sample(pool, min(k, len(pool)))
            context = "\n\n".join(f"[{i+1}] {s}" for i, s in enumerate(sample))
            user = f"Context posts:\n{context}\n\n{cfg.query}"
            text = call_llm(
                client,
                model=agent["model"],
                system=agent["system"],
                user=user,
                max_tokens=cfg.max_tokens,
            )
            new_posts.append(text)

        # Compute metrics over the new posts produced this iteration:
        emb = embed_texts(new_posts) if len(new_posts) >= 1 else None
        record = {
            "iter": t,
            "n_pool_before": len(pool),
            "k_retrieved": k,
            "n_new_posts": len(new_posts),
            "distinct_2": distinct_n(new_posts, n=2),
            "mean_pairwise_dist": float(mean_pairwise_distance(emb)) if emb is not None else 0.0,
            "frobenius": float(frobenius_distance(emb)) if emb is not None else 0.0,
            "hsd": float(hill_shannon_diversity(emb)) if emb is not None else 1.0,
            "agent_models": [a["model"] for a in agents],
        }
        history.append(record)
        if log_path:
            with open(log_path, "a") as f:
                f.write(json.dumps({"cfg": asdict(cfg), "iter_record": record, "samples": new_posts[:2]}) + "\n")
        print(
            f"  N={cfg.n} mode={cfg.mode} seed={cfg.seed} t={t:>2d}: "
            f"distinct2={record['distinct_2']:.3f}  "
            f"HSD={record['hsd']:.2f}/{cfg.n}  "
            f"meanPD={record['mean_pairwise_dist']:.3f}  "
            f"frob={record['frobenius']:.3f}",
            flush=True,
        )

        # In data_segment mode, also extend each agent's mask with its new
        # post (so peers cannot retrieve it unless we add cross-pollination).
        if seed_mask is not None:
            new_indices = list(range(len(pool), len(pool) + len(new_posts)))
            for ai, ni in enumerate(new_indices):
                seed_mask[ai].append(ni)
        pool.extend(new_posts)

    return {"cfg": asdict(cfg), "history": history}


# --------------------------------------------------------------------------- #
# Experiment driver
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/rag_ecosystem.json")
    parser.add_argument("--ns", default="1,2,3,5,8", help="comma-separated population sizes")
    parser.add_argument("--iters", type=int, default=12)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--seeds", default="0,1,2", help="random seeds")
    parser.add_argument("--mode", default="model_family", choices=["model_family", "persona", "single", "data_segment"])
    parser.add_argument("--n_seed_corpus", type=int, default=20)
    args = parser.parse_args()

    Ns = [int(x) for x in args.ns.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = Path("logs") / (out_path.stem + ".jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # truncate log
    if log_path.exists():
        log_path.unlink()

    seed_corpus = load_seed_corpus(n_seeds=args.n_seed_corpus)
    print(f"Loaded {len(seed_corpus)} seed paragraphs (avg len {np.mean([len(t) for t in seed_corpus]):.0f} chars)")

    all_runs = []
    for n in Ns:
        for seed in seeds:
            cfg = RAGRunConfig(
                n=n,
                iters=args.iters,
                beta=args.beta,
                seed=seed,
                mode=args.mode,
                seed_corpus=seed_corpus,
            )
            print(f"\n=== Running N={n} seed={seed} mode={args.mode} ===", flush=True)
            t0 = time.time()
            try:
                result = run_ecosystem(cfg, log_path=log_path)
                result["wall_seconds"] = time.time() - t0
                all_runs.append(result)
            except Exception as e:  # noqa: BLE001
                print(f"  FAILED: {e!r}")
            # incremental save after every run
            with open(out_path, "w") as f:
                json.dump(all_runs, f, indent=2)
    print(f"\nWrote {len(all_runs)} runs to {out_path}")


if __name__ == "__main__":
    main()
