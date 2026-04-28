"""E1 — Fine-tuning ecosystem experiment (Hodel & West 2026-style).

A population of N agents are initialised from the same pretrained GPT-2-small
checkpoint. Each agent gets a non-overlapping slice of WikiText-2 train as its
initial "epistemology" and is fine-tuned once on that slice.

Then, for T iterations:
  1. Each of N agents *generates* a slice of synthetic tokens.
  2. The N slices are pooled, shuffled, and re-split into N new slices
     (Hodel & West "shuffle and redistribute" mixing).
  3. Each agent continues fine-tuning on its new slice ("replace" mode: no
     real data refresh).

After every iteration we evaluate:
  * perplexity of each agent on the *fixed* WikiText-2 test set.
  * lexical diversity (distinct-2) of the pooled new generations.
  * mean pairwise embedding distance + Hill-Shannon Diversity (Vendi Score)
    over a sample of the new generations.

Per-ecosystem token budget is held constant across N, so larger N means each
agent sees fewer tokens. This isolates *diversity* as the variable.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, os.path.dirname(__file__))
from metrics import (
    distinct_n,
    embed_texts,
    frobenius_distance,
    hill_shannon_diversity,
    mean_pairwise_distance,
)

# --------------------------------------------------------------------------- #
# Setup utilities
# --------------------------------------------------------------------------- #

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_wikitext_train_text(min_chars: int = 50) -> str:
    """Load WikiText-2 train as a single concatenated string (skipping empty
    rows and section headers)."""
    from datasets import load_from_disk

    ds = load_from_disk("datasets/wikitext2/wikitext-2-raw-v1")["train"]
    chunks: List[str] = []
    for row in ds:
        t = row["text"].strip()
        if len(t) >= min_chars and not t.startswith("="):
            chunks.append(t)
    return "\n\n".join(chunks)


def load_wikitext_test_text(min_chars: int = 50) -> str:
    from datasets import load_from_disk

    ds = load_from_disk("datasets/wikitext2/wikitext-2-raw-v1")["test"]
    chunks = []
    for row in ds:
        t = row["text"].strip()
        if len(t) >= min_chars and not t.startswith("="):
            chunks.append(t)
    return "\n\n".join(chunks)


# --------------------------------------------------------------------------- #
# Data slicing
# --------------------------------------------------------------------------- #

def tokenize_to_blocks(text: str, tokenizer, block_size: int) -> torch.Tensor:
    """Tokenise `text` and split into non-overlapping blocks of `block_size`.

    Returns LongTensor of shape (n_blocks, block_size). Trailing tokens that
    don't fill a block are dropped.
    """
    ids = tokenizer(text, return_tensors="pt", truncation=False).input_ids[0]
    n_blocks = ids.size(0) // block_size
    if n_blocks == 0:
        return torch.empty((0, block_size), dtype=torch.long)
    return ids[: n_blocks * block_size].view(n_blocks, block_size)


def split_blocks_into_slices(blocks: torch.Tensor, n: int, rng: random.Random) -> List[torch.Tensor]:
    """Shuffle `blocks` and split (as evenly as possible) into n slices."""
    if blocks.size(0) == 0:
        return [blocks.clone() for _ in range(n)]
    perm = list(range(blocks.size(0)))
    rng.shuffle(perm)
    blocks = blocks[perm]
    slice_sizes = [blocks.size(0) // n] * n
    for i in range(blocks.size(0) % n):
        slice_sizes[i] += 1
    out = []
    cursor = 0
    for s in slice_sizes:
        out.append(blocks[cursor : cursor + s].clone())
        cursor += s
    return out


# --------------------------------------------------------------------------- #
# Train + generate per agent
# --------------------------------------------------------------------------- #

def train_one_epoch(
    model: torch.nn.Module,
    blocks: torch.Tensor,
    device: str,
    batch_size: int,
    lr: float,
    weight_decay: float = 0.01,
):
    """One epoch of standard causal-LM training over `blocks`."""
    if blocks.size(0) == 0:
        return float("nan")

    model.train()
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    n_steps = (blocks.size(0) + batch_size - 1) // batch_size
    total = 0.0
    n_batches = 0
    perm = torch.randperm(blocks.size(0))
    blocks = blocks[perm]
    for i in range(n_steps):
        batch = blocks[i * batch_size : (i + 1) * batch_size].to(device)
        if batch.size(0) == 0:
            continue
        out = model(batch, labels=batch)
        loss = out.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        optim.zero_grad()
        total += float(loss.detach().cpu())
        n_batches += 1
    return total / max(n_batches, 1)


@torch.no_grad()
def generate_tokens(
    model: torch.nn.Module,
    tokenizer,
    n_tokens: int,
    device: str,
    block_size: int = 64,
    batch_size: int = 8,
    temperature: float = 1.0,
    top_p: float = 0.95,
    seed_text: str = "\n\n",
) -> str:
    """Sample `n_tokens` tokens from `model`, batched.

    We seed each generation with a short text token (here a couple of
    newlines) and let the model write a new ~64-token continuation. We repeat
    enough times to reach `n_tokens` total output.
    """
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    seed_ids = tokenizer(seed_text, return_tensors="pt").input_ids.to(device)
    needed_calls = (n_tokens + (block_size * batch_size) - 1) // (block_size * batch_size)

    pieces: List[str] = []
    total = 0
    for _ in range(needed_calls):
        bs = batch_size
        input_ids = seed_ids.repeat(bs, 1)
        out = model.generate(
            input_ids,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=block_size,
            pad_token_id=tokenizer.eos_token_id,
        )
        new = out[:, input_ids.size(1) :]
        for row in new:
            text = tokenizer.decode(row, skip_special_tokens=True)
            pieces.append(text)
            total += row.numel()
            if total >= n_tokens:
                break
        if total >= n_tokens:
            break
    return "\n\n".join(pieces)


@torch.no_grad()
def perplexity_of_test(
    model: torch.nn.Module, test_blocks: torch.Tensor, device: str, batch_size: int = 4
) -> float:
    """Stride-free perplexity over fixed test blocks (1024-token).

    NLLs are aggregated by summing batch losses weighted by number of tokens
    per batch, then exponentiating mean per-token NLL.
    """
    model.eval()
    if test_blocks.size(0) == 0:
        return float("inf")
    nll_sum = 0.0
    tok_sum = 0
    for i in range(0, test_blocks.size(0), batch_size):
        batch = test_blocks[i : i + batch_size].to(device)
        out = model(batch, labels=batch)
        # out.loss is already mean per-token NLL; scale by tokens predicted:
        n_predicted = (batch.size(1) - 1) * batch.size(0)  # shift by 1
        nll_sum += float(out.loss.detach().cpu()) * n_predicted
        tok_sum += n_predicted
    return float(np.exp(nll_sum / max(tok_sum, 1)))


# --------------------------------------------------------------------------- #
# Run-level driver
# --------------------------------------------------------------------------- #

@dataclass
class FTRunConfig:
    n: int                         # population size
    iters: int                     # number of self-training iterations
    base_model: str = "distilgpt2"
    block_size: int = 64
    batch_size: int = 16
    lr: float = 5e-5
    seed: int = 0
    ecosystem_token_budget: int = 100_000  # total tokens of synthetic data per generation
    diversity_sample: int = 32     # # generated *strings* per agent for diversity stats
    test_max_blocks: int = 80      # # 64-token test blocks for perplexity (fixed)
    device: str = "cuda"


def run_ecosystem(cfg: FTRunConfig, log_path: Path) -> dict:
    set_seed(cfg.seed)
    device = cfg.device if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Initial real data
    train_text = load_wikitext_train_text()
    test_text = load_wikitext_test_text()
    train_blocks_all = tokenize_to_blocks(train_text, tokenizer, cfg.block_size)
    test_blocks_all = tokenize_to_blocks(test_text, tokenizer, cfg.block_size)
    test_blocks_eval = test_blocks_all[: cfg.test_max_blocks]

    # Cap initial real-data slices so each agent sees ~budget/N tokens
    initial_blocks_total = cfg.ecosystem_token_budget // cfg.block_size
    initial_blocks_total = min(initial_blocks_total, train_blocks_all.size(0))
    rng = random.Random(cfg.seed)

    perm = list(range(train_blocks_all.size(0)))
    rng.shuffle(perm)
    train_subset = train_blocks_all[perm[:initial_blocks_total]]
    initial_slices = split_blocks_into_slices(train_subset, cfg.n, rng)

    # Instantiate N agents from the same pretrained checkpoint
    print(f"  loading {cfg.n} copies of {cfg.base_model} on {device}")
    agents = [
        AutoModelForCausalLM.from_pretrained(cfg.base_model).to(device) for _ in range(cfg.n)
    ]

    # Iteration 0: warm-up training on real data slices
    for i, model in enumerate(agents):
        loss = train_one_epoch(model, initial_slices[i], device, cfg.batch_size, cfg.lr)
        print(f"    init agent {i}: real-data ep loss={loss:.3f} (n_blocks={initial_slices[i].size(0)})")

    history = []
    initial_record = {
        "iter": 0,
        "phase": "post_initial",
    }
    pps = [perplexity_of_test(m, test_blocks_eval, device) for m in agents]
    initial_record["perplexity_per_agent"] = pps
    initial_record["mean_perplexity"] = float(np.mean(pps))
    history.append(initial_record)
    print(f"  iter 0 mean ppl={initial_record['mean_perplexity']:.2f}")

    # Self-training loop
    for t in range(1, cfg.iters + 1):
        # 1. Each agent generates synthetic tokens
        per_agent_tokens = cfg.ecosystem_token_budget // cfg.n
        all_generated_text: List[str] = []
        per_agent_samples: List[List[str]] = []
        for i, model in enumerate(agents):
            text = generate_tokens(
                model,
                tokenizer,
                n_tokens=per_agent_tokens,
                device=device,
                block_size=cfg.block_size,
                batch_size=cfg.batch_size,
            )
            all_generated_text.append(text)
            # Save short pieces for diversity computation
            pieces = [p for p in text.split("\n\n") if p.strip()]
            per_agent_samples.append(pieces[: cfg.diversity_sample])

        # 2. Re-tokenise pooled output, shuffle, split
        pooled = "\n\n".join(all_generated_text)
        pooled_blocks = tokenize_to_blocks(pooled, tokenizer, cfg.block_size)
        new_slices = split_blocks_into_slices(pooled_blocks, cfg.n, rng)

        # 3. Each agent fine-tunes on its new slice (replace mode)
        train_losses = []
        for i, model in enumerate(agents):
            loss = train_one_epoch(model, new_slices[i], device, cfg.batch_size, cfg.lr)
            train_losses.append(loss)

        # 4. Evaluate
        pps = [perplexity_of_test(m, test_blocks_eval, device) for m in agents]
        # Diversity over a sample of the new generated text
        sample_texts: List[str] = []
        for samples in per_agent_samples:
            sample_texts.extend(samples)
        # Drop empties / very short
        sample_texts = [s for s in sample_texts if len(s.split()) >= 4]
        # cap to keep embedding cheap
        sample_texts = sample_texts[: 256]
        if len(sample_texts) >= 2:
            emb = embed_texts(sample_texts)
            mpd = mean_pairwise_distance(emb)
            frob = frobenius_distance(emb)
            hsd = hill_shannon_diversity(emb)
        else:
            mpd, frob, hsd = 0.0, 0.0, 1.0
        d2 = distinct_n(sample_texts, n=2)

        record = {
            "iter": t,
            "perplexity_per_agent": pps,
            "mean_perplexity": float(np.mean(pps)),
            "std_perplexity": float(np.std(pps)),
            "median_perplexity": float(np.median(pps)),
            "train_loss_per_agent": train_losses,
            "mean_train_loss": float(np.mean([x for x in train_losses if not np.isnan(x)])) if train_losses else float("nan"),
            "distinct_2": float(d2),
            "mean_pairwise_dist": float(mpd),
            "frobenius": float(frob),
            "hsd": float(hsd),
            "n_diversity_samples": len(sample_texts),
            "pooled_blocks": pooled_blocks.size(0),
            "per_agent_blocks": [s.size(0) for s in new_slices],
        }
        history.append(record)
        print(
            f"  iter {t}: ppl={record['mean_perplexity']:.2f}  "
            f"d2={record['distinct_2']:.3f}  "
            f"HSD={record['hsd']:.2f}/{cfg.n}  "
            f"meanPD={record['mean_pairwise_dist']:.3f}",
            flush=True,
        )
        if log_path:
            with open(log_path, "a") as f:
                f.write(json.dumps({"cfg": asdict(cfg), "iter_record": record}) + "\n")

    # Free GPU memory
    for m in agents:
        del m
    torch.cuda.empty_cache()
    return {"cfg": asdict(cfg), "history": history}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ns", default="1,2,4,8,16")
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--seeds", default="0,1")
    parser.add_argument("--base_model", default="distilgpt2")
    parser.add_argument("--budget", type=int, default=100_000)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--out", default="results/finetune_ecosystem.json")
    args = parser.parse_args()

    Ns = [int(x) for x in args.ns.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = Path("logs") / (out_path.stem + ".jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        log_path.unlink()

    all_runs = []
    for n in Ns:
        for seed in seeds:
            cfg = FTRunConfig(
                n=n,
                iters=args.iters,
                base_model=args.base_model,
                ecosystem_token_budget=args.budget,
                lr=args.lr,
                batch_size=args.batch_size,
                seed=seed,
            )
            print(f"\n=== Running N={n} seed={seed} base={args.base_model} ===", flush=True)
            t0 = time.time()
            try:
                result = run_ecosystem(cfg, log_path=log_path)
                result["wall_seconds"] = time.time() - t0
                all_runs.append(result)
            except Exception as e:  # noqa: BLE001
                import traceback; traceback.print_exc()
                print(f"  FAILED: {e!r}")
            with open(out_path, "w") as f:
                json.dump(all_runs, f, indent=2)
    print(f"\nWrote {len(all_runs)} runs to {out_path}")


if __name__ == "__main__":
    main()
