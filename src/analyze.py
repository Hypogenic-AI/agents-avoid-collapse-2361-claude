"""Aggregate experiment results and produce headline figures.

Reads JSON outputs from src/finetune_ecosystem.py, src/rag_ecosystem.py and
src/axis_compare.py, and writes:

  figures/finetune_perplexity_vs_iter.png
  figures/finetune_diversity_vs_iter.png
  figures/finetune_mvp_curve.png
  figures/rag_diversity_vs_iter.png
  figures/rag_mvp_curve.png
  figures/axis_compare.png

  results/finetune_summary.csv
  results/rag_summary.csv
  results/axis_summary.csv
  results/mvp_estimates.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS = Path("results")
FIGURES = Path("figures")
FIGURES.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def load_runs(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


def runs_to_df(runs: List[Dict], extra_cols: List[str] = None) -> pd.DataFrame:
    """Flatten list of run dicts into a long-format DataFrame keyed on
    (n, seed, mode, iter)."""
    rows = []
    for run in runs:
        cfg = run["cfg"]
        for record in run["history"]:
            row = {
                "n": cfg.get("n"),
                "seed": cfg.get("seed"),
                "mode": cfg.get("mode", "default"),
                "base_model": cfg.get("base_model", "n/a"),
                "iters": cfg.get("iters"),
                "iter": record.get("iter"),
                "mean_perplexity": record.get("mean_perplexity"),
                "median_perplexity": record.get("median_perplexity"),
                "std_perplexity": record.get("std_perplexity"),
                "distinct_2": record.get("distinct_2"),
                "mean_pairwise_dist": record.get("mean_pairwise_dist"),
                "frobenius": record.get("frobenius"),
                "hsd": record.get("hsd"),
            }
            if extra_cols:
                for c in extra_cols:
                    row[c] = record.get(c)
            rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Plot helpers
# --------------------------------------------------------------------------- #

def _color(n: int, ns: List[int]) -> tuple:
    cmap = plt.get_cmap("viridis")
    if len(ns) == 1:
        return cmap(0.5)
    idx = ns.index(n) / (len(ns) - 1)
    return cmap(idx)


def plot_metric_vs_iter(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    out_path: Path,
    log_y: bool = False,
):
    fig, ax = plt.subplots(figsize=(7, 5))
    ns = sorted(df["n"].unique())
    for n in ns:
        sub = df[df["n"] == n]
        agg = sub.groupby("iter")[metric].agg(["mean", "std", "count"]).reset_index()
        c = _color(n, ns)
        ax.plot(agg["iter"], agg["mean"], "-o", color=c, label=f"N={n}")
        if (agg["count"] > 1).any():
            ax.fill_between(
                agg["iter"],
                agg["mean"] - agg["std"],
                agg["mean"] + agg["std"],
                color=c,
                alpha=0.15,
            )
    ax.set_xlabel("Self-training iteration t")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if log_y:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Population", loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_mvp_curve(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    out_path: Path,
    iters_to_show: List[int] = None,
):
    """For each iteration, show the metric as a function of N."""
    fig, ax = plt.subplots(figsize=(7, 5))
    if iters_to_show is None:
        iters_to_show = sorted(df["iter"].unique())
        # Use a few snapshots, ensuring the final iter is included exactly once
        if len(iters_to_show) > 5:
            step = max(1, len(iters_to_show) // 5)
            picked = iters_to_show[::step]
            if iters_to_show[-1] not in picked:
                picked.append(iters_to_show[-1])
            iters_to_show = picked
    cmap = plt.get_cmap("plasma")
    for k, t in enumerate(iters_to_show):
        sub = df[df["iter"] == t]
        agg = sub.groupby("n")[metric].agg(["mean", "std"]).reset_index()
        c = cmap(k / max(len(iters_to_show) - 1, 1))
        ax.errorbar(agg["n"], agg["mean"], yerr=agg["std"], fmt="-o", color=c, label=f"t={t}")
    ax.set_xlabel("Population size N")
    ax.set_ylabel(ylabel)
    ax.set_xscale("log")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Iteration", loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  wrote {out_path}")


# --------------------------------------------------------------------------- #
# MVP estimation
# --------------------------------------------------------------------------- #

def estimate_mvp(
    df: pd.DataFrame,
    metric: str,
    direction: str = "lower_better",
    horizon_frac: float = 0.5,
) -> Dict:
    """Estimate the minimum N such that the *terminal* value of the metric
    is significantly less harmful than the N=1 baseline.

    `direction`:
      * "lower_better" (perplexity): MVP = smallest N s.t.
        terminal(N) ≤ initial(N=1) + horizon_frac × (terminal(N=1) - initial(N=1)).
        Intuition: the ecosystem only "moved" half as far toward catastrophic
        N=1 collapse.
      * "higher_better" (diversity): MVP = smallest N s.t. the slope of the
        metric over t is non-negative (95%-CI lower bound ≥ 0). Reflects
        "diversity does not decay", which is the operational definition of
        "no collapse" on a diversity axis.
    """
    last_t = df["iter"].max()
    if pd.isna(last_t):
        return {"mvp": None, "reason": "no data"}

    snap = df[df["iter"] == last_t].copy()
    by_n_terminal = snap.groupby("n")[metric].mean().sort_index()

    if direction == "lower_better":
        n1_iter0 = df[(df["n"] == 1) & (df["iter"] == 0)][metric].mean()
        if pd.isna(n1_iter0):
            n1_iter0 = df[df["iter"] == 0][metric].mean()
        n1_terminal = by_n_terminal.get(1, by_n_terminal.iloc[0])
        slack = n1_terminal - n1_iter0
        threshold = n1_iter0 + horizon_frac * slack
        qualifying = by_n_terminal[by_n_terminal <= threshold]
        mvp = int(qualifying.index.min()) if not qualifying.empty else None
        return {
            "metric": metric,
            "direction": direction,
            "last_t": int(last_t),
            "n1_iter0": float(n1_iter0),
            "n1_terminal": float(n1_terminal),
            "horizon_frac": horizon_frac,
            "threshold": float(threshold),
            "metric_terminal_by_n": {int(k): float(v) for k, v in by_n_terminal.items()},
            "mvp": mvp,
        }

    # higher_better: slope-based test
    from scipy.stats import linregress
    slopes = {}
    for n_val, sub in df[df["iter"] >= 1].groupby("n"):
        # Aggregate to per-iter mean across seeds, then regress on iter
        agg = sub.groupby("iter")[metric].mean().reset_index()
        if len(agg) >= 3:
            res = linregress(agg["iter"].values, agg[metric].values)
            slopes[int(n_val)] = {
                "slope": float(res.slope),
                "se": float(res.stderr),
                "p": float(res.pvalue),
            }
    # MVP definition: smallest N such that the metric does not significantly
    # decay (p > 0.05 OR slope >= 0). Equivalently: 95% CI of the slope
    # touches or crosses 0 — *and* the metric is non-trivially > 0 at
    # iteration T (otherwise N=1 trivially "qualifies" because HSD=1 forever).
    qualifying_ns = []
    for n_val, info in sorted(slopes.items()):
        if info["slope"] >= 0 or info["p"] > 0.05:
            terminal = by_n_terminal.get(n_val, 0.0)
            if terminal > 0.05:  # exclude degenerate flat-at-zero cases
                qualifying_ns.append(n_val)
    mvp = qualifying_ns[0] if qualifying_ns else None
    return {
        "metric": metric,
        "direction": direction,
        "last_t": int(last_t),
        "metric_terminal_by_n": {int(k): float(v) for k, v in by_n_terminal.items()},
        "slopes_by_n": slopes,
        "qualifying_ns": qualifying_ns,
        "mvp": mvp,
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    mvp_records = []

    # E1: fine-tuning ecosystem
    ft_runs = load_runs(RESULTS / "finetune_ecosystem.json")
    if ft_runs:
        ft = runs_to_df(ft_runs)
        ft.to_csv(RESULTS / "finetune_summary.csv", index=False)
        print(f"E1 fine-tune ecosystem: {len(ft)} rows from {len(ft_runs)} runs")
        # Drop t=0 for plots that start at t=1 (but include iter=0 baseline in PPL plot)
        plot_metric_vs_iter(
            ft, "mean_perplexity", "Mean perplexity on WikiText-2 test",
            "E1: Perplexity vs. iteration (fine-tuning ecosystem)",
            FIGURES / "finetune_perplexity_vs_iter.png",
            log_y=True,
        )
        ft_t1 = ft[ft["iter"] >= 1]
        plot_metric_vs_iter(
            ft_t1, "distinct_2", "Distinct-bigram ratio of new outputs",
            "E1: Lexical diversity vs. iteration",
            FIGURES / "finetune_distinct2_vs_iter.png",
        )
        plot_metric_vs_iter(
            ft_t1, "mean_pairwise_dist", "Mean pairwise cosine distance of new outputs",
            "E1: Semantic diversity vs. iteration",
            FIGURES / "finetune_meanpd_vs_iter.png",
        )
        plot_metric_vs_iter(
            ft_t1, "hsd", "Hill–Shannon Diversity (Vendi Score) of new outputs",
            "E1: Effective # distinct outputs vs. iteration",
            FIGURES / "finetune_hsd_vs_iter.png",
        )
        plot_mvp_curve(
            ft_t1, "mean_perplexity", "Mean perplexity at iteration t",
            "E1: Perplexity as a function of population size N",
            FIGURES / "finetune_mvp_curve.png",
        )
        mvp_records.append(("E1_perplexity", estimate_mvp(ft, "mean_perplexity", direction="lower_better")))
        mvp_records.append(("E1_distinct2", estimate_mvp(ft_t1, "distinct_2", direction="higher_better")))
        mvp_records.append(("E1_meanpd", estimate_mvp(ft_t1, "mean_pairwise_dist", direction="higher_better")))
        mvp_records.append(("E1_hsd", estimate_mvp(ft_t1, "hsd", direction="higher_better")))
    else:
        print("E1 fine-tune ecosystem: no results yet")

    # E2: RAG ecosystem.
    # For "model_family", merge the main sweep (N=1,2,3) with the N=4 axis-
    # comparison run so the across-N curve has all population sizes.
    for path, tag, extra_paths in [
        (RESULTS / "rag_model_family.json", "model_family", [RESULTS / "rag_model_family_n4.json"]),
        (RESULTS / "rag_persona.json", "persona", []),
        (RESULTS / "rag_single.json", "single", []),
    ]:
        rag_runs = load_runs(path)
        for extra in extra_paths:
            rag_runs = rag_runs + load_runs(extra)
        if not rag_runs:
            continue
        rag = runs_to_df(rag_runs)
        rag["mode"] = tag
        rag.to_csv(RESULTS / f"rag_{tag}_summary.csv", index=False)
        print(f"E2 RAG ecosystem ({tag}): {len(rag)} rows from {len(rag_runs)} runs")
        for m, ylabel, fn in [
            ("hsd", "Hill–Shannon Diversity (effective # distinct posts)", f"rag_{tag}_hsd_vs_iter.png"),
            ("mean_pairwise_dist", "Mean pairwise cosine distance", f"rag_{tag}_meanpd_vs_iter.png"),
            ("frobenius", "Frobenius norm of pairwise distance matrix", f"rag_{tag}_frob_vs_iter.png"),
            ("distinct_2", "Distinct-bigram ratio", f"rag_{tag}_distinct2_vs_iter.png"),
        ]:
            plot_metric_vs_iter(
                rag, m, ylabel,
                f"E2 ({tag}): {ylabel} vs. iteration",
                FIGURES / fn,
            )
        plot_mvp_curve(
            rag, "hsd", "HSD at iteration t",
            f"E2 ({tag}): HSD as a function of population size N",
            FIGURES / f"rag_{tag}_mvp_curve.png",
        )
        mvp_records.append((f"E2_{tag}_hsd", estimate_mvp(rag, "hsd", direction="higher_better")))
        mvp_records.append((f"E2_{tag}_meanpd", estimate_mvp(rag, "mean_pairwise_dist", direction="higher_better")))

    # E3: axis comparison — collected from a separate file with mode field
    axis_path = RESULTS / "axis_compare.json"
    axis_runs = load_runs(axis_path)
    if axis_runs:
        ax_df = runs_to_df(axis_runs)
        ax_df.to_csv(RESULTS / "axis_summary.csv", index=False)
        print(f"E3 axis compare: {len(ax_df)} rows")
        mode_order = ["single", "data_segment", "persona", "model_family"]
        mode_palette = {
            "single": "tab:gray",
            "data_segment": "tab:blue",
            "persona": "tab:orange",
            "model_family": "tab:red",
        }
        for metric, ylabel, fname in [
            ("hsd", "HSD (Vendi Score) of new outputs", "axis_compare_hsd.png"),
            ("mean_pairwise_dist", "Mean pairwise cosine distance of new outputs", "axis_compare_meanpd.png"),
            ("frobenius", "Frobenius norm of pairwise distance matrix", "axis_compare_frob.png"),
            ("distinct_2", "Distinct-bigram ratio of new outputs", "axis_compare_distinct2.png"),
        ]:
            fig, ax = plt.subplots(figsize=(7, 5))
            for mode in mode_order:
                sub = ax_df[(ax_df["mode"] == mode) & (ax_df["iter"] >= 1)]
                if sub.empty:
                    continue
                agg = sub.groupby("iter")[metric].agg(["mean", "std", "count"]).reset_index()
                c = mode_palette[mode]
                ax.plot(agg["iter"], agg["mean"], "-o", color=c, label=mode)
                if (agg["count"] > 1).any():
                    ax.fill_between(
                        agg["iter"],
                        agg["mean"] - agg["std"],
                        agg["mean"] + agg["std"],
                        color=c, alpha=0.15,
                    )
            ax.set_xlabel("Iteration t")
            ax.set_ylabel(ylabel)
            ax.set_title(f"E3 (N=4): {ylabel} vs. iteration")
            ax.grid(True, alpha=0.3)
            ax.legend(title="Diversity axis")
            fig.tight_layout()
            fig.savefig(FIGURES / fname, dpi=130)
            plt.close(fig)
            print(f"  wrote {FIGURES / fname}")
        # Per-mode slope MVP-style: terminal mean ± std bar chart
        fig, ax = plt.subplots(figsize=(7, 5))
        last_t = ax_df["iter"].max()
        snap = ax_df[ax_df["iter"] == last_t]
        means = snap.groupby("mode")["mean_pairwise_dist"].agg(["mean", "std"]).reindex(mode_order)
        x = np.arange(len(mode_order))
        ax.bar(x, means["mean"], yerr=means["std"], capsize=4,
               color=[mode_palette[m] for m in mode_order])
        ax.set_xticks(x)
        ax.set_xticklabels(mode_order)
        ax.set_ylabel(f"Mean pairwise cosine distance @ iter {last_t}")
        ax.set_title(f"E3 (N=4): Terminal inter-agent distance by axis")
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(FIGURES / "axis_compare_terminal_meanpd.png", dpi=130)
        plt.close(fig)
        print(f"  wrote {FIGURES / 'axis_compare_terminal_meanpd.png'}")
    else:
        print("E3 axis compare: no results yet")

    # MVP estimates
    out = {tag: rec for tag, rec in mvp_records}
    with open(RESULTS / "mvp_estimates.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote MVP estimates → {RESULTS / 'mvp_estimates.json'}")


if __name__ == "__main__":
    main()
