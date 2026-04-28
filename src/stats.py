"""Statistical analysis for the collapse experiments.

For each (experiment, metric, N) we fit a simple linear model
    y_t = a + b·t + ε
and report the slope b along with its 95% CI. A non-positive slope at the
final iteration window is consistent with "this N is large enough to
prevent collapse" (H1). A monotonically decreasing slope as N grows is
consistent with "MVP grows with horizon" (H2).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

R = Path("results")


def slope_ci(xs: np.ndarray, ys: np.ndarray) -> dict:
    if len(xs) < 3 or np.std(xs) == 0:
        return {"slope": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"), "p": float("nan"), "n": int(len(xs))}
    res = stats.linregress(xs, ys)
    slope = res.slope
    se = res.stderr
    df = max(len(xs) - 2, 1)
    tcrit = stats.t.ppf(0.975, df)
    return {
        "slope": float(slope),
        "ci_lo": float(slope - tcrit * se),
        "ci_hi": float(slope + tcrit * se),
        "p": float(res.pvalue),
        "n": int(len(xs)),
        "intercept": float(res.intercept),
        "r2": float(res.rvalue ** 2),
    }


def analyze_csv(path: Path, metric: str, group: str = "n") -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["iter"] >= 1].copy()
    rows = []
    for n_val, sub in df.groupby(group):
        # Aggregate across seeds: pool per (iter)
        slope_in = slope_ci(sub["iter"].values, sub[metric].values)
        rows.append({"group": group, "value": n_val, "metric": metric, **slope_in})
    return pd.DataFrame(rows)


def main():
    out = {}
    for fname, metrics in [
        ("finetune_summary.csv", ["mean_perplexity", "distinct_2", "mean_pairwise_dist", "hsd"]),
        ("rag_model_family_summary.csv", ["hsd", "mean_pairwise_dist", "frobenius", "distinct_2"]),
    ]:
        path = R / fname
        if not path.exists():
            continue
        for metric in metrics:
            df = analyze_csv(path, metric)
            df.to_csv(R / f"slopes_{fname.replace('.csv','')}_{metric}.csv", index=False)
            print(f"\n=== {fname} | metric={metric} ===")
            print(df.to_string(index=False))
            out[f"{fname}::{metric}"] = df.to_dict(orient="records")

    # Axis comparison: paired comparison at fixed N=4
    ax_path = R / "axis_summary.csv"
    if ax_path.exists():
        df = pd.read_csv(ax_path)
        df = df[df["iter"] >= 1].copy()
        # Compute slopes per mode
        rows = []
        for mode, sub in df.groupby("mode"):
            slope_in = slope_ci(sub["iter"].values, sub["hsd"].values)
            rows.append({"mode": mode, "metric": "hsd", **slope_in})
            slope_md = slope_ci(sub["iter"].values, sub["mean_pairwise_dist"].values)
            rows.append({"mode": mode, "metric": "mean_pairwise_dist", **slope_md})
        ax_slopes = pd.DataFrame(rows)
        ax_slopes.to_csv(R / "axis_slopes.csv", index=False)
        print("\n=== axis comparison slopes ===")
        print(ax_slopes.to_string(index=False))
        out["axis_slopes"] = ax_slopes.to_dict(orient="records")

    with open(R / "stats_summary.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {R / 'stats_summary.json'}")


if __name__ == "__main__":
    main()
