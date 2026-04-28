"""Build markdown tables for the REPORT.md from results CSVs.

Outputs:
  results/tables/finetune_table.md
  results/tables/rag_model_family_table.md
  results/tables/axis_table.md

Each table is mean ± std across seeds.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

R = Path("results")
T = R / "tables"
T.mkdir(parents=True, exist_ok=True)


def fmt(mean, std):
    if pd.isna(mean):
        return "—"
    if pd.isna(std) or std == 0:
        return f"{mean:.2f}"
    return f"{mean:.2f} ± {std:.2f}"


def finetune_table():
    path = R / "finetune_summary.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    iters = sorted(df["iter"].unique())
    rows = []
    for n in sorted(df["n"].unique()):
        row = {"N": int(n)}
        for t in iters:
            sub = df[(df["n"] == n) & (df["iter"] == t)]
            row[f"t={t}"] = fmt(sub["mean_perplexity"].mean(), sub["mean_perplexity"].std())
        rows.append(row)
    out = pd.DataFrame(rows)
    md = "## E1 — Mean perplexity (lower = better) on WikiText-2 test\n\n"
    md += out.to_markdown(index=False)
    md += "\n\n*Numbers are mean ± std across 2 random seeds.*\n"
    (T / "finetune_table.md").write_text(md)
    print("wrote", T / "finetune_table.md")
    return out


def rag_table():
    path = R / "rag_model_family_summary.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    rows = []
    for n in sorted(df["n"].unique()):
        sub_n = df[df["n"] == n]
        terminal_t = int(sub_n["iter"].max())
        row = {
            "N": int(n),
            f"HSD@t=1": fmt(*_grab(df, n, 1, "hsd")),
            f"HSD@terminal": fmt(*_grab(df, n, terminal_t, "hsd")) + f" (t={terminal_t})",
            f"meanPD@t=1": fmt(*_grab(df, n, 1, "mean_pairwise_dist")),
            f"meanPD@terminal": fmt(*_grab(df, n, terminal_t, "mean_pairwise_dist")) + f" (t={terminal_t})",
            f"frob@t=1": fmt(*_grab(df, n, 1, "frobenius")),
            f"frob@terminal": fmt(*_grab(df, n, terminal_t, "frobenius")) + f" (t={terminal_t})",
        }
        rows.append(row)
    out = pd.DataFrame(rows)
    md = "## E2 — Diversity at iteration 1 vs terminal iteration (RAG ecosystem, model_family axis)\n\n"
    md += out.to_markdown(index=False)
    md += "\n\n*N=1,2,3 ran for 12 iterations; N=4 ran for 10 iterations (from the E3 axis-comparison data). β=0.2 retrieval, mean ± std across 3 seeds.*\n"
    (T / "rag_model_family_table.md").write_text(md)
    print("wrote", T / "rag_model_family_table.md")
    return out


def _grab(df, n, t, metric):
    sub = df[(df["n"] == n) & (df["iter"] == t)]
    return sub[metric].mean(), sub[metric].std()


def axis_table():
    path = R / "axis_summary.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    last_t = df["iter"].max()
    rows = []
    for mode in ["single", "data_segment", "persona", "model_family"]:
        sub = df[df["mode"] == mode]
        if sub.empty:
            continue
        row = {"axis": mode}
        for t in [1, last_t]:
            for m in ["hsd", "mean_pairwise_dist", "frobenius", "distinct_2"]:
                v = sub[sub["iter"] == t][m]
                row[f"{m}@t={t}"] = fmt(v.mean(), v.std())
        rows.append(row)
    out = pd.DataFrame(rows)
    md = f"## E3 — Diversity-axis comparison at N=4, iteration 1 vs iteration {last_t}\n\n"
    md += out.to_markdown(index=False)
    md += f"\n\n*N=4 agents, {last_t} iterations, mean ± std across 3 seeds.*\n"
    (T / "axis_table.md").write_text(md)
    print("wrote", T / "axis_table.md")
    return out


def main():
    finetune_table()
    rag_table()
    axis_table()


if __name__ == "__main__":
    main()
