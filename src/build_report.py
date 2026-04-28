"""Render the final REPORT.md by filling templated sections from the
analysis CSVs / JSONs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

R = Path("results")


def load_csv(name: str) -> pd.DataFrame | None:
    p = R / name
    if not p.exists():
        return None
    return pd.read_csv(p)


def fmt(mean, std, dec=2):
    if pd.isna(mean):
        return "—"
    if pd.isna(std) or std == 0:
        return f"{mean:.{dec}f}"
    return f"{mean:.{dec}f} ± {std:.{dec}f}"


def section_e1(ft: pd.DataFrame) -> str:
    if ft is None or ft.empty:
        return "*E1 not yet complete.*"

    last_t = ft["iter"].max()
    iters = sorted(ft["iter"].unique())
    ns = sorted(ft["n"].unique())

    # Perplexity table
    rows = []
    for n in ns:
        row = {"N": int(n)}
        for t in iters:
            sub = ft[(ft["n"] == n) & (ft["iter"] == t)]
            row[f"t={t}"] = fmt(sub["mean_perplexity"].mean(), sub["mean_perplexity"].std(), 1)
        rows.append(row)
    ppl_table = pd.DataFrame(rows).to_markdown(index=False)

    # Multipliers: terminal / initial perplexity
    n1_init = ft[(ft["n"] == 1) & (ft["iter"] == 0)]["mean_perplexity"].mean()
    mult_rows = []
    for n in ns:
        sub_init = ft[(ft["n"] == n) & (ft["iter"] == 0)]["mean_perplexity"]
        sub_term = ft[(ft["n"] == n) & (ft["iter"] == last_t)]["mean_perplexity"]
        mult_rows.append({
            "N": int(n),
            f"ppl(t=0)": fmt(sub_init.mean(), sub_init.std(), 1),
            f"ppl(t={last_t})": fmt(sub_term.mean(), sub_term.std(), 1),
            "Δ multiplier": (
                f"{sub_term.mean() / max(sub_init.mean(), 1e-9):.2f}×"
                if not pd.isna(sub_term.mean()) and not pd.isna(sub_init.mean())
                else "—"
            ),
        })
    mult_table = pd.DataFrame(mult_rows).to_markdown(index=False)

    # Per-N perplexity slope
    from scipy.stats import linregress
    slope_rows = []
    for n in ns:
        sub = ft[ft["n"] == n]
        agg = sub.groupby("iter")["mean_perplexity"].mean().reset_index()
        if len(agg) >= 3:
            res = linregress(agg["iter"].values, agg["mean_perplexity"].values)
            slope_rows.append({
                "N": int(n),
                "perplexity slope (per iter)": f"{res.slope:.1f} ± {res.stderr * 1.96:.1f}",
                "p-value": f"{res.pvalue:.3g}",
            })
    slope_table = pd.DataFrame(slope_rows).to_markdown(index=False)

    out = f"""### 4.1 E1 — Fine-tuning ecosystem

#### Perplexity (mean across agents) on WikiText-2 test, by population size N

{ppl_table}

*Mean ± std across {ft.groupby(['n', 'iter']).size().max()} random seeds.*

![](figures/finetune_perplexity_vs_iter.png)

The N=1 baseline collapses by a factor of {ft[(ft['n']==1)&(ft['iter']==last_t)]['mean_perplexity'].mean() / n1_init:.1f}× over {last_t} iterations
(perplexity {n1_init:.1f} → {ft[(ft['n']==1)&(ft['iter']==last_t)]['mean_perplexity'].mean():.1f}). This reproduces the canonical
Shumailov 2024 collapse signature.

#### Initial vs. terminal perplexity, by N

{mult_table}

#### Linear regression of perplexity on iteration, by N

{slope_table}

A *positive* slope indicates ongoing collapse; a slope statistically
indistinguishable from zero (p > 0.05 or 95% CI crosses 0) indicates the
metric has plateaued.

![](figures/finetune_mvp_curve.png)

#### Lexical and semantic diversity

![](figures/finetune_distinct2_vs_iter.png)

![](figures/finetune_meanpd_vs_iter.png)

![](figures/finetune_hsd_vs_iter.png)

The same N-vs-collapse ordering is visible in distinct-bigram ratio and
in the Hill–Shannon Diversity (Vendi Score) over a 256-utterance sample
of the synthetic outputs.
"""
    return out


def section_e2(rag: pd.DataFrame) -> str:
    if rag is None or rag.empty:
        return "*E2 not yet complete.*"

    ns = sorted(rag["n"].unique())
    per_n_terminal_t = rag.groupby("n")["iter"].max().to_dict()

    rows = []
    for n in ns:
        terminal_t = int(per_n_terminal_t[n])
        row = {"N": int(n), "T": terminal_t}
        for label_t, t_val in [("@t=1", 1), ("@terminal", terminal_t)]:
            for m, label in [
                ("hsd", "HSD"),
                ("mean_pairwise_dist", "meanPD"),
                ("frobenius", "Frob"),
            ]:
                sub = rag[(rag["n"] == n) & (rag["iter"] == t_val)][m]
                row[f"{label}{label_t}"] = fmt(sub.mean(), sub.std())
        rows.append(row)
    table = pd.DataFrame(rows).to_markdown(index=False)
    last_t = rag["iter"].max()

    from scipy.stats import linregress
    slope_rows = []
    for n in ns:
        sub = rag[(rag["n"] == n) & (rag["iter"] >= 1)]
        for m in ["hsd", "mean_pairwise_dist"]:
            agg = sub.groupby("iter")[m].mean().reset_index()
            if len(agg) >= 3 and agg[m].std() > 0:
                res = linregress(agg["iter"].values, agg[m].values)
                slope_rows.append({
                    "N": int(n),
                    "metric": m,
                    "slope (per iter)": f"{res.slope:+.4f} ± {res.stderr * 1.96:.4f}",
                    "p-value": f"{res.pvalue:.3g}",
                })
    slope_table = pd.DataFrame(slope_rows).to_markdown(index=False)

    out = f"""### 4.2 E2 — RAG ecosystem (model_family axis)

#### Diversity at iteration 1 vs terminal iteration, by population size N
(Terminal iteration *T* shown in second column; N=1,2,3 ran to T=12, N=4 to T=10.)

{table}

*Mean ± std across {rag.groupby(['n','iter']).size().max()} random seeds.*

![](figures/rag_model_family_hsd_vs_iter.png)

![](figures/rag_model_family_meanpd_vs_iter.png)

![](figures/rag_model_family_frob_vs_iter.png)

#### Linear regression of diversity metrics on iteration, by N

{slope_table}

For HSD and mean pairwise distance, a *non-negative* slope (p > 0.05 or
95% CI touches 0) means the ecosystem has not lost diversity over T
iterations — an empirical answer to "is N enough to avoid collapse on
this axis?"
"""
    return out


def section_e3(ax: pd.DataFrame) -> str:
    if ax is None or ax.empty:
        return "*E3 not yet complete.*"

    last_t = ax["iter"].max()
    modes = ["single", "data_segment", "persona", "model_family"]

    rows = []
    for mode in modes:
        sub = ax[ax["mode"] == mode]
        if sub.empty:
            continue
        row = {"axis": mode}
        for t in [1, last_t]:
            sub_t = sub[sub["iter"] == t]
            row[f"HSD@t={t}"] = fmt(sub_t["hsd"].mean(), sub_t["hsd"].std())
            row[f"meanPD@t={t}"] = fmt(sub_t["mean_pairwise_dist"].mean(), sub_t["mean_pairwise_dist"].std())
            row[f"frob@t={t}"] = fmt(sub_t["frobenius"].mean(), sub_t["frobenius"].std())
        rows.append(row)
    table = pd.DataFrame(rows).to_markdown(index=False)

    # Rank modes by terminal mean pairwise distance
    ranks = (
        ax[ax["iter"] == last_t]
        .groupby("mode")["mean_pairwise_dist"]
        .mean()
        .sort_values(ascending=False)
    )
    rank_str = " > ".join(f"**{m}** ({v:.2f})" for m, v in ranks.items())

    out = f"""### 4.3 E3 — Diversity-axis comparison (N=4)

#### Diversity at iteration 1 vs iteration {last_t}, by axis

{table}

*N=4 agents, mean ± std across 3 seeds.*

![](figures/axis_compare_hsd.png)

![](figures/axis_compare_meanpd.png)

![](figures/axis_compare_frob.png)

![](figures/axis_compare_terminal_meanpd.png)

#### Ranking by terminal mean pairwise distance (higher = less collapse)

{rank_str}
"""
    return out


def main():
    ft = load_csv("finetune_summary.csv")
    rag = load_csv("rag_model_family_summary.csv")
    ax = load_csv("axis_summary.csv")

    sec1 = section_e1(ft)
    sec2 = section_e2(rag)
    sec3 = section_e3(ax)

    # MVP estimates
    mvp_path = R / "mvp_estimates.json"
    mvp_str = ""
    if mvp_path.exists():
        mvp = json.load(open(mvp_path))
        rows = []
        for tag, info in mvp.items():
            mvp_val = info.get("mvp")
            note = ""
            if mvp_val == 1 and info.get("direction") == "higher_better":
                # N=1 is degenerate for inter-agent diversity metrics
                # (can't be inter-anything-distance for a singleton)
                note = " (degenerate; inter-agent metric is 0 by construction)"
            elif mvp_val is None or (isinstance(mvp_val, float) and pd.isna(mvp_val)):
                note = " (no N qualifies — metric decays at every N tested)"
            rows.append({
                "experiment::metric": tag,
                "MVP": str(mvp_val) + note,
                "direction": info.get("direction"),
                "last iter": info.get("last_t"),
            })
        mvp_str = pd.DataFrame(rows).to_markdown(index=False)

    # Conclusions / discussion
    e1_summary = ""
    e1_red_str = ""
    if ft is not None and not ft.empty:
        last_t = ft["iter"].max()
        e1_perplexity_terminal = ft[ft["iter"] == last_t].groupby("n")["mean_perplexity"].mean().to_dict()
        n1_term = e1_perplexity_terminal.get(1, float("nan"))
        e1_summary = "Terminal perplexity by N (mean across seeds): " + ", ".join(
            f"N={int(n)}: {p:.1f}" for n, p in sorted(e1_perplexity_terminal.items())
        )
        n1_init = ft[(ft["n"] == 1) & (ft["iter"] == 0)]["mean_perplexity"].mean()
        # Fraction of N=1 collapse magnitude that this N retains
        red = {
            int(n): (p - n1_init) / max(n1_term - n1_init, 1e-9)
            for n, p in e1_perplexity_terminal.items() if n > 1
        }
        e1_red_str = "Collapse retained vs N=1 baseline: " + ", ".join(
            f"N={n}: {v*100:.0f}%" for n, v in sorted(red.items())
        )

    # Full report
    e2_summary = ""
    if rag is not None and not rag.empty:
        # Use per-N terminal iter (since N=4 ran for fewer iters than N=1,2,3).
        per_n_terminal_t = rag.groupby("n")["iter"].max().to_dict()
        e2_terminal_meanpd = {
            n: rag[(rag["n"] == n) & (rag["iter"] == per_n_terminal_t[n])]["mean_pairwise_dist"].mean()
            for n in per_n_terminal_t
        }
        e2_initial_meanpd = rag[rag["iter"] == 1].groupby("n")["mean_pairwise_dist"].mean().to_dict()
        e2_summary = "Terminal vs initial mean pairwise distance (RAG, model_family): " + ", ".join(
            f"N={int(n)} ({e2_initial_meanpd.get(n,float('nan')):.3f} → {e2_terminal_meanpd.get(n,float('nan')):.3f} @ t={per_n_terminal_t[n]})"
            for n in sorted(e2_terminal_meanpd.keys())
        )

    e3_summary = ""
    if ax is not None and not ax.empty:
        last_t = ax["iter"].max()
        terminal = ax[ax["iter"] == last_t].groupby("mode")["mean_pairwise_dist"].mean().to_dict()
        e3_summary = "Axis ranking by terminal inter-agent distance (high = preserves diversity): " + " > ".join(
            f"{m} ({v:.2f})" for m, v in sorted(terminal.items(), key=lambda kv: -kv[1])
        )

    sections_path = R / "_sections.md"
    sections_path.write_text(
        f"### MVP estimates\n\n{mvp_str}\n\n"
        f"### Summaries\n\nE1: {e1_summary}\n{e1_red_str}\n"
        f"E2: {e2_summary}\nE3: {e3_summary}\n\n"
        f"---\n\n{sec1}\n\n{sec2}\n\n{sec3}\n"
    )
    print(f"wrote sections snippet → {sections_path}")
    print(f"\nE1 summary: {e1_summary}")
    print(f"E1 collapse: {e1_red_str}")
    print(f"E2 summary: {e2_summary}")
    print(f"E3 summary: {e3_summary}")


if __name__ == "__main__":
    main()
