### MVP estimates

| experiment::metric     | MVP                                                     | direction     |   last iter |
|:-----------------------|:--------------------------------------------------------|:--------------|------------:|
| E1_perplexity          | 4                                                       | lower_better  |           6 |
| E1_distinct2           | None (no N qualifies — metric decays at every N tested) | higher_better |           6 |
| E1_meanpd              | None (no N qualifies — metric decays at every N tested) | higher_better |           6 |
| E1_hsd                 | 1 (degenerate; inter-agent metric is 0 by construction) | higher_better |           6 |
| E2_model_family_hsd    | 1 (degenerate; inter-agent metric is 0 by construction) | higher_better |          12 |
| E2_model_family_meanpd | 2                                                       | higher_better |          12 |

### Summaries

E1: Terminal perplexity by N (mean across seeds): N=1: 495.2, N=2: 308.4, N=4: 275.1, N=8: 245.4, N=16: 180.5
Collapse retained vs N=1 baseline: N=2: 56%, N=4: 49%, N=8: 42%, N=16: 27%
E2: Terminal vs initial mean pairwise distance (RAG, model_family): N=1 (0.000 → 0.000 @ t=12), N=2 (0.885 → 0.900 @ t=12), N=3 (0.807 → 0.741 @ t=12), N=4 (0.787 → 0.746 @ t=10)
E3: Axis ranking by terminal inter-agent distance (high = preserves diversity): model_family (0.75) > data_segment (0.40) > persona (0.40) > single (0.32)

---

### 4.1 E1 — Fine-tuning ecosystem

#### Perplexity (mean across agents) on WikiText-2 test, by population size N

|   N | t=0        | t=1        | t=2         | t=3         | t=4         | t=5          | t=6          |
|----:|:-----------|:-----------|:------------|:------------|:------------|:-------------|:-------------|
|   1 | 66.4 ± 0.0 | 87.6 ± 2.1 | 125.5 ± 5.3 | 180.6 ± 2.9 | 248.3 ± 7.9 | 362.2 ± 35.7 | 495.2 ± 57.8 |
|   2 | 69.6 ± 0.2 | 87.9 ± 0.4 | 117.3 ± 0.5 | 153.7 ± 0.9 | 196.8 ± 3.7 | 253.6 ± 2.4  | 308.4 ± 2.4  |
|   4 | 75.0 ± 0.0 | 91.1 ± 0.3 | 118.8 ± 1.8 | 152.6 ± 3.5 | 188.4 ± 3.6 | 232.0 ± 2.9  | 275.1 ± 12.0 |
|   8 | 81.0 ± 0.1 | 95.7 ± 0.3 | 119.5 ± 0.5 | 148.3 ± 0.6 | 180.4 ± 2.1 | 210.4 ± 0.7  | 245.4 ± 4.3  |
|  16 | 85.8 ± 0.1 | 93.9 ± 0.2 | 108.8 ± 0.1 | 126.0 ± 0.8 | 143.7 ± 1.2 | 161.9 ± 0.9  | 180.5 ± 0.6  |

*Mean ± std across 2 random seeds.*

![](figures/finetune_perplexity_vs_iter.png)

The N=1 baseline collapses by a factor of 7.5× over 6 iterations
(perplexity 66.4 → 495.2). This reproduces the canonical
Shumailov 2024 collapse signature.

#### Initial vs. terminal perplexity, by N

|   N | ppl(t=0)   | ppl(t=6)     | Δ multiplier   |
|----:|:-----------|:-------------|:---------------|
|   1 | 66.4 ± 0.0 | 495.2 ± 57.8 | 7.45×          |
|   2 | 69.6 ± 0.2 | 308.4 ± 2.4  | 4.43×          |
|   4 | 75.0 ± 0.0 | 275.1 ± 12.0 | 3.67×          |
|   8 | 81.0 ± 0.1 | 245.4 ± 4.3  | 3.03×          |
|  16 | 85.8 ± 0.1 | 180.5 ± 0.6  | 2.10×          |

#### Linear regression of perplexity on iteration, by N

|   N | perplexity slope (per iter)   |   p-value |
|----:|:------------------------------|----------:|
|   1 | 69.9 ± 17.6                   |  0.00056  |
|   2 | 40.3 ± 6.1                    |  4.79e-05 |
|   4 | 34.0 ± 4.0                    |  1.46e-05 |
|   8 | 28.0 ± 2.7                    |  4.95e-06 |
|  16 | 16.3 ± 1.4                    |  3.09e-06 |

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


### 4.2 E2 — RAG ecosystem (model_family axis)

#### Diversity at iteration 1 vs terminal iteration, by population size N
(Terminal iteration *T* shown in second column; N=1,2,3 ran to T=12, N=4 to T=10.)

|   N |   T | HSD@t=1     | meanPD@t=1   | Frob@t=1    | HSD@terminal   | meanPD@terminal   | Frob@terminal   |
|----:|----:|:------------|:-------------|:------------|:---------------|:------------------|:----------------|
|   1 |  12 | 1.00        | 0.00         | 0.00        | 1.00           | 0.00              | 0.00            |
|   2 |  12 | 1.99 ± 0.01 | 0.89 ± 0.04  | 1.25 ± 0.06 | 1.99 ± 0.00    | 0.90 ± 0.01       | 1.27 ± 0.01     |
|   3 |  12 | 2.81 ± 0.14 | 0.81 ± 0.08  | 2.01 ± 0.16 | 2.63 ± 0.24    | 0.74 ± 0.09       | 1.91 ± 0.17     |
|   4 |  10 | 3.32 ± 0.27 | 0.79 ± 0.04  | 2.87 ± 0.08 | 3.15 ± 0.20    | 0.75 ± 0.02       | 2.77 ± 0.02     |

*Mean ± std across 3 random seeds.*

![](figures/rag_model_family_hsd_vs_iter.png)

![](figures/rag_model_family_meanpd_vs_iter.png)

![](figures/rag_model_family_frob_vs_iter.png)

#### Linear regression of diversity metrics on iteration, by N

|   N | metric             | slope (per iter)   |   p-value |
|----:|:-------------------|:-------------------|----------:|
|   2 | hsd                | -0.0004 ± 0.0008   |     0.407 |
|   2 | mean_pairwise_dist | -0.0016 ± 0.0038   |     0.439 |
|   3 | hsd                | -0.0101 ± 0.0150   |     0.215 |
|   3 | mean_pairwise_dist | -0.0043 ± 0.0047   |     0.107 |
|   4 | hsd                | -0.0170 ± 0.0194   |     0.125 |
|   4 | mean_pairwise_dist | -0.0042 ± 0.0041   |     0.082 |

For HSD and mean pairwise distance, a *non-negative* slope (p > 0.05 or
95% CI touches 0) means the ecosystem has not lost diversity over T
iterations — an empirical answer to "is N enough to avoid collapse on
this axis?"


### 4.3 E3 — Diversity-axis comparison (N=4)

#### Diversity at iteration 1 vs iteration 10, by axis

| axis         | HSD@t=1     | meanPD@t=1   | frob@t=1    | HSD@t=10    | meanPD@t=10   | frob@t=10   |
|:-------------|:------------|:-------------|:------------|:------------|:--------------|:------------|
| single       | 3.03 ± 0.32 | 0.56 ± 0.11  | 1.98 ± 0.41 | 2.23 ± 0.28 | 0.32 ± 0.07   | 1.14 ± 0.26 |
| data_segment | 2.63 ± 0.50 | 0.46 ± 0.15  | 1.67 ± 0.55 | 2.45 ± 0.20 | 0.40 ± 0.08   | 1.48 ± 0.35 |
| persona      | 2.68 ± 0.34 | 0.49 ± 0.16  | 1.78 ± 0.66 | 2.45 ± 0.45 | 0.40 ± 0.13   | 1.42 ± 0.47 |
| model_family | 3.32 ± 0.27 | 0.79 ± 0.04  | 2.87 ± 0.08 | 3.15 ± 0.20 | 0.75 ± 0.02   | 2.77 ± 0.02 |

*N=4 agents, mean ± std across 3 seeds.*

![](figures/axis_compare_hsd.png)

![](figures/axis_compare_meanpd.png)

![](figures/axis_compare_frob.png)

![](figures/axis_compare_terminal_meanpd.png)

#### Ranking by terminal mean pairwise distance (higher = less collapse)

**model_family** (0.75) > **data_segment** (0.40) > **persona** (0.40) > **single** (0.32)

