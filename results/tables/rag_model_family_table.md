## E2 — Diversity at iteration 1 vs terminal iteration (RAG ecosystem, model_family axis)

|   N | HSD@t=1     | HSD@terminal       | meanPD@t=1   | meanPD@terminal    | frob@t=1    | frob@terminal      |
|----:|:------------|:-------------------|:-------------|:-------------------|:------------|:-------------------|
|   1 | 1.00        | 1.00 (t=12)        | 0.00         | 0.00 (t=12)        | 0.00        | 0.00 (t=12)        |
|   2 | 1.99 ± 0.01 | 1.99 ± 0.00 (t=12) | 0.89 ± 0.04  | 0.90 ± 0.01 (t=12) | 1.25 ± 0.06 | 1.27 ± 0.01 (t=12) |
|   3 | 2.81 ± 0.14 | 2.63 ± 0.24 (t=12) | 0.81 ± 0.08  | 0.74 ± 0.09 (t=12) | 2.01 ± 0.16 | 1.91 ± 0.17 (t=12) |
|   4 | 3.32 ± 0.27 | 3.15 ± 0.20 (t=10) | 0.79 ± 0.04  | 0.75 ± 0.02 (t=10) | 2.87 ± 0.08 | 2.77 ± 0.02 (t=10) |

*N=1,2,3 ran for 12 iterations; N=4 ran for 10 iterations (from the E3 axis-comparison data). β=0.2 retrieval, mean ± std across 3 seeds.*
