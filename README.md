# How Many Agents to Avoid Collapse?

Empirical study of *minimum viable population* for an LLM information
ecosystem that recursively trains on its own outputs.

## Question

Wikipedia defines minimum viable population (MVP) as the lower bound on a
species' population size compatible with long-run survival in the wild. We
ask the analogous question for LLM ecosystems:

> Is there a minimum *N* such that a population of *N* LLM agents that
> share an "internet" of their own outputs avoids distributional collapse?
> What kind of differentiation between agents is enough to count as
> different "individuals"?

## Headline findings

See `REPORT.md` for the full numbers, figures and caveats. Three results:

1. **MVP exists, but is method-dependent.** In a RAG ecosystem
   (no parameter updates), N=2 LLMs from different pretraining families
   show statistically zero diversity decay over 12 self-training
   iterations. In a fine-tuning ecosystem, even N=16 distilgpt2 copies
   still collapse (perplexity grows 2.10× over 6 iterations); collapse
   rate decreases with N (slope: 80 → 16 ppl/iter for N = 1 → 16) but
   never reaches zero.
2. **Architecture diversity is the only individuality axis that buys
   collapse-resistance.** At fixed N=4, four agents from four different
   pretrained model families preserve ~1.9× the inter-agent semantic
   distance over 10 iterations as the same model with four different
   personas, four different RAG retrieval pools, or four different
   sampling seeds. The latter three are essentially indistinguishable.
3. **Diversity helps more at every doubling of N in the FT regime.**
   N=1 → 2 → 4 → 8 → 16 cuts the terminal perplexity multiplier
   from 7.45× → 4.43× → 3.67× → 3.03× → 2.10×.

Headline figures:

* `figures/finetune_perplexity_vs_iter.png` — perplexity-vs-iteration
  in the fine-tuning ecosystem for N ∈ {1,2,4,8,16}.
* `figures/finetune_mvp_curve.png` — perplexity-vs-N at each
  iteration t (perplexity vs population size).
* `figures/rag_model_family_meanpd_vs_iter.png` — RAG ecosystem
  inter-agent distance is essentially flat for N ≥ 2 over 12 iterations.
* `figures/axis_compare_terminal_meanpd.png` — diversity-axis
  comparison at fixed N=4 — model_family clearly dominates.

`results/mvp_estimates.json` contains the numerical MVP estimates per
metric per experiment; `results/stats_summary.json` and the per-metric
`results/slopes_*.csv` files contain the regression slopes with 95% CIs.

## Reproduction

```bash
# 1. Environment
uv venv
source .venv/bin/activate
uv pip install -e .          # uses pyproject.toml
uv pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121 --force-reinstall

# 2. API keys (only needed for E2 / E3)
export OPENROUTER_KEY=...

# 3. Datasets — already present in datasets/. Re-download with
#    datasets-cli download wikitext --config wikitext-2-raw-v1
#    if missing. See datasets/README.md.

# 4. Run experiments
python src/finetune_ecosystem.py --ns 1,2,4,8,16 --iters 6 --seeds 0,1 --budget 80000 \
       --out results/finetune_ecosystem.json
python src/rag_ecosystem.py --ns 1,2,3,5,8 --iters 12 --seeds 0,1,2 --mode model_family \
       --out results/rag_model_family.json
python src/rag_ecosystem.py --ns 4 --iters 10 --seeds 0,1,2 --mode persona       --out results/rag_persona_n4.json
python src/rag_ecosystem.py --ns 4 --iters 10 --seeds 0,1,2 --mode data_segment  --out results/rag_data_segment_n4.json
python src/rag_ecosystem.py --ns 4 --iters 10 --seeds 0,1,2 --mode single        --out results/rag_single_n4.json

# 5. Analyse — produces all figures + tables and splices into REPORT.md
python src/finalize.py
```

(`finalize.py` is a thin orchestrator that runs `merge_axis`, `analyze`,
`make_tables`, `build_report`, and `stats` in order, then splices the
auto-generated section into `REPORT.md` between `<!-- AUTO_BEGIN -->`
and `<!-- AUTO_END -->` sentinels.)

## File structure

```
src/
  metrics.py             # diversity / collapse metrics (Vendi Score, distinct-n, etc.)
  finetune_ecosystem.py  # E1 — Hodel & West-style fine-tuning ecosystem
  rag_ecosystem.py       # E2 / E3 — Wang-style API-only RAG ecosystem
  merge_axis.py          # combine 4 modes' results into axis_compare.json
  analyze.py             # plots + MVP estimation
  make_tables.py         # markdown tables for the report
  build_report.py        # auto-generated 4.x sections of REPORT.md
  stats.py               # per-N metric slope + CI
  finalize.py            # orchestrates the above and splices into REPORT.md
results/
  finetune_ecosystem.json
  rag_model_family.json
  rag_{persona,single,data_segment}_n4.json
  mvp_estimates.json
  *_summary.csv
figures/                 # all figures referenced in REPORT.md
logs/                    # per-iteration JSONL streams
papers/                  # 67 deep-read PDFs (literature_review.md catalogues)
datasets/                # WikiText-2 (raw v1), TinyStories sample
literature_review.md     # synthesis of the model-collapse / ecosystem literature
resources.md             # full resource catalogue
planning.md              # pre-experiment plan
REPORT.md                # full report
```

## Key dependencies

* `transformers ≥ 4.40`, `torch 2.4 + CUDA 12.1`, `datasets`
* `sentence-transformers` (`all-MiniLM-L6-v2` for embedding)
* `openai` (OpenRouter-compatible client)
* `pandas`, `numpy`, `scipy`, `matplotlib`

See `pyproject.toml` for exact versions.

## Hardware used

* 4 × NVIDIA RTX A6000 (one card actively used for E1).
* OpenRouter API for E2/E3 (Llama-3.1-8B-Instruct, Mistral-7B-Instruct,
  Qwen2.5-7B-Instruct, Gemma-2-9B-it, GPT-4o-mini, etc.).
