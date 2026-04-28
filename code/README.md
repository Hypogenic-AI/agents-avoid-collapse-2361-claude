# Cloned Repositories

Scaffolding for our model-collapse / multi-agent experiments.

## 1. KoyejoLab-Collapse-or-Thrive (Schaeffer / Gerstgrasser)

- **URL**: https://github.com/RylanSchaeffer/KoyejoLab-Collapse-or-Thrive
- **Paper(s)**: 
  - Gerstgrasser et al. 2024, *Is Model Collapse Inevitable?* (arXiv 2404.01413)
  - Schaeffer et al. 2024, *Collapse or Thrive?* (ICML 2025; arXiv 2410.16713)
- **Path**: `code/KoyejoLab-Collapse-or-Thrive/`
- **What it provides**:
  - Toy distribution-fitting (Gaussian, KDE, linear regression) collapse simulations under replace-vs-accumulate dynamics: `src/fit_gaussians`, `src/fit_kdes`, `src/fit_linear_regressions`
  - Supervised fine-tuning of language models with self-generated data: `src/sft_language_model/sft_language_model.py` (and a `_mixed_data` variant)
  - Pretraining loop: `src/pretrain_language_model/`
  - Sampling generation `src/sample_language_model/`
  - W&B-tracked `sweeps/` for hyperparameter scans
- **Stack**: PyTorch + HuggingFace; conda env via `environment.yml`; environment named `model_collapse_20240911`
- **Entry point for our experiments**: their `sft_language_model.py` is a drop-in for the Gerstgrasser data-accumulation baseline that we should compare M ≥ 1 ecosystems against.

## 2. ce_llms (Kovač / FlowersTeam @ INRIA)

- **URL**: https://github.com/flowersteam/ce_llms
- **Paper**: Kovač et al. 2025, *Recursive Training Loops in LLMs: How training data properties modulate distribution shift* (arXiv 2504.03814)
- **Path**: `code/ce_llms/`
- **What it provides**:
  - **Iterative-chain training scripts**: `iterative_train.sh`, `dev_iterative_train.sh`, `clusters_iterative_train.sh` — fine-tunes a fresh base LLM each generation on a mix of accumulated + fresh human data, then generates more data
  - **LLM training**: `ft_and_gen.py` — LoRA fine-tuning via Unsloth on LLaMA / Qwen / SmolLM / Falcon (1B-class models)
  - **Dataset utilities**: `dataset_utils.py`, `create_per_cluster_webis_datasets.py`, `create_webis_miniclusters.py`
  - **Diversity / quality metrics**: `text_clustering.py`, `add_qualities_to_dataset.py`, `correlate_metrics.py`
  - **Evaluation**: `evaluate_generations.py`, `eval_utils.py`, `eval_openmeva.py`, `evaluate_webis_clusters.py`, `batch_eval.sh`
  - **Plotting**: `plot_scaling_law.py` and other scripts
- **Stack**: 3 separate conda envs (Unsloth-based training, sklearn-based evaluation, plus a third). Requires CUDA GPUs; SLURM scripts included for cluster runs.
- **Entry point for our experiments**: 
  - For **iterative chain experiments with rotated base models** (which maps cleanly to our "what counts as different enough" question), this is the cleanest existing scaffolding.
  - The `evaluate_generations.py` + `text_clustering.py` modules give us reusable diversity/quality/cluster metrics.
- **Notes / blockers**:
  - Heavy GPU stack (Unsloth, bitsandbytes 4-bit). Will need GPU compute — won't run on CPU.
  - Their per-cluster experiments use the WEBIS-WIKIPEDIA / WEBIS-Reddit / WEBIS-Twitter datasets (not yet downloaded; some are on HuggingFace).

## 3. linguistic-diversity (Guo et al.)

- **URL**: https://github.com/yanzhuguo/linguistic-diversity
- **Paper**: Guo et al. 2023, *The Curious Decline of Linguistic Diversity* (arXiv 2311.09807)
- **Path**: `code/linguistic-diversity/`
- **What it provides**: Three tightly-scoped diversity metric scripts:
  - `lexical_diversity.py` — n-gram Type-Token Ratio
  - `semantic_diversity.py` — sentence-embedding cosine similarity
  - `syntactic_diversity.py` — dependency-graph kernels
- **Stack**: small `requirements.txt`. CC0 license.
- **Entry point**: drop-in metric library. Each script expects `data/<category>/outputs/` for categories `{story, dialogue, summary, translation, wiki}`. We can call these scripts (or import the functions directly) on our own ecosystem outputs.

## 4. nanoGPT (Karpathy)

- **URL**: https://github.com/karpathy/nanoGPT
- **Paper(s)**: Used by Briesch et al. 2023 (*Large Language Models Suffer From Their Own Output*); not paper-specific.
- **Path**: `code/nanoGPT/`
- **What it provides**: A minimal, well-engineered GPT-2 / GPT-style LM training implementation. ~300 LOC for `train.py`. Standard HuggingFace-free pipeline.
- **Stack**: PyTorch only. CPU works for tiny models; reasonable single-GPU for ~125M.
- **Entry point**: useful if we want a from-scratch tiny-LM training loop without pulling in HuggingFace's full stack. For the `OPT-125m` baseline, HuggingFace Transformers is the natural choice though.

## What is NOT cloned (and why)

- **Shumailov "curse_recurse" code (Zenodo only)** — the original is distributed
  as a zip on Zenodo, not GitHub. Reproducing OPT-125m on WikiText-2 from the
  paper description plus HuggingFace defaults is straightforward; pulling the
  zip is optional. Download URL: https://zenodo.org/records/10866595
- **Hodel & West 2026 code** — not publicly released as of 2026-04-28. Their
  experimental recipe is fully described in the paper (see literature_review.md);
  we will need to reimplement, but the recipe is concrete: OPT-125m / GPT-2 +
  WikiText-2 segmented M-ways + standard HuggingFace fine-tuning.
- **LLM Web Dynamics (Wang et al.) code** — not publicly released.
- **Self-Consuming Generative Models Go MAD (Alemohammad et al.) code** — not
  publicly released.

## Recommended workflow for the experiment runner

1. Use `code/KoyejoLab-Collapse-or-Thrive/` as the single-model baseline reference
   (Gerstgrasser-style replace-vs-accumulate).
2. Implement the **Hodel & West M-segmented WikiText-2** experiment from scratch
   on top of HuggingFace Transformers — the recipe is fully specified in
   literature_review.md / `papers/2512.15011_*.pdf`.
3. Use `code/linguistic-diversity/` for off-the-shelf diversity metrics.
4. Use `code/ce_llms/` as the reference for richer multi-base-model rotations
   and dataset-property manipulations if the basic Hodel & West extension yields
   clean results.
5. For cheap large-N scans, prototype a Wang-style RAG-based ecosystem in pure
   API calls (no training) — small custom code, no clone needed.
