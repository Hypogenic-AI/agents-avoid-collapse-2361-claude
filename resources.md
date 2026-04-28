# Resources Catalog

## Summary

| Category | Count | Notes |
|---|---|---|
| Papers downloaded | 67 PDFs | All from arXiv, paper-finder relevance ≥ 2 |
| Datasets downloaded | 2 | WikiText-2 (12 MB), TinyStories sample (5K rows) |
| Code repositories cloned | 4 | Schaeffer, ce_llms, linguistic-diversity, nanoGPT |

## Papers

67 PDFs downloaded from arXiv, ranked by paper-finder relevance.
See `papers/README.md` for the per-paper breakdown.

### Top 15 by relevance + citation count

| Rank | arXiv ID | Title | Year | Cit. |
|---|---|---|---|---|
| 1 | 2305.17493 | The Curse of Recursion / AI Models Collapse (Shumailov; Nature 2024) | 2024 | 697 |
| 2 | 2404.01413 | Is Model Collapse Inevitable? (Gerstgrasser et al.) | 2024 | 123 |
| 3 | 2402.07043 | A Tale of Tails: Model Collapse as Scaling Laws (Dohmatob et al.) | 2024 | 120 |
| 4 | 2311.09807 | The Curious Decline of Linguistic Diversity (Guo et al.) | 2023 | 116 |
| 5 | 2311.16822 | LLMs Suffer From Their Own Output (Briesch et al.) | 2023 | 86 |
| 6 | 2404.05090 | How Bad is Training on Synthetic Data? (Seddik et al.) | 2024 | 73 |
| 7 | 2410.18982 | O1 Replication Journey (Qin et al.) | 2024 | 150 |
| 8 | 2406.14532 | RL on Incorrect Synthetic Data (Setlur et al.) | 2024 | 121 |
| 9 | 2404.14387 | A Survey on Self-Evolution of LLMs (Tao et al.) | 2024 | 61 |
| 10 | 2410.16713 | Collapse or Thrive? (Schaeffer et al.) | 2024 | – |
| 11 | 2307.01850 | Self-Consuming Generative Models Go MAD (Alemohammad et al.) | 2023 | – |
| 12 | 2402.11778 | Theoretical Understandings of Self-Consuming GMs (Bertrand et al.) | 2024 | – |
| 13 | 2506.15690 | LLM Web Dynamics: Network of LLMs (Wang et al.) | 2025 | – |
| 14 | 2512.15011 | **Epistemic diversity mitigates knowledge collapse (Hodel & West)** | 2026 | – |
| 15 | 2505.21677 | Generative AI Models Train on Each Others' Outputs (Vu et al.) | 2025 | – |

The full list with abstracts is in `papers/README.md` and `papers/paper_finder_resolved.json`.

## Datasets

| Name | Source | Size | Task | Location | Notes |
|---|---|---|---|---|---|
| WikiText-2 (raw v1) | HuggingFace `wikitext` | ~12 MB; 36k/3.7k/4.4k rows | Causal LM | `datasets/wikitext2/wikitext-2-raw-v1/` | Canonical benchmark used by Shumailov, Hodel & West |
| TinyStories (5K subset) | HuggingFace `roneneldan/TinyStories` | ~2 MB local; 1.5 GB full | Pretraining-from-scratch LM | `datasets/tinystories/data/` | Used by Gerstgrasser. Local copy is a 5K sample; full set downloadable |

Detailed download instructions in `datasets/README.md`. Data files are excluded
from git via `datasets/.gitignore`; only documentation and small samples are
committed.

## Code Repositories

| Name | URL | Purpose | Path |
|---|---|---|---|
| Collapse-or-Thrive (Schaeffer) | https://github.com/RylanSchaeffer/KoyejoLab-Collapse-or-Thrive | Single-model replace-vs-accumulate; Gerstgrasser baseline implementation | `code/KoyejoLab-Collapse-or-Thrive/` |
| ce_llms (FlowersTeam) | https://github.com/flowersteam/ce_llms | Iterative-chain LoRA fine-tuning across rotated base models, dataset clustering, evaluation | `code/ce_llms/` |
| linguistic-diversity (Guo) | https://github.com/yanzhuguo/linguistic-diversity | Lexical/semantic/syntactic diversity metric scripts | `code/linguistic-diversity/` |
| nanoGPT (Karpathy) | https://github.com/karpathy/nanoGPT | Minimal GPT-style training code (used by Briesch et al.) | `code/nanoGPT/` |

Full per-repo breakdown in `code/README.md`.

## Search strategy

1. **Paper-finder API (Allen Institute) — primary**: queried `"model collapse LLM training synthetic generated data recursive"` in `diligent` mode. Returned 133 papers ranked by relevance; 74 had relevance ≥ 2 and we captured their full metadata (`papers/paper_finder_results.json`, `papers/paper_finder_resolved.json`).
2. **arXiv direct search — supplementary**: ran 8 targeted queries (e.g. `Shumailov model collapse`, `self-consuming generative models`, `population dynamics multi-agent language model`, `model autophagy disorder`, `cultural evolution language models`, etc.). Captured in `papers/arxiv_search_*.json`. Mostly redundant with paper-finder, but caught a few foundational hits (e.g. arXiv ID lookup for the Shumailov Nature paper itself).
3. **Semantic Scholar API**: used to resolve arXiv IDs from paper-finder's Semantic Scholar URLs / Corpus IDs.
4. **Web search agent**: used to find GitHub repositories for the 9 most-relevant papers.

## Selection criteria

For paper download, we filtered to relevance ≥ 2/3 from paper-finder *and* with
a resolvable arXiv ID. Within that pool, we downloaded all 67 papers with
successful arXiv PDFs (no failures). We deep-read 8 of the most directly
relevant to our research question:

- 2305.17493 (Shumailov; foundational)
- 2404.01413 (Gerstgrasser; data accumulation)
- 2402.07043 (Dohmatob; scaling laws)
- 2311.16822 (Briesch; LLM-specific)
- 2311.09807 (Guo; diversity decline)
- 2504.03814 (Kovač; data-property modulation, multi-base-model chain)
- 2506.15690 (Wang; multi-LLM RAG ecosystem)
- 2512.15011 (Hodel & West; **most directly relevant**: ecosystem diversity vs collapse)
- 2505.21677 (Vu, Reeves & Wenger; model-on-model recursive training)

Chunks for these are in `papers/pages/`.

## Challenges encountered

1. **Paper-finder service initially returned `httpx not installed` fallback**:
   resolved by adding `httpx` to the project. After install, the service ran
   in `diligent` mode (~3 min) and produced 133 ranked results.
2. **Many Semantic Scholar URLs lack `openAccessPdf`**: only 1 of 41 missing
   arXiv IDs had a PDF URL in the SS API. We fell back to arXiv title-search,
   which recovered 17 more arXiv IDs. After that, two papers' SS title-search
   collided ("AI models collapse" vs "A Note on Shumailov"); we manually fixed
   those mappings.
3. **Some PDFs are very recent (2026)**: a handful of 2026 papers exist on arXiv
   but not yet in HuggingFace dataset collections. These are downloaded as PDFs
   only.
4. **Code releases are inconsistent**: of 9 priority papers, only 3 have
   identifiable, de-anonymized public GitHub repos (Schaeffer, ce_llms, Guo).
   Hodel & West, Wang, Briesch, Alemohammad have no public code; Shumailov is
   Zenodo-only. This is a meaningful research-process risk: the most relevant
   ecosystem-level paper to our question (Hodel & West) needs reimplementation.

## Gaps and workarounds

- **Hodel & West 2026 code unavailable** → we have the full experimental recipe
  from the paper. Reimplementing on HuggingFace Transformers + the WikiText-2
  data we downloaded is straightforward (M-way segmentation, fine-tune,
  generate, redistribute, repeat).
- **LLM Web Dynamics code unavailable** → recipe is API-only; reimplementing in
  ~200 LOC is feasible. We have the embedding model spec (`nomic-embed-v1.5`)
  and the algorithm.
- **WEBIS dataset / Reddit / Twitter clusters not yet downloaded** → these are
  Kovač-specific; if we don't run the per-cluster experiments, we don't need
  them. If we do, `code/ce_llms/create_per_cluster_webis_datasets.py` shows
  how to build them.

## Recommendations for experiment design

Distilled from the literature review. See `literature_review.md` for the full
reasoning and citations.

### Primary experiment: multi-agent ecosystem with M-segmentation
- Models: OPT-125m and GPT-2 (or similar small base LMs)
- Data: WikiText-2 (already local)
- M ∈ {1, 2, 4, 8, 16, 32, 64} — extends Hodel & West beyond their M=16 ceiling
- 10–20 self-training iterations
- Per-ecosystem token budget held constant
- Metrics: perplexity on WikiText-2 test, lexical TTR, semantic embedding-cosine variance, Hill-Shannon Diversity over output embeddings

### Companion experiment: API-only RAG ecosystem
- 3+ pretrained-different LLMs (HuggingFace small models)
- Shared "Internet" text database, growing over iterations
- Wang-style retrieval policy with k_t = ⌊β·|A_t|⌋
- Sweep N ∈ {2, 3, 5, 10, 20, 50, 100} (cheap; no training)
- Metric: Frobenius norm of pairwise embedding-distance matrix

### Baselines (single-model)
- Replace-only (Shumailov): expected divergence
- Accumulate (Gerstgrasser): expected π²/6-style bounded plateau
- 10% real-data refresh (Shumailov regime b): realistic upper bound for single-model

### Diversity-instantiation axes (orthogonal experiments)
- Same architecture, different data segments (Hodel & West)
- Different architecture, same data
- Same architecture+data, different prompts/personas
- Same model, different RAG sources

Each axis answers a different version of "what counts as different enough."
