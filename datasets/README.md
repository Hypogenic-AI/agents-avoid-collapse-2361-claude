# Datasets

Data files are NOT committed to git due to size. Follow the download instructions
below to reproduce the local copy. All datasets here are public, redistributable,
and well-suited to the model-collapse experimental paradigm (small enough to allow
many self-training iterations on commodity hardware).

## Dataset 1: WikiText-2 (raw, v1)

### Overview
- **Source**: HuggingFace `wikitext` / `wikitext-2-raw-v1`
- **Size**: ~12 MB (raw); 36,718 train / 3,760 validation / 4,358 test rows
- **Format**: HuggingFace `Dataset` (Arrow), one Wikipedia paragraph per row
- **Task**: causal language modeling, perplexity evaluation
- **License**: CC-BY-SA 3.0 (Wikipedia)

### Why this dataset for the project
This is the canonical benchmark in the model-collapse literature. Used by:
- Shumailov et al. 2024 (the seminal "Curse of Recursion" paper) — fine-tunes
  OPT-125m on WikiText-2 across 9 generations.
- Hodel & West 2026 (most directly relevant to our research question) —
  segments WikiText-2 across M={1,2,4,16} models and runs 10 iterations.

Reusing this dataset lets us compare directly against published baselines.

### Download Instructions
```python
from datasets import load_dataset
ds = load_dataset("wikitext", "wikitext-2-raw-v1")
ds.save_to_disk("datasets/wikitext2/wikitext-2-raw-v1")
```

### Loading
```python
from datasets import load_from_disk
ds = load_from_disk("datasets/wikitext2/wikitext-2-raw-v1")
print(ds)  # DatasetDict with train/validation/test
```

## Dataset 2: TinyStories

### Overview
- **Source**: HuggingFace `roneneldan/TinyStories`
- **Size**: ~1.5 GB full (2.1M short stories train + 22K val); first 5K downloaded here for quick iteration
- **Format**: HuggingFace `Dataset`, one short story (kindergarten level) per row
- **Task**: causal language modeling on simple text — used for from-scratch pretraining experiments
- **License**: CDLA-Sharing-1.0

### Why this dataset for the project
Used by Gerstgrasser et al. 2024 ("Is Model Collapse Inevitable?") to pretrain
small (9M-125M) GPT-2 / Llama2 from scratch across multiple data-replace vs
data-accumulate iterations. Small enough that pretraining a 9M model for one
epoch is feasible.

### Download Instructions

For the full dataset:
```python
from datasets import load_dataset
ds = load_dataset("roneneldan/TinyStories")
ds.save_to_disk("datasets/tinystories/full")
```

The local copy here only has the first 5,000 stories (a quick-iteration subset):
```python
from datasets import load_dataset
ds = load_dataset("roneneldan/TinyStories", split='train[:5000]')
ds.save_to_disk("datasets/tinystories/data")
```

### Loading
```python
from datasets import load_from_disk
ds = load_from_disk("datasets/tinystories/data")
```

## Dataset gaps / candidates not yet downloaded

These are mentioned in the literature but not pulled here. The experiment runner
can grab them on demand:

- **C4** (Common Crawl) — large pretraining corpus; used in some collapse studies
  for higher-capacity models. Tens of GB; pull a subset only.
- **Reddit / Twitter snapshots** — used by Kovač et al. 2025 to study how data
  properties (lexical/semantic diversity, quality, political lean) modulate
  distribution shift. They ran on five datasets across three domains
  (Twitter ×2, Reddit ×2, Wikipedia ×1). See `code/ce_llms/` for the loaders.
- **Crypto-news set** — the LLM Web Dynamics paper (Wang et al. 2025) uses 20
  crypto posts from Ebrahimi 2024 as the seeded "Internet" for their
  RAG-based multi-LLM simulation. Tiny (~20 short posts).
