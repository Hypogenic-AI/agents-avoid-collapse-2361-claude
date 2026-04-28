# Planning: How Many Agents to Avoid Collapse?

## Motivation & Novelty Assessment

### Why This Research Matters
Generative models are increasingly trained on web text that itself contains LLM
output. If a single recursive loop reliably collapses ("model collapse",
Shumailov 2024), the same phenomenon at the scale of *the entire web* would
degrade future foundation models. Whether *population diversity* among the
LLM authors of that web text is enough to prevent collapse — and how much
diversity is needed — directly informs (a) data-curation strategy,
(b) anti-monoculture policy for AI deployment, and (c) the long-run
sustainability of the open web as a training substrate.

### Gap in Existing Work
Almost all collapse work studies a *single* self-training model. The few
ecosystem-level results in the corpus (Hodel & West 2026 with M ∈ {1,2,4,16},
Wang et al. 2025 with N=3, Vu et al. 2025 with N=2 model families) leave three
specific gaps that map onto our question:

1. **No published number for "minimum viable population" (MVP).** Hodel & West
   show optimal M *grows monotonically with iteration count*, but never report
   the *minimum* M at which a chosen horizon T stays bounded.
2. **Only one axis of "individuality" tested.** Hodel & West vary
   training-data segments only. Wang et al. observe that even
   pretrained-different LLMs converge under shared-RAG dynamics. The
   *what-counts-as-different* sub-question is wide open.
3. **No cross-method comparison.** Fine-tuning ecosystems and API-only RAG
   ecosystems have not been benchmarked on the same metric in the same paper.

### Our Novel Contribution
We answer all three sub-questions in one experimental program:

* **MVP curves**: For each of three diversity axes (data segments, prompt
  personas, model families), estimate the smallest population N that keeps a
  collapse-related metric bounded over T iterations.
* **Axis comparison**: Holding N fixed, which axis of differentiation buys
  the most resistance to collapse?
* **Cross-method consistency**: Do the MVP and ranking conclusions agree
  between expensive fine-tuning ecosystems (Hodel & West-style) and cheap
  API-only RAG ecosystems (Wang-style)?

### Experiment Justification
| Experiment | Why needed |
|---|---|
| E1 — Fine-tuning ecosystem with GPT-2 on WikiText-2, M ∈ {1,2,4,8,16} | Direct extension of Hodel & West with explicit MVP analysis; reproduces and pushes the only published ecosystem-level result. |
| E2 — RAG ecosystem with real LLMs via API, N ∈ {1,2,3,5,8} | Cheap large-N sweep; tests whether the MVP claim is method-dependent or robust across radically different "ecosystem" implementations. |
| E3 — Diversity-axis comparison at fixed N=4 | Disentangles "what counts as different": same-model-different-prompt vs. different-model-family vs. different-data-segment, on identical metrics. |

## Research Question (restated)

> Is there a minimum viable population *N* for an LLM information ecosystem
> such that, as the ecosystem self-trains on its own outputs over T
> iterations, distributional collapse (rising perplexity, falling diversity,
> converging cross-agent embeddings) does **not** occur? How does *N* depend
> on the form of differentiation between agents?

## Hypothesis Decomposition

* **H1 (MVP exists, finite-N regime)**: There is a finite *N* > 1 such that an
  ecosystem of *N* agents stays bounded in a collapse metric M(t) for
  t ∈ [0, T] with T moderate (T = 8–10). Equivalently, *the curve M(t) for
  this N flattens rather than monotonically increases*.
* **H2 (MVP grows with horizon)**: For any finite *N*, there exists a horizon
  T*(N) at which M(T*) exceeds a chosen threshold; T*(N) is increasing in N.
  (This generalises Hodel & West's "optimal M grows with t" finding.)
* **H3 (axis matters)**: Conditional on N, the order of axes from most-to-
  least collapse-resistant is:
  *different model families* > *different data segments* > *different prompt
  personas* > *same-everything-different-seeds* (control).

Independent variables: population size N; differentiation axis; iteration t.
Dependent variables: perplexity on a held-out human test set; lexical
diversity (distinct-n / TTR); semantic diversity (embedding distance
Frobenius norm and Hill–Shannon Diversity).

## Proposed Methodology

### Approach
Two complementary experimental harnesses, both driven by the *same* metric
suite, so results can be compared apples-to-apples.

### E1: Fine-tuning ecosystem (Hodel & West-style)
* **Backbone**: `distilgpt2` (82M params). Compromise between expressiveness
  and the need to fine-tune ~5×8×N models per sweep (N up to 16, T=8).
* **Data**: WikiText-2-raw-v1 train (already local). Fixed test split for
  perplexity.
* **Per-ecosystem training-token budget**: ~250K tokens per generation,
  partitioned into N equal slices. This holds compute roughly constant
  across N.
* **Loop**: For t = 1..T:
  1. Each of N agents generates a slice of synthetic text (target tokens =
     budget/N).
  2. All slices are pooled, shuffled, and re-split into N new slices.
  3. Each agent fine-tunes 1 epoch on its new slice (LR 5e-5, batch 4).
* **Population sizes**: N ∈ {1, 2, 4, 8, 16}.
* **Iterations**: T = 6 (extendable if compute allows).
* **Replace mode** (no real-data refresh): we explicitly want to see whether
  diversity *alone* prevents collapse, with no Gerstgrasser-style data
  accumulation as a confound.

### E2: RAG-style ecosystem (Wang-style)
* **Models**: Mix of OpenRouter-served models from different families:
  Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3, Qwen2.5-7B-Instruct,
  Gemma-2-9B-it, GPT-4o-mini.
* **Internet seed**: ~30 short factual paragraphs sampled from WikiText-2.
* **Loop**: For t = 1..T:
  1. Each of N agents retrieves k = ⌊β·|A_t|⌋ paragraphs from the pool
     (β = 0.2), uses them as RAG context, and generates a paragraph in
     response to a fixed query.
  2. New paragraphs are appended to the pool.
* **Population sizes**: N ∈ {1, 2, 3, 5, 8}; we instantiate N agents by
  cycling through the available real models (so larger N exposes the
  ecosystem to more pretraining-data diversity).
* **Iterations**: T = 12.
* **Embedding**: `sentence-transformers/all-MiniLM-L6-v2` for diversity
  computations. (Substitute for nomic-embed-v1.5 — same use case, faster.)

### E3: Diversity-axis comparison
At fixed N=4, run E1's loop with four agents differentiated by:
* (a) random seed only (control)
* (b) different non-overlapping data segments
* (c) different system-prompt personas (single base model, four personas)
* (d) different base model checkpoints (`distilgpt2`, `gpt2`, `gpt2-medium`,
  `EleutherAI/pythia-160m`)

Run T=6 iterations and compare collapse curves.

### Baselines
* **N=1 replace** (Shumailov): worst-case collapse curve.
* **N=1 accumulate** (Gerstgrasser): single-model upper bound.
* **N=M segmented** (Hodel & West): multi-agent baseline at moderate M.

### Evaluation Metrics
* **Perplexity** of generation t on held-out WikiText-2 test (E1, E3).
* **Lexical diversity** = distinct-bigram ratio over each generation's text
  pool.
* **Semantic diversity** = mean pairwise cosine distance over sentence
  embeddings of generated text (E1, E2, E3).
* **Pairwise embedding-distance Frobenius norm** of the inter-agent matrix
  (E2 — Wang's diagnostic).
* **Hill–Shannon Diversity** (effective number of distinct outputs) over
  embedded outputs (E1, E2).

### Statistical Analysis
* **Per-condition**: 3 random seeds, report mean ± std.
* **Comparison**: paired t-test of "metric at iteration T" between
  successive N values to identify the smallest N for which the metric is
  statistically indistinguishable from N_max.
* **Trend**: regress metric against log(N) at each iteration.

## Expected Outcomes
* H1 supported if E1 and E2 both show metric curves that flatten or invert
  trend at some N* < N_max within T iterations.
* H2 supported if even the largest N exhibits a positive metric slope, with
  a smaller slope for larger N.
* H3 supported if the E3 axis ranking matches the predicted order.

## Timeline and Milestones
| Phase | Time |
|---|---|
| Planning | 30 min (now done) |
| Setup + load datasets, sanity-check API | 15 min |
| E1 implementation + run | 90 min |
| E2 implementation + run | 60 min |
| E3 run | 45 min |
| Analysis + figures | 45 min |
| REPORT.md + README.md | 30 min |
| Buffer | 30 min |

## Potential Challenges
* **Fine-tuning compute** at N=16, T=6 → mitigated by tiny per-agent data
  slice and `distilgpt2`. If still slow we drop to N_max = 8.
* **Generation quality at small training data** → distilgpt2 starts from a
  non-trivially pretrained checkpoint, so small fine-tunes mostly *shift*
  rather than learn from scratch. Acceptable for the collapse signal we want.
* **API rate limits / cost** in E2 → keep generations short (~80 tokens),
  cache responses, prefer cheap routed models.
* **Repeating-text artefact** (Shumailov caveat) → use sampling
  (temperature 1.0, top_p 0.95), no repetition penalty.

## Success Criteria
* All five population sizes complete in E1 and E2.
* Each metric is computed at every iteration for every condition.
* REPORT.md contains MVP estimates with statistical support, axis ranking,
  and an explicit answer to "will it always collapse?".
