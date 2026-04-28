# Literature Review: How Many Agents to Avoid Collapse?

## Research question

**Is there a minimum viable population for an LLM information ecosystem such that
it does not collapse? Will such an ecosystem always collapse? What counts as
different enough to be an individual in the population?**

## Research area overview

Since Shumailov et al. (2023, *Nature* 2024) coined "model collapse," a fast-growing
literature has established that generative models recursively trained on their
own outputs progressively lose information about the tails of the original
distribution and converge toward low-variance, homogeneous outputs. Almost the
entire literature, however, studies a **single self-training model** — only a few
recent papers shift the unit of analysis from "the model" to "the ecosystem" of
multiple, possibly distinct, models that share a synthetic-data substrate (the
internet).

The research question maps directly onto this newer ecosystem-level work and asks
two extensions that have not yet been answered comprehensively:

1. **Minimum viable population** — at what number of agents *N* does a multi-agent
   ecosystem cease to collapse? Does any *finite* *N* suffice without other
   interventions (data accumulation, real-data refresh, curation), or must *N*
   grow with iteration count?
2. **Individual identity** — what kind of differentiation between agents counts?
   Different model families? Different training-data segments? Different prompts?
   Different RAG sources?

The most directly relevant empirical answer published to date is **Hodel & West
(2026, arXiv 2512.15011)**: under fixed per-ecosystem compute and data budget,
they segment WikiText-2 across M ∈ {1, 2, 4, 16} fine-tuned OPT-125m / GPT-2
models and run 10 self-training iterations. They find an **optimal diversity that
grows monotonically with iteration count** — suggesting no fixed *N* is "enough"
indefinitely, and motivating the question of which axes of differentiation give
the most mileage. Our project can directly extend Hodel & West along the
dimension they did *not* vary (architecture, RAG sources, prompt regimes) and
along the dimension they only stub (M up to 16).

## Key papers (selected; full catalogue in `resources.md`)

### Foundational

#### Shumailov et al. 2024 — *AI models collapse when trained on recursively generated data* (Nature; arXiv 2305.17493)
- **Contribution**: First formalization of model collapse. Defines early- vs late-stage collapse: tails of the data distribution disappear first, then the model converges to low-variance point estimates.
- **Methodology**: 
  - GMM and VAE toy demonstrations.
  - **LLM experiment**: fine-tune OPT-125m on WikiText-2; for each generation, generate a same-sized synthetic dataset via 5-way beam search over 64-token blocks; train next generation on that. Two regimes: (a) 5 epochs, no original data preserved; (b) 10 epochs, 10% of original data preserved per round.
- **Key results**: Perplexity rises from ~34 (gen 0) toward ~52 over 9 generations under regime (a); regime (b) — preserving 10% real data — strongly attenuates collapse. Histograms over generations show emerging long tails of "errors that would never appear under the real distribution."
- **Code**: Zenodo zip (no GitHub mirror). https://zenodo.org/records/10866595
- **Relevance to our research**: Establishes the *single*-model baseline our N=1 condition reproduces. Their OPT-125m / WikiText-2 setup is the de-facto reference design.

#### Alemohammad et al. 2023 — *Self-Consuming Generative Models Go MAD* (arXiv 2307.01850)
- **Contribution**: Coined "Model Autophagy Disorder" (MAD); demonstrated collapse in image generative models (GANs, diffusion). Categorized self-consuming loops into fully-synthetic, synthetic-augmented, and fresh-data regimes.
- **Relevance**: Shows model collapse is a cross-modality phenomenon, not specific to language. Their framework (loop variants) is a useful conceptual scaffolding.

### Mitigation through real-data dynamics (single model)

#### Gerstgrasser et al. 2024 — *Is Model Collapse Inevitable? Breaking the Curse of Recursion by Accumulating Real and Synthetic Data* (arXiv 2404.01413)
- **Contribution**: Shows that **accumulating** synthetic data alongside real data (rather than replacing real with synthetic each generation) yields a finite, bounded test error independent of iteration count — both empirically (transformers on TinyStories, diffusion on molecular conformation, VAE on images) and theoretically (linear-regression chain).
- **Methodology**: 
  - Pretrain 9M GPT-2 / 12M-42M-125M Llama2 from scratch on TinyStories (470M tokens).
  - For each generation n ≥ 2, generate a TinyStories-sized synthetic dataset, then either replace or concatenate with prior data; pretrain a fresh init.
  - Sampling temperatures 0.3 and 1.0.
- **Key results**: "Replace" diverges; "accumulate" plateaus at a bounded error. This holds across model architecture, parameter count, and modality.
- **Code**: https://github.com/RylanSchaeffer/KoyejoLab-Collapse-or-Thrive (companion repo).
- **Relevance**: A non-population mitigation. Our N agents experiment must hold this constant or contrast with it. They use TinyStories which is now downloaded locally.

#### Briesch et al. 2023 — *Large Language Models Suffer From Their Own Output* (arXiv 2311.16822)
- **Contribution**: First LLM-only empirical study of self-consuming loops, uses logic-expression tasks (where correctness is verifiable) to disentangle quality decay from diversity decay. Finds correctness preserved, diversity declines.
- **Relevance**: Decoupling quality from diversity matters for picking a metric. We'll need a diversity measure that's not entangled with task accuracy.

#### Guo et al. 2023 — *The Curious Decline of Linguistic Diversity* (arXiv 2311.09807)
- **Contribution**: Defines lexical, syntactic, and semantic diversity metrics and tracks their decay under recursive fine-tuning across multiple generation tasks.
- **Code**: https://github.com/yanzhuguo/linguistic-diversity (metric scripts only).
- **Relevance**: Provides off-the-shelf diversity metrics we can reuse to evaluate ecosystem-level diversity.

### Theoretical results

#### Dohmatob et al. 2024 — *A Tale of Tails: Model Collapse as a Change of Scaling Laws* (arXiv 2402.07043)
- **Contribution**: Unifies model collapse with scaling-law theory. Shows synthetic data shifts the scaling exponent and induces "un-learning" of low-frequency skills. Validated empirically on transformer arithmetic and Llama2 text generation.
- **Relevance**: Provides an analytical framework for predicting how a ratio of synthetic-to-real data degrades a fixed-architecture model — relevant when our N agents differ in their relative synthetic-share.

#### Seddik et al. 2024 — *How Bad is Training on Synthetic Data?* (arXiv 2404.05090)
- **Contribution**: Statistical analysis showing collapse is unavoidable on pure synthetic training, but a maximal synthetic ratio exists below which collapse can be eventually avoided when mixed with real data.
- **Relevance**: Gives a quantitative target: there's a per-iteration synthetic-budget threshold. Our ecosystem can be framed as N agents trading per-agent synthetic exposure.

### Multi-model / ecosystem-level (most relevant to our research question)

#### Hodel & West 2026 — *Epistemic diversity across language models mitigates knowledge collapse* (arXiv 2512.15011) — **central reference**
- **Contribution**: Direct test of the research question for an ecosystem of fine-tuned single-family models, using **training-data segmentation** as the diversity knob.
- **Methodology**:
  - Start from one pretrained model (OPT-125m, GPT-2). Make M = 1, 2, 4, 16 identical copies.
  - Segment WikiText-2 into M equal-sized non-overlapping subsets; fine-tune each model on its segment ("epistemology").
  - At each iteration, all M models generate text; outputs are concatenated, shuffled, redistributed uniformly into M equal training slices for the next iteration.
  - Per-ecosystem total tokens N is held constant; n = N/M per model.
  - Run 10 iterations. Evaluate perplexity on WikiText-2 test (fixed).
  - HSD diversity D simplifies to M.
- **Key findings**:
  1. The **optimal M increases monotonically with iteration count**: a single big model wins short-term but is dominated by 16 small specialized models long-term.
  2. Robust across model families (OPT vs GPT-2), parameter sizes, mixing real data (10% Wikitext at each iteration), and temperature sampling τ ∈ {0.5, 1, 2}.
  3. Scaling up the system (V1: bigger models / more data) *amplifies* collapse in homogeneous ecosystems, *increasing* diversity benefits.
- **Code**: Not yet publicly released as of 2026-04-28 (paper Dec 2025 / Mar 2026).
- **Relevance**: This is the closest published answer to our research question and the natural baseline. Open extensions:
  - They cap at M=16; we could push further and also test compute scaling laws.
  - They instantiate diversity *only* through training data segmentation. Other axes — architecture, prompt, RAG corpus — were enumerated but not tested.
  - They use a fixed "shuffle and redistribute" mixing scheme; alternative mixing topologies (e.g., social-network style with limited information sharing) are unexplored.

#### Wang et al. 2025 — *LLM Web Dynamics: Tracing Model Collapse in a Network of LLMs* (arXiv 2506.15690)
- **Contribution**: API-only ecosystem simulator using RAG instead of fine-tuning, hugely cheaper. n LLMs from different families share a common "Internet" (text database); each retrieves k_t = ⌊β·|A_t|⌋ sentences for context, generates responses, and posts back. They prove (and show empirically) that the Frobenius norm of pairwise embedding-distance matrix converges to a small constant — so even pretrained-only LLMs without any fine-tuning collapse toward each other in this multi-model RAG dynamic. Provides an analogous GMM-based system as theoretical backbone.
- **Methodology**: 3 LLMs (Llama-3.1-8B, DeepSeek-7B-chat, Mistral-7B-Instruct), pretrained-different. Seeded "Internet" = 20 crypto posts. Single fixed query repeated for T=60 iterations. Embed via nomic-embed-v1.5. Distance matrix evolves over time.
- **Code**: Not publicly released.
- **Relevance**: An entirely different methodology (RAG-based, no training) for the same research question, and a much cheaper way to run many-N experiments. Their result that even *pretrained-different* models converge implies the answer to "what counts as different" is non-trivial: model-family alone is not sufficient differentiation under an aggressive RAG-mixing scheme.

#### Vu, Reeves & Wenger 2025 — *What happens when generative AI models train recursively on each others' outputs?* (arXiv 2505.21677)
- **Contribution**: Theoretical framework for *data-mediated interactions* between heterogeneous models: derives concise formulas for the dynamics of mixed-source recursive training, validates with empirical LLM experiments. Documents the substantial overlap of pretraining datasets across major commercial LLMs (Llama, GPT, Phi, etc.) — key empirical motivation for studying ecosystem-level collapse.
- **Key finding**: Multi-model recursive training has *both* benefits (exposure to novel concepts ≈ transfer learning) and harms (homogenization on shared tasks). Replicates collapse but qualifies it.
- **Relevance**: Provides a useful theoretical lens when we vary the *origin* of synthetic data each agent ingests — own-output vs. peer-output vs. mixed.

#### Kovač et al. 2025 — *Recursive Training Loops in LLMs: How training data properties modulate distribution shift* (arXiv 2504.03814)
- **Contribution**: Treats the data side, not the model side, as the diversity knob. Uses an "iterative chain" framework where at each generation a fresh base LLM (sampled uniformly from {Llama-3.2-1B, Qwen2.5-1.5B, SmolLM-1.7B, Falcon3-1B}) is fine-tuned via LoRA on a mix of pre-existing accumulated content and fresh human samples, then generates more content. Evaluates 5 datasets (Twitter ×2, Reddit ×2, Wikipedia ×1) across 3 domains; runs regression analyses on 800 clusters.
- **Key findings**: Lexical diversity *amplifies* shift; semantic diversity and data quality *mitigate* it; influences are highly modular (cross-domain spillover is small); political bias amplification depends on baseline lean.
- **Code**: https://github.com/flowersteam/ce_llms (cloned locally).
- **Relevance**: Closest existing scaffold for an experiment runner — already implements iterative chains, multi-base-model rotation, LoRA fine-tuning, dataset clustering, evaluation pipelines.

### Other related (catalogued but not deep-read)

- **Bias and fairness loops** — Wyllie et al. 2024 (Fairness Feedback Loops, arXiv 2403.07857), Bohacek & Farid 2024 (Nepotistically Trained, arXiv 2311.12202): show systematic biases compound under recursive training.
- **Theoretical bounds** — Dey & Donoho 2024 (Universality of π²/6, arXiv 2410.22812): if data accumulates, error is bounded by a universal π²/6 constant. Bertrand et al. (Heat Death, 2402.07043): closed-loop learning has heat-death dynamics.
- **Practical mitigations** — machine-generated text detection (Drayson & Lampos 2502.15654), reinforcement-style verification (Feng et al. 2406.07515 / 2410.16713), curated data (Ferbach et al. 2407.09499), text synthesis without collapse (Zhang et al. 2412.14689).
- **Self-improvement** — papers in our results corpus (RL-based self-training, Think-Prune-Train, Importance Weighting, Spend Wisely, etc.) study the *engineered* recursive training case. Provide diversity/curation tricks our ecosystems could employ within an agent.

## Common experimental methodologies

Across the corpus, the dominant single-model recipe is:

1. Pick a base model (usually OPT-125m, GPT-2, or small Llama variant).
2. Pick a base dataset (WikiText-2 for fine-tuning; TinyStories / C4 for pretraining).
3. For T generations:
   - Sample N synthetic tokens from generation t-1's model.
   - (Optional) mix in real data — replacement / accumulation / partial accumulation.
   - Fine-tune (or pretrain from scratch) generation t.
4. Evaluate generation t on a fixed held-out test set — perplexity is the standard metric.

The ecosystem variants substitute step (3) with multi-agent versions:
- **Hodel & West**: M models, segment + shuffle + redistribute pool every iteration.
- **Wang et al.**: M models, no fine-tuning, share via RAG over a growing text pool.
- **Kovač et al.**: rotate base models across iterations, accumulate posts, mix with fresh human posts.

## Standard baselines

- **N=1, replace, no real data refresh**: the Shumailov-style "worst-case" curve (rising perplexity, vanishing tails). Our ecosystem with M ≥ 2 should be compared against this.
- **N=1, accumulate**: Gerstgrasser π²/6 bounded curve. Tells us how much value pure data accumulation buys without diversity.
- **N=1, 10% real data refresh**: Shumailov regime (b). Realistic upper bound of "what a careful single-actor pipeline can do."
- **N=M, segmented data**: Hodel & West baseline.
- **N=M, RAG-only (no fine-tuning)**: Wang et al. baseline; cheap to run.

## Standard metrics

- **Perplexity on held-out human data** (most common; quality+coverage proxy).
- **Lexical diversity (Type-Token Ratio, n-gram TTR)** — Guo et al.
- **Semantic diversity (cosine sim over sentence embeddings)** — Guo et al., Hodel & West.
- **Embedding distance matrix** (Frobenius norm of pairwise distances over embedded outputs) — Wang et al.
- **Hill-Shannon Diversity (HSD)** — Hodel & West, from ecology, captures "effective number of distinct equally-frequent agents."
- **Distribution-shift metrics** — KL / Wasserstein from baseline distribution.
- **Bias / political-lean drift** — Kovač et al., Wang et al.

## Datasets in the literature

- **WikiText-2 (raw, v1)** — the dominant fine-tuning benchmark; downloaded.
- **TinyStories** — small kindergarten-level corpus suitable for pretraining-from-scratch loops; downloaded.
- **C4** — large-scale; used for higher-capacity models, not yet downloaded (large).
- **Reddit / Twitter / Wikipedia clusters** — Kovač et al.; available via `code/ce_llms/`.
- **Crypto-news posts (Ebrahimi 2024)** — Wang et al. (~20 short posts).

## Gaps and opportunities (specific to our research question)

1. **Population-size scaling beyond M=16.** Hodel & West's largest ecosystem is 16; the slope of optimal-M vs iteration suggests larger N pays off. Empirically pinning down *minimum* viable N for a target horizon is open.
2. **Identity along architecture / pretraining axis, not just data segments.** Hodel & West vary data segmentation only. Wang et al. show that even pretrained-different models converge under aggressive RAG mixing. The question — *what differentiation suffices* — needs orthogonal axis tests:
   - Same architecture, different training-data segments (Hodel & West).
   - Different architectures, same training data.
   - Different prompts / personas, same model.
   - Different RAG sources / retrieval policies.
3. **Mixing-topology dependence.** Existing work uses uniform redistribution or full RAG sharing. Real ecosystems have structured information networks. A graph-topology study (clustered networks, hub-spoke, small-world) is open.
4. **Cheap-eval pipeline.** Wang-style RAG-only simulations let us scan large N (say 2…100) at API cost, and validate at small N with full fine-tuning à la Hodel & West.
5. **Notion of "minimum viable population" formalism.** Borrow from population genetics / ecology — currently no paper formalizes a *threshold N* below which extinction (full collapse) is inevitable. Hill-Shannon (Hodel & West) is the start.

## Recommendations for our experiment

### Recommended primary setup
- **Backbone (cheap)**: Hodel & West-style segmented WikiText-2 with OPT-125m, M ∈ {1, 2, 4, 8, 16, 32, 64}. This extends their grid into the "minimum viable" regime they only touched.
- **Iterations**: at least 10, ideally 20+, to expose the monotonic optimal-M trend.
- **Per-ecosystem token budget N held constant** — this controls for compute and isolates diversity as the variable.

### Recommended companion setup
- **API-only RAG ecosystem**: Wang-style 3+ LLMs differing in pretraining (e.g., HuggingFace small variants) sharing a seeded text DB. Sweep N up cheaply. Use Frobenius distance matrix as the diversity-loss metric.

### Recommended baselines
- **N=1 replace** (Shumailov regime a): worst-case lower bound.
- **N=1 accumulate** (Gerstgrasser): single-model upper-bound mitigation.
- **N=M segmented** (Hodel & West): published ecosystem baseline.

### Recommended metrics
- Perplexity on WikiText-2 test.
- Lexical TTR + semantic embedding cosine-sim variance (Guo).
- Pairwise embedding-distance matrix Frobenius norm (Wang).
- Hill-Shannon Diversity computed over output embeddings (effective-number-of-distinct-models).

### Methodological warnings (from the literature)
- **Repeating-text artifact**: Shumailov found that LMs naturally produce repetitive output; using a repetition penalty actually *worsens* collapse. Don't naively penalize repetition.
- **Generation temperature matters**: Gerstgrasser test τ ∈ {0.3, 1.0}; Hodel & West test τ ∈ {0.5, 1, 2}. Fix or sweep deliberately.
- **Test set must be from the original real distribution**, fixed across all generations — perplexity on a synthetic test set is the wrong measurement (Shumailov, Hodel & West).
- **Cross-generation comparison requires equal-size training data** (Gerstgrasser footnote 1) — careful when comparing replace vs accumulate at fixed iteration n.
- **Compute realism**: Hodel & West's 10-iteration OPT-125m experiment ran on consumer-scale GPUs in hours; Wang et al.'s 60-iteration RAG simulation took ~8 hours on an A100. Either is feasible without a cluster.

## Bottom line for our project

The literature *strongly* suggests population diversity mitigates collapse, but
the existing answer (Hodel & West's M ∈ {1,2,4,16}) is incomplete:

- It does not pin down a *minimum* N for a fixed horizon.
- It tests only one axis of differentiation (training-data segments).
- It does not contrast cheap RAG-style ecosystems (Wang et al.) against fine-tuning-style ecosystems.

Our project sits naturally in this gap. The most directly extensible experimental
scaffold is Hodel & West's pipeline (which has no public code yet — we will need
to reimplement, but the recipe is fully spec'd in the paper) augmented with the
Kovač et al. iterative-chain code (`code/ce_llms/`) for richer dataset / model
manipulation, and validated cheaply in a Wang-style RAG harness.
