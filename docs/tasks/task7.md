# Task 7: Advanced Generation and Preference Recipes

## Purpose

Extend the pipeline beyond the core single-turn and trajectory recipes with techniques adopted from a set of code-LLM papers: WizardCoder (2306.08568), Magicoder / OSS-Instruct (2312.02120), DeepSeek-Coder (2401.14196), DeepSeek-Coder-V2 (2406.11931), Magpie (2406.08464), CodeDPO (2410.05605), and SelfCodeAlign (2410.24198). These recipes raise data diversity, harden preference data against untrusted synthetic tests, and add data formats the core recipes do not produce.

This task assumes the foundation, prompt design, validation, single-turn generation, trajectory collection, and release tasks are in place. It depends on the cross-compiled test infrastructure from Task 3 and Task 4.

## Inputs Needed

- [`context.md`](../context.md) for strategy and seeding axes.
- [`task1.md`](task1.md) for metadata, naming, and `seed_source` values.
- [`task3.md`](task3.md) for validation, credibility scoring, and the reward-model data note.
- [`task6.md`](task6.md) for deduplication and DPO pair construction.
- The filtered Python source pool and deterministic Python-to-Jac test compiler.
- A trained v0 Jac model (for zero-seed extraction and self-distillation recipes).

## Artifacts To Produce

- Snippet-seeded generation batches with `seed_source: oss_instruct`.
- Zero-seed extraction batches with `seed_source: zero_seed` (post-v0).
- Credibility-ranked DPO preference pairs with `test_credibility_score` and `solution_credibility_score`.
- Runtime-efficiency preference pairs with `runtime_ms`.
- Fill-in-the-Middle (FIM) infill samples.
- Repo-level multi-file synthetic Jac projects.
- Semantic-domain coverage report.
- Token-accounting report (per-example and aggregate).

## Step-By-Step Checklist

### Snippet-seeded generation (OSS-Instruct, Magicoder)

- [ ] Sample 1--15 random consecutive lines of real Python code from the source pool.
- [ ] Abstract the snippet to high-level concepts before generating (SelfCodeAlign: seed→concepts beats seed→instruction directly, 65.2 vs 59.8).
- [ ] Prompt for a self-contained, unrelated Jac problem inspired by the concepts.
- [ ] Validate through the full pipeline; record `seed_source: oss_instruct` and `semantic_domain`.
- [ ] Track mean cosine-to-holdout; prefer this method while it stays novel and passes gates.

### Zero-seed template extraction (Magpie)

- [ ] Requires a v0 Jac model. Prompt with only the pre-query chat-template prefix (no seed).
- [ ] Decode instructions at high temperature (1.0--1.25, top-p 0.99) for diversity.
- [ ] Decode responses greedily (highest-probability tokens best reflect the correct distribution).
- [ ] Validate through the full pipeline; record `seed_source: zero_seed`.

### Credibility-ranked DPO (CodeDPO)

- [ ] For each task, generate 15 candidate solutions at temperature 1.5 and multiple candidate tests.
- [ ] Build a bipartite pass/fail graph; iterate coupled PageRank scores (damping 0.85, ~10 iterations).
- [ ] Record `solution_credibility_score` and `test_credibility_score`.
- [ ] Select DPO winners (high credibility) and losers (low credibility); drop pairs with near-identical scores.
- [ ] Cross-compiled tests remain the strongest gate; credibility scoring applies where only generated tests exist.

### Runtime-efficiency preference pairs (CodeDPO)

- [ ] Among solutions passing the top-credibility tests, measure execution time; record `runtime_ms`.
- [ ] Pair faster vs slower correct solutions as preferred/rejected.
- [ ] Measure timing only on credibility-vetted tests so timing is not polluted by wrong tests.

### Fill-in-the-Middle data (DeepSeek-Coder)

- [ ] Generate FIM/infill samples at document level before packing, PSM mode, 0.5 rate.
- [ ] Use the sentinel layout `<|fim_start|>pre<|fim_hole|>suf<|fim_end|>middle<|eos|>`.
- [ ] Do not exceed ~50% FIM; 100% FIM hurts left-to-right completion.

### Repo-level synthetic projects (DeepSeek-Coder)

- [ ] Synthesize multi-file Jac projects, not only isolated snippets.
- [ ] Build a file-dependency graph from `import`/`include`/node-walker references.
- [ ] Topologically sort (minimal-in-degree variant to tolerate import cycles; handle disconnected subgraphs separately) so each file appears after its dependencies in one sample.
- [ ] Prepend a file-path comment to every file before packing.
- [ ] Run repo-level MinHash near-dedup over the concatenated project.

### Evolution control (WizardCoder)

- [ ] Merge every Evol-Instruct round with all prior rounds and the seed (preserves difficulty gradient).
- [ ] Stop evolution via a held-out Jac dev set (~3 rounds optimal; gains non-monotonic).
- [ ] Use a bounded ~10-word constraint budget per round.

### Token accounting

- [ ] Record `token_count` (and optional `prompt_token_count` / `completion_token_count`) on every example.
- [ ] Log aggregate token usage per batch and per run, by generator and recipe, to `dataset/logs/generation/`.

## Testing And Validation Checklist

- [ ] Confirm snippet-seeded and zero-seed batches carry the correct `seed_source`.
- [ ] Confirm credibility scores are recorded and DPO pairs were filtered for ambiguity.
- [ ] Confirm runtime-efficiency pairs measured timing only on credibility-vetted tests.
- [ ] Confirm FIM rate does not exceed 50% and uses PSM mode.
- [ ] Confirm repo-level projects pack files in dependency order with path comments.
- [ ] Confirm semantic-domain distribution is balanced across the ~10 domains.
- [ ] Confirm token counts are present per example and aggregated per batch.

## Failure Conditions And Retry Guidance

- If snippet-seeded output reuses the seed verbatim, strengthen the concept-abstraction step.
- If zero-seed extraction produces low-quality Jac, the v0 model is too weak; bootstrap it on seed data first (WizardCoder Appendix E) before using it as a generator.
- If DPO pairs are noisy, widen the credibility spread (higher temperature) and drop near-identical-score pairs; use RPO loss if training is unstable.
- If FIM hurts left-to-right completion, lower the FIM rate toward 50% PSM.
- If repo packing breaks on import cycles, use the minimal-in-degree topsort variant and split disconnected subgraphs.

## Completion Criteria

This task is complete when each advanced recipe has documented prompts, validation paths, metadata, and pilot results, and the team has decided which recipes graduate into volume generation based on quality and diversity gains over the core recipes.
