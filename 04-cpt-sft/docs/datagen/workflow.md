# datagen/workflow.md — Pipeline Mechanics

Companion to `spec.md` (task catalog) and `../spec.md` (architecture). This
file is the mechanical run order, module dependency graph, and per-call
sequence for `model-experiments/01-sft-dpo/sft_dpo/jacgen2/`.

## 1. Module dependency graph

```mermaid
flowchart TD
    subgraph reused["reused from jacgen/ (unmodified)"]
        writer[writer.jac<br/>make_sft_example, append_jsonl,<br/>run_jac gate, revalidate_example]
        dedup[dedup.jac<br/>ROUGE-L near-dup guard]
        decontam[decontam.jac<br/>14-token shingle overlap]
        convpipe[existing conversion pipeline<br/>mine / scale_conversion / idiomatic_batch* / graph_seeds]
    end

    subgraph new["jacgen2/ (new)"]
        seedpool[seed_pool.jac]
        llmopus[llm.jac: opus wrapper<br/>bulk / token-heavy]
        llmfable[llm.jac: fable wrapper<br/>precision / error-prone]
        gencg[gen_code_gen.jac]
        gendbg[gen_debug.jac]
        genexp[gen_explanation.jac]
        gentraj[gen_trajectory.jac]
        gendoc[gen_documentation.jac]
        genmig[gen_migration.jac]
        gendpo[gen_dpo.jac]
        manifest[build_manifest_v2.jac]
        stats[dataset_stats_v2.jac]
        decontamv2[decontam_v2.jac]
    end

    mcp[(jac-mcp<br/>list_examples / get_example /<br/>search_docs / get_resource)]

    mcp --> seedpool
    seedpool --> gencg
    seedpool --> gendbg
    seedpool --> genexp
    seedpool --> gentraj
    seedpool --> gendoc
    seedpool --> genmig
    seedpool --> gendpo

    llmopus --> gencg
    llmopus --> gentraj
    llmopus --> genmig
    llmfable --> gendbg
    llmfable --> genexp
    llmfable --> gendoc
    llmfable --> gendpo

    gencg --> writer
    gendbg --> writer
    gentraj --> writer
    genmig --> writer
    gendpo --> writer
    genexp -.groundedness check only, no compiler.-> writer
    gendoc -.symbol-existence check only, no compiler.-> writer

    writer --> dedup --> decontam

    gencg --> manifest
    gendbg --> manifest
    genexp --> manifest
    gentraj --> manifest
    gendoc --> manifest
    genmig --> manifest
    convpipe --> manifest
    gendpo --> manifest

    manifest --> stats
    manifest --> decontamv2
```

## 2. Run order

Seed pool is built once per run-tag pass (or reused if unchanged — see §4).
Generators can run in any order relative to each other; they don't depend on
each other's output, only on the seed pool. `build_manifest_v2.jac` must run
last since it consumes every category's clean output.

```bash
export RUN_TAG=fresh   # or post_cptv2

jac run model-experiments/01-sft-dpo/sft_dpo/jacgen2/seed_pool.jac          # seed_pool.jsonl (shared, not run-tag-scoped)

jac run model-experiments/01-sft-dpo/sft_dpo/jacgen2/gen_code_gen.jac        # -> model-experiments/04-cpt-sft/dataset/$RUN_TAG/clean_dataset/code_gen/
jac run model-experiments/01-sft-dpo/sft_dpo/jacgen2/gen_debug.jac           # -> .../debug/
jac run model-experiments/01-sft-dpo/sft_dpo/jacgen2/gen_explanation.jac     # -> .../explanation/
jac run model-experiments/01-sft-dpo/sft_dpo/jacgen2/gen_trajectory.jac      # -> .../trajectory/
jac run model-experiments/01-sft-dpo/sft_dpo/jacgen2/gen_documentation.jac   # -> .../documentation/
jac run model-experiments/01-sft-dpo/sft_dpo/jacgen2/gen_migration.jac       # -> .../migration/
jac run model-experiments/01-sft-dpo/sft_dpo/jacgen2/gen_dpo.jac             # -> .../dpo/ (see dpo-plan.md)

# conversion category: existing jacgen/ pipeline, unchanged, run separately
# (already produces model-experiments/01-sft-dpo/dataset/sft.jsonl + sft_auto.jsonl — build_manifest_v2
#  reads those directly, does not regenerate them per run-tag)

jac run model-experiments/01-sft-dpo/sft_dpo/jacgen2/build_manifest_v2.jac   # -> model-experiments/04-cpt-sft/dataset/$RUN_TAG/releases/sft_train.jsonl
jac run model-experiments/01-sft-dpo/sft_dpo/jacgen2/dataset_stats_v2.jac    # composition report
jac run model-experiments/01-sft-dpo/sft_dpo/jacgen2/decontam_v2.jac         # contamination audit vs eval holdouts
```

Note on `conversion`: because it's reused unmodified from `jacgen/`, it is
**not** independently regenerated per run-tag the way the other six
categories are. `fresh` and `post_cptv2` releases share the same conversion
slice. This is a deliberate asymmetry — re-running the conversion miner
against a live HF dataset a second time would introduce corpus-drift noise
(HF row ordering / availability can change) with no benefit, since conversion
generation has no LLM-creative component to vary between runs in the first
place (it's transpile + compiler gate, deterministic given the same source
rows).

## 3. Per-example generation sequence (one `gen_code_gen.jac` call)

```mermaid
sequenceDiagram
    participant Seed as seed_pool.jsonl
    participant Gen as gen_code_gen.jac
    participant LLM as llm.jac (Opus, by llm())
    participant Gate as writer.jac (jac run)
    participant Dedup as dedup.jac / decontam.jac
    participant Out as clean_dataset/code_gen/

    Seed->>Gen: next seed (jac_code, domain, task_type)
    Gen->>LLM: reverse-generate NL instruction for this seed
    LLM-->>Gen: instruction text
    Gen->>Gate: jac run seed's own code (confirm still compiles+behaves)
    alt seed fails gate
        Gate-->>Gen: reject, log to rejected/code_gen/ with reason
    else seed passes
        Gate-->>Gen: ok
        Gen->>Dedup: check instruction+code against existing clean set
        alt near-duplicate or contaminated vs holdout
            Dedup-->>Gen: reject, log with reason
        else unique and clean
            Dedup-->>Gen: ok
            Gen->>Out: append {instruction, jac_code, meta...}
        end
    end
```

`gen_debug.jac`, `gen_explanation.jac`, and `gen_dpo.jac` follow the same
shape but call the Fable wrapper instead of Opus (`../spec.md` §4.1).
`gen_debug.jac` has two gate calls (buggy variant must fail, fixed variant
must pass) instead of one. `gen_trajectory.jac` calls the Opus wrapper and
gates only the final turn. `gen_explanation.jac` replaces the `jac run` gate
with the lexical groundedness check described in `../spec.md` §7.

## 4. Run-tag isolation, visually

```mermaid
flowchart LR
    seedpool[seed_pool.jsonl<br/>shared, not tagged]

    seedpool --> freshgen[RUN_TAG=fresh<br/>generators + Opus/Fable calls]
    seedpool --> postgen[RUN_TAG=post_cptv2<br/>generators + Opus/Fable calls]

    freshgen --> freshout["model-experiments/04-cpt-sft/dataset/fresh/releases/<br/>sft_train.jsonl, dpo_train.jsonl"]
    postgen --> postout["model-experiments/04-cpt-sft/dataset/post_cptv2/releases/<br/>sft_train.jsonl, dpo_train.jsonl"]

    freshout --> compare[dataset_stats_v2.jac diff report<br/>fresh vs post_cptv2 composition]
    postout --> compare
```

The diff report in step 4 is the confound-mitigation step referenced in
`../workflow.md` §comparison protocol: since the two datasets are
independently LLM-generated (not the same content reused), some of the
downstream eval delta between the two SFT runs could be dataset-generation
noise rather than a real CPT effect. Comparing `task_type` distribution,
per-category example counts, and rejection rates between the two releases
gives a sanity check — if the two datasets look statistically similar in
composition, the shared-seed-pool design has done its job and the eval delta
is more likely attributable to the base model, not the data.

## 5. Cost / scale accounting

10,000-15,000 examples × 2 independent run-tags, split by model per `../spec.md`
§4.1:

- **Opus** (`code_gen` + `trajectory` + `migration`): ~4,500 `code_gen`
  (1 call/example, minus the `error_message_authoring` slice which is
  Fable) + ~1,250 `trajectory` (up to 6 calls/example for multi-turn
  unrolling, budget at ~4x raw example count in call volume) + ~500
  `migration` (1 call/example, but whole-file outputs — budget ~2x normal
  tokens/call) → roughly 10,000-10,500 calls per run-tag, ~20,000-21,000
  total across both tags. This is the bulk of total call volume, by
  design — Opus carries the token-heavy load.
- **Fable** (`debug` + `explanation` + `documentation` + `gen_dpo` + the
  ungated-prose task-type overrides): ~2,000 `debug` + ~1,250
  `explanation` + ~750 `documentation` + ~2,500 DPO pairs (1 call/example
  each) + the `error_message_authoring`/`code_critique` slices → roughly
  6,500-7,000 calls per run-tag, ~13,000-14,000 total across both tags.

`conversion` contributes no LLM calls at all (reused deterministic
pipeline).

Recommended sequencing to control spend: run the pilot (§5 of `../spec.md`
rollout plan, ~20-30 examples per category, both models) first, read
`dataset_stats_v2.jac`'s token-usage-per-batch log
(`model-experiments/04-cpt-sft/dataset/$RUN_TAG/logs/generation/`, following
the existing `dataset/logs/generation/` convention from `jac-context-v1.md`,
broken out per model) to get actual per-example cost for Opus and Fable
separately, then extrapolate before committing to the full 10,000-15,000 run
for either tag.

## 6. Idempotency / resumability

Every generator appends rather than overwrites (matching `writer.jac`'s
`append_jsonl` convention), and every example carries a `seed_id`. A
generator run can be safely re-invoked after a partial failure (network
error, rate limit) — it should skip `seed_id`s already present in that
run-tag's `clean_dataset/<category>/` output before making a fresh Opus or
Fable call. This mirrors the existing `jacgen/verify_dataset.jac` non-destructive
re-validation pattern rather than introducing a new resumability mechanism.
