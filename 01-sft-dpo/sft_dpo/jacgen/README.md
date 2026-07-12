# jacgen — Jac-native conversion data generation + probe harness

Synthetic **Python → Jac conversion** training data plus the eval/instrumentation
harness for the mini finetune probe. Every example is gated by `jac run` (exit 0)
**and** a behavioral check (program output matches expected across multiple distinct
test cases) — never by `jac check`, which over-rejects untyped-but-runnable Jac. The
idiomatic core is agentic (Claude + jac-mcp); volume is automated transpile +
behavioral gate (MultiPL-T style). **All tooling is written in Jac.**

> Full project handoff (env, run, gotchas, rebuild order): **`../docs/modeltesting/HANDOFF.md`**.

## Modules (24)

**Shared libraries**

| File | Role |
|---|---|
| `writer.jac` | `make_sft_example`, `append_jsonl`, `run_jac` (subprocess `jac run` — the gate), `extract_jac`, `revalidate_example`. |
| `corpus.jac` | HF mining + transpile lib: `fetch_page/fetch_rows` (+retry), `is_clean` filter, `first_func`, `py_cases` (exec in builtins-only ns, SIGALRM 2s timeout, ≤3 distinct-output cases), `transpile` (`jac py2jac`), `sanitize_transpile`, `normalize`. |
| `decontam.jac` | 14-token shingles, `is_contaminated` (≥0.5 overlap), `extract_python`. |
| `dedup.jac` | ROUGE-L (LCS) near-duplicate guard on the Jac body. |

**Generators / builders** (run order matters — see HANDOFF §6)

| File | Role |
|---|---|
| `mine.jac` | Cleaned runnable functions → `dataset/source_pool/mined.jsonl`. |
| `seed_conversion.jac` | ⚠️ **TRUNCATES** `sft.jsonl` → 32 idiomatic seeds + 2 seed DPO pairs. |
| `idiomatic_batch.jac` / `…batch2.jac` / `…batch3.jac` | **APPEND** +30 / +23 / +31 idiomatic examples (`source: generated_idiomatic`). |
| `scale_conversion.jac` | Mine Vezora → `jac py2jac` → **jac-run gate** → 1500 → `sft_auto.jsonl`. |
| `dpo_conversion.jac` | Idiomatic (chosen) vs transpiled Python-shaped (rejected), compile-gated → `dpo.jsonl`. |
| `build_manifest.jac` | All idiomatic + stride-sampled transpile at **1:3** → `sft_train.jsonl`. |
| `build_splits.jac` | `sft_train.jsonl` → `01-sft-dpo/dataset/mlx/{train,valid}.jsonl` (messages-only). |
| `build_dpo_splits.jac` | `dpo.jsonl` → `01-sft-dpo/dataset/mlx_dpo/{train,valid}.jsonl` (`{prompt,chosen,rejected}` for mlx-lm-lora DPO). |
| `graph_seeds.jac` | **Graph tier (real idiom headroom):** reads `graph_data/train.json` (31), appends node/edge/walker SFT examples to `sft.jsonl`. Idiomatic-vs-transpile sim ~0.26 (vs 0.97 for functions). |
| `graph_holdout.jac` | Reads `graph_data/holdout.json` (13 disjoint tasks) → `eval_holdout/graph_conversion.jsonl`. Target via `JAC_HOLDOUT`. |
| `holdout.jac` | Disjoint offsets (12000+) + 14-gram decontam → `eval_holdout/conversion.jsonl` (150). |

**Eval + instrumentation**

| File | Role |
|---|---|
| `eval_probe.jac` | The harness: load model **once** (mlx), score holdout via `jac run`, report run% / test-pass% / token metrics. Modes `py2jac` (no-LLM smoke) \| `mlx`. ⚠️ lazy-imports `mlx_lm`; parse-checked only. |
| `idiom_eval.jac` | Idiom-quality judge: per correct output, `rouge_l(output, py2jac(python))` — high sim = Python-shaped (learned to transpile), low = idiomatic-divergent. Buckets idiomatic vs python_shaped. ⚠️ lazy-imports `mlx_lm`; parse-checked only. |
| `dashboard.jac` | Zero-dep ASCII live view (loss/LR/throughput/learning curve). |
| `plot_metrics.jac` | matplotlib → 8 PNGs. |
| `dataset_stats.jac` | Composition report by difficulty / generator / source. |
| `verify_dataset.jac` | **Non-destructive** sampled re-validation (`JAC_SAMPLE_EVERY`). check.sh's behavior gate. |
| `decontam_report.jac` | Holdout-vs-train contamination audit. |

## Data tiers

- **Idiomatic core** (`sft.jsonl`, 147 incl 31 graph, `generator: claude-code-mcp/graph`) — true
  graph-spatial Jac (walkers, nodes, edges, abilities, `with entry`). The quality bar.
- **Volume tier** (`sft_auto.jsonl`, 1500, `generator: jac-py2jac`) — transpiled
  corpus functions, compiler + behaviorally validated. Python-shaped Jac; breadth.
- **DPO** (`dpo.jsonl`, 147) — idiomatic (chosen) vs Python-shaped (rejected):
  teaches de-Python-ification.

Total SFT = **1647** (147 + 1500). Manifest `sft_train.jsonl` = 588 (1:3).
mlx splits = 529/59 (SFT) and 132/15 (DPO). Holdouts = 150 function + 13 graph.

## Source corpus

`Vezora/Tested-22k-Python-Alpaca` via the HuggingFace datasets-server rows API
(curl). CodeSearchNet was rejected — repo-coupled, ~4/600 runnable standalone;
Vezora is ~28% runnable. Functions kept only if self-contained, single-argument,
stdlib-only, returning a value across distinct test inputs.

## Run

Full rebuild order (the batches append, `seed_conversion` truncates — order matters):

```bash
jac run 01-sft-dpo/sft_dpo/jacgen/seed_conversion.jac     # sft.jsonl -> 32 (TRUNCATES)
jac run 01-sft-dpo/sft_dpo/jacgen/idiomatic_batch.jac     # -> 62  (append)
jac run 01-sft-dpo/sft_dpo/jacgen/idiomatic_batch2.jac    # -> 85
jac run 01-sft-dpo/sft_dpo/jacgen/idiomatic_batch3.jac    # -> 116
jac run 01-sft-dpo/sft_dpo/jacgen/graph_seeds.jac         # + 31 graph (node/edge/walker) -> 147
jac run 01-sft-dpo/sft_dpo/jacgen/scale_conversion.jac    # sft_auto.jsonl -> 1500 (slow)
jac run 01-sft-dpo/sft_dpo/jacgen/dpo_conversion.jac      # dpo.jsonl -> ~147
jac run 01-sft-dpo/sft_dpo/jacgen/build_manifest.jac      # sft_train.jsonl -> 588 (1:3)
jac run 01-sft-dpo/sft_dpo/jacgen/build_splits.jac        # 01-sft-dpo/dataset/mlx/{train,valid}.jsonl -> 529/59
jac run 01-sft-dpo/sft_dpo/jacgen/build_dpo_splits.jac    # 01-sft-dpo/dataset/mlx_dpo/{train,valid}.jsonl -> 132/15
jac run 01-sft-dpo/sft_dpo/jacgen/holdout.jac             # function eval holdout -> 150
jac run 01-sft-dpo/sft_dpo/jacgen/graph_holdout.jac       # graph eval holdout -> 13
jac run 01-sft-dpo/sft_dpo/jacgen/dataset_stats.jac       # report
jac run 01-sft-dpo/sft_dpo/jacgen/verify_dataset.jac      # non-destructive audit (JAC_SAMPLE_EVERY=40)
```

Generated data lands in gitignored `dataset/`.
