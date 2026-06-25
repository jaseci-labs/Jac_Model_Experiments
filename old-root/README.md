# Jac Data Generation — Synthetic Python→Jac Training Data + Finetune Probe

Build **100% synthetic, compiler-validated Python→Jac conversion training data**, then
prove it works: LoRA-finetune a small open base model on that data on a single
Apple-Silicon Mac (48 GB, MLX) and measure **base-vs-finetuned** correctness on a
held-out, decontaminated test set.

**The result, in one line:** a stock 30B / 26B model produces **0% runnable Jac**;
after one LoRA pass on our data it produces **93–94% behaviorally-correct Jac** — and on
graph-shaped tasks, **100% idiomatic** (nodes/edges/walkers, not Python-shaped) after DPO.

> **Confidential — Jaseci Labs.** Every tool in this repo is itself written in Jac
> (`sft_dpo/jacgen/`) — we dogfood the language we generate data for.

---

## Table of contents

- [Why this exists](#why-this-exists)
- [What Jac is (and why models fail at it)](#what-jac-is-and-why-models-fail-at-it)
- [The core idea](#the-core-idea)
- [Quickstart](#quickstart)
- [Results](#results)
- [How the data is built](#how-the-data-is-built)
- [The dataset on disk](#the-dataset-on-disk)
- [The probe](#the-probe)
- [Repository layout](#repository-layout)
- [The all-Jac pipeline (24 modules)](#the-all-jac-pipeline-24-modules)
- [Rebuilding the dataset (order matters)](#rebuilding-the-dataset-order-matters)
- [Environment](#environment)
- [Gotchas](#gotchas)
- [Documentation map](#documentation-map)
- [Glossary](#glossary)

---

## Why this exists

The goal is a **coding agent for Jac** — what Claude Code is for Python: generate,
debug, explain, and convert to **idiomatic, compiler-correct Jac**. Deployed via the
Jac MCP inside coding assistants. The quality bar is **compiles + runs + idiomatic**,
not "Jac-looking."

The blocker: **no real Jac corpus exists.** You cannot scrape your way to a Jac
dataset. So all training data is synthesized — and the entire value of this repo is the
machinery that makes synthetic Jac data *trustworthy* (compiler- and behavior-gated)
plus a probe that **measures** whether finetuning on it actually works.

This repo is the **model-testing phase**: data pipeline + a runnable finetune probe +
measured results on two candidate base models.

---

## What Jac is (and why models fail at it)

Jac (Jaseci Labs) is a programming language built **on top of Python** with a
**data-spatial / object-spatial** model: computation is expressed with **nodes, edges,
walkers, and abilities** instead of plain functions and classes. It compiles to Python
and interops with the ecosystem, but its idioms are distinct enough that a model trained
on Python/JS/C has a **very weak prior** on correct Jac.

| Jac construct | Role | Python analogue |
|---|---|---|
| `walker` | a traversal agent that moves through the graph | (no direct equivalent) |
| `node` / `edge` | graph primitives — data + typed connections | object + reference |
| `can … with <Node> entry` | an **ability** — event-triggered behavior | method (sort of) |
| `def` | a plain method | method |
| `obj` | preferred data archetype | `class` |
| `with entry` | module entry block | `if __name__ == "__main__"` |
| `spawn` / `++>` / `visit [-->]` / `disengage` | launch a walker / create edge / traverse / stop | — |
| `has` | typed field declaration | typed attribute |

A non-finetuned model produces Python-shaped code that *looks* plausible but is
syntactically or semantically wrong Jac. Closing that gap — cheaply, verifiably — is the
entire project.

---

## The core idea

**Three anchors substitute for a real-data distribution:**

1. **Jac grammar** = the distribution anchor (every construct must appear in the data).
2. **Jac compiler + cross-compiled tests** = an unlimited oracle. Rejection sampling is
   free; a **behavioral test pass** is the real gate, not mere compilation.
3. **Python** = the proxy distribution. Translate validated Python → idiomatic Jac
   (the **MultiPL-T** methodology).

**One load-bearing rule — the gate is `jac run`, never `jac check`.** "The conversion
works" means it **compiles, executes, and its output matches** the recorded behavioral
cases — behavioral equivalence. The strict type-checker (`jac check`) *rejects
untyped-but-runnable Jac*, which is correct Jac we want to keep. Switching the gate from
`jac check` to `jac run` moved a smoke-eval score from **26% → 96%**. This rule is
everywhere in the pipeline.

---

## Quickstart

```bash
./setup_env.sh && source .venv/bin/activate   # venv + jaclang + mlx-lm + matplotlib (NO anaconda)
./check.sh                                     # 19× jac check + parse-check + non-destructive behavioral re-validation
./run_probe.sh Qwen/Qwen3-Coder-30B-A3B-Instruct qwen   # quantize → base eval → train → fuse → finetuned eval → graphs
```

`./check.sh` should print `19 passed` + `eval_probe PASSED` (parse-only) + `39/39`
re-validated. If green, the toolchain and data are healthy. It **does not** mutate the
dataset.

Full operator runbook (setup, run, pause/resume, timings): **[`sft_dpo/process.md`](sft_dpo/process.md)**.
Full architecture handoff (every module, every gotcha): **[`sft_dpo/docs/modeltesting/HANDOFF.md`](sft_dpo/docs/modeltesting/HANDOFF.md)**.

---

## Results

Two base models run end-to-end. Primary metric = **cross-compiled test-pass %** on
unseen, decontaminated holdouts (compiles + runs + output matches behavioral cases).
**Full results, all 16 training graphs, side-by-side analysis → [`resultsft/RESULTS.md`](resultsft/RESULTS.md).**

### Function tier — 150 unseen function tasks (correctness)

| stage | Qwen test-pass | Gemma test-pass |
|---|---|---|
| **base** (stock model) | **0%** (0/150) | **0%** (0/150) |
| **SFT** (after finetune) | **94%** (141/150) | **93%** (140/150) |
| **DPO** | 93% (140/150) | 93% (140/150) |

Both stock models produce essentially **zero** runnable Jac → confirming the premise.
One LoRA-SFT pass takes both above **90% behaviorally correct**, converging to ~96% by
checkpoint 200. Generation also gets **leaner**: Qwen 34.7k→16.3k tokens (−53%), Gemma
76.8k→19.2k (−75%).

### Function tier — idiom judge (the honest caveat)

On standalone functions the model learns to **transpile** — outputs stay Python-shaped
(transpile-similarity **0.968**), and DPO doesn't move it. This is **not a defect**:
pure functions have **no idiom headroom** (for `factorial`, idiomatic Jac *is* the
mechanical transpile). The idiom axis only exists for **graph-shaped** problems →

### Graph tier — 13 unseen graph/tree tasks (idiom)

Idiomatic Jac here means **nodes + edges + a walker**, diverging hard from a naive
dict+stack transpile. This is where DPO can work.

| metric (13 graph tasks) | base | Qwen SFT | **Qwen DPO** | Gemma SFT | Gemma DPO |
|---|---|---|---|---|---|
| correct | **0%** | 46% | **61%** | 15% | 15% |
| of-correct idiomatic | — | 83% | **100%** | 100% | 100% |
| transpile-similarity | — | 0.457 | **0.338** | 0.667 | 0.667 |
| idiom constructs / output | 0 | 4.5 | **6.75** | 0.0 | 0.0 |

Qwen goes from *cannot do graph conversion at all* → SFT (46%, mostly idiomatic) →
**DPO lifts correctness to 61% and makes 100% of correct outputs idiomatic**, similarity
falling toward the 0.26 idiomatic reference. **Gemma learns the graph idiom far more
weakly** (15%, 0 detected constructs) — idiom acquisition is **model-dependent**, and
**Qwen3-Coder is the stronger base** for the idiom-sensitive agent.

**What this proves:** (1) synthetic compiler-validated data → correct Jac (0→94%);
(2) data *with* idiom headroom + DPO on real-divergence pairs → measurably *idiomatic*
Jac. The DPO machinery is proven and reusable.

---

## How the data is built

```
  HF corpus (Vezora/Tested-22k-Python-Alpaca, via datasets-server rows API)
        │  corpus.jac: fetch → is_clean filter → first_func → py_cases (exec, SIGALRM 2s)
        ▼
  mine.jac ──► dataset/source_pool/mined.jsonl        (cleaned Python + behavioral cases)
        │
        ├──► scale_conversion.jac ──► sft_auto.jsonl   (1500: py2jac transpile, jac-run gated)
        │
  seed + idiomatic_batch{,2,3} + graph_seeds ──► sft.jsonl  (147 idiomatic, incl 31 graph)
        │
        ├──► dpo_conversion.jac ──► dpo.jsonl          (147: idiomatic chosen vs transpile rejected)
        │
        ├──► build_manifest.jac ──► sft_train.jsonl    (588: 1:3 idiom:transpile, de-skew)
        │         └──► build_splits.jac ──► dataset/mlx/{train,valid}.jsonl (529/59, messages-only)
        │
        ├──► holdout.jac ──► eval_holdout/conversion.jsonl       (150 function, decontaminated)
        └──► graph_holdout.jac ──► eval_holdout/graph_conversion.jsonl (13 graph, disjoint)

  RUN: run_probe.sh ──► quantize Q4+Q8 ──► base eval ──► dry-run ──► LoRA train
        │                                    (eval_probe.jac: load model ONCE, jac-run gate)
        ├──► live: dashboard.jac (ASCII) + plot_metrics.jac (PNG) per checkpoint
        └──► fuse ──► finetuned eval ──► graphs
```

**Two data tiers, on purpose:**

- **Idiomatic core** (`sft.jsonl`, 147) — hand-written / agentically-written Jac using
  walkers, nodes, edges, abilities, `with entry`, typed edges. *Jac-shaped.* Includes
  **31 graph-tier** examples with real idiom headroom.
- **Transpile volume** (`sft_auto.jsonl`, 1500) — `jac py2jac` of mined Python,
  behaviorally gated. Correct but *Python-shaped*. Cheap volume.
- **DPO pairs** (`dpo.jsonl`, 147) — teach **de-Python-ification**: `chosen` =
  idiomatic, `rejected` = the transpiled Python-shaped version of the same function.

The manifest mixes idiom:transpile at **1:3** — enough idiom that the model doesn't just
learn "transpile-ese," without starving on the cheap volume.

**Verification order:** compiler gate → cross-compiled behavioral tests → idiom judge →
sampled manual review. The 12 generation recipes (R1–R12) are documented in
[`docs/initmodelchoice/strat.md`](docs/initmodelchoice/strat.md).

---

## The dataset on disk

`dataset/` is **gitignored** but fully **regenerable** from the Jac builders (see
[Rebuilding](#rebuilding-the-dataset-order-matters)). Confirm any time with
`jac run sft_dpo/jacgen/dataset_stats.jac`.

| Artifact | Path | Count |
|---|---|---|
| Idiomatic SFT core (hand/agentic + 31 graph) | `dataset/conversion/sft.jsonl` | **147** |
| Transpile SFT volume (py2jac, behaviorally gated) | `dataset/conversion/sft_auto.jsonl` | **1500** |
| **Total SFT** | both above | **1647** |
| DPO pairs (idiomatic vs Python-shaped; 24 graph) | `dataset/conversion/dpo.jsonl` | **147** |
| Balanced manifest (1:3 idiom:transpile) | `dataset/conversion/sft_train.jsonl` | **588** |
| mlx-lm SFT split (messages-only) | `dataset/mlx/{train,valid}.jsonl` | **529 / 59** |
| mlx-lm DPO split (`{prompt,chosen,rejected}`) | `dataset/mlx_dpo/{train,valid}.jsonl` | **132 / 15** |
| Function eval holdout (behavioral `test_cases`) | `dataset/eval_holdout/conversion.jsonl` | **150** |
| Graph eval holdout (idiom headroom) | `dataset/eval_holdout/graph_conversion.jsonl` | **13** |

Idiomatic core composition: 24 seed + 84 idiomatic batches + 8 mined + 31 graph = 147;
difficulty mix atomic 41 / idiomatic 37 / composed 69.

---

## The probe

`./run_probe.sh <model-id> <name>` runs, in order (each stage **skippable + resumable**):

1. **Quantize** the model → Q4 (train) + Q8 (eval).
2. **Base eval** on the 150 holdout → `results/<name>/base.txt`.
3. **30-iter dry-run** — bail check (loss drops, no NaN/OOM); Ctrl-C within 8s to abort.
4. **LoRA train** (`configs/lora.yaml`, 600 iters) with a live ASCII dashboard + PNG
   graphs refreshed per checkpoint.
5. **Fuse** adapter → Q8.
6. **Finetuned eval** on the 150 holdout → `results/<name>/finetuned.txt`.
7. **Graphs** → `results/<name>/*.png` (8 series + learning curve).

Per-model namespaced (`results/qwen/`, `results/gemma/`) — two models never clash.

**Metrics measured** (per checkpoint + base/finetuned): runs% (compiles+executes),
**cross-compiled test-pass%** (primary), generation tokens, eval tok/s,
tokens-to-correct, plus training-log series (train/val loss, LR, tok/s, it/s, trained
tokens, peak mem).

**Resumability:** runs under `caffeinate` (no idle sleep; lid-close suspends, resumes on
wake). Kill/shutdown/crash → re-run the **same command**: finished stages skip via
`results/<name>/.<stage>.done` markers, and LoRA training resumes from the last saved
checkpoint (mlx saves every 100 steps to `adapters/<name>-probe/`).

**DPO stage:** `./run_dpo.sh <name>` (needs `mlx-lm-lora`; mlx-lm has no native DPO).
Fuses the SFT adapter, runs `--train-mode dpo`, fuses onto Q8, evals with both
`eval_probe` (behavior must hold) and `idiom_eval` (similarity should drop).

Rough time on M-series, first run: **~3–6 hr** (dominated by download + train + 30B
generation); subsequent runs skip download/quantize → **~2–4 hr**.

---

## Repository layout

| Path | What |
|---|---|
| `sft_dpo/jacgen/*.jac` | the all-Jac pipeline: generate, validate, dedup, decontaminate, split, eval harness, dashboards (24 modules) |
| `sft_dpo/jacgen/graph_data/` | authored graph/tree tasks (`train.json` 31, `holdout.json` 13) + the Python generators |
| `dataset/` *(gitignored)* | generated data — see [the dataset table](#the-dataset-on-disk) |
| `configs/lora.yaml` | LoRA SFT config (mlx-lm) |
| `run_probe.sh` / `run_dpo.sh` | SFT probe runner / DPO runner (resumable) |
| `setup_env.sh` / `check.sh` | venv + installs / type + behavior gate (non-destructive) |
| `results/` *(gitignored)* | per-model run outputs (`base.txt`, `finetuned.txt`, `*.png`, `metrics.jsonl`) |
| `resultsft/` | **committed copies of all results + graphs** → [`RESULTS.md`](resultsft/RESULTS.md) |
| `models/` / `adapters/` *(gitignored)* | quantized/fused models / LoRA adapters |
| `docs/` | strategy, model-testing, datagen plans → [map below](#documentation-map) |
| `sft_dpo/process.md` | operator runbook (setup → check → run, pause/resume) |
| `context.md` | durable project framing |
| `papers/` | reference papers (MultiPL-T, WizardCoder, Magicoder, SelfCodeAlign, DeepSeek-Coder, CodeDPO, Magpie) |

---

## The all-Jac pipeline (24 modules)

In `sft_dpo/jacgen/` (full reference: its [`README.md`](sft_dpo/jacgen/README.md)
and HANDOFF §5).

**Shared libraries**
- `writer.jac` — dataset I/O + **the behavioral gate**: `run_jac(code) -> (rc, out, err)`
  (subprocess `jac run` on a tempfile), `make_sft_example`, `append_jsonl`, `extract_jac`.
- `corpus.jac` — HF mining + transpile: `fetch_rows` (curl+retry), `is_clean`,
  `first_func`, `py_cases` (exec in builtins-only ns, **SIGALRM 2s** timeout, ≤3
  distinct-output cases), `transpile` (`jac py2jac`), `sanitize_transpile`.
- `decontam.jac` — 14-token shingles, `is_contaminated` (≥0.5 overlap). Keeps the
  holdout honest.
- `dedup.jac` — ROUGE-L (LCS) near-duplicate filter.

**Generators / builders** *(run order matters)*
- `mine.jac` — corpus → `source_pool/mined.jsonl`.
- `seed_conversion.jac` — ⚠️ **TRUNCATES** `sft.jsonl` → 32 seeds + 2 seed DPO pairs.
- `idiomatic_batch{,2,3}.jac` — **APPEND** +30/+23/+31 idiomatic examples.
- `scale_conversion.jac` — mine → py2jac → **jac-run gate** → 1500 → `sft_auto.jsonl` (slow).
- `dpo_conversion.jac` — chosen=idiomatic, rejected=transpile, **parse-gated** → `dpo.jsonl`.
- `build_manifest.jac` — all idiomatic + stride-sampled transpile at 1:3 → `sft_train.jsonl`.
- `build_splits.jac` / `build_dpo_splits.jac` — → `dataset/mlx/` / `dataset/mlx_dpo/`.
- `graph_seeds.jac` — reads `graph_data/train.json`, appends idiomatic graph SFT (node/edge/walker).
- `graph_holdout.jac` — reads `graph_data/holdout.json` (disjoint) → graph eval holdout.
- `holdout.jac` — mine from disjoint offsets + decontam → 150 function holdout.

**Eval + instrumentation**
- `eval_probe.jac` — **the harness.** Loads the model **once** in-process
  (`mlx_lm.load` + `stream_generate`), scores the holdout via the `jac run` gate.
- `idiom_eval.jac` — **idiom judge.** Per correct output, `rouge_l(output, py2jac(python))`:
  high = Python-shaped, low+runs = idiomatic. Buckets idiomatic vs python_shaped + counts
  graph constructs.
- `dashboard.jac` — zero-dep ASCII live view. `plot_metrics.jac` — 8 matplotlib PNGs.
- `dataset_stats.jac` — composition report. `verify_dataset.jac` — non-destructive
  sampled re-validation (check.sh's behavior gate). `decontam_report.jac` — holdout audit.

---

## Rebuilding the dataset (order matters)

`dataset/` is gitignored. To regenerate from scratch, run in **this exact order**
(`seed_conversion` truncates; the batches append, so they must follow it):

```bash
source .venv/bin/activate
jac run sft_dpo/jacgen/mine.jac              # (optional) refresh source_pool/mined.jsonl
jac run sft_dpo/jacgen/seed_conversion.jac   # sft.jsonl -> 32 (TRUNCATES), dpo seed -> 2
jac run sft_dpo/jacgen/idiomatic_batch.jac   # -> 62  (appends)
jac run sft_dpo/jacgen/idiomatic_batch2.jac  # -> 85
jac run sft_dpo/jacgen/idiomatic_batch3.jac  # -> 116
jac run sft_dpo/jacgen/graph_seeds.jac       # + graph-tier idiomatic (node/edge/walker) -> 147
jac run sft_dpo/jacgen/scale_conversion.jac  # transpile volume -> 1500 (SLOW: mines+gates)
jac run sft_dpo/jacgen/dpo_conversion.jac    # dpo.jsonl -> 147 (parse-gated, regenerates from sft.jsonl)
jac run sft_dpo/jacgen/build_manifest.jac    # sft_train.jsonl -> 588 (1:3)
jac run sft_dpo/jacgen/build_splits.jac      # dataset/mlx/{train,valid}.jsonl -> 529/59
jac run sft_dpo/jacgen/build_dpo_splits.jac  # dataset/mlx_dpo/{train,valid}.jsonl -> 132/15
jac run sft_dpo/jacgen/holdout.jac           # function eval holdout (decontaminated, 150)
jac run sft_dpo/jacgen/graph_holdout.jac     # graph eval holdout (13)
jac run sft_dpo/jacgen/dataset_stats.jac     # verify composition
```

> **If `sft.jsonl` shows 32 and `dpo.jsonl` shows 2**, the idiomatic batches got wiped
> (something ran `seed_conversion` last). Re-run from `idiomatic_batch` onward. Skip
> `scale_conversion` if `sft_auto.jsonl` already has 1500 (it's the slow network step).

---

## Environment

**Anaconda was removed on purpose — do not reinstall it.** The project runs on a venv
over Homebrew `python3.14`:

```bash
./setup_env.sh                 # python3 -m venv .venv + pip install jaclang mlx-lm matplotlib
source .venv/bin/activate      # puts jac + mlx_lm.* on PATH
```

- `jaclang` **0.16.0** (strict `Any` handling — see gotchas).
- `mlx-lm` (`mlx_lm.convert` / `lora` / `fuse` / `generate` + Python API).
- `mlx-lm-lora` **2.1.0** (DPO only; mlx-lm has no native DPO).
- `matplotlib` (PNG graphs), `caffeinate` (macOS built-in; keeps runs awake).

`check.sh` and `run_probe.sh` prepend `.venv/bin` to `PATH` so subprocess `jac` resolves
even without `source`. You need **~50–60 GB free disk per model** (download + quantize).

---

## Gotchas

Read before touching anything — these will bite. Full list: HANDOFF §9.

1. **`seed_conversion.jac` truncates the dataset** → `sft.jsonl` to 32, `dpo.jsonl` to 2.
   The idiomatic batches *append*, so any later run of `seed_conversion` silently wipes
   them. `check.sh` is now **non-destructive** (runs `verify_dataset.jac`, not seed).
2. **jaclang 0.16.0 is strict about `Any`.** Python-interop calls return `Any`, rejected
   in typed positions. Fix: cast at the boundary — `str(...)`, `list(...)`, `int(...)`,
   `Path(str(d))`; a few untypeable stdlib calls carry `# jac:ignore[E10xx]`. Don't chase
   `--disable-error-code` (flaky in this version).
3. **The type-checker CRASHES on `mlx_lm`.** So `eval_probe.jac` / `idiom_eval.jac`
   **lazy-import `mlx_lm` inside functions** (never top-level) and are **parse-checked
   only** (`jac check -p`) in check.sh, while the other modules get full `jac check`.
4. **The gate is `jac run`, never `jac check`.** Idiomatic Jac is often
   untyped-but-runnable. Every behavioral path runs the code and compares stdout.
5. **Model id needs `-Instruct`:** `Qwen/Qwen3-Coder-30B-A3B` 401s on HF;
   `Qwen/Qwen3-Coder-30B-A3B-Instruct` is correct.
6. **bash 3.2 (macOS) errors on empty-array expansion under `set -u`.** `run_probe.sh`
   avoids `"${ARR[@]}"` — the resume launch is two explicit branches. Don't "simplify."
7. **Eval loads the model ONCE** (in-process, reused across the whole holdout). An earlier
   per-task reload was unusably slow.
8. **No `timeout` on stock macOS.** Use the Python SIGALRM pattern (`corpus.py_cases`) or
   `gtimeout` from coreutils.
9. **Live per-checkpoint eval OOM-deadlocks a 30B run on 48 GB → off by default.** It loads
   a *second* full model copy; 30B + training copy exceeds 48 GB → swap thrash → both wedge.
   Gated behind `LIVE_EVAL=1` (default 0). With it off, watch **val loss** (free, same
   model); test-pass% is measured at base-vs-finetuned. Only enable for models ≲8B.

---

## Documentation map

| Doc | What |
|---|---|
| **[`sft_dpo/process.md`](sft_dpo/process.md)** | operator runbook — setup → check → run, pause/resume, launchd, timings |
| **[`sft_dpo/docs/modeltesting/HANDOFF.md`](sft_dpo/docs/modeltesting/HANDOFF.md)** | **single source of truth** — architecture, every module, every gotcha, rebuild order |
| **[`resultsft/RESULTS.md`](resultsft/RESULTS.md)** | full measured results + all 16 training graphs, both models, side by side |
| [`context.md`](context.md) | durable project framing (what Jac is, goal, constraints) |
| [`docs/initmodelchoice/strat.md`](docs/initmodelchoice/strat.md) | the 12 data-generation recipes (R1–R12) |
| [`docs/wholestack/strat.md`](docs/wholestack/strat.md) | whole-stack strategy |
| [`sft_dpo/docs/modeltesting/`](sft_dpo/docs/modeltesting/) | strategy, evaluation, conversion-probe, per-model notes (Qwen / Gemma) |
| [`sft_dpo/jacgen/README.md`](sft_dpo/jacgen/README.md) | module-by-module pipeline reference |

---

## Glossary

| Term | Meaning |
|---|---|
| **SFT** | supervised finetuning — train on input→output pairs |
| **DPO** | direct preference optimization — train on (chosen vs rejected) pairs to push toward the preferred style |
| **LoRA** | low-rank adapter finetuning — cheap, small, fusable into the base weights |
| **MLX** | Apple's array/ML framework for Apple Silicon; `mlx-lm` runs LLM train/infer locally |
| **py2jac** | `jac` subcommand that mechanically transpiles Python → Jac (Python-shaped output) |
| **transpile-similarity** | ROUGE-L of model output vs `py2jac` of the same Python: high = Python-shaped, low = rewritten/idiomatic |
| **idiom headroom** | how much an idiomatic answer can diverge from a mechanical transpile (large for graphs, ~zero for pure functions) |
| **cross-compiled test-pass** | the primary metric: converted Jac compiles, runs, and output matches the recorded behavioral cases |
| **holdout** | unseen, decontaminated eval set (150 function + 13 graph tasks) |
