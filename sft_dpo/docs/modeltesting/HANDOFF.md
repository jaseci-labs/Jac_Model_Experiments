# HANDOFF — Python→Jac conversion probe (full state of the build)

Single source of truth for picking this work up cold. Covers what was built, why,
how to run it, the exact data on disk, every Jac module, the environment, and
**every gotcha** that will bite the next session. Read this first.

---

## 1. What this project is

Build synthetic **Python→Jac conversion** training data, then run a **mini probe**:
LoRA-finetune one small open base model on that data on Apple Silicon (Mac M-series,
48 GB, MLX) and measure **base-vs-finetuned** on a held-out, decontaminated eval set.
The signal we want: *did finetuning on our Jac data make the model produce more
correct Jac?* (cross-compiled behavioral test-pass rate, base vs finetuned).

Two design rules, both load-bearing:

- **Everything is written in Jac** (the data pipeline, the eval harness, the
  dashboards) — we dogfood the language we're generating data for.
- **The gate is `jac run`, not `jac check`.** "The conversion works" means it
  *compiles and runs and the output matches* — behavioral equivalence. `jac check`
  (the strict type-checker) rejects untyped-but-runnable code, which is not what we
  mean. This realization moved a smoke-eval score from 26% → 96%. Keep it.

Candidate base models (narrowed to two — DeepSeek/Kimi dropped):
- **Qwen** — `Qwen/Qwen3-Coder-30B-A3B-Instruct`  ← note the `-Instruct` suffix (see gotchas)
- **Gemma** — `google/gemma-4-26b-a4b-it`

Both candidates have now been run end-to-end (SFT + graph + DPO) — see §14. Qwen3-Coder
wins on graph idiom; functions are a tie.

---

## 2. Current state (as of this handoff)

**Dataset is rebuilt, complete, and consistent on disk:**

| Artifact | Path | Count |
|---|---|---|
| Idiomatic SFT core (hand/agentic + 31 graph) | `dataset/conversion/sft.jsonl` | **147** |
| Transpile SFT volume (py2jac, behaviorally gated) | `dataset/conversion/sft_auto.jsonl` | **1500** |
| **Total SFT** | (both above) | **1647** |
| DPO pairs (idiomatic chosen vs Python-shaped rejected; 31 graph w/ real divergence) | `dataset/conversion/dpo.jsonl` | **147** |
| Balanced manifest (1:3 idiom:transpile) | `dataset/conversion/sft_train.jsonl` | **588** (147 + 441) |
| mlx-lm train split (messages-only) | `dataset/mlx/train.jsonl` | **529** |
| mlx-lm valid split (messages-only) | `dataset/mlx/valid.jsonl` | **59** |
| Function eval holdout (behavioral `test_cases`) | `dataset/eval_holdout/conversion.jsonl` | **150** |
| Graph eval holdout (idiom headroom, §13) | `dataset/eval_holdout/graph_conversion.jsonl` | **13** |

Idiomatic core composition: 24 seed + 84 idiomatic_batch{,2,3} + 8 mined + 31 graph = 147;
difficulty mix atomic 41 / idiomatic 37 / composed 38.

**Toolchain is green:** `./check.sh` → `19 passed` (full `jac check`) + `eval_probe`
parse-check passed + `39/39` sampled behavioral re-validation, and it is now
**non-destructive** (it no longer truncates the dataset — see gotcha #1).

**The Qwen SFT probe HAS been run (success).** `Qwen/Qwen3-Coder-30B-A3B-Instruct`
quantized, LoRA-SFT'd (600 iters), fused, evaluated. Results in `results/qwen/`:
- **base 0% → finetuned 94%** cross-compiled test-pass on the 150 holdout (stock Qwen
  produces zero runnable Jac; finetuned is behaviorally correct). Generation also
  halved (34.7k→15.7k tokens) — less rambling.
- **Learning curve** (`results/qwen/learning_curve.png`): ~96% by iter 100, then flat
  — the model learns Jac almost immediately; more iters don't help.
- **Idiom judge** (`idiom_eval.jac`): on FUNCTIONS, 141/142 Python-shaped (sim 0.968) —
  idiom ≈ 0 because functions have no idiom headroom (idiomatic ≈ transpile).
- **GRAPH tier (§13): the model learns idiom, and DPO sharpens it.** 31 graph seeds →
  graph holdout (13 tasks) **0% → SFT 46% → DPO 61% correct**, of-correct idiomatic
  **83% → 100%**, similarity **0.457 → 0.338** (function holdout held ~94%). Proves
  data-with-headroom → measurable idiomatic Jac, and DPO on real-divergence pairs lifts it.
**The Gemma run is also done** (`results/gemma/`): function 93%, graph SFT 15% / DPO 15%.
Full head-to-head in §14 — Qwen wins graph idiom decisively; functions tie. ⚠️ Gemma id is
gated + case-sensitive: use `google/gemma-4-26B-A4B-it` (capital B/A4B) and accept the
license + `huggingface-cli login` first; lowercase 307-redirects, no token = 401.

`dataset/` is gitignored — it is **regenerable** from the Jac builders (section 6).

---

## 3. Environment (no anaconda — intentional)

Anaconda was removed on purpose mid-build. **Do not reinstall it.** The project runs
on a venv over Homebrew python3.14:

```bash
./setup_env.sh                 # python3 -m venv .venv + pip install jaclang mlx-lm matplotlib
source .venv/bin/activate      # puts jac + mlx_lm.* on PATH
```

- `jaclang` **0.16.0** (strict `Any` handling — see gotcha #2).
- `mlx-lm` (`mlx_lm.convert` / `lora` / `fuse` / `generate`, and the Python API
  `mlx_lm.load` / `stream_generate`).
- `matplotlib` (PNG graphs; the ASCII dashboard needs nothing).
- `caffeinate` (macOS built-in; keeps the run awake).

`check.sh` and `run_probe.sh` both prepend `.venv/bin` to `PATH` so subprocess `jac`
resolves even without `source`.

---

## 4. Architecture — the all-Jac pipeline

```
  HF corpus (Vezora/Tested-22k-Python-Alpaca, via datasets-server rows API)
        │  corpus.jac: fetch → is_clean filter → first_func → py_cases (exec, SIGALRM 2s)
        ▼
  mine.jac ──► dataset/source_pool/mined.jsonl        (cleaned Python + behavioral cases)
        │
        ├──► scale_conversion.jac ──► sft_auto.jsonl   (1500: py2jac transpile, jac-run gated)
        │
  seed + idiomatic_batch{,2,3} + graph_seeds ──► sft.jsonl  (140 idiomatic, incl 24 graph)
        │
        ├──► dpo_conversion.jac ──► dpo.jsonl          (60: idiomatic chosen vs transpile rejected)
        │
        ├──► build_manifest.jac ──► sft_train.jsonl    (560: 1:3 idiom:transpile, de-skew)
        │         └──► build_splits.jac ──► dataset/mlx/{train,valid}.jsonl (504/56, messages-only)
        │
        └──► holdout.jac ──► dataset/eval_holdout/conversion.jsonl (150, disjoint offsets + decontam)

  RUN: run_probe.sh ──► quantize Q4+Q8 ──► base eval ──► dry-run ──► LoRA train
        │                                    (eval_probe.jac: load model ONCE, jac-run gate)
        ├──► live: dashboard.jac (ASCII) + plot_metrics.jac (PNG) per checkpoint
        └──► fuse ──► finetuned eval ──► graphs
```

**Two data tiers, on purpose:**
- *Idiomatic core* (`sft.jsonl`) — hand-written / agentically-written Jac that uses
  walkers, nodes, edges, abilities, `with entry`, typed edges — *Jac-shaped*.
- *Transpile volume* (`sft_auto.jsonl`) — `jac py2jac` of mined Python, behaviorally
  gated — correct but *Python-shaped*. Cheap volume.
- *DPO* (`dpo.jsonl`) — teaches **de-Python-ification**: `chosen` = idiomatic,
  `rejected` = the transpiled Python-shaped version of the same function.

The manifest mixes idiom:transpile at **1:3** so the model sees enough idiom not to
just learn "transpile-ese," without starving on the cheap volume.

---

## 5. Every Jac module (24) — `srccurrent/jacgen/`

**Shared libraries**
- `writer.jac` — dataset I/O + the behavioral gate. `make_sft_example(...)`,
  `append_jsonl`, `run_jac(jac_code, timeout) -> (rc, stdout, stderr)` (subprocess
  `jac run` on a tempfile — *this is the gate*), `extract_jac`, `revalidate_example`.
- `corpus.jac` — HF mining + transpile lib. `fetch_page/fetch_rows` (curl + retry),
  `is_clean` (BANNED/ALLOWED-imports filter), `first_func`, `py_cases` (exec in a
  builtins-only namespace with a **SIGALRM 2s timeout**, collects up to 3
  distinct-output cases), `transpile` (`jac py2jac`), `sanitize_transpile`, `normalize`.
- `decontam.jac` — 14-token shingles, `build_training_shingles`, `is_contaminated`
  (≥0.5 overlap), `extract_python`. Keeps the holdout honest.
- `dedup.jac` — ROUGE-L (LCS) near-duplicate filter.

**Generators / builders** (run order matters — see section 6)
- `mine.jac` — corpus → `source_pool/mined.jsonl`.
- `seed_conversion.jac` — **TRUNCATES** `sft.jsonl` and writes 32 hand-crafted
  idiomatic seeds + 2 seed DPO pairs. ⚠️ destructive — see gotcha #1.
- `idiomatic_batch.jac` / `idiomatic_batch2.jac` / `idiomatic_batch3.jac` — **APPEND**
  +30 / +23 / +31 idiomatic examples (`source: generated_idiomatic`).
- `scale_conversion.jac` — mine Vezora → py2jac → **jac-run gate** → 1500 → `sft_auto.jsonl`.
- `dpo_conversion.jac` — read `sft.jsonl`, chosen = idiomatic, rejected =
  `sanitize_transpile(transpile(py))`, compile-gated → `dpo.jsonl` (60 this build).
- `build_manifest.jac` — all idiomatic + stride-sampled transpile at 1:3 → `sft_train.jsonl`.
- `build_splits.jac` — `sft_train.jsonl` → `dataset/mlx/{train,valid}.jsonl`
  (messages-only; every 10th → valid).
- `build_dpo_splits.jac` — `dpo.jsonl` → `dataset/mlx_dpo/{train,valid}.jsonl`
  (`{prompt,chosen,rejected}`, the schema mlx-lm-lora's DPODataset expects; 54/6).
- `graph_seeds.jac` — **graph tier with real idiom headroom** (see §13). Reads
  `graph_data/train.json` (24 validated tasks), appends idiomatic graph SFT examples
  (node/edge/walker, single-dict-arg `def`) to `sft.jsonl`, re-validating each by running.
- `graph_holdout.jac` — reads `graph_data/holdout.json` (10 DISJOINT tasks) → writes
  `dataset/eval_holdout/graph_conversion.jsonl` (same schema as the function holdout).
- `holdout.jac` — mine from offset 12000+ (disjoint from training's 0–8200) +
  `is_contaminated` 14-gram → `eval_holdout/conversion.jsonl` (150).

**Eval + instrumentation**
- `idiom_eval.jac` — **idiom-quality judge.** Behavioral pass% (eval_probe) can't tell
  idiomatic Jac from Python-shaped Jac that merely runs. This judge, per correct
  holdout output, computes `rouge_l(model_output, py2jac(python))`: HIGH similarity =
  the model reproduced the mechanical transpile (Python-shaped); LOW + still runs =
  it rewrote into cleaner/idiomatic Jac. Buckets `idiomatic` vs `python_shaped` (by
  `JAC_IDIOM_SIM`, default 0.7) + counts graph-spatial constructs. No extra model.
  Same env as eval_probe; lazy-imports `mlx_lm` (parse-checked only, like eval_probe).
- `eval_probe.jac` — **the harness.** Loads the model **once** in-process
  (`mlx_lm.load` + `stream_generate`) and scores the holdout: per task, generate Jac
  → extract → append `with entry { print(func(input)); }` per case → `jac run` →
  compare stdout. Reports run% (compiles+executes), cross-compiled test-pass%,
  generation tokens, eval tok/s, tokens-to-correct. Modes via `JAC_EVAL_MODE`:
  `py2jac` (no-LLM smoke that proves the loop) | `mlx` (real model). Env:
  `JAC_EVAL_MODEL`, `JAC_EVAL_ADAPTER`, `JAC_EVAL_LIMIT` (subset during training),
  `JAC_EVAL_METRICS_OUT`, `JAC_EVAL_STEP`. ⚠️ lazy-imports `mlx_lm` inside functions —
  gotcha #3.
- `dashboard.jac` — zero-dep ASCII live view (train/val loss, LR, tok/s, it/s,
  trained tokens, peak mem, holdout learning curve). Reads `JAC_TRAIN_LOG` +
  `JAC_METRICS`.
- `plot_metrics.jac` — matplotlib; writes 8 PNGs (same series + learning curve).
- `dataset_stats.jac` — composition report (counts by difficulty/generator/source).
- `verify_dataset.jac` — **non-destructive** sampled re-validation; re-runs every Nth
  stored example and checks output still matches. `JAC_SAMPLE_EVERY` (default 1; check.sh
  uses 40). This is what check.sh's behavior gate now runs.
- `decontam_report.jac` — audits holdout-vs-train contamination.

---

## 6. Rebuilding the dataset (order is load-bearing)

`dataset/` is gitignored. To regenerate the **full** dataset from scratch, run in
**this exact order** (because `seed_conversion` truncates and the batches append):

```bash
source .venv/bin/activate
jac run srccurrent/jacgen/mine.jac              # (optional) refresh source_pool/mined.jsonl
jac run srccurrent/jacgen/seed_conversion.jac   # sft.jsonl -> 32 (TRUNCATES), dpo seed -> 2
jac run srccurrent/jacgen/idiomatic_batch.jac   # sft.jsonl -> 62  (appends)
jac run srccurrent/jacgen/idiomatic_batch2.jac  # sft.jsonl -> 85
jac run srccurrent/jacgen/idiomatic_batch3.jac  # sft.jsonl -> 116
jac run srccurrent/jacgen/scale_conversion.jac  # sft_auto.jsonl -> 1500  (SLOW: mines+transpiles+gates)
jac run srccurrent/jacgen/dpo_conversion.jac    # dpo.jsonl -> ~60 (regenerates from sft.jsonl)
jac run srccurrent/jacgen/build_manifest.jac    # sft_train.jsonl -> 560 (1:3)
jac run srccurrent/jacgen/build_splits.jac      # dataset/mlx/{train,valid}.jsonl -> 504/56
jac run srccurrent/jacgen/holdout.jac           # eval_holdout/conversion.jsonl -> 150
jac run srccurrent/jacgen/dataset_stats.jac     # verify composition
```

If `sft_auto.jsonl` (1500) is already present, skip `scale_conversion` (it's the slow
one — network mining + transpile + per-example `jac run` gate). Everything else is fast.

> **If `sft.jsonl` ever shows 32 and `dpo.jsonl` shows 2**, the idiomatic batches were
> wiped (something ran `seed_conversion` last). Re-run from `idiomatic_batch` onward.

---

## 7. Running the probe

```bash
./run_probe.sh Qwen/Qwen3-Coder-30B-A3B-Instruct qwen      # note -Instruct
# or: ./run_probe.sh google/gemma-4-26b-a4b-it gemma
```

Stages (each skippable/resumable): quantize Q4 (train) + Q8 (eval) → **base eval** on
150 holdout → 30-iter **dry-run** bail check → **LoRA train** (`configs/lora.yaml`)
with a **live ASCII dashboard + PNG graphs** refreshed per checkpoint (per-checkpoint
holdout eval = the learning curve) → **fuse** adapter into Q8 → **finetuned eval** →
graphs. Outputs in `results/`:
- `*-base.txt` / `*-finetuned.txt` — the headline base-vs-finetuned comparison.
- `*-metrics.jsonl` — per-checkpoint learning curve + token metrics.
- `*.png` — `learning_curve`, `train_loss`, `val_loss`, `learning_rate`,
  `tokens_per_sec`, `iters_per_sec`, `trained_tokens`, `peak_mem` (open in Preview;
  auto-refresh live).

**Tunables (env):** `EVAL_EVERY` (dashboard secs), `SUBSET` (tasks/checkpoint eval),
`DRY_ITERS`, `SKIP_DRY=1`. The per-checkpoint eval on a 30B model is the hidden cost
(shares the GPU with training) — `SUBSET=20 EVAL_EVERY=300` to cut it, or
`EVAL_EVERY=99999` to skip live evals and just read base-vs-finetuned at the end.

**Resumability:** runs under `caffeinate` (no idle sleep; lid-close suspends and
continues on wake). Kill/shutdown/crash → re-run the **same command**: finished stages
skip via `results/.<name>.*.done` markers, and LoRA training **resumes from the last
saved checkpoint** (mlx saves every 100 steps to `adapters/<name>-probe/`), training
only the remaining iters; the learning-curve metrics append rather than reset.
`process.md` has an optional launchd plist for auto-resume after a full shutdown.

**Rough time (M-series, first run):** ~3–6 hr, dominated by download + train + 30B
generation; subsequent runs skip download/quantize → ~2–4 hr.

---

## 7b. Results layout + learning curve (per model)

`run_probe.sh` namespaces everything under **`results/<name>/`** (e.g. `results/qwen/`,
`results/gemma/`) so two models never clash: `base.txt`, `finetuned.txt`, `train.log`,
`metrics.jsonl`, all `*.png`, and `.<stage>.done` markers live there. `plot_metrics.jac`
honors `JAC_PLOT_DIR`.

The **holdout learning curve** is built by a **post-training checkpoint sweep** (stage 6
in run_probe.sh), NOT a concurrent live eval (that OOMs — gotcha #10). After training,
it evaluates each saved adapter checkpoint (`adapters/<name>-probe/NNNN_adapters.safetensors`
+ the final `adapters.safetensors`) on `SUBSET` holdout tasks **sequentially** — one
model in RAM at a time — writing a `metrics.jsonl` row per checkpoint, then
`plot_metrics.jac` renders `results/<name>/learning_curve.png`. Safe on 48 GB; the
x-axis is iters (100, 200, …, final), y is holdout test-pass%.

## 8. Metrics measured

The eval (`eval_probe.jac`) and dashboards report, per checkpoint and at base/finetuned:
- **runs%** — compiles + executes (a model that emits no callable scores 0, fair).
- **cross-compiled test-pass%** — *the primary signal*; output matches the recorded
  behavioral cases.
- **generation tokens** (total), **eval tokens/sec** (avg `generation_tps`),
  **tokens-to-correct** (avg gen tokens per correct conversion).
- Training-log series: train/val loss, learning rate, tokens/sec, iters/sec,
  cumulative trained tokens, peak memory.

---

## 9. Gotchas — read before touching anything

1. **`seed_conversion.jac` truncates the dataset.** It rewrites `sft.jsonl` to 32 and
   `dpo.jsonl` to 2. The idiomatic batches *append*, so any run of `seed_conversion`
   after them silently wipes 84 idiomatic examples. **check.sh used to run
   seed_conversion as its behavior gate and degraded the dataset every invocation** —
   now fixed: check.sh runs `verify_dataset.jac` (non-destructive sampled re-run). If
   you script the pipeline, respect the order in section 6.
2. **jaclang 0.16.0 is strict about `Any`.** Python-interop calls (json/subprocess/
   regex/matplotlib) return `Any`, which the checker rejects in typed positions. The
   fix used throughout: wrap at the boundary — `str(...)`, `list(...)`, `dict(...)`,
   `int(...)`, `float(...)`, `Path(str(d))`; a few genuinely-untypeable stdlib calls
   (`re.findall`, `inspect.signature`, `signal`, matplotlib stubs) carry
   `# jac:ignore[E10xx]`. Do **not** chase the `--disable-error-code` flag (flaky in
   this version) — use casts.
3. **The jac type-checker CRASHES on `mlx_lm`** (`'NoneType' object has no attribute
   'is_instantiable_class'` — an internal jaclang bug resolving mlx's model types).
   So `eval_probe.jac` (a) **lazy-imports `mlx_lm` *inside* functions**, never at top
   level, and (b) is **parse-checked only** (`jac check -p`) in check.sh while the
   other 19 get full `jac check`. Keep both or check.sh will crash.
4. **`jac run` is the gate, never `jac check`.** Idiomatic Jac is often untyped-but-
   runnable; `jac check` would reject correct conversions. Every behavioral validation
   path runs the code and compares stdout.
5. **Model id needs the `-Instruct` suffix:** `Qwen/Qwen3-Coder-30B-A3B` 401s on HF;
   `Qwen/Qwen3-Coder-30B-A3B-Instruct` is correct (verified 200).
6. **bash 3.2 (macOS) errors on empty-array expansion under `set -u`.** `run_probe.sh`
   avoids `"${ARR[@]}"` entirely — the resume launch is **two explicit branches** (with
   vs without `--resume-adapter-file`). Don't "simplify" it back into an array.
7. **`run_probe.sh` captures the trainer PID, not `tee`.** Training is redirected
   (`> "$TRAIN_LOG" 2>&1 &`) not piped, so `$!` is the trainer; "done" requires a real
   `$ADAPTER/adapters.safetensors` on disk (self-heals a stale `.done` marker from an
   interrupt). A previous bug ("training already complete 0/600" + fuse
   FileNotFoundError) came from `$!` capturing `tee`.
8. **Eval loads the model ONCE.** An earlier version reloaded the 30B model per task
   (150×/checkpoint) — unusably slow. `eval_probe.jac` now loads in-process once and
   reuses it across the whole holdout via `stream_generate`.
9. **No `timeout` on stock macOS.** Don't rely on it in scripts (use the Python
   SIGALRM pattern in `corpus.py_cases`, or `gtimeout` from coreutils if installed).
10. **Live per-checkpoint eval OOM-deadlocks a 30B run on 48 GB — now off by default.**
    The training loop's optional holdout eval loads a *second* full model copy
    in-process (`mlx_lm.load`); training already holds one (~27 GB peak), so a 30B
    model + the eval copy exceeds 48 GB → swap thrash → both wedge at ~0% CPU. It bit
    the first real run (froze at iter 150). `run_probe.sh` now gates that eval behind
    **`LIVE_EVAL=1` (default 0)**. With it off, the live learning signal is **val loss
    from the train log** (free, same model, plotted by dashboard/plot_metrics); the
    holdout test-pass% is still measured at **base vs finetuned**. Only set
    `LIVE_EVAL=1` for small models (≲8B) that fit in memory twice. Symptom to
    recognize: dashboard stuck on the first "no data yet" frame + `vm.swapusage` near
    100% used + the trainer and a `jac run` both at ~0% CPU.

---

## 10. Files outside `srccurrent/jacgen/`

- `run_probe.sh` — the resumable SFT runner (section 7); per-model `results/<name>/`.
- `run_dpo.sh` — the DPO runner (section 12); needs `mlx-lm-lora`.
- `check.sh` — type + behavior gate (non-destructive; full-checks 20, parse-checks the
  2 mlx-importing modules `eval_probe`/`idiom_eval`).
- `setup_env.sh` — venv + pip installs.
- `configs/lora.yaml` — LoRA SFT config: fine_tune_type lora, num_layers 16,
  rank 16 / scale 2.0 / dropout 0.05, batch_size 2, iters 600, lr 2.0e-5,
  save_every 100, steps_per_eval 50, steps_per_report 10. `--model` overrides the
  placeholder `model:` path. ⚠️ keys vary by mlx-lm version — verify against
  `mlx_lm.lora --help` before the real run.
- `process.md` — operator runbook (setup → check → run, pause/resume, launchd, time).
- `README.md` — front door → process.md.
- `context.md` — trimmed project context.
- `docs/modeltesting/mini_probe.md` — readiness doc (prebuilt vs needs-install).
- `docs/modeltesting/{strategy,evaluation,conversion_probe,workflow,gemma4-26b,qwen3-coder-30b}.md`
  — strategy + per-model notes.
- `srccurrent/jacgen/README.md` — module reference (mirrors section 5).

---

## 11. Next session — suggested first moves

1. `source .venv/bin/activate && ./check.sh` → expect `19 passed` + parse pass +
   `39/39` sampled. Confirms toolchain.
2. `jac run srccurrent/jacgen/dataset_stats.jac` → expect SFT 1640 (140 incl 24 graph + 1500),
   DPO 60. If it shows 32/2, rebuild from section 6.
3. `./run_probe.sh Qwen/Qwen3-Coder-30B-A3B-Instruct qwen` → first real probe. Watch
   `results/learning_curve.png` (rising = learning Jac; flat-while-loss-drops =
   memorizing format, not idiom).
4. After SFT signal: run the idiom judge (§5 `idiom_eval.jac`) to split the 94% into
   idiomatic vs Python-shaped — that tells you how much DPO is needed.
5. DPO stage (see §12).

## 12. DPO stage — no-op on FUNCTIONS (no headroom), WORKS on GRAPH (real divergence)

**Ran `./run_dpo.sh qwen` (2026-06-05).** DPO trained *perfectly* — `acc 1.000,
margin 7.4, chosen_r +3.3 / rejected_r −4.7` (it learned to strongly prefer the
idiomatic chosen over the transpile rejected). **But generation didn't change:**

| | base | SFT | DPO |
|---|---|---|---|
| behavior test-pass | 0% | 94% | 94% (held) |
| avg transpile-similarity | — | 0.968 | 0.968 (identical) |

**Why (the real finding):** DPO maximized the chosen-vs-rejected *scoring* margin, but
the argmax generation is unchanged because **the holdout tasks have no idiom headroom**.
For `factorial`/`fib`/`normalize_vector`, idiomatic Jac ≈ the mechanical transpile —
there's no meaningfully different idiomatic answer to push toward. So 0.968 "Python-shaped"
is **not a model defect**; it's the nature of standalone-function conversion (Python-shaped
Jac *is* idiomatic for a pure function). The idiom axis only exists for **graph-shaped**
problems (walker/node/edge as the right tool, diverging hard from a naive transpile), which
the current holdout lacks. Cranking `BETA`/`LR` won't help — the ceiling is the task.

**→ Real next lever (not DPO tuning):** build a **graph-shaped eval holdout + idiomatic
training data** (tree/graph/state-machine problems where idiomatic Jac means walkers/nodes).
Then idiom has room to move and DPO / idiom-weighted SFT show real separation. The DPO
machinery is proven working and reusable once such data exists.

Artifacts: `models/qwen-jac-dpo-fused-q8` (DPO model ≈ SFT, kept), `adapters/qwen-dpo/`
(adapter; the auto-fused 57 GB duplicate was deleted to reclaim disk). Results in
`results/qwen/dpo/`.

### How it was set up (reusable)

The SFT probe worked behaviorally (0%→94%) but the **idiom judge proved the gap**:
of the 142 correct outputs, **141 are Python-shaped** (avg transpile-similarity
**0.968**, 0 graph constructs) — the SFT model essentially learned to *transpile*.
DPO on the R3 de-Python-ification pairs is the lever to move idiom.

Fully scaffolded:
- **`mlx-lm-lora` 2.1.0** installed in `.venv` (mlx-lm has NO native DPO). Verified it
  does **not** churn `mlx`/`mlx-lm` (its floors `>=0.30.6` are already met).
- Data: `jac run srccurrent/jacgen/build_dpo_splits.jac` → `dataset/mlx_dpo/{train,valid}.jsonl`
  (54/6, `{prompt,chosen,rejected}`). Confirmed this matches mlx-lm-lora's **`DPODataset`**
  (datasets.py:177 — wraps prompt as user turn, chosen/rejected as assistant turns via
  chat template). ⚠️ note: the package's `PreferenceDataset` (datasets.py:54) has a bug
  (`tokenizer.encode(rejected_key)` encodes the literal key string), but DPO mode uses
  `DPODataset`, not that class — so it's fine.
- Runner: **`run_dpo.sh`** — fuses the SFT adapter into a Q4 base (`models/<name>-jac-fused-q4`),
  runs `mlx_lm_lora.train --train-mode dpo` (LoRA, **reference left unset = base frozen**,
  so only ONE 30B weight set in RAM → fits 48 GB), fuses onto the Q8 SFT model, then evals
  the DPO model with **both** `eval_probe.jac` (behavior must hold) and `idiom_eval.jac`
  (avg_sim must drop). Outputs namespaced under `results/<name>/dpo/`.
  Env: `DPO_ITERS`(200) `DPO_LR`(1e-6) `DPO_BETA`(0.1) `SUBSET`(50).
- **Win condition:** behavior stays ~94% AND avg transpile-similarity drops below 0.968
  (model rewrites toward idiom instead of reproducing the transpile).

Caveat: the holdout is standalone-function conversions, where the idiom ceiling is modest
(pure functions legitimately need only `def`+`with entry`, not walkers/nodes). To exercise
+ measure idiom fully, a future holdout should include graph-shaped problems → **done, §13**.

## 13. Graph tier — idiom headroom that exists, AND a retrain that learned it ✅

The function tasks have ~no idiom headroom (idiomatic ≈ transpile, sim 0.97). The fix is
**graph-shaped tasks**: the Python is a dict+stack traversal, but idiomatic Jac builds
**nodes/edges and spawns a walker** — while keeping a single-dict-arg `def` signature so
the behavioral eval harness drives them unchanged. Measured idiomatic-vs-transpile
similarity **~0.26** (vs 0.97) with **8 graph constructs** each → a large, real idiom axis.

Built + validated (all via `jac run`, the gate):
- `srccurrent/jacgen/graph_data/{train.json (24), holdout.json (10)}` — authored, committed
  task sets; every task passes on every test case. Holdout aggregations are DISJOINT from
  training (min/odd/sum-above/branches/range/negative vs count/sum/max/above/path/leaves/
  product/even) → generalization, not memorization.
- `graph_seeds.jac` → appends 24 idiomatic graph SFT examples to `sft.jsonl` (`source:
  generated_graph`). `graph_holdout.jac` → `dataset/eval_holdout/graph_conversion.jsonl` (10).
- `eval_probe.jac` / `idiom_eval.jac` take **`JAC_HOLDOUT`** to target the graph holdout:
  `JAC_HOLDOUT=dataset/eval_holdout/graph_conversion.jsonl JAC_EVAL_MODE=mlx ... jac run ...`.

**RESULT — the full SFT→DPO progression on the graph holdout (2026-06-05):**

Graph data scaled to 31 train / 13 holdout tasks across 3 idioms (accumulator-walker over
adj, typed/weighted edges, linked-list chains). Holdout = 13 tasks.

| metric (graph holdout, 13 tasks) | base | SFT (31 graph) | **DPO (31 graph pairs)** |
|---|---|---|---|
| correct (runs+matches) | **0%** | 46% | **61%** |
| of correct → idiomatic | — | 83% | **100%** |
| avg transpile-similarity | — | 0.457 | **0.338** (→ 0.26 ref) |
| graph constructs / output | 0.0 | 4.5 | **6.75** |

The model went from *cannot do graph conversion at all* → SFT produces it (46%, mostly
idiomatic) → DPO **lifts correctness to 61% AND makes 100% of correct outputs idiomatic**,
with similarity dropping toward the 0.26 reference. Function holdout held ~94% throughout.
Full thesis proven: (1) data with real idiom headroom → the model learns idiomatic Jac
(unlike the function tier where idiomatic ≈ transpile, so SFT can't and DPO is a no-op);
(2) DPO on real-divergence graph pairs measurably pushes "runs" → "idiomatic".
Results: `results/qwen/graph-idiom-retrain2.txt` (SFT), `results/qwen/dpo/graph-idiom.txt` (DPO).

**Next levers (both proven-ready):** (1) scale graph tasks with more STRUCTURAL variety
(binary trees, linked-list chains, weighted edges), not just predicates, via
`gen_graph_tasks.py`; (2) graph DPO — the 24 graph pairs now exist (DPO 140, parse-gated,
real 0.26 divergence), so `run_dpo.sh` would push idiom further on top of this SFT.

✅ **Resolved — graph DPO pairs:** `dpo_conversion.jac` now gates on `jac check -p`
(parse), not strict `jac check`. Strict check rejected the graph transpile (untyped
dict/set Any) even though it parses+runs, which had dropped all graph DPO pairs (and 56
function pairs). With the parse gate: **DPO 60 → 140 pairs, 24 graph-sourced**, real 0.26
divergence. `dataset/mlx_dpo/` rebuilt 126/14. Consistent with "gate = runnable, not
type-checker-approved Jac".

---

## 14. Model comparison — Qwen vs Gemma (both run, same data/config)

Both finetuned on the identical graph-inclusive dataset (529/56 split) + same eval.

| metric | Qwen3-Coder-30B-A3B | Gemma-4-26B-A4B |
|---|---|---|
| function holdout — base | 0% | 0% |
| function holdout — finetuned | **94%** | **93%** |
| graph holdout — SFT correct | **46%** | 15% |
| graph holdout — DPO correct | **61%** | 15% (no change) |
| graph — of-correct idiomatic | 83% → **100%** (DPO) | 100% (but only 2/13) |
| graph — transpile-similarity | 0.457 → **0.338** (DPO) | 0.667 (flat) |

**Findings:**
- **Function conversion: a tie** — both go 0% → ~94%. Either model learns plain Python→Jac
  equally well from our data.
- **Graph idiom: Qwen wins decisively** — 46%/61% vs Gemma's 15%. Qwen3-Coder (code-
  specialized) learns the walker/node/edge structure far better than Gemma.
- **DPO only pays off where SFT already produces idiom**: Qwen had enough correct graph
  outputs (46%) for DPO to push (→61%, sim 0.457→0.338, 100% idiomatic). Gemma's graph SFT
  was too weak (15%, ~2 correct) → DPO had nothing to move (15%, flat). Confirms the §12
  pattern: DPO sharpens existing idiom, it can't manufacture it.
- **Pick for Jac graph idiom: Qwen3-Coder-30B-A3B-Instruct.**

Graphs: `results/qwen/*.png` and `results/gemma/*.png` (learning_curve, train_loss,
val_loss, learning_rate, tokens_per_sec, iters_per_sec, trained_tokens, peak_mem).
