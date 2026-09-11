# Playbook: running a new SFT experiment

This is the template for every new experiment. It follows what
[`08-nitin-new2-ds`](experiments/08-nitin-new2-ds/) actually did: **scaffold, Spectrum
SFT, eval, report**, one directory per step inside the experiment itself:

```
experiments/NN-name/
  scaffold/   build + split the dataset                    -> dataset/
  train/      Spectrum SFT (+ optional stock / DPO arms)  -> adapter/, results/
  eval/       score on the holdouts                        -> results/
              (+ adapter/, results/, report.md, docs/ once you've written them)
```

08 is the template: a new experiment is `mkdir experiments/NN-name && cp -r
experiments/08-nitin-new2-ds/{scaffold,train,eval} experiments/NN-name/` —
see [Step 0.1](#01-create-the-experiment-directory). 04, 06 and 07 have no
scripts of their own (only `adapter/`, `results/`, `report.md`); they predate
this recipe.

Every script, flag and env var below was read out of the scripts before being
written down, and the inline Python snippets were run against 08's data. Where
08 hit a problem, the fix is written into the step.

Background on why the recipe looks like this is in [docs/HISTORY.md](docs/HISTORY.md).
The short version: plain LoRA SFT on 16 Spectrum-selected blocks is the only
thing that has reliably moved this model. CPT, GRPO and DPO have never beaten it.

Contents:

- [Prerequisites](#prerequisites)
- [Step 0: scaffold and build the dataset](#step-0-scaffold-and-build-the-dataset)
- [Step 1: Spectrum SFT](#step-1-spectrum-sft)
- [Step 2: eval](#step-2-eval)
- [Step 3: report](#step-3-report)
- [Optional: DPO pass](#optional-dpo-pass)
- [Lessons and gotchas](#lessons-and-gotchas)

---

## Prerequisites

### Where things run from

Every script `cd`s to the **repo root** itself before doing anything — the
directory that contains `experiments/`, `models/` and `.venv/` — and derives
which experiment it belongs to from its own location. There is **no `EXP` env
var**: a script under `experiments/NN-name/{scaffold,train,eval}/` always
reads and writes that same experiment's own `dataset/`, `adapter/` and
`results/`. So:

- Clone the repo under any name, anywhere.
- Run every command in this playbook **by path, from anywhere** — e.g.
  `experiments/08-nitin-new2-ds/scaffold/prep_training_dirs.sh` or
  `jac run experiments/08-nitin-new2-ds/eval/eval_functional.jac`.
- To work on a different experiment, run its own copy of the scripts instead
  of 08's — e.g. `experiments/09-my-dataset/train/run_sft_spectrum.sh`. See
  [Step 0.1](#01-create-the-experiment-directory) for how a new experiment
  gets its own copies.
- Call `jac` as `.venv/bin/jac` (or `source .venv/bin/activate` first). The
  wrappers prepend `.venv/bin` to `PATH` themselves, and the eval harness shells
  out to whatever `jac` is on `PATH`, so a stale global `jac` gives wrong scores.

### Hardware

- Apple Silicon with 48 GB unified memory (the work here was done on an M5 Pro).
  Spectrum SFT peaked at 28.2 GB; batched eval peaks around 22.4 GB. Only one
  model can be resident at a time.
- Disk: the base model is 16 GB. Each adapter snapshot is 1.1 GB, and an SFT run
  keeps about 20 of them (mlx writes one per 820 steps, and the watchdog copies
  each into `checkpoints/`). Budget about 25 GB per SFT run.
- Keep the laptop lid open or use an external display. `caffeinate -dimsu` (the
  run scripts re-exec themselves under it) does not stop clamshell sleep, and a
  lid-close killed 08's stock run mid-training.

### Environment

```bash
./setup_env.sh      # creates .venv at the repo root, installs, sanity-checks the pipeline
source .venv/bin/activate
```

`setup_env.sh` pins `jaclang==0.16.1 mlx==0.31.2 mlx-lm==0.31.3 mlx-lm-lora==2.1.0`,
the versions every reported number was produced with. Don't bump them casually.
The Spectrum driver refuses to run if `mlx_lm`'s `linear_to_lora_layers` source
hash differs from the one it was written against.

(Python 3.14.5, numpy 2.4.6, matplotlib 3.10.9 in the known-good venv.)

### Base model

`models/qwen-q4` is Qwen3-Coder-30B-A3B-Instruct converted to
MLX at 4 bits, group size 64. It is gitignored and exists as a single copy.
Every experiment since 04 has used this exact checkpoint, so every holdout-(a)
number is comparable. Do not replace it. If it is ever lost, rebuild it with
`.venv/bin/python -m mlx_lm convert --hf-path Qwen/Qwen3-Coder-30B-A3B-Instruct -q --q-bits 4 --q-group-size 64 --mlx-path models/qwen-q4`,
then re-score the base and at least one existing adapter before trusting any
comparison, because a new quantization is a new base.

### Preflight (before every training or eval launch)

```bash
pgrep -fl "jac start|mlx_lm"     # must print nothing
```

A second resident model (a JMS server, a leftover eval) will OOM the machine.
The training scripts run this check themselves and refuse to start. The eval
scripts do not, so run it yourself.

The check matches on command lines, so anything whose argv contains `mlx_lm`
blocks a launch. That includes a monitoring loop started as
`bash -c "<script that mentions mlx_lm>"`. Put watchdog scripts in a file and
run `bash <file>`.

---

## Step 0: scaffold and build the dataset

### 0.1 Create the experiment directory

Pick a name `NN-short-name` (next number, e.g. `09-...`). 08 is the template:
copy its `scaffold/`, `train/` and `eval/` into the new experiment directory —
the copies target the new experiment automatically, no `EXP`, no edits needed
just to point them at the right place.

```bash
mkdir experiments/09-my-dataset
cp -r experiments/08-nitin-new2-ds/{scaffold,train,eval} experiments/09-my-dataset/
mkdir experiments/09-my-dataset/dataset experiments/09-my-dataset/docs
cp experiments/08-nitin-new2-ds/dataset/holdout_a_shared855.jsonl experiments/09-my-dataset/dataset/
```

The rest of this playbook says **"your experiment dir"** for
`experiments/09-my-dataset/` — swap in whatever you actually named it.

What the copied `scaffold/`, `train/` and `eval/` give you, and what to edit
for a new experiment (paths below are relative to your experiment dir):

| Path | Edit needed |
|---|---|
| `eval/eval_functional.jac` | none. This is the one scoring harness. Never change how it grades, or no number is comparable any more. |
| `scaffold/nan_guard.jac`, `scaffold/prep_training_dirs.sh` | none |
| `eval/plot_metrics.jac`, `eval/plot_progress.jac` | none |
| `eval/gen_eval_detail.jac`, `eval/grade_eval_detail.jac`, `eval/grade_reference.jac` | `gen_eval_detail.jac` reads holdout (a) (`dataset/holdout_a_shared855.jsonl`) and holdout (b) (`dataset/nitin_holdout_eval.jsonl`) from its `HOLDOUTS` dict. Point `B` at your holdout (b) file. |
| `scaffold/pipeline.jac` | 08-specific: merges 7 `jac-data-gen` sources at a pinned commit. Rewrite for your source (08's version stays in git history). Edit `COMMIT`, `CLONE`, `WORKTREE` (or set `JDG_ROOT`), `DATASET_VERSION`, `RUN_TAG`, and the per-source blocks. |
| `scaffold/copy_holdout.jac` | `HOLDOUT_FILES` names the holdout (b) files (`nitin_holdout.jsonl`, `nitin_holdout_eval.jsonl`). Rename if yours differ. |
| `scaffold/release.jac` | pool schema to trainer schema. Edit `DATASET_VERSION`, `RUN_TAG`, `TASK_TYPE`, `REGISTER` for your sources. |
| `train/` (`run_sft_spectrum.sh`, `spectrum/`, `configs/`) | nothing. |
| `train/stock/` | only if you want a stock control arm; nothing to edit |
| `docs/` | write your own `CONTEXT_BRIEF.md` (settled facts) and `README.md`/`spec.md` (question, design, decision rule) before training |

Verify holdout (a) is byte-identical to every prior experiment's:

```bash
shasum -a 256 experiments/09-my-dataset/dataset/holdout_a_shared855.jsonl
# 51ad3bb36a31725a54de0db8a168f58d76e7cf52d96a708c87e4c581929594cc
```

### 0.2 Holdouts

Two holdouts, both scored at every final checkpoint.

**(a) `dataset/holdout_a_shared855.jsonl`: the shared regression baseline.**
1,428 rows, 855 code-graded (the rest are `explanation`/`documentation` prose
rows, generated but not graded). Mixed categories: 322 conversion, 313
code_gen, 183 trajectory, 36 debug, 1 migration. It came from the 04 fresh arm's
`sft/valid.jsonl` and has been used unchanged since 04. It is the only reason
numbers from 04, 06, 07 and 08 can be put in one table. Never regenerate,
filter, or edit it.

**(b) experiment-specific, in-distribution for your new data.**
Carve it from your own source pool with a fixed seed before the training data
is finalized, and freeze it. Aim for 855 code-graded rows so its power matches (a). What 08
learned the hard way: 08 reused 07's holdout (b) verbatim for comparability,
but 07's holdout was 100% py2jac-shaped and 08's training mix was not, so 08's
holdout-(b) score (37.7%) could not be interpreted. Reusing an old (b) as a
secondary number is fine; it cannot be your only in-distribution check.

Every holdout row the harness reads needs:

- `messages`: at least the user turn (`messages[0]`); the harness sends only
  `messages[0].content`. A reference assistant turn is needed for the leak
  check and for `grade_reference.jac`.
- `category` and `gate_class`. Rows with `gate_class` in `behavioral`,
  `compile_project`, `compile_only` are graded; anything else is generated and
  counted as prose.

06 shipped a holdout (b) with no `messages` field. Its eval "finished" in 30
seconds and scored nothing. If a full eval finishes in seconds, it did not run.

### 0.3 Build the dataset (08's sequence)

Put the holdout (b) files in your experiment dir's `dataset/` first. 08's
`pipeline.jac` reads `dataset/nitin_holdout.jsonl` as its holdout gate, so it
must exist before the pipeline runs. All four scripts derive the experiment
from their own location.

```bash
.venv/bin/jac run experiments/09-my-dataset/scaffold/pipeline.jac
.venv/bin/jac run experiments/09-my-dataset/scaffold/copy_holdout.jac    # HOLDOUT_SRC_DIR=<dir> to import holdout files from elsewhere
.venv/bin/jac run experiments/09-my-dataset/scaffold/release.jac
experiments/09-my-dataset/scaffold/prep_training_dirs.sh                  # FORCE=1 to overwrite an existing split
```

What each does in 08:

1. **`pipeline.jac`**: merge, filter, dedup, decontaminate. Reads the pinned
   `jac-data-gen` checkout: a detached worktree created on demand inside a
   `.jdg-worktrees/` directory that's a sibling of the repo root, off the
   clone at `../jac-data-gen` (also a sibling; override the clone location
   with `$JDG_CLONE`), or `$JDG_ROOT` to point at an existing checkout
   directly; it raises if HEAD is not `COMMIT`. In order: eval-denylist drop (upstream
   `evals/function/v1/denylist_ids.txt`), composer quality filter
   (`test_count >= 18`), js2jac license filter, per-source dedup, holdout-(b)
   leak drop (numeric id + normalized target hash), global cross-source dedup,
   then a final assertion that nothing in the pool matches the holdout or the
   denylist. Writes `dataset/candidate_pool.jsonl`,
   `dataset/rejected/{sft,dpo}/rejected.jsonl` (every drop with a reason), and
   funnel stats to your experiment dir's `docs/funnel/stats.json` (persisted, commit them with the experiment).
2. **`copy_holdout.jac`**: re-runs the leak check against the pool **as
   written to disk**, on four surfaces: numeric id, `jac` target hash,
   `jac_rejected` hash, holdout prompt text. If anything collides it rewrites
   the pool and keeps `candidate_pool.jsonl.preleak`. Expected: 0 collisions.
3. **`release.jac`**: splits the pool by `task_kind` into
   `dataset/sft_train.jsonl` (completion + translation) and
   `dataset/dpo_train.jsonl` (`pair_dpo`). Wire format: SFT `messages` =
   `[user prompt, assistant "```jac\n<code>\n```"]` (fenced; the harness strips
   fences), DPO `prompt`/`chosen`/`rejected` as raw unfenced code. Hard-asserts
   required fields.
4. **`prep_training_dirs.sh`**: 85/15 split, seed 42, sort-then-shuffle, into
   `dataset/sft/{train,valid}.jsonl` and `dataset/dpo/{train,valid}.jsonl` (the
   directories the configs point at). `valid.jsonl` is mlx's training-time
   validation split, not a scored eval. Refuses to overwrite a non-empty split
   without `FORCE=1`. `SPLIT_ONLY=sft|dpo` does one track. Then it runs the NaN
   guard.
5. **`nan_guard.jac`** (runs inside step 4; standalone:
   `.venv/bin/jac run experiments/09-my-dataset/scaffold/nan_guard.jac`). Drops SFT
   rows whose prompt alone is at least `max_seq_length` (3072) tokens, measured
   with the base tokenizer exactly as mlx computes its prompt mask. Such rows
   have zero loss tokens after truncation and turn the whole run to NaN (see
   gotchas). Rewrites `dataset/sft/{train,valid}.jsonl` in place, no backup,
   idempotent. Env: `DATA_DIR`, `MODEL`, `MAX_SEQ_LENGTH`. It reads `messages`,
   so it does not work on DPO rows.

If your source is not `jac-data-gen`, you can replace steps 1 to 3 with
anything that writes `dataset/sft_train.jsonl` in the wire format above, as
long as it counts and records every drop and runs the leak checks below.

### 0.4 Leak checks

`pipeline.jac` and `copy_holdout.jac` check the pool against **holdout (b)
only**. No 08 script checks against holdout (a). 08 has zero exact overlap with
(a) (checked for this playbook), but a new source could collide. After
`prep_training_dirs.sh`, run this exact-match check of prompts and code
targets against holdout (a):

````bash
python3 - experiments/09-my-dataset <<'PY'
import json, re, hashlib, sys
E = sys.argv[1]
def norm(s): return " ".join(str(s).split()).lower()
def code(s):
    m = re.search(r"```(?:jac)?\s*\n(.*?)```", s, re.S)
    body = m.group(1) if m else s
    return " ".join(l.split("#", 1)[0].strip() for l in body.splitlines() if l.split("#", 1)[0].strip())
def h(s): return hashlib.sha1(code(s).encode()).hexdigest()
H = [json.loads(l) for l in open(f"{E}/dataset/holdout_a_shared855.jsonl")]
hp = {norm(r["messages"][0]["content"]) for r in H}
ht = {h(r["messages"][-1]["content"]) for r in H if r["messages"][-1]["role"] == "assistant"}
for split in ["train", "valid"]:
    p = t = 0
    for l in open(f"{E}/dataset/sft/{split}.jsonl"):
        m = json.loads(l)["messages"]
        p += norm(m[0]["content"]) in hp
        t += h(m[-1]["content"]) in ht
    print(f"sft/{split}: prompt hits {p}, target hits {t}")   # both must be 0
PY
````

Also check train and valid are disjoint and that their union equals the
release (08's gate). Exact match only catches copies. The 14-token shingle
near-duplicate check from 06/07 lived in scripts deleted with those
experiments; 08's `pipeline.jac` does exact and normalized-hash dedup only.

### 0.5 Gate before training

- Funnel closes: every source count minus every logged drop equals the pool size.
- 0 leaks against both holdouts.
- `nan_guard` reported its drops (0 is fine).
- The NN experiment's `CONTEXT_BRIEF.md` states the question, the confounds
  (pool size, content type), and the decision rule, written before any number exists.

---

## Step 1: Spectrum SFT

### 1.1 What Spectrum is

`mlx_lm` puts LoRA on the last `num_layers` decoder blocks by default (blocks
32 to 47 of 48). Spectrum picks the 16 blocks with the most learned structure
instead. For each block it takes the five dense matrices
(`self_attn.{q,k,v,o}_proj`, `mlp.gate`) and computes a signal-to-noise ratio
against the Marchenko-Pastur bulk edge, the largest eigenvalue a random matrix
of the same shape and variance would produce. Energy above the edge counts as
signal, energy below it as noise. Each module type is z-scored across layers,
the five z-scores are averaged, and the top 16 blocks are kept. Expert matrices
are recorded but not ranked. Every selected block still gets LoRA on all 8
module types.

The selection for this base, in
`train/spectrum/spectrum_layers.json` (inside your experiment dir):

```
[0, 22, 23, 27, 30, 34, 36, 37, 38, 39, 41, 42, 43, 44, 45, 47]
```

11 of the 16 overlap the trailing slice. Capacity is unchanged: rank 16, 256
LoRA tensors, 281,837,568 trainable parameters, exactly the same as stock. Only
placement moves. The file is sha256-identical from 04 through 08:
`349efc2fbeaff7b57a91aef990c9d2c1a2a948f4e6c7f5d085b1fac8a354a9bb`.

Design detail: [docs/reference/spectrum-layer-sft-probe-design.md](docs/reference/spectrum-layer-sft-probe-design.md).
The original result (+4.9 pp at SFT, p=0.023):
[experiments/04-cpt-sft/report.md](experiments/04-cpt-sft/report.md).

### 1.2 Reuse the layer list, or re-scan

**Reuse `spectrum_layers.json` unchanged** as long as the base weights are
unchanged. The selection depends only on the base model, not on your data.

Re-scan only for a new base model. The scan reads the **bf16 Hugging Face
snapshot** (57 GB, streamed one tensor at a time), not the q4 MLX copy:

```bash
S=experiments/09-my-dataset/train/spectrum
.venv/bin/python $S/snr_scan.py \
  --snapshot ~/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-30B-A3B-Instruct/snapshots/b2cff646eb4bb1d68355c01b18ae02e7cf42d120 \
  --out $S/snr/snr_raw.json
.venv/bin/python $S/layer_select.py --snr $S/snr/snr_raw.json \
  --layer-scores-out $S/snr/layer_scores.json --selection-out $S/spectrum_layers.json
```

(`layer_select.py` also takes `--k` (default 16) and `--rule dense-z-mean|q_proj-only`.)
A new base also means updating the constants pinned in
`spectrum_lora_layers.jac` (`EXPECTED_TRAINABLE_PARAMS`, `DEFAULT_NUM_BLOCKS`,
module suffixes) and the `adapter_config_fix.jac` assumptions. Those are
written for Qwen3-Coder-30B-A3B.

### 1.3 Hyperparameters

From `train/configs/sft_spectrum.yaml` (inside your experiment dir; identical
to `train/configs/sft_stock.yaml` except `adapter_path`). Unchanged since 04.
Change none of them if you want your number in the lineage table.

| key | value |
|---|---|
| model | `models/qwen-q4` |
| data | your experiment dir's `dataset/sft` (reads `train.jsonl` + `valid.jsonl`; the runner passes `--data`, which overrides the yaml's own default) |
| fine_tune_type | lora |
| num_layers | 16 (the driver asserts this equals the layer-list length) |
| lora rank / scale / dropout | 16 / 2.0 / 0.05 |
| batch_size | 1 |
| iters | 8200 |
| learning_rate | 2.0e-5, cosine_decay, warmup 820, decays to 1.0e-6 at 8200 |
| max_seq_length | 3072 |
| mask_prompt | true (loss on the assistant turn only) |
| grad_checkpoint | true |
| save_every | 820 (10 checkpoints) |
| steps_per_eval / steps_per_report / val_batches | 500 / 50 / 8 |
| seed | 42 |

On 08's pool this ran at about 0.53 it/s, so roughly 4.3 hours for 8200 steps.

### 1.4 Launch

First run, without confirmation: self-test and dry run only.

```bash
pgrep -fl "jac start|mlx_lm"                                  # empty
experiments/09-my-dataset/train/run_sft_spectrum.sh
```

This does two things and exits:

1. **`--verify-layers` gate** (loads the base twice, a few minutes). Output in
   `results/verify_layers.txt` (inside your experiment dir). It must end in
   `VERIFY: PASS`, with both the spectrum and control lines showing
   `lora tensors = 256` and `trainable = 281.838M / 5054.233M (5.576%)`, and
   `upstream guard: OK (mlx-lm 0.31.3, source hash matches pin)`. (The
   percentage uses packed q4 words as the denominator. mlx's own training
   banner prints `281.838M/30532.123M` (0.923%). Same count.) If it fails,
   training does not start.
2. **Dry run**, 30 iters into `tmp/dry-spectrum`
   (advisory; check the loss lines are finite). `tmp/` is throwaway and gitignored.

Then the real run:

```bash
CONFIRM_FULL_RUN=1 experiments/09-my-dataset/train/run_sft_spectrum.sh
```

Env knobs: `SKIP_VERIFY=1`, `SKIP_DRY=1`, `DRY_ITERS` (30), `EVAL_EVERY`
(watchdog poll seconds, 60), `SFT_STALL_SECS` (900), `SFT_OOM_RECOVERY_ITERS`
(100).

Sequential only: one training process on the box. If you also run a stock
arm, finish one before starting the other.

### 1.5 What the watchdog does (markers, crash-resume)

Run state lives in your experiment dir's `results/`:

| file | meaning |
|---|---|
| `.verify.done`, `.dry.done` | gates already passed; delete to force a re-run |
| `.sft_progress_steps` | true global step reached so far |
| `.segment.log` | stdout of the current attempt (appended to `train.log` when the attempt ends) |
| `.train.done` | training complete; a re-run exits immediately |
| `train.log` | all attempts concatenated |

- Training runs as one continuous `mlx_lm.lora` process. The loop relaunches
  only after a crash, a stall (no log growth for 900 s), or an OOM.
- On relaunch it resumes with `--resume-adapter-file adapters.safetensors` for
  the remaining steps. That restores weights only. The optimizer state and the
  LR schedule restart. This is accepted, but note it in the report if it happens.
- mlx saves `NNNNNNN_adapters.safetensors` numbered from the start of each
  attempt. After an attempt ends, the loop copies them into
  `adapter/checkpoints/` (gitignored, inside your experiment dir) under their
  **true** global step. The eval sweep reads that folder.
- OOM signature: the next attempts are capped at 100 iters, at most twice, then
  it gives up. Five consecutive failures at the same step also give up.
- **If the whole process tree dies** (lid-close sleep, power), the loop never
  gets to update `.sft_progress_steps` or copy the new checkpoints. Before
  relaunching: true step = the value in `.sft_progress_steps` + the highest
  local number among `adapter/*_adapters.safetensors` from the dead
  attempt. Write that into `.sft_progress_steps` and copy the missing
  checkpoints into `checkpoints/` with true-step names. Then run the same
  `CONFIRM_FULL_RUN=1` command.
- **Guard against training over a finished adapter:** if
  `adapter/adapters.safetensors` exists but neither `.sft_progress_steps` nor
  `.train.done` does (in `results/`), the script refuses to run. This is
  what protects every finished adapter under `experiments/`. To retrain, move the
  adapter directory aside. Do not delete it; adapters here are single-copy.

### 1.6 Monitoring

- The watchdog regenerates PNGs in `results/` (inside your experiment dir)
  every poll via `eval/plot_metrics.jac`: `train_loss.png`, `val_loss.png`,
  `learning_rate.png`, `tokens_per_sec.png`, `iters_per_sec.png`,
  `peak_mem.png`, `trained_tokens.png`.
- On demand:

  ```bash
  R=experiments/09-my-dataset/results
  .venv/bin/jac run experiments/09-my-dataset/eval/plot_progress.jac \
    --train-log $R/train.log --out $R/plots/loss.png \
    --eval-curve $R/metrics_functional.jsonl --eval-out $R/plots/eval.png
  ```

- **Grep for NaN.** `mlx_lm.lora` does not crash or exit non-zero when loss
  goes NaN; the log keeps growing and the stall detector sees nothing.
  `grep -n "nan" experiments/09-my-dataset/results/.segment.log`
  should print nothing. If it does, stop, find the over-length rows
  (`nan_guard.jac`), and restart from scratch. NaN poisons Adam's state, so a
  resume will not recover.
- Healthy end state for 08: `Iter 8200: Train loss 0.083`, `Val loss 0.048`,
  one `>>> SFT attempt` line in `train.log`. A low loss does not mean a good
  eval. 08 had the lowest loss of the lineage and the lowest holdout score.

### 1.7 Optional: stock control arm

To measure Spectrum against stock on a new dataset (RQ1 in 06/07/08), train
the stock arm with the same data and recipe on blocks 32 to 47:

```bash
CONFIRM_FULL_RUN=1 experiments/09-my-dataset/train/stock/run_sft.sh     # -> experiments/09-my-dataset/stock/{adapter,results}
experiments/09-my-dataset/eval/eval_sft_sweep.sh                         # same HOLDOUT=/RDIR= overrides as below
```

It doubles training time. On the three datasets where both arms were scored
(04, 06, 07), Spectrum never lost significantly on holdout (a) and won
significantly in two (04 at every stage, 07 at the DPO stages), so it is the
default without a control. Run the control when the dataset is very different from
anything before.

---

## Step 2: eval

### 2.1 Commands

```bash
pgrep -fl "jac start|mlx_lm"     # empty. The eval script does not check.

# holdout (a): defaults are HOLDOUT=<experiment dir>/dataset/holdout_a_shared855.jsonl, RDIR=<experiment dir>/results
experiments/09-my-dataset/eval/eval_sft_spectrum.sh

# holdout (b)
HOLDOUT=experiments/09-my-dataset/dataset/<your_holdout_b_eval>.jsonl \
RDIR=experiments/09-my-dataset/results/holdoutB \
  experiments/09-my-dataset/eval/eval_sft_spectrum.sh
```

Run them one after the other, not in parallel. On 08: holdout (a) took about
2h15m, holdout (b) about 1h20m.

### 2.2 What `eval_sft_spectrum.sh` does

1. **Rewrites `adapter_config.json`** (`train/spectrum/adapter_config_fix.jac`, inside your experiment dir):
   `num_layers` 16 becomes 48 and the layer list is added. Without this,
   `mlx_lm.load` rebuilds LoRA on blocks 32 to 47 from the adapter's own
   `num_layers: 16`, and `load_weights(strict=False)` silently drops blocks 0,
   22, 23, 27 and 30. The model scores as a partly trained one. Blocks covered
   but not trained stay at LoRA init (`lora_b` = 0, zero delta), so the rewrite
   is exact. It costs about 2.2 GB of extra memory. Log: `adapter_config_rewrite.txt`.
   The rewrite is persistent and required for any later `mlx_lm.load`.
2. **Asserts all 256 adapter keys land in the loaded model.** Must print
   `OK: all 256 adapter keys present in the loaded model` (`key_assertion.txt`).
3. **Base model, full holdout**, step 0 → `base.txt`. The floor: 10.5%
   (90/855) on (a); 0% on the py2jac holdouts.
4. **Checkpoint sweep**: every file in `adapter/checkpoints/` (inside your experiment dir), on the
   first `SUBSET` rows (default 100). This is the first 100 rows, not a random
   sample: on holdout (a) they are all `code_gen` (65 compile_only, 35
   behavioral). Use it for the trend only, never as a headline number.
5. **Final adapter, full holdout** at step 8200 → `final.txt`.

It truncates `metrics_functional.jsonl` at the start, so re-running overwrites
the previous eval in that `RDIR`.

Outputs in `RDIR`: `adapter_config_rewrite.txt`, `key_assertion.txt`,
`base.txt`, `final.txt`, `metrics_functional.jsonl`, and an empty `images/`.

### 2.3 The harness: `eval/eval_functional.jac`

Env vars (the wrappers set them):

| var | meaning |
|---|---|
| `JAC_EVAL_MODEL` | base model path |
| `JAC_EVAL_ADAPTER` | adapter dir, empty for base |
| `JAC_HOLDOUT` | holdout JSONL (default: the running script's own experiment dir's `dataset/holdout_a_shared855.jsonl`) |
| `JAC_EVAL_LIMIT` | first N rows only; 0 = all |
| `JAC_EVAL_BATCH_SIZE` | batch size for `mlx_lm.batch_generate` (32) |
| `JAC_EVAL_METRICS_OUT` | JSONL to append results to |
| `JAC_EVAL_STEP` | step label written into each metrics row |
| `JAC_EVAL_MODE` | set to `mlx` by the wrappers, not read by the harness |

What counts as a pass: the harness sends `messages[0]` through the chat
template, generates up to 768 tokens in batches of 32, extracts the first
` ```jac ` block (or first fenced block, or the raw text), writes it to
`snippet.jac` in a fresh temp dir, and runs `jac run snippet.jac`. **Pass =
exit code 0 within 30 s.** Despite the `behavioral` label, no output is
compared against expected values. The metric is "compiles and executes".
Batched 4-bit generation is not bit-reproducible, so compare aggregate rates,
not individual rows, across runs.

Standalone (e.g. scoring an earlier experiment's adapter):

```bash
JAC_EVAL_MODEL=models/qwen-q4 \
JAC_EVAL_ADAPTER=experiments/06-nitin-ds-sft/adapter \
JAC_HOLDOUT=experiments/08-nitin-new2-ds/dataset/holdout_a_shared855.jsonl \
  .venv/bin/jac run experiments/08-nitin-new2-ds/eval/eval_functional.jac
```

(Activate the venv first so the harness's inner `jac run` finds the venv `jac`.)

### 2.4 Reading the numbers

`final.txt` ends with the per-bucket lines and `OVERALL runs: 70% (603/855)`.
The printed percentages are **floored integers**. Always report the fraction
and compute one decimal yourself (603/855 = 70.5%).

`metrics_functional.jsonl` has one row per `(category, gate_class)` per step,
plus an `__overall__` row: `{"step", "category", "gate_class", "total", "runs", "runs_pct"}`.
Full-holdout rows have `total` 855 for the overall; subset rows have 100.

Per-category table for the final checkpoint, with one decimal (the last block
of `final.txt` has the same numbers, floored):

```bash
python3 - experiments/09-my-dataset/results/metrics_functional.jsonl <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
ends = [i for i, r in enumerate(rows) if r["category"] == "__overall__"]
for r in rows[ends[-2] + 1 : ends[-1] + 1]:          # last eval written = full-holdout final
    print(f'{r["category"]:>12} / {r["gate_class"]:<14} {100 * r["runs"] / r["total"]:5.1f}%  ({r["runs"]}/{r["total"]})')
PY
```

On 08 this prints `code_gen / compile_only 30.8% (70/227)` ...
`__overall__ / __all__ 70.5% (603/855)`. On holdout (a) the
buckets that move are `code_gen/compile_only` (n=227) and `code_gen/behavioral`
(n=86). `conversion/behavioral` (n=322) sits at 89 to 97% for every trained
adapter so far and barely separates anything.

### 2.5 Significance

**What prior reports did:** every verdict in 04, 06 and 07 used an **unpaired
two-proportion z-test**, two-sided, p < 0.05. The specs also required a paired
McNemar test wherever both sides answered the same items, but it was never
computed because no per-row pass/fail data was saved (07's report says so
directly). Treat that as a known gap, and close it in your experiment.

z-test (reproduces the published numbers, e.g. 06 spectrum vs stock SFT gives
z=0.930, p=0.352; 04 gives z=2.269, p=0.023):

```bash
python3 -c '
from math import sqrt, erfc
def z(x1, n1, x2, n2):
    p = (x1 + x2) / (n1 + n2)
    z = (x1/n1 - x2/n2) / sqrt(p*(1-p)*(1/n1 + 1/n2))
    return z, erfc(abs(z)/sqrt(2))
print(z(603, 855, 636, 855))   # 08 vs 06 on holdout (a): z=-1.79, p=0.074
'
```

For McNemar, dump per-row results for both adapters, then count discordant
pairs by `id`:

```bash
E=experiments/09-my-dataset
ADAPTER=$E/adapter \
OUT_PREFIX=$E/results/failure_data/spectrum_sft HOLDOUTS=A,B \
  .venv/bin/jac run $E/eval/gen_eval_detail.jac
IN=$E/results/failure_data/spectrum_sft_holdoutA.gen.jsonl \
OUT=$E/results/failure_data/spectrum_sft_holdoutA.graded.jsonl \
  .venv/bin/jac run $E/eval/grade_eval_detail.jac
```

`gen_eval_detail.jac` uses the same generation path as the harness (batch 32,
768 tokens) and writes one `{id, category, gate_class, prompt, generated}` row
per holdout row. Run it after `eval_sft_spectrum.sh` so the adapter config is
already rewritten. `grade_eval_detail.jac` takes `WORKERS` (8) and `JAC_BIN`
(default `.venv/bin/jac`). Each
`*.graded.jsonl` row has `id`, `category`, `gate_class`, `graded`, `pass`, plus
the error text for failures. With `b` = rows only adapter X passes and `c` =
rows only Y passes, the exact McNemar p is
`min(1, 2 * sum(C(b+c, i) for i <= min(b, c)) / 2**(b+c))`.

`grade_reference.jac` (`HOLDOUT=<file> OUT=<jsonl>`) runs the holdout's own
reference answers through the same gate. It tells you how many rows no model
can pass.

**Decision rule used since 06:** a claim ("Spectrum beats stock", "dataset X
beats dataset Y") holds if at least 4 of 6 comparisons are significant wins
with no significant losses. It is worse if the mirror holds, and
indistinguishable otherwise. Indistinguishable is a valid result.

**Power:** at n=855 and around 70%, the unpaired test detects about 4 to 5 pp.
Smaller differences need the paired test or more seeds. Every run so far is a
single seed, so p-values cover sampling noise, not training-seed noise.

---

## Step 3: report

Write `report.md` in your experiment dir. Structure, following
[08's report](experiments/08-nitin-new2-ds/report.md) and [07's](experiments/07-nitin-ds-new-sft/report.md):

1. **Scope note.** What ran, what was cut, and why, if the design changed.
2. **Bottom line up front.** One table: this experiment vs the lineage on both
   holdouts, with fractions. Then one or two sentences with the verdict against
   the decision rule you wrote before training.
3. **What ran.** Stage-by-stage status, iterations completed, `VERIFY: PASS`
   line, layer list and sha256.
4. **Headline numbers.** Per holdout: base, final, per-category table, the
   subset sweep curve (labelled as a 100-row trend).
5. **Comparison.** z-test (and McNemar if you dumped per-row data) against the
   baseline table below. List every comparison, not only the significant ones.
6. **Incidents.** Every real problem, cause, fix, and how you verified nothing
   was corrupted.
7. **Interpretation.** Why the result came out that way, with the evidence
   level stated (measured vs inferred).
8. **Named confounds.** Pool size, content type, holdout distribution, single seed.
9. **Artifacts.** Only paths that exist on disk when you finish.
10. **Open items.**

Baseline table, holdout (a), 855 code-graded rows, same base and harness:

| experiment | adapter | holdout (a) | where the adapter is |
|---|---|---|---|
| base (untrained) | none | 10.5% (90/855) | `models/qwen-q4` |
| 04-cpt-sft | fresh-arm Spectrum SFT-final | **74.7% (639/855)** | [`experiments/04-cpt-sft/adapter/`](experiments/04-cpt-sft/adapter/) |
| 06-nitin-ds-sft | Spectrum SFT-final | 74.4% (636/855) | [`experiments/06-nitin-ds-sft/adapter/`](experiments/06-nitin-ds-sft/adapter/) |
| 07-nitin-ds-new-sft | Spectrum DPO-best (step 120); its SFT-final was 72.4% (619/855) | 73.0% (624/855) | [`experiments/07-nitin-ds-new-sft/adapter/`](experiments/07-nitin-ds-new-sft/adapter/) |
| 08-nitin-new2-ds | Spectrum SFT-final | 70.5% (603/855) | [`experiments/08-nitin-new2-ds/adapter/`](experiments/08-nitin-new2-ds/adapter/) |

None of the differences among 04, 06 and 07 is significant. 08 vs 04 is
z=-1.95, p=0.051. A new experiment "wins" only if it beats 04's 639/855
significantly, which at this n means 674/855 (78.8%) or better unpaired.

Holdout (b) numbers are only comparable when (b) is the same file. 07 and 08
share one: 07 Spectrum SFT-final 98.2% (840/855), 07 DPO-best 97.9% (837/855),
08 37.7% (322/855).

---

## Optional: DPO pass

The scripts exist and work: `train/dpo/run_dpo_spectrum.sh` and
`eval/eval_dpo_spectrum.sh` (stock twins: `train/stock/run_dpo_nofuse.sh`,
`eval/eval_dpo_nofuse.sh`), all inside your experiment dir. Outputs:
`dpo/{adapter,adapter-best,results}` (stock: `stock-dpo/...`), also inside
your experiment dir. 08 skipped DPO entirely.

History says to expect little. DPO's best checkpoint has tied SFT in every
experiment (04, 06, 07). Running it to the full budget has lost up to 8 pp
(04). The one practical upside is that a Spectrum adapter held its score
through DPO better than a stock one. Run it only with a new kind of preference
data and a reason to think it teaches something SFT did not.

```bash
pgrep -fl "jac start|mlx_lm"
experiments/09-my-dataset/train/dpo/run_dpo_spectrum.sh                     # gates + 8-iter dry run, then exits
CONFIRM_FULL_RUN=1 experiments/09-my-dataset/train/dpo/run_dpo_spectrum.sh
experiments/09-my-dataset/eval/eval_dpo_spectrum.sh                         # HOLDOUT= / RDIR= as in step 2
```

- Needs the SFT adapter (it refuses to start without it). Run the holdout-(a)
  `eval_sft_spectrum.sh` first too: the collapse gate reads the SFT baseline from
  your experiment dir's `results/metrics_functional.jsonl`, and only rows with
  `total == 855`. Train with the default `HOLDOUT` (holdout a). Pointing it at
  another holdout silently leaves only the 30% absolute floor.
- Gates: `--verify-patches` (chat-template fix and layer rebind both live),
  `--verify-layers` (281.838M trainable, and every SFT-adapter key lands in the
  converted model). Both must print `VERIFY: PASS`.
- Defaults: `DPO_ITERS=250`, `DPO_LR=1e-6`, `DPO_BETA=0.1`, sigmoid loss,
  `DPO_MAXLEN=512` (OOM ladder can drop it to 384), 20-iter segments, a
  100-row subset eval per snapshot, best snapshot tracked in
  `dpo/results/.best_step` and copied to
  `dpo/adapter-best` (both inside your experiment dir).
- The DPO LoRA is seeded from the SFT adapter with `--resume-adapter-file`.
  **Never `mlx_lm.fuse` a LoRA into the 4-bit base**: re-quantization drops the
  SFT delta, and every DPO run then trains on an effectively untrained base
  (04's "DPO collapse" to 2 to 12%). `train/configs/dpo_lora.yaml` sets `fuse: false`
  for the same reason.
- `eval_dpo_spectrum.sh` scores both the last and the best snapshot:
  `final_last.txt`, `final_best.txt`.
- The DPO scripts never write `.train.done`. Completion is the
  `=== DPO reached full 250 iters ===` line in `train.log`.
- `.best_step` does not survive a crash-resume. After any resume, diff
  `runs_pct` across all log segments and fix `-best` by hand if needed (06
  incident 2).
- `nan_guard.jac` does not cover DPO rows. Check your experiment dir's
  `dataset/dpo/*.jsonl` prompt lengths against `DPO_MAXLEN` yourself first.
  08 never did.

---

## Lessons and gotchas

From [docs/HISTORY.md](docs/HISTORY.md), the per-experiment reports under [experiments/](experiments/) and [docs/history/](docs/history/), and 08.

**What works and what doesn't**

- **SFT is the lever.** Every stage that moved accuracy was SFT: 01 (0 to 94%
  on functions), 02 (39 to 61% greedy), 04 (10.5 to 74.7% on holdout a).
- **Spectrum layer selection is a free, modest win.** Same capacity, different
  placement. Significant at every stage in 04, mixed in 06, significant at the
  DPO stages in 07, never a significant loss on holdout (a). Keep it on.
- **Second passes on top of SFT have not paid off.** GRPO added nothing over
  SFT (02). CPT's +37 pp base-stage head start washed out to +2.8 pp, p≈0.20,
  once SFT ran (04). DPO ties SFT at best and regresses if run long (04, 06, 07).
- **A bigger pool is not a better pool.** 07 had 24% more rows than 06 and was
  numerically worse in all six cells. 08 had 2.18x 07's rows plus a quality
  filter aimed at exactly that problem, and scored lower still (70.5% vs
  74.4%). Mix and fit to the eval surface matter more than count.
- **Dilution is the likely cause of 08's holdout-(b) collapse** (98.2% for 07
  down to 37.7% for 08 on the same file). Only about 3,580 of 08's 14,792 SFT
  rows (composer + step4_work) were py2jac-shaped. The rest were js2jac
  translation and graph-native completions. This is inferred, not measured. The per-source breakdown
  (`gen_eval_detail.jac`) was never run.
- **The small-subset trap from 02:** a small, focused SFT set (20 examples)
  beat the full mix. More varied data can regress an already-learned task
  (task interference).

**Measurement**

- **Verify the grader before believing a null.** 02 spent three weeks on a flat
  result caused by an extraction helper that graded the driver's docstring
  instead of the model's answer (a ~3.5x undercount). The GRPO reward shared
  that helper, so training was also wrong.
- **A full eval that finishes in seconds did not run** (06: holdout with no
  `messages`).
- **Spectrum adapters must have `adapter_config.json` rewritten before load**,
  or the out-of-slice blocks drop silently. `eval_sft_spectrum.sh` does this;
  anything else that loads the adapter needs the rewritten config.
- **Subset sweep = first 100 rows**, all `code_gen` on holdout (a). Trend only.
- **The harness floors percentages.** Report fractions.

**Training**

- **NaN from over-length prompts (08).** With `mask_prompt: true` and
  `max_seq_length: 3072`, a row whose prompt alone is 3072 tokens or more has
  zero loss tokens after truncation. Loss becomes NaN, Adam's state is poisoned
  for good, and `mlx_lm.lora` keeps running without error. 08's first stock run
  went NaN at iter ~2300 from 5 js2jac rows up to 7,521 tokens. `nan_guard.jac`
  now runs inside `prep_training_dirs.sh`. Grep the log for `nan` anyway.
- **GRPO σ=0 trap (02).** If every rollout in a group scores the same (all
  fail), the advantage is zero and so is the gradient, at any learning rate.
  RL cannot bootstrap a skill the model has none of. SFT first.
- **CPT on doc prose teaches fluent fabrication (03).** CPT-v2 cleared 0 of 3
  acceptance gates. The model invented plausible wrong-domain syntax instead of
  learning semantics. Not worth another run.
- **Never fuse a LoRA into the 4-bit base** (04). See the DPO section.
- **`--resume-adapter-file` restores weights, not optimizer or LR schedule.**
  Don't segment SFT on purpose; the SFT runner only relaunches after a real failure.
- **The best-checkpoint tracker resets across crash-resume** (06). Re-derive
  the best step from all log segments.

**Operations**

- **External drive drops.** `/Volumes/ExtremePro` disconnected three times
  (06 twice, 08's dataset build once). After any `Input/output error`, reseat,
  then verify every artifact (row counts, funnel arithmetic, duplicate ids,
  checksums) before resuming. Don't assume the last write finished.
- **Lid-close kills runs** even under `caffeinate`. Keep the lid open.
- **Watchdog self-collision** (08): a monitor whose argv contains `mlx_lm`
  makes the runner's preflight refuse to relaunch. Run monitors from a file.
- **One process at a time.** No `jac start` (JMS) while training or evaluating.
- **Adapters, datasets and results are single-copy.** Move aside, never delete.
  Check before any destructive command.
