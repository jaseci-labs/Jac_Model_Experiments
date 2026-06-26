# SFT+DPO Model Bake-off Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the exact SFT+DPO treatment Qwen3-Coder already got on 5 same-size candidate base models, then rank all 6 to decide whether anything displaces Qwen3-Coder as the Jac base model.

**Architecture:** This is an experiment runbook, not new software. The pipeline already exists: `sft_dpo/run_probe.sh <hf-id> <name>` (quantize → base eval → LoRA SFT → fuse → learning curve → finetuned eval) and `sft_dpo/run_dpo.sh <name>` (fuse SFT → LoRA-DPO → behavior re-eval → idiom eval). One controlled variable: the base model. Each candidate runs sequentially (each saturates 48GB RAM), produces a `results/<name>/` dir, and contributes one row to a comparison matrix. The only code written is a parser extension to `make_comparison.py` and the final report.

**Tech Stack:** MLX (`mlx_lm`, `mlx_lm_lora`), Jac (`jac run` eval scripts), Python 3, bash. Mac M5 Pro, 48GB unified RAM.

## Global Constraints

These apply to every task. Copied verbatim from `docs/superpowers/specs/2026-06-25-sft-dpo-model-bakeoff-design.md`.

- **Hard gate: NO OOM.** Train Q4+LoRA, eval Q8. Any step that would exceed 48GB RAM is a failure to stop on, not push through. Keep `LIVE_EVAL` OFF (default) — it loads a second model copy and deadlocks 30B.
- **One variable only.** Identical `sft_dpo/configs/lora.yaml` for all models (rank 16, scale 2.0, 600 iters, lr 2e-5, batch 2, max-seq 2048). Identical DPO params (8 layers, beta 0.1, lr 1e-6, 200 iters, grad-checkpoint, max-seq 384). Identical data: `dataset/mlx` (SFT) + `dataset/mlx_dpo` (DPO). `num_layers` may be lowered **only** if MLX rejects 16 on a shallow model — log any such deviation.
- **Gate: none.** Full SFT+DPO on every surviving candidate regardless of SFT result. The user wants the complete matrix, not an early-drop screen. (Exception: gpt-oss may be dropped if its MXFP4 conversion path is broken — see Task 2.)
- **Sequential only.** No parallel runs — each model saturates RAM.
- **Never delete adapters, datasets, or results.** Cleanup between runs deletes only re-derivable `models/` (quantized weights), never `adapters/<name>-*` or `results/<name>/`. See [[feedback-never-delete-models]].
- **Decision rule.** A candidate displaces Qwen3-Coder (94% behavioral / idiom avg-sim baseline) only if it beats it on finetuned behavioral test-pass-% by more than run-to-run noise AND matches/beats idiom gain (DPO avg-sim drops while behavior holds). Ties → keep Qwen3-Coder (incumbent, Apache-2.0, proven).

## Candidate run order (locked)

MoE-first, dense last; gpt-oss de-risked first because its conversion path is the one unknown.

| order | short-name | HF id (verify exact tag in Task 1) | type | notes |
|---|---|---|---|---|
| — | `qwen` | — | MoE | incumbent, already measured — **not re-run**, parsed from existing `results/qwen/` |
| 1 | `gptoss` | `openai/gpt-oss-20b` | MoE | MXFP4 — convert dry-run gate first |
| 2 | `qwen3i` | `Qwen/Qwen3-30B-A3B-Instruct-2507` | MoE | sibling, isolates coder-pretrain value |
| 3 | `dscoder` | `deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct` | MoE | coder MoE |
| 4 | `ling` | `inclusionAI/Ling-Coder-lite` | MoE | coder MoE, native SFT→DPO recipe |
| 5 | `qwen25c` | `Qwen/Qwen2.5-Coder-14B-Instruct` | dense | dense-vs-MoE LoRA control |

Excluded (OOM): GLM-4.5-Air, Llama-4-Scout, Kimi K2, any dense ≥27B. Gemma3-26B reuses existing `results/gemma/` numbers.

---

### Task 1: Pre-flight — verify environment, HF tags, data, disk

**Files:**
- Read only: `sft_dpo/configs/lora.yaml`, `dataset/mlx/`, `dataset/mlx_dpo/`, `dataset/eval_holdout/conversion.jsonl`
- Produces: a confirmation note (no commit needed — gating step)

**Interfaces:**
- Consumes: nothing.
- Produces: a verified candidate table (exact HF ids + revisions) used by Tasks 2–6, and a go/no-go on disk + toolchain.

- [ ] **Step 1: Confirm the toolchain is installed**

```bash
mlx_lm.lora --help | head -5          # SFT trainer present
python -c "import mlx_lm_lora; print(mlx_lm_lora.__file__)"   # DPO trainer present
jac --version                          # eval runner present
python -c "import matplotlib, numpy, scipy; print('plot deps ok')"
```
Expected: each prints without error. If `mlx_lm_lora` is missing: `pip install mlx-lm-lora`. If `scipy` missing (needed by `make_comparison.py` pchip): `pip install scipy`.

- [ ] **Step 2: Confirm the fixed inputs exist**

```bash
wc -l dataset/mlx/train.jsonl dataset/mlx/valid.jsonl \
      dataset/mlx_dpo/train.jsonl dataset/mlx_dpo/valid.jsonl \
      dataset/eval_holdout/conversion.jsonl
test -f sft_dpo/configs/lora.yaml && echo "config ok"
```
Expected: `dataset/mlx` ~529/59, `dataset/mlx_dpo` ~132/15, holdout 150, `config ok`. If any file is missing, STOP — the spec forbids regeneration; investigate before proceeding.

- [ ] **Step 3: Verify each HF repo id + revision resolves**

```bash
python - <<'PY'
from huggingface_hub import model_info
ids = ["openai/gpt-oss-20b",
       "Qwen/Qwen3-30B-A3B-Instruct-2507",
       "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
       "inclusionAI/Ling-Coder-lite",
       "Qwen/Qwen2.5-Coder-14B-Instruct"]
for i in ids:
    try:
        mi = model_info(i)
        print(f"OK  {i:55s} sha={mi.sha[:8]}")
    except Exception as e:
        print(f"ERR {i:55s} {e}")
PY
```
Expected: 5 `OK` lines. For any `ERR` (tag drift / renamed repo), find the correct current id on huggingface.co and update the candidate table in this plan before that model's task. Record the resolved `sha` per model — that is the revision to cite in the final report.

- [ ] **Step 4: Confirm disk headroom**

```bash
df -h /Volumes/ExtremePro
```
Expected: ≥120GB free to start (each model is ~50–60GB of Q4+Q8+fused; cleanup between runs keeps the working set to ~2 models). If <120GB, plan to run cleanup (Task 2–6 final step) aggressively after each model.

---

### Task 2: gpt-oss-20b (`gptoss`) — MXFP4 de-risk gate, then full run

gpt-oss is run first because its MXFP4-native quantization is the one unproven path. If conversion or the 30-iter dry run fails, drop it and record an N/A row — do not let it block the other four.

**Files:**
- Create: `results/gptoss/` (base.txt, finetuned.txt, metrics.jsonl, idiom-finetuned.txt, idiom-metrics.jsonl, *.png), `results/gptoss/dpo/` (finetuned.txt, idiom.txt, idiom-metrics.jsonl), `adapters/gptoss-probe/`, `adapters/gptoss-dpo/`
- Read: `sft_dpo/run_probe.sh`, `sft_dpo/run_dpo.sh`, `sft_dpo/jacgen/idiom_eval.jac`

**Interfaces:**
- Consumes: verified HF id from Task 1.
- Produces: matrix row `gptoss` = {base%, SFT%, DPO%, SFT avg-sim, DPO avg-sim} — or a documented "dropped: MXFP4 conversion unsupported".

- [ ] **Step 1: Quantize + dry-run only (the gate)**

Run `run_probe.sh`; it quantizes Q4+Q8 first, then does a 30-iter LoRA dry run (8s manual-abort pause). Watch those two stages:

```bash
caffeinate -s ./sft_dpo/run_probe.sh openai/gpt-oss-20b gptoss
```
Expected through the gate: `models/gptoss-q4/` and `models/gptoss-q8/` each contain `config.json` + `*.safetensors`, and the dry run reaches a finite (non-NaN) train loss. 

**If quantize errors (MXFP4 unsupported by this mlx version) or the dry run NaNs/OOMs:** stop the run, then:
```bash
rm -rf models/gptoss-q4 models/gptoss-q8 adapters/gptoss-dry
```
Mark `gptoss` dropped, note the exact error in the final report (Task 8), and **skip to Task 3**. Do not retry with altered config — that would break the one-variable rule.

- [ ] **Step 2: Let the full SFT run continue**

If the gate passed, the same `run_probe.sh` invocation continues through train → fuse → learning curve → finetuned eval (it is resumable via `.done` markers, so a re-invoke picks up where it left off). Confirm completion:

```bash
ls results/gptoss/.train.done results/gptoss/.curve.done results/gptoss/.finetuned.done
grep "cross-compiled test pass" results/gptoss/base.txt results/gptoss/finetuned.txt
```
Expected: 3 marker files exist; `finetuned.txt` test-pass% > `base.txt` test-pass% (SFT moved the model). If finetuned ≤ base, note it — that is a real (surprising) result, not a bug to fix.

- [ ] **Step 3: Capture the SFT idiom baseline** (scripts don't do this; DPO gain needs a pre-DPO number)

```bash
JAC_EVAL_MODE=mlx JAC_EVAL_MODEL="models/gptoss-jac-fused-q8" \
  JAC_IDIOM_OUT="results/gptoss/idiom-metrics.jsonl" \
  jac run sft_dpo/jacgen/idiom_eval.jac | tee results/gptoss/idiom-finetuned.txt
```
Expected: prints `avg transpile-similarity: <x>` and appends one JSON row to `results/gptoss/idiom-metrics.jsonl`.

- [ ] **Step 4: Run DPO**

```bash
caffeinate -s ./sft_dpo/run_dpo.sh gptoss
```
Expected: creates `models/gptoss-jac-dpo-fused-q8/`, `results/gptoss/dpo/finetuned.txt`, `results/gptoss/dpo/idiom.txt`, `results/gptoss/dpo/idiom-metrics.jsonl`. Exit code 0.

- [ ] **Step 5: Verify the row is sane**

```bash
grep "cross-compiled test pass" results/gptoss/finetuned.txt results/gptoss/dpo/finetuned.txt
echo "SFT idiom:"; tail -1 results/gptoss/idiom-metrics.jsonl
echo "DPO idiom:"; tail -1 results/gptoss/dpo/idiom-metrics.jsonl
```
Expected: DPO test-pass% within run-to-run noise of SFT (behavior holds), and DPO `avg_sim` < SFT `avg_sim` (idiom gain). Record all five numbers.

- [ ] **Step 6: Cleanup if disk-bound, then commit results**

```bash
df -h /Volumes/ExtremePro    # if free < 80GB, run the rm below; else skip
rm -rf models/gptoss-q4 models/gptoss-q8 models/gptoss-jac-fused-q4 \
       models/gptoss-jac-fused-q8 models/gptoss-jac-dpo-fused-q8
git add results/gptoss adapters/gptoss-probe adapters/gptoss-dpo
git commit -m "results: gptoss SFT+DPO bake-off run"
```
Never `rm` `adapters/gptoss-*` or `results/gptoss/` — they are single-copy and re-fusing needs the adapter. (Quantized `models/` are re-derivable from HF + adapter.)

---

### Task 3: Qwen3-30B-A3B-Instruct (`qwen3i`) — full SFT+DPO

The coder-sibling. Same procedure as Task 2 with no de-risk gate (proven 30B-A3B path — `qwen` already ran on it).

**Files:**
- Create: `results/qwen3i/` + `results/qwen3i/dpo/`, `adapters/qwen3i-probe/`, `adapters/qwen3i-dpo/`

**Interfaces:**
- Consumes: verified HF id `Qwen/Qwen3-30B-A3B-Instruct-2507`.
- Produces: matrix row `qwen3i`.

- [ ] **Step 1: SFT**

```bash
caffeinate -s ./sft_dpo/run_probe.sh Qwen/Qwen3-30B-A3B-Instruct-2507 qwen3i
ls results/qwen3i/.train.done results/qwen3i/.curve.done results/qwen3i/.finetuned.done
grep "cross-compiled test pass" results/qwen3i/base.txt results/qwen3i/finetuned.txt
```
Expected: 3 markers; finetuned% > base%.

- [ ] **Step 2: SFT idiom baseline**

```bash
JAC_EVAL_MODE=mlx JAC_EVAL_MODEL="models/qwen3i-jac-fused-q8" \
  JAC_IDIOM_OUT="results/qwen3i/idiom-metrics.jsonl" \
  jac run sft_dpo/jacgen/idiom_eval.jac | tee results/qwen3i/idiom-finetuned.txt
```
Expected: appends one JSON row with `avg_sim`.

- [ ] **Step 3: DPO**

```bash
caffeinate -s ./sft_dpo/run_dpo.sh qwen3i
grep "cross-compiled test pass" results/qwen3i/finetuned.txt results/qwen3i/dpo/finetuned.txt
tail -1 results/qwen3i/idiom-metrics.jsonl results/qwen3i/dpo/idiom-metrics.jsonl
```
Expected: DPO behavior holds vs SFT; DPO `avg_sim` < SFT `avg_sim`. Record five numbers.

- [ ] **Step 4: Cleanup if disk-bound, commit**

```bash
df -h /Volumes/ExtremePro
rm -rf models/qwen3i-q4 models/qwen3i-q8 models/qwen3i-jac-fused-q4 \
       models/qwen3i-jac-fused-q8 models/qwen3i-jac-dpo-fused-q8   # only if free < 80GB
git add results/qwen3i adapters/qwen3i-probe adapters/qwen3i-dpo
git commit -m "results: qwen3i SFT+DPO bake-off run"
```

---

### Task 4: DeepSeek-Coder-V2-Lite-Instruct (`dscoder`) — full SFT+DPO

**Files:**
- Create: `results/dscoder/` + `results/dscoder/dpo/`, `adapters/dscoder-probe/`, `adapters/dscoder-dpo/`

**Interfaces:**
- Consumes: verified HF id `deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct`.
- Produces: matrix row `dscoder`.

- [ ] **Step 1: SFT**

```bash
caffeinate -s ./sft_dpo/run_probe.sh deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct dscoder
ls results/dscoder/.train.done results/dscoder/.curve.done results/dscoder/.finetuned.done
grep "cross-compiled test pass" results/dscoder/base.txt results/dscoder/finetuned.txt
```
Expected: 3 markers; finetuned% > base%. **If `run_probe.sh` aborts at the dry run with an MLX error about `num_layers` exceeding model depth:** lower it — temporarily edit `sft_dpo/configs/lora.yaml` `num_layers: 16` to the model's layer count (DeepSeek-Coder-V2-Lite has 27 layers, so 16 is valid and this should not trigger), re-run, then revert the edit and **log the deviation** in the final report. This is the one sanctioned departure from one-variable.

- [ ] **Step 2: SFT idiom baseline**

```bash
JAC_EVAL_MODE=mlx JAC_EVAL_MODEL="models/dscoder-jac-fused-q8" \
  JAC_IDIOM_OUT="results/dscoder/idiom-metrics.jsonl" \
  jac run sft_dpo/jacgen/idiom_eval.jac | tee results/dscoder/idiom-finetuned.txt
```

- [ ] **Step 3: DPO**

```bash
caffeinate -s ./sft_dpo/run_dpo.sh dscoder
grep "cross-compiled test pass" results/dscoder/finetuned.txt results/dscoder/dpo/finetuned.txt
tail -1 results/dscoder/idiom-metrics.jsonl results/dscoder/dpo/idiom-metrics.jsonl
```
Expected: behavior holds; DPO `avg_sim` < SFT `avg_sim`. Record five numbers.

- [ ] **Step 4: Cleanup if disk-bound, commit**

```bash
df -h /Volumes/ExtremePro
rm -rf models/dscoder-q4 models/dscoder-q8 models/dscoder-jac-fused-q4 \
       models/dscoder-jac-fused-q8 models/dscoder-jac-dpo-fused-q8   # only if free < 80GB
git add results/dscoder adapters/dscoder-probe adapters/dscoder-dpo
git commit -m "results: dscoder SFT+DPO bake-off run"
```

---

### Task 5: Ling-Coder-lite (`ling`) — full SFT+DPO

**Files:**
- Create: `results/ling/` + `results/ling/dpo/`, `adapters/ling-probe/`, `adapters/ling-dpo/`

**Interfaces:**
- Consumes: verified HF id `inclusionAI/Ling-Coder-lite`.
- Produces: matrix row `ling`.

- [ ] **Step 1: SFT**

```bash
caffeinate -s ./sft_dpo/run_probe.sh inclusionAI/Ling-Coder-lite ling
ls results/ling/.train.done results/ling/.curve.done results/ling/.finetuned.done
grep "cross-compiled test pass" results/ling/base.txt results/ling/finetuned.txt
```
Expected: 3 markers; finetuned% > base%. (Ling depth ~28 layers → num_layers 16 valid; apply the same num_layers fallback as Task 4 only if MLX errors.)

- [ ] **Step 2: SFT idiom baseline**

```bash
JAC_EVAL_MODE=mlx JAC_EVAL_MODEL="models/ling-jac-fused-q8" \
  JAC_IDIOM_OUT="results/ling/idiom-metrics.jsonl" \
  jac run sft_dpo/jacgen/idiom_eval.jac | tee results/ling/idiom-finetuned.txt
```

- [ ] **Step 3: DPO**

```bash
caffeinate -s ./sft_dpo/run_dpo.sh ling
grep "cross-compiled test pass" results/ling/finetuned.txt results/ling/dpo/finetuned.txt
tail -1 results/ling/idiom-metrics.jsonl results/ling/dpo/idiom-metrics.jsonl
```
Expected: behavior holds; DPO `avg_sim` < SFT `avg_sim`. Record five numbers.

- [ ] **Step 4: Cleanup if disk-bound, commit**

```bash
df -h /Volumes/ExtremePro
rm -rf models/ling-q4 models/ling-q8 models/ling-jac-fused-q4 \
       models/ling-jac-fused-q8 models/ling-jac-dpo-fused-q8   # only if free < 80GB
git add results/ling adapters/ling-probe adapters/ling-dpo
git commit -m "results: ling SFT+DPO bake-off run"
```

---

### Task 6: Qwen2.5-Coder-14B-Instruct (`qwen25c`) — dense control, full SFT+DPO

The dense-vs-MoE LoRA-learnability control. Run last.

**Files:**
- Create: `results/qwen25c/` + `results/qwen25c/dpo/`, `adapters/qwen25c-probe/`, `adapters/qwen25c-dpo/`

**Interfaces:**
- Consumes: verified HF id `Qwen/Qwen2.5-Coder-14B-Instruct`.
- Produces: matrix row `qwen25c`.

- [ ] **Step 1: SFT**

```bash
caffeinate -s ./sft_dpo/run_probe.sh Qwen/Qwen2.5-Coder-14B-Instruct qwen25c
ls results/qwen25c/.train.done results/qwen25c/.curve.done results/qwen25c/.finetuned.done
grep "cross-compiled test pass" results/qwen25c/base.txt results/qwen25c/finetuned.txt
```
Expected: 3 markers; finetuned% > base%. (Qwen2.5-Coder-14B is 48 layers → num_layers 16 valid.)

- [ ] **Step 2: SFT idiom baseline**

```bash
JAC_EVAL_MODE=mlx JAC_EVAL_MODEL="models/qwen25c-jac-fused-q8" \
  JAC_IDIOM_OUT="results/qwen25c/idiom-metrics.jsonl" \
  jac run sft_dpo/jacgen/idiom_eval.jac | tee results/qwen25c/idiom-finetuned.txt
```

- [ ] **Step 3: DPO**

```bash
caffeinate -s ./sft_dpo/run_dpo.sh qwen25c
grep "cross-compiled test pass" results/qwen25c/finetuned.txt results/qwen25c/dpo/finetuned.txt
tail -1 results/qwen25c/idiom-metrics.jsonl results/qwen25c/dpo/idiom-metrics.jsonl
```
Expected: behavior holds; DPO `avg_sim` < SFT `avg_sim`. Record five numbers.

- [ ] **Step 4: Cleanup if disk-bound, commit**

```bash
df -h /Volumes/ExtremePro
rm -rf models/qwen25c-q4 models/qwen25c-q8 models/qwen25c-jac-fused-q4 \
       models/qwen25c-jac-fused-q8 models/qwen25c-jac-dpo-fused-q8   # only if free < 80GB
git add results/qwen25c adapters/qwen25c-probe adapters/qwen25c-dpo
git commit -m "results: qwen25c SFT+DPO bake-off run"
```

---

### Task 7: Extend `make_comparison.py` to tabulate all 6 models from result files

The current `make_comparison.py` hardcodes accuracy/idiom numbers for `qwen`/`gemma` (lines 101–107, 121–122) — error-prone to hand-extend to 6 models. Add a parser that reads the numbers straight from the result files and emits the comparison matrix. Leave the existing PNG-plotting code untouched (it still works for the qwen/gemma curves).

**Files:**
- Modify: `sft_dpo/make_comparison.py` (append the matrix block + `__main__` call)
- Create: `sft_dpo/test_matrix.py` (parser self-check)
- Output: `results/comparison/matrix.md`

**Interfaces:**
- Consumes: each `results/<name>/{base.txt,finetuned.txt,idiom-metrics.jsonl}` and `results/<name>/dpo/{finetuned.txt,idiom-metrics.jsonl}`.
- Produces: `results/comparison/matrix.md` and stdout matrix; functions `pass_pct(path)->int|None`, `avg_sim(path)->float|None`, `build_matrix()->list[tuple]`, `write_matrix()->None`.

- [ ] **Step 1: Write the failing parser self-check**

Create `sft_dpo/test_matrix.py`:

```python
"""Self-check for the bake-off matrix parser (ponytail: asserts, no framework)."""
import make_comparison as mc

def test_pass_pct(tmp):
    p = tmp / "finetuned.txt"
    p.write_text("=== conversion probe (mlx) on 150 holdout tasks ===\n"
                 "runs (compiles+executes): 96%  (145/150)\n"
                 "cross-compiled test pass: 93%  (140/150)\n")
    assert mc.pass_pct(p) == 93
    assert mc.pass_pct(tmp / "missing.txt") is None

def test_avg_sim(tmp):
    p = tmp / "idiom-metrics.jsonl"
    p.write_text('{"total":150,"avg_sim":0.968}\n{"total":150,"avg_sim":0.338}\n')
    assert mc.avg_sim(p) == 0.338          # last row wins
    assert mc.avg_sim(tmp / "missing.jsonl") is None

if __name__ == "__main__":
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as d:
        test_pass_pct(pathlib.Path(d))
        test_avg_sim(pathlib.Path(d))
    print("matrix parser self-check: PASS")
```

- [ ] **Step 2: Run it to confirm it fails (functions don't exist yet)**

Run: `cd sft_dpo && python test_matrix.py`
Expected: FAIL with `AttributeError: module 'make_comparison' has no attribute 'pass_pct'`.

- [ ] **Step 3: Append the matrix parser to `make_comparison.py`**

Add at the end of `sft_dpo/make_comparison.py`, before any existing `if __name__ == "__main__":` block (if none exists, this introduces one):

```python
# --- bake-off matrix (6 models, parsed from result files) ---
import json as _json, re as _re, pathlib as _pl

BAKEOFF = {
    "qwen":    "Qwen3-Coder-30B-A3B (incumbent)",
    "gptoss":  "gpt-oss-20b",
    "qwen3i":  "Qwen3-30B-A3B-Instruct",
    "dscoder": "DeepSeek-Coder-V2-Lite",
    "ling":    "Ling-Coder-lite",
    "qwen25c": "Qwen2.5-Coder-14B",
}
_RES = _pl.Path("results")
_PASS_RE = _re.compile(r"cross-compiled test pass:\s*(\d+)%")

def pass_pct(path):
    """Behavioral test-pass % from an eval_probe stdout dump, or None if absent."""
    try:
        m = _PASS_RE.search(_pl.Path(path).read_text())
        return int(m.group(1)) if m else None
    except FileNotFoundError:
        return None

def avg_sim(path):
    """Last-row avg transpile-similarity from an idiom-metrics jsonl, or None."""
    try:
        rows = [_json.loads(l) for l in _pl.Path(path).read_text().splitlines() if l.strip()]
        return rows[-1].get("avg_sim") if rows else None
    except FileNotFoundError:
        return None

def build_matrix():
    rows = []
    for name, label in BAKEOFF.items():
        d = _RES / name
        rows.append((
            label,
            pass_pct(d / "base.txt"),
            pass_pct(d / "finetuned.txt"),
            pass_pct(d / "dpo" / "finetuned.txt"),
            avg_sim(d / "idiom-metrics.jsonl"),
            avg_sim(d / "dpo" / "idiom-metrics.jsonl"),
        ))
    return rows

def write_matrix():
    pct = lambda v: "—" if v is None else f"{v}%"
    sim = lambda v: "—" if v is None else f"{v:.3f}"
    out = ["| model | base | SFT | DPO | SFT sim | DPO sim | idiom gain |",
           "|---|---|---|---|---|---|---|"]
    for label, base, sft, dpo, ss, sd in build_matrix():
        gain = f"{ss - sd:+.3f}" if (ss is not None and sd is not None) else "—"
        out.append(f"| {label} | {pct(base)} | {pct(sft)} | {pct(dpo)} | "
                   f"{sim(ss)} | {sim(sd)} | {gain} |")
    text = "\n".join(out)
    (_RES / "comparison").mkdir(exist_ok=True)
    (_RES / "comparison" / "matrix.md").write_text(text + "\n")
    print(text)

if __name__ == "__main__":
    write_matrix()
```

Note: if `make_comparison.py` already ends with a top-level call to its plotting routine (not guarded by `__main__`), leave that call where it is — the new `__main__` block only adds the matrix step when the file is run directly.

- [ ] **Step 4: Run the self-check to confirm it passes**

Run: `cd sft_dpo && python test_matrix.py`
Expected: `matrix parser self-check: PASS`

- [ ] **Step 5: Generate the real matrix from completed runs**

Run: `cd sft_dpo && python make_comparison.py`
Expected: prints a 6-row markdown table (rows with no results yet show `—`) and writes `results/comparison/matrix.md`. Spot-check the `qwen` row against the known incumbent (94% behavioral).

- [ ] **Step 6: Commit**

```bash
git add sft_dpo/make_comparison.py sft_dpo/test_matrix.py results/comparison/matrix.md
git commit -m "feat: tabulate 6-model bake-off matrix from result files"
```

---

### Task 8: Write the comparison report + recommendation

**Files:**
- Create: `docs/initmodelchoice/2026-06-26-sft-dpo-bakeoff-results.md`
- Read: `results/comparison/matrix.md`, each `results/<name>/train.log` (val-loss stability + tok/s), each model's license

**Interfaces:**
- Consumes: the populated matrix and per-model logs.
- Produces: the final recommendation — which base model to commit the generation budget to.

- [ ] **Step 1: Assemble the matrix + secondary metrics**

For each of the 6 models, pull from the result files:
- behavioral base% / SFT% / DPO% and idiom SFT-sim / DPO-sim → from `results/comparison/matrix.md`
- training stability: eyeball `results/<name>/val_loss.png` (or grep `Val loss` in `results/<name>/train.log`) — note smooth-converging vs spiky/diverging
- inference tok/s: `grep -i "tok/s" results/<name>/finetuned.txt` (eval_probe prints avg tok/s)
- license: from each HF model card (gpt-oss Apache-2.0; Qwen2.5-Coder Apache-2.0; record the rest as resolved in Task 1)

- [ ] **Step 2: Write the report**

Create `docs/initmodelchoice/2026-06-26-sft-dpo-bakeoff-results.md` with:
- The 6-row matrix (paste `results/comparison/matrix.md`) plus columns for train-stability, tok/s, license.
- One paragraph per candidate: did SFT move it, did DPO improve idiom while holding behavior, any anomalies.
- If `gptoss` was dropped at the Task 2 gate, a line stating the exact MXFP4 error and that it is excluded.
- Any `num_layers` deviation logged in Tasks 4–6.
- **Recommendation**, applying the Global-Constraints decision rule verbatim: name the winner; if no candidate beats Qwen3-Coder on behavioral-% beyond run-to-run noise AND on idiom, the recommendation is "keep Qwen3-Coder."

- [ ] **Step 3: Update the spec status + memory**

- Edit `docs/superpowers/specs/2026-06-25-sft-dpo-model-bakeoff-design.md` header `Status: design — awaiting review` → `Status: complete — see docs/initmodelchoice/2026-06-26-sft-dpo-bakeoff-results.md`.
- Update memory `project-sft-dpo-bakeoff.md`: status → done, record the winner and the headline numbers. Add a one-line MEMORY.md hook update.

- [ ] **Step 4: Commit**

```bash
git add docs/initmodelchoice/2026-06-26-sft-dpo-bakeoff-results.md \
        docs/superpowers/specs/2026-06-25-sft-dpo-model-bakeoff-design.md
git commit -m "docs: SFT+DPO bake-off results + base-model recommendation"
```

---

## Self-Review

**Spec coverage:**
- Objective (rank 6 models, one variable) → Tasks 2–6 + 8. ✓
- Hardware no-OOM gate → Global Constraints + LIVE_EVAL-off note. ✓
- Candidate pool (5 + incumbent) → run-order table + Tasks 2–6; `qwen` parsed not re-run (Task 7 `BAKEOFF["qwen"]`). ✓
- Protocol (run_probe → run_dpo → record) → each task Steps 1–5. ✓
- Config/data control → Global Constraints; scripts pass `--model` so `lora.yaml` needs no per-model edit. ✓
- num_layers fallback → Task 4 Step 1 (with revert + log). ✓
- gpt-oss MXFP4 risk → Task 2 de-risk gate with drop path. ✓
- HF tag-drift risk → Task 1 Step 3. ✓
- Disk risk → Task 1 Step 4 + per-task cleanup. ✓
- Metrics (behavioral pass-% + idiom gain) → captured in every task; idiom SFT-baseline step added (scripts omit it). ✓
- make_comparison extension → Task 7. ✓
- Deliverables (per-model dirs, comparison report) → Tasks 2–6, 8. ✓
- Decision rule (ties keep Qwen3-Coder) → Global Constraints + Task 8 Step 2. ✓
- Out-of-scope (no RL, no regen) → respected; no such tasks. ✓

**Placeholder scan:** all commands use real paths/ids; no TBD/TODO. ✓

**Type consistency:** `pass_pct`/`avg_sim`/`build_matrix`/`write_matrix` used identically in `test_matrix.py`, the appended code, and Interfaces blocks. Short-names (`gptoss`,`qwen3i`,`dscoder`,`ling`,`qwen25c`,`qwen`) consistent across run-order table, tasks, and `BAKEOFF` dict. ✓
