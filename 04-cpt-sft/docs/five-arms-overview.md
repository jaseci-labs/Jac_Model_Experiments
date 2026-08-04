# The five arms — CPT × layer-selection, full picture

One master index for everything this phase has run: two independent variables
(does CPT run first, and which 16 of 48 decoder blocks carry the LoRA), five
arms covering the combinations that matter, plus exactly how to pick this back
up if you're returning cold.

**Read this file first.** It links out to every other doc instead of
repeating them. If you only read one thing before resuming work, read this.

---

## 1. The two variables

| | Trailing-16 (`mlx_lm` default, blocks 32–47) | Spectrum (Arcee's SNR-picked 16, `[0,22,23,27,30,34,36,37,38,39,41,42,43,44,45,47]`) |
|---|---|---|
| **No CPT** (fresh base) | **Arm 1** | **Arm 3** |
| **CPT-v2 already trained on trailing-16** | **Arm 2** | **Arm 4** (union/freeze workaround) |
| **CPT itself re-run on Spectrum's picks** | *(not run — CPT-v2 already is a trailing-16 control)* | **Arm 5** (clean chain, IN PROGRESS) |

"Spectrum" = Arcee AI's layer-selection method: rank each decoder block's
weight-matrix singular-value spectrum against what pure noise (Marchenko-Pastur
random matrix theory) would produce, and LoRA-target the 16 blocks with the
most real signal, instead of blindly taking the last 16. It's a placement
change at constant capacity (same rank 16 / scale 2.0 / dropout 0.05, same
281,837,568 trainable params) — never a capacity change. Full mechanism:
`spectrum-plan.md` §2.

---

## 2. The five arms, one table

All functional pass rates are on the same frozen 855-row code-graded holdout
(`dataset/shared/holdouts/`), same SFT/DPO dataset, same recipe, so every
number below is directly comparable. n=855 unless noted.

| # | Name | Directory | CPT layers | SFT/DPO layers | Base | SFT | DPO-best | DPO-final | Status |
|---|---|---|---|---|---|---|---|---|---|
| 1 | fresh, stock | `sft_fresh_probe/` | none | 32–47 | 10.5% (90) | 69.8% (597) | 69.8% (597, step20) | 62.1% (531) | **DONE** |
| 2 | cptv2, stock (incumbent) | `sft_cptv2_probe/` | 32–47 | 32–47 | 47.3% (404) | 72.6% (621) | 71.7% (613, step40) | 64.9% (555) | **DONE** |
| 3 | fresh, Spectrum | `sft_fresh_probe/spectrum/` | none | picks | 10.5% (90) | 74.7% (639) | 74.2% (634, step20) | 72.7% (622) | **DONE** |
| 4 | cptv2, Spectrum (union/freeze) | `sft_cptv2_probe/spectrum/` | 32–47 | picks (21-block union, 5 frozen) | 47.3% (404) | 70.5% (603) | 71.7% (613, step20) | 69.0% (590) | **DONE** |
| 5 | cptspectrum, clean chain | `sft_cptspectrum_probe/` + `03-cpt-only/cpt_train/spectrum/` | **picks** | **picks** | TBD | TBD | TBD | TBD | **PAUSED — see §5** |

Read as prose:

- **Arm 1 vs Arm 2** answers "does CPT-v2 help?" (the phase's original question,
  `docs/reports/2026-07-cpt-vs-fresh-comparison.md`) — CPT gives a huge base-stage
  edge (+37pp) that's statistically absorbed by SFT (+2.8pp, not significant)
  but survives structurally in the SFT adapter's own weight geometry (that
  report's §4, SVD probe).
- **Arm 1 vs Arm 3**, and **Arm 2 vs Arm 4**, answer "does Spectrum layer
  selection help?" (`docs/reports/2026-08-spectrum-vs-stock-comparison.md`) —
  Spectrum **significantly** beats stock on the fresh base at every stage
  (SFT +4.9pp p=0.023, DPO-best +4.3pp p=0.046, DPO-final +10.6pp p<0.0001),
  but shows **no significant difference** once CPT-v2 is already in the chain
  (SFT −2.1pp p=0.34, DPO-best exact tie, DPO-final +4.1pp p=0.07 — marginal).
- **Arm 4 vs Arm 5** is the open question Arm 5 exists to answer: was Arm 4's
  null result caused by the union/freeze *confound* (CPT trained on one 16-block
  set, SFT/DPO trained on a different, only-11/16-overlapping set), or does
  Spectrum genuinely not help once any CPT-style adaptation is in the chain?
  Arm 5 removes the confound by re-running CPT itself on Spectrum's exact
  picks, so CPT → SFT → DPO is one continuous 16-block lineage with no union,
  no freeze, no merge step anywhere.

---

## 3. Where each arm's full detail lives

| Arm | Design docs | Results | Full report |
|---|---|---|---|
| 1, 2 | `spec.md`, `workflow.md`, `dpo-plan.md` | `RESULTS.md` §2–3 | `docs/reports/2026-07-cpt-vs-fresh-comparison.md` |
| 3, 4 | `spectrum-plan.md`, `spectrum-workflow.md` | `RESULTS.md` (Spectrum callout) | `docs/reports/2026-08-spectrum-vs-stock-comparison.md` |
| 5 | `cpt-spectrum-plan.md` (design, ~1050 lines, exact recipe/paths/mechanism) | *(not yet written — arm incomplete)* | `docs/reports/2026-08-cpt-spectrum-three-way.md` *(does not exist yet — write this when arm 5 finishes, see §6)* |

`cpt-spectrum-plan.md` is the one to actually read before resuming arm 5 — it
has the exact file layout, the exact bug fixes already applied, the exact
gate sequence, and the exact 6 z-tests the final report needs to run.

---

## 4. Two real engineering incidents, both resolved (worth knowing before touching this code again)

1. **MLX in-place-safetensors-merge bug** (hit building arm 4). `mx.load()` on
   a safetensors file is a *lazy* memory map. `merge_frozen_keys.py` (arm 4's
   union/freeze merge step) read a file with `mx.load()` then wrote the merged
   result back to the *same path* — truncating the mmap the loaded arrays were
   still lazily reading from, silently zeroing every trained key. Fixed with
   `mx.eval(list(trained.values()))` before any write. Full incident writeup:
   `docs/reports/2026-08-spectrum-vs-stock-comparison.md` §3.1. This is why
   arm 5's design deliberately has **no merge step at all** — removing the
   union/freeze machinery removes the whole bug class, not just this instance
   (`cpt-spectrum-plan.md` §2.2c audits every remaining safetensors read site
   in arm 5's pipeline and confirms none is a read-then-overwrite-in-place).
2. **DPO out-of-memory on arm 4.** DPO holds both policy and reference model;
   arm 4's extra ~1.9GB union-conversion overhead (21 converted blocks vs 16)
   pushed it over this machine's GPU wired-memory ceiling at the default
   `DPO_MAXLEN=512`. Fixed by dropping to `DPO_MAXLEN=384` (same value the
   scripts' own built-in OOM-recovery ladder already uses as its last resort).
   Arm 5 should **not** hit this — no union conversion, so ~1.9GB less
   overhead — but if it does, the fix is the same.

Also: two independent import-order bugs were found and fixed in the layer-
selection rebind mechanism this phase relies on throughout (`spectrum_lora_layers.py`
composing `mlx_lm`'s `linear_to_lora_layers`, and `mlx_lm_lora`'s separate DPO
conversion path needing 3 rebind sites, not 1) — see `spectrum-plan.md` §6.2
and `dpo_spectrum_train.py`'s header. Arm 5's CPT driver
(`run_cpt_leg_spectrum.py`) hit this same class again and handles it the same
way: patch before import, then assert the patch took, with 4 independent
detectors (`cpt-spectrum-plan.md` §2.1, R7).

---

## 5. Arm 5's exact status, and how to resume it

**As of 2026-08-03 22:07 EDT: PAUSED intentionally, mid-leg-1, by explicit
request** (to write this document and stop background compute) — **not** a
crash, **not** a bug. Nothing is broken. Everything below is what to do to
pick it back up.

### 5.1 What's already done (all committed and pushed, commits `80516a2`, `4e6a382`)

- Full design: `cpt-spectrum-plan.md`, written by an opus-model planning
  agent, reviewed and judged sound.
- **5 real bugs found and fixed** in the previously-broken CPT orchestration
  code (stale `03-new` → `03-cpt-only` rename paths that made it
  unexercisable): `run_cf_check.py`'s temp-file path, `run_cf_check.py`'s
  `ROOT` being off-by-one (`parents[3]` → `parents[4]`, so `models/qwen-q4`
  never resolved), `test_run_leg_cf_check.py`'s matching off-by-one in its own
  `BASE_MODEL` constant, and `run_leg_cf_check.py`'s hardcoded output dir
  (would have silently overwritten CPT-v2's own precious per-leg CF evidence
  at identical step numbers — now `JAC_CF_RESULTS_DIR`-overridable). All 65
  existing + 5 new tests pass.
- New CPT driver tree, fully built: `03-cpt-only/cpt_train/spectrum/`
  (`run_cpt_leg_spectrum.py`, `run_epoch_loop_spectrum.py`,
  `run_cpt_spectrum.sh`, `configs/config_v2_leg1..12.yaml`,
  `configs/spectrum_layers.json` symlink).
- Downstream tree, fully built: `04-cpt-sft/sft_cptspectrum_probe/` (dataset
  symlinks MD5-verified against the other arms, `sft_spectrum.yaml` with
  `resume_adapter_file` already pointing at where the CPT stage's final
  adapter *will* land, all 4 run/eval shell scripts copied and retargeted from
  the fresh arm's, verified clean).
- **All 3 pre-flight gates passed clean on the real 30B model**:
  - Gate 1 (`--verify-layers`): PASS — 256 tensors, 48 `LoRASwitchLinear`,
    exact 281.838M trainable params, placement-invariant capacity confirmed.
  - Gate 2 (2-iter dry leg): PASS — **peak mem 30.058 GB**, a comfortable
    ~6GB margin below this machine's ~36GB estimated GPU wired-memory
    ceiling. This was the single biggest risk in the whole plan (R2 — CPT
    runs at `max_seq_length: 4096` *and* backprops all 48 blocks, higher
    activation memory than anything else tested this phase) and it's
    resolved with real headroom.
  - Gate 3 (end-to-end CF-check on the dry adapter): PASS 16/16, and
    CPT-v2's own historical CF evidence verified **untouched** (file count
    unchanged) — proving the `JAC_CF_RESULTS_DIR` fix actually prevents the
    data-loss hazard it exists for.
- The real ~26–30h run was launched (`CONFIRM_FULL_RUN=1 nohup bash
  run_cpt_spectrum.sh`), leg 1/12 started cleanly (confirmed correct command,
  correct flags), then **killed intentionally** ~1 minute in, before any
  checkpoint was written (`0000544_adapters.safetensors` does not exist —
  only `adapter_config.json` was written, which gets overwritten harmlessly
  on the next launch). Confirmed clean: no orphaned processes, no partial
  checkpoint files.

### 5.2 Honest time budget — read before relaunching

This is **not** a quick job. Measured, not guessed (`cpt-spectrum-plan.md` §6):
Spectrum's picks include layer 0, so backprop must traverse all 48 decoder
blocks instead of the trailing 16 — a measured **1.59× slowdown** vs CPT-v2's
own pace. Projected: **~26–30 hours for the 12-leg CPT stage alone**
(9.06 s/iter × 1.59 ≈ 14.4 s/iter × 6528 total iters, plus 12 CF-checks at
~8–20 min each), **~37–41 hours end to end** including the downstream SFT
(~3.3h) and DPO (~1.2h) stages and their evals. Plan the resume around a
multi-day window, not one evening. The loop is resumable by construction —
a kill/crash/sleep costs at most the in-flight leg (~2.2h) — so it's safe to
run it in chunks across multiple sessions rather than needing one unbroken
stretch.

### 5.3 Exact commands to resume

```bash
cd /Volumes/ExtremePro/JaseciLabs/jac_model_studio

# 1. Sanity check nothing else is running that would compete for GPU memory
ps aux | grep -E "jac start|mlx_lm" | grep -v grep   # must be empty

# 2. Confirm where the loop will resume from (should say "leg 1" again,
#    since no checkpoint was written before the intentional stop)
cat model-experiments/03-cpt-only/results/cpt-v2-spectrum/json/training_state.json 2>/dev/null \
  || echo "no legs completed yet -- will start at leg 1"
ls model-experiments/03-cpt-only/adapters/cpt-v2-spectrum/*.safetensors 2>/dev/null \
  || echo "no checkpoints yet -- confirms leg 1 restart"

# 3. Launch (ONE background mechanism only -- no trailing '&' AND
#    run_in_background together, that orphans duplicate processes)
mkdir -p model-experiments/03-cpt-only/results/cpt-v2-spectrum/logs
CONFIRM_FULL_RUN=1 nohup bash model-experiments/03-cpt-only/cpt_train/spectrum/run_cpt_spectrum.sh \
  > model-experiments/03-cpt-only/results/cpt-v2-spectrum/logs/orchestrator.log 2>&1 &
disown

# 4. Watch it
tail -f model-experiments/03-cpt-only/results/cpt-v2-spectrum/logs/orchestrator.log
# or, less noisy, poll training_state.json every so often:
watch -n 60 'cat model-experiments/03-cpt-only/results/cpt-v2-spectrum/json/training_state.json 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print(f\"{len(d[\\\"legs\\\"])}/12 legs, status={d[\\\"status\\\"]}\")" 2>/dev/null || echo "leg 1 still running (no legs finished yet)"'
```

If it was interrupted again mid-leg with a partial checkpoint written, the
same launch command just works — `resume_point()` re-derives the correct
leg from whatever's on disk. Never delete anything under
`adapters/cpt-v2-spectrum/` to "clean up" a partial run; the loop needs those
files to resume correctly.

### 5.4 What happens automatically once it's running, and what still needs a human/agent to drive it

The 12-leg loop itself is fully automatic (training → CF-check → stop-loss
gate decision → next leg, or halt). **Nothing after the loop finishes is
automatic** — once it completes (or halts at the floor/ceiling gate), the
following still need to be run, in order, each gated the same way (dry run
first, `CONFIRM_FULL_RUN=1` for the real run):

```
 5. snapshot accepted checkpoint -> adapters/cpt-v2-spectrum-final/     (cpt-spectrum-plan.md §2.7)
 6. (optional, ~35min) CPT-only functional eval of cpt-v2-spectrum-final
 7. sft_cptspectrum_probe/run_sft_spectrum.sh          dry-gate, then CONFIRM_FULL_RUN=1  (~3.3h)
 8. sft_cptspectrum_probe/eval_sft_spectrum.sh                                            (~1.3h)
 9. sft_cptspectrum_probe/run_dpo_spectrum.sh          dry-gate, then CONFIRM_FULL_RUN=1  (~1.2h)
10. sft_cptspectrum_probe/eval_dpo_spectrum.sh                                            (~1.2h)
11. write docs/reports/2026-08-cpt-spectrum-three-way.md                                  (~1-2h)
```

The exact commands, exact expected outputs, and exact six z-tests the final
report needs (Arm 5 vs Arm 2, Arm 5 vs Arm 4, at each of SFT/DPO-best/
DPO-final) are all spelled out in `cpt-spectrum-plan.md` §5–§7 — that document
is the actual runbook, this file is just the map to it.

### 5.5 If the CF-check stop-loss gate fires before leg 12

This is a **real, reportable finding**, not a failure to work around
(`cpt-spectrum-plan.md` R10). CPT-v2 itself passed CF-check 16/16 on all 12
legs; Spectrum's picks include layer 0, right next to the embedding, which is
a plausible route to more general-coding forgetting than a trailing-16 delta
ever risked. If the gate halts early: continue to steps 5–11 above anyway,
using whatever checkpoint was accepted, and report the halt itself as part of
the answer (e.g. "Spectrum CPT triggered the stop-loss at leg N; CPT-v2 never
did") rather than treating it as blocking.

---

## 6. Once arm 5 finishes: write-up checklist

1. New report: `model-experiments/04-cpt-sft/docs/reports/2026-08-cpt-spectrum-three-way.md`,
   structure mirrors `2026-08-spectrum-vs-stock-comparison.md`.
2. Headline table: arms 2, 4, 5 (+ arm 3 as a descriptive reference) × {SFT,
   DPO-best, DPO-final} × {pass rate, z, p, significant?}. Six z-tests: arm 5
   vs arm 2, arm 5 vs arm 4, at each of the 3 stages. Two-proportion z-test,
   this project's standing convention: z = (p1−p2) / sqrt(p_pool·(1−p_pool)·(1/n1+1/n2)),
   p_pool = (x1+x2)/(n1+n2).
3. CPT-stage table: arm 5's 12 (or fewer) CF-check scores and per-leg train/val
   loss, next to CPT-v2's own 16/16×12 record and per-leg losses (already in
   `03-cpt-only/results/cpt-v2/json/training_state.json`) — a like-for-like fit
   comparison independent of anything downstream.
4. State plainly which of the four outcomes in `cpt-spectrum-plan.md` §7.5
   actually happened (the plan pre-commits what each result licenses as a
   conclusion, specifically so the reading isn't chosen after the fact).
5. Cross-link from this file (§3's table), from `RESULTS.md`, and from
   `spectrum-plan.md`'s status section.
6. Commit + push — verify `.gitignore` coverage with real `git check-ignore -v`
   calls before staging (never assume), never `git add -A`. The CPT stage
   alone produces ~41.7GB of checkpoints (12 legs × ~3.38GB adapter+optimizer
   each) that must never be staged.
