# Spectrum vs Stock Trailing-16 — Does SNR-Based Layer Selection Beat mlx_lm's Default?

**Question:** `mlx_lm`'s default LoRA target is the trailing 16 decoder blocks (32–47). Arcee's
Spectrum method picks 16 blocks by signal-to-noise ratio (Marchenko-Pastur random matrix
theory) instead — a fixed, non-contiguous set: `[0, 22, 23, 27, 30, 34, 36, 37, 38, 39, 41,
42, 43, 44, 45, 47]` (11/16 overlap with the trailing slice). Same SFT/DPO recipe, same
dataset, same holdout, only the LoRA target set changes. Both `04-cpt-sft` arms (fresh base,
CPT-v2 base) ran this probe end to end.

**Bottom line:** Spectrum gave a real, statistically significant lift on the **fresh** arm at
every stage (SFT, DPO-best, DPO-final). It gave **no significant lift** on the **CPT-v2**
arm's SFT stage, and a mixed, largely-non-significant picture at DPO (tied at best-checkpoint,
a suggestive-but-not-quite-significant edge at final-checkpoint). This is an honest,
asymmetric result, not a clean win — see §4 for why that asymmetry is plausible.

---

## 1. Headline table

Two-proportion z-test throughout, same method as `2026-07-cpt-vs-fresh-comparison.md`:
z = (p1−p2) / sqrt(p_pool·(1−p_pool)·(1/n1+1/n2)), p_pool = (x1+x2)/(n1+n2). All n=855
(full holdout) unless noted.

| Arm | Stage | Spectrum | Stock (trailing-16) | Δ | z | p | Significant? |
|---|---|---|---|---|---|---|---|
| fresh | SFT | 74.7% (639/855) | 69.8% (597/855) | **+4.9pp** | 2.269 | 0.023 | **Yes** |
| fresh | DPO-best | 74.2% (634/855) | 69.8% (597/855, step20) | **+4.3pp** | 1.993 | 0.046 | **Yes** |
| fresh | DPO-final | 72.7% (622/855) | 62.1% (531/855, step250) | **+10.6pp** | 4.696 | <0.0001 | **Yes** |
| cptv2 | SFT | 70.5% (603/855) | 72.6% (621/855) | −2.1pp | −0.965 | 0.335 | No |
| cptv2 | DPO-best | 71.7% (613/855, step20) | 71.7% (613/855, step40) | 0.0pp | 0.000 | 1.000 | No (exact tie) |
| cptv2 | DPO-final | 69.0% (590/855, step250) | 64.9% (555/855, step250) | +4.1pp | 1.799 | 0.072 | Marginal (not <0.05) |

Fresh arm: Spectrum wins clearly, at every stage, by a growing margin as training progresses.
CPT-v2 arm: SFT is a wash (slightly negative, not significant), DPO-best is an exact tie,
DPO-final leans positive but doesn't clear the conventional 0.05 bar.

---

## 2. Per-stage detail

### 2.1 — Fresh arm (`sft_fresh_probe/`)

No incidents on this arm — SFT and DPO both ran clean the first time. Full detail already
lives in `spectrum-plan.md`/`spectrum-workflow.md`'s per-phase logs; the numbers above are
the full 855-row holdout scores pulled from `sft_fresh_probe/results/sft-spectrum/final.txt`
and `.../dpo-spectrum/final_best.txt` + `final_last.txt`.

### 2.2 — CPT-v2 arm (`sft_cptv2_probe/`) — SFT

SFT-spectrum checkpoint sweep (subset=100 rows, `metrics_functional.jsonl`), full 8200-iter
run, `resume_adapter_file` seeded from CPT-v2's real adapter, union-convert-and-freeze
discipline (§8.3):

| step | 820 | 1640 | 2460 | 3280 | 4100 | 4920 | 5740 | 6560 | 7380 | 8200 |
|---|---|---|---|---|---|---|---|---|---|---|
| subset pass% | 63 | 71 | 73 | 73 | 76 | **79** | 74 | 77 | 76 | 76 |

Peak was step 4920 (79%, subset), not the final checkpoint — but the full-holdout score at
the final checkpoint (70.5%, 603/855) is what's reported in §1, matching this project's
"final checkpoint is the number that counts unless a sweet-spot search says otherwise"
convention. The subset curve is noisy enough (100 rows) that step 4920's apparent peak
isn't treated as a real stopping-point recommendation without a full-holdout confirmation,
which wasn't run for intermediate checkpoints on this arm (time-boxed).

### 2.3 — CPT-v2 arm — DPO

DPO-spectrum segment curve (subset=100, 250 iters, `DPO_MAXLEN=384` — see §3.2):

| step | 20 | 40 | 60 | 80 | 100 | 120 | 140 | 160 | 180 | 200 | 220 | 240 | 250 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| subset pass% | **81** | 78 | 73 | 75 | 72 | 71 | 73 | 72 | 70 | 67 | 69 | 69 | 68 |

Best snapshot by this run's own gate: step 20 (81% subset). Full-holdout at step 20: 71.7%
(613/855) — an exact tie with stock's own best checkpoint (613/855, step 40). Full-holdout
at step 250 (final): 69.0% (590/855) vs stock's 64.9% (555/855) — Spectrum's final checkpoint
held up better than stock's did, though not quite to p<0.05.

---

## 3. Incidents

### 3.1 — The cptv2-arm SFT merge-corruption bug

**Observation.** After the first real 8200-iter cptv2 SFT-spectrum run completed cleanly
(exit 0, normal loss curve), every one of the 16 trainable layers' LoRA weights came back
**exactly zero** — both `lora_a` and `lora_b`, bit-for-bit — when loaded for eval.

**False lead, ruled out.** The first hypothesis was `mlx_lm/tuner/trainer.py`'s
`grad_checkpoint()` — it monkeypatches `type(model.layers[0]).__call__` at the *class* level
(affecting all 48 layer instances, since they share a class) and calls
`model.trainable_parameters()` from inside the checkpointed closure. This looked plausible
for a LoRA-converted-then-refrozen setup like the union-convert-and-freeze pattern this arm
uses. An isolated diagnostic — identical code, identical config, a short real run, checked
at the same checkpoint boundary — did **not** reproduce the zero weights. Layer 0 came out
correctly nonzero. This ruled the hypothesis out; the diagnosis moved to "probably a one-off
environmental issue" as the leading (wrong) explanation, and the training was retried clean.

**The retry reproduced it anyway — but at a different point.** The clean retry's raw
training checkpoints (`0000820_adapters.safetensors` through `0008200_adapters.safetensors`,
saved directly by `mlx_lm`'s own trainer) were all individually verified nonzero, at every
one of the 10 checkpoints, including the final one. But after `run_sft_spectrum.sh`'s
post-training merge step (§8.3 step 5, `merge_frozen_keys.py`) ran, the **merged**
`adapters.safetensors` — the file eval actually reads — had every trained key back to zero
again, while the 80 frozen CPT-v2 keys (loaded from a *separate* source file) were still
correct.

That asymmetry — frozen keys fine, trained keys zero, and only *after* the merge step — is
what pinned the real cause: **`merge_frozen_keys.py` calls `mx.load(--in)` and later writes
the merged dict to `--out`, and both call sites in `run_sft_spectrum.sh` pass `--in FILE
--out FILE` — the same path, in place.** MLX's `mx.load` on a safetensors file is a *lazy*
memory map: the returned arrays aren't materialized until something actually reads them.
Writing `--out` (the same file) *before* those lazy `trained` arrays are evaluated truncates
the very mmap they read from — so every trained key silently reads back as zero. No
exception, no warning. The frozen keys were unaffected because they were loaded from the
CPT-v2 adapter file, a completely different path never touched by the write.

**Fix.** One line in `merge_frozen_keys.py`'s `merge()`:

```python
trained = dict(mx.load(str(trained_file)))
mx.eval(list(trained.values()))  # force materialization before any write can truncate
                                   # the file these arrays are still lazily reading from
```

**Recovery.** Re-merged the main adapter and all 10 archived checkpoints from the clean raw
(never-merged) numbered checkpoint files, verified zero all-zero-keys across every one before
proceeding to eval. The DPO run later exercised the same code path (its own in-place merge
call) and came back clean on the first try, confirming the fix generalizes.

This is a genuine MLX gotcha worth remembering: **`mx.load()` + in-place overwrite of the
same path, with no `mx.eval()` in between, silently zeros whatever hadn't been read yet** —
worth checking for in any script that post-processes a safetensors file in place with
`mlx.core`.

### 3.2 — DPO out-of-memory on the cptv2 arm

The cptv2 DPO dry-run (8 iters) crashed at iter 7/8 with
`libc++abi: ... std::runtime_error: [METAL] Command buffer execution failed: Insufficient
Memory (kIOGPUCommandBufferCallbackErrorOutOfMemory)` at the default `DPO_MAXLEN=512`.
`vm_stat` confirmed 37GB+ of free system RAM at the time — this was not a system memory
shortage. It's macOS's GPU wired-memory ceiling (`iogpu.wired_limit_mb`, reported as `0` /
system-default, which resolves to roughly 70-75% of physical RAM on a 48GB machine) being
hit by DPO's requirement to hold **both** the policy and reference model in memory, plus
this arm's ~1.9GB extra union-conversion overhead (21 converted blocks vs the fresh arm's
16) that the fresh arm's DPO never had to carry.

Raising the OS-level ceiling requires `sudo sysctl iogpu.wired_limit_mb=...`, which needs an
interactive password this environment can't supply — not attempted. Instead, retried at
`DPO_MAXLEN=384`, the same value `run_dpo_spectrum.sh`'s own built-in OOM-recovery ladder
already uses as its documented last resort for the real run (the ladder only wraps the
*real* segmented run, not the one-shot dry-run, which is why the dry-run didn't self-heal).
The dry-run then completed cleanly (peak_mem 39.8GB), and the full 250-iter real run
completed with **zero** further OOMs across all 13 segments at this shorter sequence length.

This is a recipe deviation from the fresh arm's DPO (which ran at the default 512) worth
flagging honestly — the cptv2 arm's DPO numbers above were produced at `DPO_MAXLEN=384`,
not 512. Given DPO pairs in this dataset were generated without a specific length target,
some longer training pairs may have been truncated; this wasn't checked pair-by-pair and is
a known limitation of the comparison, not just a footnote.

---

## 4. Reading the asymmetry honestly

Spectrum's lift on the fresh arm is large, consistent, and grows through training. Its
effect on the cptv2 arm is small, inconsistent in direction, and not statistically confirmed
at any stage. One plausible (not proven) explanation: on the fresh arm, Spectrum's 16 picks
are the *only* adaptation happening — there's no prior training to interact with, so a
better-chosen 16 blocks has clean room to matter. On the cptv2 arm, the base model already
carries a real, structurally significant CPT-v2 delta across blocks 32–47 (see
`2026-07-cpt-vs-fresh-comparison.md` §4's SVD finding — CPT-v2's fingerprint survives SFT
structurally even when it washes out behaviorally). Spectrum's non-contiguous 16 picks then
have to interact with that pre-existing, block-position-specific adaptation rather than
starting from a blank slate, which may blunt whatever advantage the SNR-based selection
otherwise offers. This is speculation, not a mechanism this comparison actually tests —
flagged as a hypothesis for a future probe, not a conclusion.

**Verdict, stated directly:** Spectrum-vs-stock layer selection is a real, useful lever when
there's no prior adaptation to work around. It is not (on this evidence) a reliable lever
once a CPT stage has already reshaped part of the same layer range.

---

## 5. Artifacts

- `sft_fresh_probe/spectrum/`, `sft_cptv2_probe/spectrum/` — drivers, configs, merge script
  (with the fix from §3.1)
- `sft_fresh_probe/results/{sft-spectrum,dpo-spectrum}/`, `sft_cptv2_probe/results/{sft-spectrum,dpo-spectrum}/`
  — per-stage metrics, full-holdout `final*.txt`, sweep curves
- `sft_cptv2_probe/adapters/sft-on-cptv2-spectrum-BROKEN-zeroweights/`,
  `results/sft-spectrum-BROKEN-zeroweights/` — the first, corrupted cptv2 SFT attempt,
  archived (not deleted) per this project's never-delete-models convention
- `docs/spectrum-plan.md`, `docs/spectrum-workflow.md` — original design + runbook
