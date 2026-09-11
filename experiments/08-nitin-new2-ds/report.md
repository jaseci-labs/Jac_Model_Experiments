# 08-nitin-new2-ds — final comparison report

**Scope note, read this first:** this phase's scope was cut twice by explicit
user decision on 2026-09-11, after stock SFT had already completed. The
original design (`CONTEXT_BRIEF.md`, `docs/workflow.md`) called for 2 arms ×
2 holdouts × 3 stages (SFT/DPO-best/DPO-final) — a 12-cell matrix, plus
RQ1 (arm-vs-arm) and RQ2 (vs 07) comparisons. What actually ran and is
reported below is **spectrum-arm SFT only, both holdouts** — no DPO (either
arm), no stock-arm evaluation. Stock SFT itself completed cleanly but was
never evaluated; its adapter was later deleted in the 2026-09-11 repo cleanup
(see §1, §6). RQ1 and the DPO-axis analysis
(spec.md §2.4/§3.1) are **out of scope this phase**; there is no stock cell
or DPO cell to compare against. Treat this as a single-arm SFT checkpoint on
the RQ2 lineage-comparison question only.

## Bottom line up front

Spectrum SFT-final **underperforms every prior phase in this lineage on both
holdouts**, despite training on the largest, most-filtered pool yet (2.18×
07's row count, plus a quality-gradient filter specifically designed to fix
the exact "bigger pool, worse result" pattern 07 hit against 06):

| Holdout | 08 spectrum SFT-final | 07 spectrum SFT-final | 06 spectrum SFT-final | 04-cpt-sft spectrum SFT-final |
|---|---|---|---|---|
| (a) shared 855 | **70.5% (603/855)** | 72.4% (619/855) | 74.4% (636/855) | 74.7% (639/855) |
| (b) 07's holdout | **37.7% (322/855)** | 98.2% (840/855) | not run on this file (97.9% on 06's own holdout B) | not run |

(07 column filled in from `../../../docs/history/07-nitin-ds-new-sft/07-final-comparison.md`
after this report was first written; 07's best adapter overall was its
Spectrum DPO-best at 73.0%, 624/855.)

Significance on holdout (a), unpaired two-proportion z-test, the method every
prior report used: vs 07 SFT-final z=-0.86, p=0.39; vs 06 z=-1.79, p=0.074;
vs 04 z=-1.95, p=0.051. 08 is numerically the lowest, but not significantly
below any of them at p<0.05. On holdout (b) the drop from 07 (98.2% to 37.7%)
is far outside noise.

Holdout (b)'s collapse — from a documented 97–99% "ceiling compression" in
06/07 down to 37.7% here — is the headline finding, and it has a specific,
plausible mechanical explanation (§4), not just "the model got worse."

## 1. What ran

| Stage | Status |
|---|---|
| Stage 0 (corpus merge/filter/dedup/holdout reuse) | DONE, unchanged from before this report |
| Stage 0b (release + splits) | DONE, unchanged, **with one post-hoc fix** — see §3 |
| Stock SFT | DONE, completed cleanly to 8200/8200 iters, 0 NaN. **Not evaluated** per user scope cut. Was archived at `stock_probe/_out_of_scope_stock_sft_20260911_004125/`; deleted in the 2026-09-11 cleanup. |
| Spectrum SFT | DONE, completed cleanly to 8200/8200 iters, 0 NaN. `--verify-layers` gate passed (VERIFY: PASS, 281.838M/30532.123M trainable params, layer set `[0,22,23,27,30,34,36,37,38,39,41,42,43,44,45,47]`, sha256-verified unchanged from 04-cpt-sft/06/07). |
| Stock DPO | **Not run** (user scope cut) |
| Spectrum DPO | **Not run** (user scope cut) |
| Eval, holdout (a) | DONE, spectrum SFT only |
| Eval, holdout (b) | DONE, spectrum SFT only |
| Eval, stock arm (either holdout) | **Not run** (user scope cut) |

## 2. Headline numbers

### 2.1 Holdout (a) — shared 855, spectrum SFT

| Step | Overall pass rate |
|---|---|
| 0 (base) | 10.5% (90/855) |
| 8200 (final) | **70.5% (603/855)** |

Per-category breakdown at step 8200 (n=855; percentages are the harness's
floored integers, the fractions are exact):

| Category / gate | Pass rate |
|---|---|
| code_gen / behavioral | 53% (46/86) |
| code_gen / compile_only | 30% (70/227) |
| debug / behavioral | 81% (27/33) |
| debug / compile_only | 100% (3/3) |
| migration / behavioral | 0% (0/1) |
| trajectory / behavioral | 70% (49/70) |
| trajectory / compile_only | 84% (96/113) |
| conversion / behavioral | 96% (312/322) |

`conversion` (dominated by `js2jac`, the largest new source) and `debug` are
the strongest categories; `code_gen` (`compile_only`, n=227, the single
largest bucket) is the weakest at 30% and pulls the overall number down hard.

### 2.2 Holdout (b) — 07's reused holdout, spectrum SFT

| Step | Overall pass rate |
|---|---|
| 0 (base) | 0% (0/855) — expected, matches 06/07 exactly (raw Qwen has never seen Jac syntax) |
| 8200 (final) | **37.7% (322/855)** |

Interim sweep (100-row subset, not the headline number, shown for trend):
step 820→0%, 1640→5%, 2460→8%, 3280→15%, 4100→14%, 4920→19%, 5740→26%,
6560→33%, 7380→34%, 8200(subset)→31%. The curve is still climbing shallowly
at the final checkpoint, unlike 06/07 which reached their ceiling well before
iter 8200.

## 3. Incidents (real problems hit, in order)

1. **SFT train loss silently went NaN, `mlx_lm.lora` does not crash on it.**
   Stock SFT's first launch hit NaN at iter ~2300 and would have completed
   all 8200 iters producing a corrupted adapter with zero external signal
   (no exception, log kept growing, watchdog's stall/OOM detectors both
   miss it). Root cause: 5 `js2jac` rows (giant translated source files, up
   to 7521 tokens vs the 3072 `max_seq_length`) had **prompt alone** longer
   than the sequence limit; with `mask_prompt: true`, truncation left zero
   unmasked completion tokens, and cross-entropy over an empty target is
   NaN — which then poisons Adam's momentum/variance permanently. Fixed by
   scanning both SFT splits with the real tokenizer
   (`mlx_lm.tokenizer_utils.load`) and dropping any row whose
   `apply_chat_template(msgs[:last_assistant], add_generation_prompt=True)`
   length ≥ `max_seq_length`. 3 rows dropped from train, 2 from valid, all
   `js2jac`; originals were backed up as `dataset/sft/{train,valid}.jsonl.pre-nanfix.bak`
   (since deleted). The check is now `1-scaffold/nan_guard.jac`, run by
   `prep_training_dirs.sh`.
   Logged in `CONTEXT_BRIEF.md` §11 so it isn't rediscovered. **Note:**
   `dataset/dpo/{train,valid}.jsonl` was never checked for the same risk
   under `DPO_MAXLEN=512`, since DPO ended up out of scope — flag before any
   future DPO run on this pool.

2. **Lid-closed sleep killed the whole training process tree mid-run.**
   `caffeinate -dimsu` blocks idle/system sleep but not a forced Clamshell
   Sleep on lid-close — confirmed via `pmset -g log` (`Entering DarkWake
   state due to 'Clamshell Sleep'` then `Entering Sleep state due to
   'Clamshell Sleep'`, both at 20:41–20:42 on 2026-09-10). This killed
   stock SFT's `mlx_lm.lora` process and its supervising wrapper together,
   ~30 min after the last saved checkpoint (iter 2460), with no crash
   signature — the run just vanished. Recovered by rebuilding the
   orchestrator to resync each SFT stage's progress-step file from the
   true-global-step-named checkpoint files on disk (ground truth) before
   every attempt, rather than trusting a progress file that only updates on
   the wrapper's own clean-exit path. Training resumed from true step 2460
   with 5740 iters remaining, `--resume-adapter-file` correctly applied.
   **Operationally: keep the lid open (or use an external display) for any
   future unattended multi-hour run on this box** — this is a real
   constraint the orchestrator cannot fully paper over (worst case, up to
   `save_every`=820 iters / ~15 min lost per incident, whatever wasn't
   checkpointed yet).

3. **Watchdog self-collision.** An inline `Monitor` script that named the
   training tool processes it watched for (`mlx_lm.lora`, `run_sft.sh`, …)
   put those literal strings into the watchdog's own process argv (since
   `bash -c "<script>"` puts the whole script text there) — `pgrep -f
   mlx_lm` in `run_sft.sh`'s own competing-process safety check then
   matched the watchdog itself and refused to relaunch training after
   incident 2. Fixed by moving the watchdog to a file and invoking it as
   `bash <file>`, keeping its own argv short.

4. **Scope-cut race.** After the DPO-scope-cut decision, the orchestrator
   still already running (from before the decision) briefly attempted to
   start stock DPO at 05:30:23 before it was stopped — it failed instantly
   with `SFT adapter missing` (the stock SFT adapter had just been archived
   out of `experiments/08-nitin-new2-ds/stock`), so zero GPU time was actually spent on
   it. No corrective action needed beyond confirming the failure was
   harmless.

## 4. Why holdout (b) likely collapsed — RQ3-adjacent read

Holdout (b) is 07's own holdout, **100% py2jac-shaped content, deliberately
not carved from 08's multi-source pool** (`CONTEXT_BRIEF.md` §4). In 07's
training pool, that shape was ~100% of the data. In 08's training pool, the
comparable shape (`composer` + `step4_work`, the two source types that are
py2jac-like) is roughly 2,900 + 683 ≈ 3,580 of 12,573 SFT-train rows — well
under a third, diluted by `js2jac` (translation, largest source), `osp` and
`farm` (graph-native completions, a genuinely new task shape this phase).
The model spends most of its 8200 steps on task shapes holdout (b) doesn't
test. That is a direct, mechanical reading of the confound
`CONTEXT_BRIEF.md` names up front ("changes dataset content type AND size
simultaneously") — not a claim that Spectrum or this base checkpoint
regressed in some general sense. Holdout (a)'s smaller, cross-category
underperformance (70.5% vs 06/04-cpt-sft's ~74–75%) is consistent with the
same dilution effect at a smaller scale, since holdout (a) at least draws
from the same multi-shape distribution the model was trained on.

This reading is **qualitative**, not backed by a per-source-type accuracy
breakdown — the optional failure-analysis pipeline (`gen_eval_detail.jac`,
`grade_eval_detail.jac`, `grade_reference.jac`, workflow.md §5.4) was not run
this phase (out of scope given the DPO/stock cuts). A follow-up phase
wanting to confirm this mechanism directly should run that pipeline and
bucket accuracy by `source_type`.

## 5. Named confounds (restated, still live)

- **Pool size and content type changed simultaneously** vs 07: 2.18× the
  SFT rows, plus three genuinely new source types (js2jac, osp, farm) that
  didn't exist at 07's pin. This report cannot separate "more data" from
  "different data" as the cause of the underperformance — both point the
  same direction here, which is itself notable (07 hit "bigger pool, worse
  result" as a *count* effect; 08's quality filter targeted that
  specifically and it did not fix the pattern, so content-type dilution is
  the more likely driver, per §4).
- **Single seed, single arm.** No stock-arm number this phase to compare
  against, no DPO stage, no repeated-seed variance estimate — every number
  above is one run.
- **Holdout (b) is out-of-distribution by design**, not a bug — see §4.
- **RQ1 (Spectrum vs stock) is unanswerable this phase** — stock SFT ran but
  was never evaluated (user scope cut), and its adapter has since been
  deleted, so answering RQ1 now needs a stock retrain.

## 6. Artifacts

- `experiments/08-nitin-new2-ds/adapter` — final adapter (8200/8200 iters)
- `experiments/08-nitin-new2-ds/results` — `train.log`, `verify_layers.txt`,
  `adapter_config_rewrite.txt`, `key_assertion.txt`, `base.txt`, `final.txt`,
  `metrics_functional.jsonl`, and the watchdog PNGs (`train_loss.png`,
  `val_loss.png`, `learning_rate.png`, `tokens_per_sec.png`,
  `iters_per_sec.png`, `peak_mem.png`, `trained_tokens.png`)
- `experiments/08-nitin-new2-ds/results/holdoutB` — `adapter_config_rewrite.txt`,
  `key_assertion.txt`, `base.txt`, `final.txt`, `metrics_functional.jsonl`
- `CONTEXT_BRIEF.md` §11 — updated failure-mode table (NaN-via-truncation, this phase's own entry)

Deleted in the 2026-09-11 cleanup and no longer available: the stock SFT
adapter and results (`stock_probe/_out_of_scope_stock_sft_20260911_004125/`),
the NaN'd first stock run, the `*.pre-nanfix.bak` dataset backups, and
`orchestrator.log`.

## 7. Open items for a future phase

1. If RQ1 becomes relevant again, retrain the stock arm
   (`2-train/stock/run_sft.sh`); the stock adapter no longer exists.
2. Check `dataset/dpo/{train,valid}.jsonl` for the same prompt-overflow-NaN
   risk under `DPO_MAXLEN=512` before ever running DPO on this pool.
3. Run the optional failure-analysis pipeline (`../workflow.md` §5.4) to confirm
   §4's per-source-type dilution hypothesis directly instead of by inference.
4. Keep the laptop lid open (or attach an external display) for any future
   unattended multi-hour run — `caffeinate` alone does not prevent
   Clamshell Sleep on this machine.
