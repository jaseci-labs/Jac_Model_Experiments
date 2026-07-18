
## Leg 1

- train loss (last): 1.756, val loss (last): 1.371, final LR: 8.328e-06
- duration: 4762.8s (~79min; longer than the ~70min dry-run extrapolation, mostly periodic-eval overhead: 11 validation passes at ~37-39s each observed in the log)
- CF-check: 16/16 (PASS)
- gate decision: continue

**Sonnet review:** No prior leg to delta against. LR at leg-end (8.328e-06) is still climbing the 652-step warmup ramp (leg 1 = steps 1-544 of that ramp, peak 1e-5 isn't reached until step 652, mid-leg-2) — expected, not a signal either way yet. Train loss noisy in the 1.1-2.2 range across the leg (normal for LoRA on raw packed text with per-document boundaries, batch_size=1), no divergence or NaN. Val loss 1.371-1.556 across the 11 eval checkpoints, trending flat-to-slightly-down, consistent with one early epoch on a domain-shifted corpus. CF-check 16/16 clean — no detectable regression to general Python coding ability yet, as expected this early (and unconditional through leg 6 regardless). **Caveat**: leg 1's CF-check ran before I added per-task result persistence (`cf_check/run_leg_cf_check.py`, landed during leg 2), so I have no sample-generation text to inspect for this leg specifically — only pass/fail counts. From leg 2 onward, `03-new/results/cpt-v2/cf_check_{step}.json` will hold full generated code/output per task for real qualitative review, not just counts. Nothing here warrants concern; floor legs (1-6) are unconditional by design regardless.

## Leg 2

- train loss (last): 1.251, val loss (last): 1.158, final LR: 9.902e-06
- duration: 5008.1s
- CF-check: 16/16 (PASS)
- gate decision: continue

**Sonnet review:** Clear improvement over leg 1 -- val loss 1.371→1.158, train loss 1.756→1.251, both moving the right direction as LR reaches its peak (9.902e-06, essentially at the 1e-5 target; warmup completed mid-leg at step 652). First leg with real generation samples to inspect (`cf_check_0001088.json`, path bug fixed this check -- see commit): all 16 CF-check tasks produce clean, correct, idiomatic Python (`is_prime` via trial division to sqrt, `gcd` via Euclidean algorithm, `caesar_cipher` with proper wraparound) -- no repetition, no gibberish, no Jac-syntax bleed into general Python generation. Model is healthy. Nothing concerning; continuing through the floor as designed.

## Leg 3

- train loss (last): 1.365, val loss (last): 1.218, final LR: 9.511e-06
- duration: 5115.6s
- CF-check: 16/16 (PASS)
- gate decision: continue

**Sonnet review:** LR now past peak (9.902e-06 leg 2 -> 9.511e-06 leg 3), confirms the schedule is in its decay phase as designed. Val loss ticked up slightly vs leg 2 (1.158 -> 1.218) and train loss too (1.251 -> 1.365) -- single-leg noise on a small eval set (20 val batches), not a trend; watching but not concerning at this magnitude, especially unconditional through leg 6. CF-check 16/16 again, sampled generations (`cf_check_0001632.json`) still clean idiomatic Python, same style/correctness as legs 1-2, no degeneration. Continuing as designed.

## Leg 4

- train loss (last): 0.898, val loss (last): 1.107, final LR: 8.846e-06
- duration: 5101.0s
- CF-check: 16/16 (PASS)
- gate decision: continue

**Sonnet review:** Leg 3's uptick resolved -- val loss 1.218 -> 1.107 (best yet, below leg 2's 1.158), train loss 1.365 -> 0.898 (biggest single-leg drop so far). Confirms leg 3 was noise, not a trend. LR now clearly decaying (9.511e-06 -> 8.846e-06). CF-check 16/16, `cf_check_0002176.json` generations still clean idiomatic Python across all 16 tasks, indistinguishable in style/correctness from legs 1-3 -- no degeneration as training progresses. Healthy trajectory through the floor.

## Leg 5

- train loss (last): 1.0, val loss (last): 1.076, final LR: 7.953e-06
- duration: 5098.0s
- CF-check: 16/16 (PASS)
- gate decision: continue

**Sonnet review:** Val loss best yet (1.107 -> 1.076), train loss essentially flat (0.898 -> 1.0, within the noise band the last 3 legs have shown). LR continuing its decay (8.846e-06 -> 7.953e-06). CF-check 16/16, `cf_check_0002720.json` generations still clean and correct across all 16 tasks -- five legs in, zero degeneration signal. This is the last unconditional floor leg; leg 6 (running now) is also floor, and leg 7 onward the stop-loss gate goes live. Nothing here changes the outlook: healthy training, no CF regression risk visible yet.

## Leg 6

- train loss (last): 1.065, val loss (last): 0.974, final LR: 6.893e-06
- duration: 4805.2s
- CF-check: 16/16 (PASS)
- gate decision: continue

**Sonnet review:** Val loss broke below 1.0 for the first time (1.076 -> 0.974), train loss flat/noisy as usual (1.0 -> 1.065). LR well into decay (7.953e-06 -> 6.893e-06, ~69% of peak). CF-check 16/16, `cf_check_0003264.json` generations still clean idiomatic Python, identical style/correctness to every prior leg -- six legs in, zero degeneration. **This was the last unconditional floor leg.** Overall floor-leg summary: val loss trajectory 1.371 -> 1.158 -> 1.218 (noise) -> 1.107 -> 1.076 -> 0.974, net improving ~29% from leg 1, CF-check 16/16 every single leg, no degenerate output at any point. Strong, clean run through the floor. Leg 7 (running now) is the first leg where the CF-check <16/16 stop-loss gate is live -- watching closely.

## Leg 7

- train loss (last): 0.886, val loss (last): 0.976, final LR: 5.738e-06
- duration: 4759.1s
- CF-check: 16/16 (PASS)
- gate decision: continue

**Sonnet review:** First leg with the stop-loss gate live -- passed clean. Train loss best yet (1.065 -> 0.886), val loss essentially flat at the floor's best level (0.974 -> 0.976). LR now under 6e-06, well into decay (~58% of peak). CF-check 16/16, `cf_check_0003808.json` still clean idiomatic Python across all 16 tasks, seven legs in with zero degeneration. Model comfortably clears the CF bar; gate correctly continued.
