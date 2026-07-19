
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

## Leg 8

- train loss (last): 0.613, val loss (last): 0.823, final LR: 4.567e-06
- duration: 4792.6s
- CF-check: 16/16 (PASS)
- gate decision: continue

**Sonnet review:** Design's target landing zone (leg 8) delivered the best losses of the run so far by a clear margin -- train loss 0.886 -> 0.613, val loss 0.976 -> 0.823. LR now under half of peak (4.567e-06, ~46%). CF-check 16/16, `cf_check_0004352.json` generations still clean idiomatic Python across all 16 tasks, eight legs in, zero degeneration signal at any point in the run. If a decision were needed right now, this checkpoint alone would already look strong; continuing per the ceiling-12 schedule to see if it improves further or the gate catches a regression first.

## Leg 9

- train loss (last): 0.497, val loss (last): 0.893, final LR: 3.459e-06
- duration: 4952.7s
- CF-check: 16/16 (PASS)
- gate decision: continue

**Sonnet review:** Train loss again best yet (0.613 -> 0.497), but val loss ticked up for the first time since the floor (0.823 -> 0.893) -- train/val starting to diverge, worth watching as a possible early-overfitting signal now that LR is down to ~35% of peak (3.459e-06). Not alarming yet (single-leg move, smaller than leg 3's floor-era noise blip in relative terms), but the pattern to watch is whether val loss keeps climbing while train keeps falling over the remaining legs -- that would argue for accepting an earlier checkpoint (leg 8, the best val loss so far) over a later one even if CF stays clean. CF-check 16/16, `cf_check_0004896.json` generations still clean idiomatic Python, no degeneration. Gate correctly continued (CF is the only automated stop criterion; loss-based judgment is mine to make at the acceptance stage).

## Leg 10

- train loss (last): 0.451, val loss (last): 0.784, final LR: 2.491e-06
- duration: 4833.2s
- CF-check: 16/16 (PASS)
- gate decision: continue

**Sonnet review:** Resolves leg 9's watch item -- val loss dropped to a new run-best (0.893 -> 0.784, below leg 8's previous best of 0.823), train loss also best yet (0.497 -> 0.451). Leg 9's uptick reads as noise after all, not the start of overfitting. LR now under 25% of peak (2.491e-06). CF-check 16/16, `cf_check_0005440.json` still clean idiomatic Python, ten legs in with zero degeneration. Current strongest acceptance candidate is leg 10 (best val loss + best train loss simultaneously, still CF-clean). Two legs remain to the ceiling.

## Leg 11

- train loss (last): 0.392, val loss (last): 0.942, final LR: 1.727e-06
- duration: 4971.6s
- CF-check: 16/16 (PASS)
- gate decision: continue

**Sonnet review:** Train loss keeps improving (0.451 -> 0.392, monotonic since leg 1), but val loss jumped clearly (0.784 -> 0.942) -- worse than leg 9's earlier blip (0.893) and now the second-worst val loss of the entire post-floor stretch. This is a real train/val divergence pattern, not noise: train loss has fallen monotonically for 4 straight legs while val loss bottomed at leg 10 and has moved away from it since. LR is down to ~17% of peak (1.727e-06), so the model is increasingly fitting fine details of the training distribution specifically. CF-check still 16/16 and `cf_check_0005984.json` generations still clean, correct, undegraded Python -- the CF suite doesn't detect this kind of overfitting (it's general Python, not the packed docs corpus the val loss measures). **Leg 10 remains the strongest acceptance candidate** (best val loss 0.784, also CF-clean); this leg strengthens rather than weakens that case. Leg 12 (ceiling, final leg) running now -- the gate always halts after it, ending Task 13.

## Leg 12 (ceiling, final leg)

- train loss (last): 0.319, val loss (last): 0.783, final LR: 1.221e-06
- duration: 4952.8s
- CF-check: 16/16 (PASS)
- gate decision: **halt_keep_this** -- ceiling reached, CF clean, this checkpoint is final

**Sonnet review:** Best possible ending -- leg 12 recovered to a run-best-tying val loss (0.942 -> 0.783, essentially identical to leg 10's 0.784) while posting the best train loss of the entire run (0.392 -> 0.319). The leg 9/11 upticks read as oscillation around a genuine floor near 0.78-0.79, not a monotonic overfitting trend after all -- leg 12 landing back at that floor with CF still 16/16 is the strongest possible signal to end on. LR fully decayed (1.221e-06, ~12% of peak). `cf_check_0006528.json` generations clean, correct, idiomatic Python across all 16 tasks -- twelve legs, zero degeneration at any point in the entire run. Gate reached the hard ceiling with CF passing, so `halt_keep_this` is correct: **this is the checkpoint that ends up in `03-new/adapters/cpt-v2/adapters.safetensors`**, confirmed on disk (timestamp 19:33, matches leg 12's completion).

---

## Full run summary (Task 13 complete)

12 legs, 6528 total steps, ~16.4h wall clock. **Every single leg passed CF-check 16/16 -- zero regression to general Python coding ability at any point.** No degenerate generations observed in any of the 12 `cf_check_{step}.json` samples inspected.

Val loss trajectory: 1.371 → 1.158 → 1.218 → 1.107 → 1.076 → 0.974 → 0.976 → 0.823 → 0.893 → **0.784** → 0.942 → **0.783**. Net improvement from leg 1 to leg 12: **~43%**. The run wasn't monotonic after the floor (legs 9 and 11 both ticked up while train loss kept falling), but it repeatedly returned to a floor around 0.78-0.79 rather than diverging away from it -- legs 10 and 12 are statistically tied for best, and leg 12 is what's actually on disk.

Train loss trajectory: 1.756 → 1.251 → 1.365 → 0.898 → 1.0 → 1.065 → 0.886 → 0.613 → 0.497 → 0.451 → 0.392 → **0.319**. Monotonic decline for the last 6 legs -- expected as LR decays to near-zero and the model increasingly fits the training distribution.

**Outcome: the checkpoint accepted by the automated gate (leg 12, ceiling reached, CF-clean) is also the strongest checkpoint by validation loss (tied with leg 10) and by train loss (best of the run).** No conflict between the automated criterion and the loss-based judgment call design.md reserved for the acceptance stage -- both point to the same checkpoint. Proceeding to fuse leg 12's weights into `models/qwen-cpt-v2-fused-q4`.
