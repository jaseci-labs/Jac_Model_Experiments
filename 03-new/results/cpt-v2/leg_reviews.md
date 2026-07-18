
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
