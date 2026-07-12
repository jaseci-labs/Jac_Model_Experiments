# RL — published graphs (CORRECTED)

All corrected (fixed eval+reward). Old broken-eval graphs deleted (~3.5× undercount).
Full write-up: root `02-rl-grpo/RL_FINDINGS.md`. Regenerate: `python3 02-rl-grpo/rl/make_summary.py && python3 02-rl-grpo/rl/make_graphs.py`.

| graph | what it shows |
|---|---|
| `corrected_journey.png` | **the story** — a measurement bug hid an 11%→94% Jac generator |
| `corrected_ladder.png` | SFT ladder, pure-fn holdout — greedy 39→61%, GRPO≈SFT |
| `corrected_all_holdouts.png` | greedy vs best-of-k, base vs SFT, across **5 holdouts** (pure-fn/graph/conversion/clean/big+fresh) |
| `corrected_kscale.png` | sampling budget → accuracy (greedy 39 → k=8 72 → k=32 89%) |
| `corrected_followup.png` | clean n=16 (94% ceiling) + bigger n=32 (SFT lift holds) |
| `studio-rl-full.png` | the live Studio RL section (all corrected charts + GENERATE panel) |

**Headline:** best-of-k + jac-compiler-verifier ships **~94% on meaningful tasks**;
SFT lifts greedy 39→61% (holds at n=32, generalizes to fresh tasks); conversion peaks 82%;
free-form NL is the one real gap. Deployable generator: `02-rl-grpo/rl/generate.py`.
