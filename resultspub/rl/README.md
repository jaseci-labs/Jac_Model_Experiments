# RL — published graphs (CORRECTED)

All graphs here are from the **corrected** pipeline (fixed eval+reward). The old
broken-eval graphs were deleted — they undercounted accuracy ~3.5× (the body extractor
grabbed the driver docstring). Full analysis: root `RL_FINDINGS.md`.

| graph | what it shows |
|---|---|
| `corrected_ladder.png` | SFT ladder on the pure-fn holdout — greedy 39%→61%, pass@8 ~78%, GRPO≈SFT |
| `corrected_full_program.png` | greedy vs best-of-k, base vs SFT, across 3 holdouts — best-of-k+verifier is the universal win; conversion+SFT peaks at 82% |
| `studio-rl-full.png` | the live Studio RL section — invalidation banner + the shipped GENERATE JAC (best-of-k) panel |

**Headline:** the model is capable; **best-of-k + the jac compiler as verifier** ships
~78% (pure-fn) / 65% (graph) / 89% (k=32); SFT and the conversion framing stack on top
(peak 82%). Regenerate: `python3 rl/make_graphs.py`. Deployable generator: `rl/generate.py`
(shipped into the Studio RL section).
