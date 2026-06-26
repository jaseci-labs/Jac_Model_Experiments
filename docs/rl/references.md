# Literature — does GRPO/RLVR move a Qwen-class base?

Papers that bear on the ladder's bet. Short version: the field largely agrees with our
weekend null — **on Qwen, GRPO/RLVR mostly sharpens what the base already does; it rarely
adds capability the base can't sample.** A minority show real expansion, but only with
scale/diversity/compute we don't have on 48GB. Mapped to [strat.md](strat.md) hypotheses.

## The "GRPO won't add much" side (supports H3, and explains the weekend pass@1==pass@8)

- **Yue et al., "Does RL Really Incentivize Reasoning Capacity Beyond the Base Model?"**
  NeurIPS'25 · [2504.13837](https://arxiv.org/abs/2504.13837) · [repo](https://github.com/LeapLabTHU/limit-of-RLVR).
  **The most directly relevant.** RLVR (incl. GRPO) beats base at **pass@1** but base **matches or
  beats** the RL model at **large pass@k** — RL narrows the output distribution toward already-
  sample-able correct paths, it doesn't widen the reasoning boundary. **This is exactly our
  weekend result** (pass@1 == pass@8 == 14.3%): nothing new to amplify. Their contrast: *distillation*
  (≈ our SFT warm-start) genuinely adds new patterns. → backs **H3** and the "SFT is the mover" thesis.

- **Shao/Li et al., "Spurious Rewards: Rethinking Training Signals in RLVR"**
  [2506.10947](https://arxiv.org/abs/2506.10947) · [interconnects writeup](https://www.interconnects.ai/p/reinforcement-learning-with-random).
  GRPO on **Qwen2.5-Math** gains ~21pts on MATH-500 with **random/spurious rewards** (vs 29 with real) —
  because GRPO's clip bias amplifies pretrained **code-reasoning priors**. **Crucial caveat: this is
  Qwen-specific — it does NOT replicate on Llama/OLMo.** So a Qwen gain may be eliciting priors, not
  learning the reward. → our **two-model probe (jac-trained vs fresh qwen3coder) is the right control**;
  if both move identically regardless of jac-SFT, suspect prior-elicitation not learning.

- **"RLVR makes models faster, not smarter"** — [promptfoo summary](https://www.promptfoo.dev/blog/rlvr-explained/).
  Practitioner-level restatement of the same: efficiency gain, not capability gain.

## The σ=0 / vanishing-advantage mechanism (validates our `body_sim` scar)

- **DAPO** · [2503.14476](https://arxiv.org/abs/2503.14476) · [site](https://dapo-sia.github.io/).
  Names our exact failure: when every rollout in a group is all-correct or all-wrong, group variance →
  0 → **zero gradient**. DAPO's fix = **dynamic sampling** (oversample, drop 0/1-accuracy groups, refill
  with mixed-quality). We solve the same problem differently: a **dense `body_sim` term** that gives
  within-group variance even at 0% pass. → validates **carried scar #2**. *Possible upgrade for the
  raw-base-GRPO control:* add DAPO dynamic-sampling so the σ=0 control fails for the right reason.
- **EBPO** ([2602.05165](https://arxiv.org/html/2602.05165)), **"Gradient Starvation in Binary-Reward
  GRPO"** ([2605.07689](https://arxiv.org/html/2605.07689)) — more variants of the same saturated-regime
  zero-gradient diagnosis + fixes. Confirms it's a known structural GRPO flaw, not our harness bug.

## The "RL CAN expand" counter-side (the falsifier for H3 — what would prove us wrong)

- **ProRL** · [2505.24864](https://arxiv.org/html/2505.24864v1).
  *Prolonged* RL (much longer training, KL control, diverse tasks) **does** push reasoning past the base —
  pass@k boundary widens, not just pass@1. The condition we likely can't meet on 48GB/LoRA: needs scale,
  many tasks, full-ish training. Reading: if any rung shows real holdout lift from GRPO, ProRL is why.
- **"RLVR Implicitly Incentivizes Correct Reasoning in Base LLMs"** · [2506.14245](https://arxiv.org/html/2506.14245v2).
  Argues RLVR does instill correct-reasoning signal (nuance against Yue). Healthy adversarial read.

## Takeaway for the ladder

The weekend null is the *expected* outcome per Yue + Spurious Rewards, not a harness failure — at our
scale (30B, LoRA, ~50 tasks, exact-stdout) GRPO has little headroom to amplify. The ladder's value is
**measuring the SFT curve** (the proven mover) and pinning GRPO's marginal lift at exactly 0 with the
controls. To give GRPO its best honest shot (and possibly falsify H3), the literature points to: more
tasks (ProRL's diversity), DAPO dynamic sampling on the control, and reading **pass@k not pass@1** (Yue).
