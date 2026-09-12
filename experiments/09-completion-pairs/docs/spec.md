# 09-completion-pairs: design note

## Why
08 (Spectrum LoRA SFT, Qwen3-Coder-30B-A3B 4-bit) scored 57.0% pass@1 on jac-data-gen's
hidden-test function eval: translation 83.2%, completion only 30.8%. Completion prompts show
a function prefix cut at about 30% of the body, often inside an open `{` block, or right after the
def line's `{`. 08 continues as if the open block were already closed (122/202 compile failures
leave a brace open) or closes the loop early. 08 only ever saw whole-function answers.

## Hypothesis and the one variable
Continuing training from 08's adapter on "continue this partial function" pairs, with a
replay slice of 08's own data, teaches the model to track the open blocks in a visible
prefix and close them. That should lift completion without costing translation or holdout (a).
**Only one thing changes vs 08:** the data (completion pairs + replay), trained starting from 08's adapter.
Model, layer set (`train/spectrum/spectrum_layers.json`, identical to 08's), LoRA rank/scale/
dropout, max_seq_length, mask_prompt, and seed all stay the same.

## Data (scaffold/build_completion_pairs.py, seed 42, deterministic)
- Source: 08 `dataset/sft/{train,valid}.jsonl` (read only). For each row, pick ONE
  single-line-signature `def ... {` (incl. `def:pub`, nested/method defs) with >= 6 body lines.
- Two cut shapes, one seeded draw per row:
  - **~25% after the def line** (`cut_after_def_line: true`): the prefix stops at the def line's
    `{` with no trailing newline, and the target is `"\n"` + body + `}`. This is the eval's
    zero-body shape: 143/500 test completion prefixes end this way, and it is now covered.
  - **~75% mid-body**: cut after round(U(0.2,0.6) * body) body lines (>= 1 shown, >= 2 left),
    moving to the nearest line that is not blank and not inside a string or comment. The prefix
    ends with a newline.
- Prompt = one of 6 own wordings + `\n\n` + the answer code from file start through the cut.
  Answer = the rest of that function through its closing brace, raw, no fences; any code after
  the function is dropped.
- Every pair verified: prefix + target == original code through the closing brace. 0 failures (also 0 on an independent re-check).
- Train: 12,570 rows scanned -> **3,089 pairs**. 24.8% cut after the def line; 42.5% cut inside
  an open inner block. Median prefix 9 lines, target 9 lines. 9,478 rows had no eligible def,
  mostly multi-line signatures and short functions.
  Parents: js2jac 1,721, composer 1,040, step4_work 307, osp 21.
- Valid: 2,217 scanned -> **539 pairs**. 25.6% after the def line; 38.4% inside an open block.
- Mixed release `dataset/sft/`: train = 3,089 pairs + 1,500 replay (08 train rows, messages
  unchanged) = **4,589**; valid = 539 + 500 replay = **1,039**. Seeded shuffle.
- NaN guard (prompt >= 3072 tokens): 0 rows dropped.
- Leak check vs jac-data-gen evals/function/v1 (private test+dev source_ids, denylist_ids.txt,
  1,404 ids): 0 hits in origin `#id=`, in id fields, or in raw text. 0 prompts contain the eval
  prompt sentence. 0/700 eval completion prefixes appear verbatim. (Previous build: 4 eval signature lines in composer rows, different bodies, also in 08.)

## Config (train/configs/sft_spectrum.yaml)
08's config with data/adapter -> 09, `iters 3000`, `learning_rate 1e-5`, cosine_decay warmup
300, arguments [1e-5, 3000, 1e-6], `save_every 500`, `steps_per_eval 250`. Unchanged: rank 16,
scale 2.0, dropout 0.05, num_layers 16, max_seq_length 3072, mask_prompt, grad_checkpoint, seed 42.
Dose: 3,000 iters at batch 1 = ~0.65 epoch of the 4,589-row mix (67% pairs; ~2,000 pairs seen).

## Init from 08 (train/run_sft_spectrum.sh)
`INIT_ADAPTER` (default `experiments/08-nitin-new2-ds/adapter/adapters.safetensors`) is passed as
`--resume-adapter-file` to the dry run and to the first attempt of a fresh start. After 09
writes its own adapter and progress markers, the stock resume branch takes over. INIT_ADAPTER is
read only. The runner refuses to start if it sits inside this experiment, or if its safetensors
layer set differs from spectrum_layers.json (mlx_lm loads resume weights with strict=False, so
a mismatch would be dropped silently). The refuse-to-retrain guard on 09's own adapter is unchanged.

    experiments/09-completion-pairs/train/run_sft_spectrum.sh                  # verify + dry run, then stops
    CONFIRM_FULL_RUN=1 experiments/09-completion-pairs/train/run_sft_spectrum.sh

## Success bar (all three)
1. Completion pass@1 > 08's 30.8% on the same 500 test completion tasks, with paired McNemar
   p < 0.05 vs 08's per-task results.
2. Translation within noise of 08's 83.2% (paired McNemar, not significantly worse).
3. Holdout (a) (855 shared) within noise of 08's 70.5%.

## Eval plan
- `eval/run_function_eval.sh`: 09's default stages are `reference adapter`. Base is the same
  untrained model as 08's, so reuse 08's base results. The adapter stage picks up
  `experiments/09-completion-pairs/adapter` automatically. Then run `eval/summarize_function_eval.py` / `rescore_tests.py`
  on `experiments/09-completion-pairs/results/function_v1_test`.
- `eval/eval_sft_spectrum.sh` for holdout (a). It also re-runs base plus a per-checkpoint subset
  sweep (every 500 steps); only the final full-holdout number counts for the bar.

## Risk: teaching to the benchmark format
We train on exactly the task shape the eval measures, now including its zero-body cut.
Mitigations: our own varied prompt wordings (none reuse the eval sentence), no eval data (0 leak
hits above), random cut points rather than the eval's cut rule, and the pairs come from 08's
existing training code, not the eval's Python sources. The translation and holdout (a) checks
guard against general regression.
