# CPT-v1 Training Results

*First real training run of attempt 03. Locked design: `design.md`. Dataset: `cpt-dataset-design.md`. Roadmap: `workflow.md` (this run is Phase 2). Live during the run in Studio's 03 CPT → TRAIN tab; this doc is the after-action record.*

## Run summary

| | |
|---|---|
| Started | 2026-07-14 04:03:33 |
| Finished | 2026-07-14 10:53:00 |
| Wall clock | 6h 49m |
| Base model | `models/qwen-q4` (Qwen3-Coder-30B-A3B-Instruct, Q4) |
| Method | LoRA, rank 16 / scale 2.0 / dropout 0.05, `num_layers` 16, `grad_checkpoint: true` |
| Data | `03-new/dataset/cpt/packed/{train,valid}.jsonl` — 862 train / 162 val windows, raw `{"text":...}` pretrain format (mlx_lm's native `TextDataset`, no reshaping) |
| Sequence length | 4096 (the packed window size) |
| Batch size | 1 (same per-step token count as the proven SFT batch=2/seq=2048 config — not a new memory regime) |
| Learning rate | 1e-5, cosine decay, warmup 259 iters, floor 1e-6 |
| Iterations | 2586 (3 epochs × 862 train windows) |
| Config | `03-new/cpt_train/config.yaml` |
| Adapter | `03-new/adapters/cpt-v1/` (27GB, gitignored — 26 checkpoints @ 100-iter cadence + final) |
| Fused model | `models/qwen-cpt-v1-fused-q4` (16GB), registered in Studio as **Qwen · CPT-v1**, chat-testable now |

Launched and monitored through `studio/cpt.sv.jac`'s `start_cpt_training`/`get_cpt_status` — the same endpoints the browser TRAIN tab calls — so the whole run was watchable live (loss/LR/tokens-sec charts, `MonoChart`) the entire 6h49m. Exit code 0, no crash, no OOM, no manual intervention needed after launch.

## Loss curve

| | Train loss | Val loss |
|---|---|---|
| Avg, first 10 evals / first 100 iters | 1.205 | 1.427 |
| Avg, last 10 evals / last 100 iters | 0.982 | 1.079 |
| Change | **−18.5%** | **−24.4%** |
| Min observed | — | 0.817 (iter 900) |
| Max observed | — | 1.742 (iter 200) |

Both curves move down together over the run, at similar magnitude — no divergence, no sign of val loss decoupling upward while train drops (the classic overfit signature). At `batch_size: 1` the raw per-iter train-loss series is single-example noise (a dense code file and a short prose paragraph have very different intrinsic perplexity, so consecutive points swing 0.8–1.4 routinely) — this was visible live during the run and tracked as expected noise, not instability, against the smoothed trend rather than any single point. Val loss (`val_batches: 20`, so each point already averages 20 examples) is calmer but still noisy at this data scale; the 10-point trailing average is the meaningful read, not the single final point (1.179 — an unremarkable point within the observed 0.82–1.74 band, not a late regression).

## Resource usage

- **Peak memory**: 29.29–30.42 GB, flat across the whole run (well inside the 48GB M5 Pro cap — no memory pressure, no swap risk observed).
- **Throughput**: 423–490 tokens/sec, avg ~455 tok/s.
- **Total tokens trained**: 10,042,071 (≈ 862 windows × 4096 tokens × 3 epochs, minus a few shorter tail windows — matches the packed corpus size, confirms all 3 epochs actually ran to completion, not truncated).

## What this run does NOT establish

Per `design.md`'s CF (catastrophic forgetting) guard and the open item already flagged in `workflow.md`: **no general-coding regression benchmark exists yet**, so this run has no automated check that CPT didn't degrade the model's non-Jac coding ability. Val loss on the CPT corpus itself improving is not evidence either way on that question — it measures fit to the CPT mix (which already includes the ~20% rehearsal slice), not held-out general-coding performance. This is an honest gap, not a silent one: building that benchmark is unstarted work, tracked in `workflow.md`'s checklist.

Similarly, this run says nothing about the actual target question — **whether CPT moved the semantic ceiling** that `01-sft-dpo`/`02-rl-grpo` both hit (SFT fixes syntax, plateaus on problem-pass ~40%; RL adds nothing on top). Loss going down confirms the model fit the corpus; it does not confirm semantic understanding improved. That's what Checkpoint 1 (semantic MCQ + human trust check, per `design.md`'s eval section) is for — not yet run.

## Next steps (per `workflow.md`)

1. **Checkpoint 1 eval** — semantic MCQ (isolated from syntax) + human-sample trust check against `qwen-cpt-v1` vs the `qwen-q4` base, to decide whether this CPT checkpoint is accepted before it gates Phase 4 (SFT/DPO redesign).
2. **CF regression check** — still unbuilt. Needs a general-coding benchmark (e.g. a small held-out non-Jac coding eval) run against both `qwen-q4` and `qwen-cpt-v1` to confirm no regression before trusting the checkpoint further.
3. If accepted: SFT/DPO redesign spec (Phase 4, currently unwritten) runs on top of `qwen-cpt-v1` instead of raw base.

## Reproducing / resuming

`03-new/cpt_train/config.yaml` is the full mlx_lm config. `studio/cpt.sv.jac`'s `start_cpt_training(name="cpt-v1")` is resume-aware (scans `03-new/adapters/cpt-v1/*_adapters.safetensors` for the latest checkpoint and runs only the remaining iters) — safe to re-invoke if a future run needs to extend this adapter. A fresh run under a new `name` starts clean.
