# Jac Data Generation

Synthetic **Python→Jac conversion** data + a mini finetuning probe. All tooling
is written in Jac (`srccurrent/jacgen/`), validated against the Jac compiler.

**Start here → [`process.md`](process.md)** — set up the env and run the probe.
**Full handoff → [`docs/modeltesting/HANDOFF.md`](docs/modeltesting/HANDOFF.md)** —
architecture, every module, all gotchas, dataset rebuild order.

```bash
./setup_env.sh && source .venv/bin/activate   # venv + jaclang/mlx-lm/matplotlib
./check.sh                                     # jac check (19) + parse + non-destructive jac run audit
./run_probe.sh Qwen/Qwen3-Coder-30B-A3B-Instruct qwen   # quantize → train → eval → graphs
```

## Layout

| Path | What |
|---|---|
| `srccurrent/jacgen/*.jac` | the pipeline: generate, validate, dedup, decontaminate, split, train-eval harness, dashboard |
| `dataset/` (gitignored) | generated data: 1616 SFT (116 idiomatic + 1500 transpile), 60 DPO, 150 decontaminated eval holdout |
| `configs/lora.yaml` | LoRA SFT config (mlx-lm) |
| `run_probe.sh` / `setup_env.sh` / `check.sh` | run / setup / validate |
| `docs/` | strategy, model-testing, datagen plans |
| `context.md`, `papers/` | background (parts of `context.md` predate the current pipeline) |
