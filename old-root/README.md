# Jac Data Generation

Synthetic **Python→Jac conversion** data + a mini finetuning probe. All tooling
is written in Jac (`srccurrent/jacgen/`), validated against the Jac compiler.

**Start here → [`process.md`](process.md)** — set up the env and run the probe.

```bash
./setup_env.sh && source .venv/bin/activate   # venv + jaclang/mlx-lm/matplotlib
./check.sh                                     # jac check (20/20) + jac run (32/32)
./run_probe.sh <hf-model-id> <name>            # quantize → train → eval → graphs
```

## Layout

| Path | What |
|---|---|
| `srccurrent/jacgen/*.jac` | the pipeline: generate, validate, dedup, decontaminate, split, train-eval harness, dashboard |
| `dataset/` (gitignored) | generated data: ~1616 SFT, 85 DPO, 150 decontaminated eval holdout |
| `configs/lora.yaml` | LoRA SFT config (mlx-lm) |
| `run_probe.sh` / `setup_env.sh` / `check.sh` | run / setup / validate |
| `docs/` | strategy, model-testing, datagen plans |
| `context.md`, `papers/` | background (parts of `context.md` predate the current pipeline) |
