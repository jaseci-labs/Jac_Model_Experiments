# Jac Model Studio

Two independent projects share this repo for now. `model-experiments/` will move to its own repo.

| Folder | What |
|---|---|
| [`jms/`](jms/README.md) | Jac ML Studio, the fullstack Jac app for dataset, train, and eval workflows |
| [`model-experiments/`](model-experiments/README.md) | Fine-tuning experiments for Qwen3-Coder-30B-A3B on Jac. The SFT playbook is in `model-experiments/docs/PLAYBOOK.md` |

Neither folder imports code from the other. JMS reaches experiment artifacts only through `jms/studio.workspace.toml`.
