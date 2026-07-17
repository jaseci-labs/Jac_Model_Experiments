"""Generate config_v2_leg{N}.yaml files sharing ONE cosine schedule computed
for the epoch ceiling (design.md section 4.2/4.3), so run_cpt_leg.py's
optimizer-state persistence (Task 8) produces one genuinely continuous LR
curve across all legs, not N independent decay-to-floor cycles."""
import argparse
import json
from pathlib import Path

import yaml

RECIPE = {
    "model": "models/qwen-q4",
    "train": True,
    "fine_tune_type": "lora",
    "num_layers": 16,
    "grad_checkpoint": True,
    "lora_parameters": {"rank": 16, "scale": 2.0, "dropout": 0.05},
    "batch_size": 1,
    "learning_rate": 1.0e-5,
    "max_seq_length": 4096,
    "steps_per_eval": 50,
    "steps_per_report": 10,
    "val_batches": 20,
    "seed": 42,
}


def windows_per_epoch(manifest_path) -> int:
    manifest = json.loads(Path(manifest_path).read_text())
    return manifest["packed"]["train"]


def build_leg_configs(windows: int, ceiling_epochs: int, data_dir: str, adapter_dir: str):
    total_iters = windows * ceiling_epochs
    warmup = max(1, int(total_iters * 0.1))
    schedule = {"name": "cosine_decay", "warmup": warmup,
                "arguments": [1.0e-5, total_iters, 1.0e-6]}
    configs = []
    for _ in range(ceiling_epochs):
        cfg = dict(RECIPE)
        cfg["data"] = data_dir
        cfg["adapter_path"] = adapter_dir
        cfg["iters"] = windows
        cfg["lr_schedule"] = dict(schedule)
        configs.append(cfg)
    return configs, total_iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--ceiling-epochs", type=int, default=12)
    ap.add_argument("--data-dir", default="03-new/dataset/cpt-v2/packed")
    ap.add_argument("--adapter-dir", default="03-new/adapters/cpt-v2")
    ap.add_argument("--out-dir", default="03-new/cpt_train")
    args = ap.parse_args()

    windows = windows_per_epoch(args.manifest)
    configs, total_iters = build_leg_configs(windows, args.ceiling_epochs, args.data_dir, args.adapter_dir)
    out_dir = Path(args.out_dir)
    for i, cfg in enumerate(configs, start=1):
        out = out_dir / f"config_v2_leg{i}.yaml"
        out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"{windows} windows/epoch x {args.ceiling_epochs} epochs = {total_iters} total iters, "
          f"wrote config_v2_leg1..{args.ceiling_epochs}.yaml -> {out_dir}")


if __name__ == "__main__":
    main()
