"""Run one CPT-v2 training leg with full optimizer-state (Adam moments + LR
schedule step) persistence across leg boundaries. mlx_lm.lora's own CLI only
resumes LoRA weights (verified against its installed source -- see
03-new/docs/cpt-2/design.md section 4.2); this driver reimplements the small
model/optimizer-setup slice of mlx_lm.lora.train_model() (Apple Inc, MIT
licensed) and adds save/restore of optimizer.state around it. Model loading,
dataset loading, and the training loop itself are mlx_lm's own public API,
unmodified.

Also fixes a latent checkpoint-numbering collision: mlx_lm.tuner.trainer's
internal periodic save numbers files by its LOCAL per-invocation iteration
counter, which restarts at 1 every process launch -- harmless for a single
uninterrupted run (CPT-v1), but silently overwrites earlier legs' checkpoints
under repeated resume. Fixed by disabling that internal save (steps_per_save
set past args.iters) and doing our own single, globally-numbered save at the
true end of the leg."""
import argparse
import types
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim
import yaml
from mlx.utils import tree_flatten, tree_unflatten

from mlx_lm.lora import CONFIG_DEFAULTS
from mlx_lm.tuner.datasets import CacheDataset, load_dataset
from mlx_lm.tuner.trainer import TrainingArgs, train
from mlx_lm.tuner.utils import build_schedule, linear_to_lora_layers, print_trainable_parameters
from mlx_lm.utils import load, save_config


def save_optimizer_state(optimizer, path: Path):
    flat = dict(tree_flatten(optimizer.state))
    mx.save_safetensors(str(path), flat)


def restore_optimizer_state(optimizer, path: Path):
    flat = mx.load(str(path))
    restored = tree_unflatten(list(flat.items()))
    optimizer.state = restored


def build_args(config_path: str, overrides: dict) -> types.SimpleNamespace:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    args = dict(config)
    args.update({k: v for k, v in overrides.items() if v is not None})
    for k, v in CONFIG_DEFAULTS.items():
        args.setdefault(k, v)
    return types.SimpleNamespace(**args)


def run_leg(config_path: str, adapter_path: str, iters: int, done_steps: int,
            resume_adapter_file: str = None, resume_optimizer_file: str = None):
    args = build_args(config_path, {
        "adapter_path": adapter_path, "iters": iters,
        "resume_adapter_file": resume_adapter_file,
    })

    print("Loading pretrained model")
    model, tokenizer = load(args.model, tokenizer_config={"trust_remote_code": True})

    print("Loading datasets")
    train_set, valid_set, _ = load_dataset(args, tokenizer)

    mx.random.seed(args.seed)
    model.freeze()
    if args.fine_tune_type not in ("lora", "dora"):
        raise ValueError(f"run_cpt_leg.py only supports lora/dora, got {args.fine_tune_type}")
    linear_to_lora_layers(model, args.num_layers, args.lora_parameters,
                           use_dora=(args.fine_tune_type == "dora"))

    if args.resume_adapter_file is not None:
        print(f"Loading fine-tuned weights from {args.resume_adapter_file}")
        model.load_weights(args.resume_adapter_file, strict=False)

    print_trainable_parameters(model)

    adapter_dir = Path(args.adapter_path)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    save_config(vars(args), adapter_dir / "adapter_config.json")

    training_args = TrainingArgs(
        batch_size=args.batch_size, iters=args.iters, val_batches=args.val_batches,
        steps_per_report=args.steps_per_report, steps_per_eval=args.steps_per_eval,
        steps_per_save=args.iters + 1,  # disable mlx_lm's own local-numbered periodic save
        adapter_file=adapter_dir / "adapters.safetensors",
        max_seq_length=args.max_seq_length, grad_checkpoint=args.grad_checkpoint,
        grad_accumulation_steps=args.grad_accumulation_steps,
    )

    lr = build_schedule(args.lr_schedule) if args.lr_schedule else args.learning_rate
    opt_class = {"adam": optim.Adam, "adamw": optim.AdamW}[args.optimizer.lower()]
    opt = opt_class(learning_rate=lr, **args.optimizer_config.get(args.optimizer.lower(), {}))

    if resume_optimizer_file:
        opt.init(model.trainable_parameters())
        restore_optimizer_state(opt, Path(resume_optimizer_file))
        print(f"Restored optimizer state from {resume_optimizer_file} "
              f"(resuming at global step {int(opt.step.item())})")

    train(model=model, args=training_args, optimizer=opt,
          train_dataset=CacheDataset(train_set), val_dataset=CacheDataset(valid_set))

    final_it = done_steps + iters
    weights = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(str(adapter_dir / f"{final_it:07d}_adapters.safetensors"), weights)
    save_optimizer_state(opt, adapter_dir / f"{final_it:07d}_optimizer.safetensors")
    print(f"Leg complete: global step {final_it}, "
          f"wrote {final_it:07d}_adapters.safetensors + {final_it:07d}_optimizer.safetensors")


def main():
    ap = argparse.ArgumentParser(description="Run one CPT-v2 leg with optimizer-state persistence")
    ap.add_argument("--config", required=True)
    ap.add_argument("--adapter-path", required=True)
    ap.add_argument("--iters", type=int, required=True)
    ap.add_argument("--done-steps", type=int, required=True)
    ap.add_argument("--resume-adapter-file", default=None)
    ap.add_argument("--resume-optimizer-file", default=None)
    args = ap.parse_args()
    run_leg(args.config, args.adapter_path, args.iters, args.done_steps,
            args.resume_adapter_file, args.resume_optimizer_file)


if __name__ == "__main__":
    main()
