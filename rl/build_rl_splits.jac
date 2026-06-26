"""Split dataset/rl/tasks.jsonl into GRPO train / valid + an eval holdout.

`mlx_lm_lora`'s GRPO trainer reads `--data <dir>` expecting `train.jsonl` and
`valid.jsonl` (same convention as the DPO stage's `dataset/mlx_dpo`). We also
write `holdout.jsonl` (== valid here) for our own post-train scorer
`rl/eval_rl.jac`.

Holdout is chosen at the task level by a deterministic stride (every Nth task,
`JAC_RL_HOLDOUT_EVERY`, default 4) so train and holdout never share a task. At
scale, prefer holding out whole *driver source files* (e.g. all guestbook tasks)
and run a 14-gram decontam pass (sft_dpo/jacgen/decontam.jac) — noted for the
bulk-authoring follow-up; with a handful of disjoint, self-contained tasks the
stride split is already non-overlapping.
"""

import os;
import json;
import from pathlib { Path }

with entry {
    tasks_path = "dataset/rl/tasks.jsonl";
    if not Path(tasks_path).exists() {
        print("MISSING " + tasks_path + " — run: jac run rl/build_tasks.jac");
        import sys;
        sys.exit(1);
    }
    every = int(os.environ.get("JAC_RL_HOLDOUT_EVERY", "4"));
    lines = [l for l in Path(tasks_path).read_text().split("\n") if l.strip()];

    train = [];
    holdout = [];
    for (i, ln) in enumerate(lines) {
        # stride pick for holdout; guarantee at least one of each when tiny
        if (i + 1) % every == 0 {
            holdout.append(ln);
        } else {
            train.append(ln);
        }
    }
    if not holdout and train {
        holdout.append(train.pop());   # tiny set: ensure a non-empty valid/holdout
    }
    if not train and holdout {
        train.append(holdout[0]);      # trainer needs a non-empty train set
    }

    Path("dataset/rl/train.jsonl").write_text("\n".join(train) + ("\n" if train else ""));
    Path("dataset/rl/valid.jsonl").write_text("\n".join(holdout) + ("\n" if holdout else ""));
    Path("dataset/rl/holdout.jsonl").write_text("\n".join(holdout) + ("\n" if holdout else ""));

    print("=== splits ===");
    print("  train.jsonl:   " + str(len(train)));
    print("  valid.jsonl:   " + str(len(holdout)) + "   (trainer eval)");
    print("  holdout.jsonl: " + str(len(holdout)) + "   (eval_rl.jac)");
}
