"""Split the conversion tasks into a one-rung train / holdout (POC).

All conversion tasks share ONE skill (python->jac), so a simple stride split is a
valid transfer test: train on most, hold out a few UNSEEN conversions. Writes
dataset/rl_conv/{train,holdout}.jsonl.
  jac run rl/build_conv_splits.jac
"""

import from pathlib { Path }

with entry {
    lines = [l for l in Path("dataset/rl_conv/tasks.jsonl").read_text().split("\n") if l.strip()];
    hold: list = []; train: list = [];
    for (i, ln) in enumerate(lines) {
        if (i + 1) % 4 == 0 { hold.append(str(ln)); } else { train.append(str(ln)); }
    }
    Path("dataset/rl_conv/holdout.jsonl").write_text("\n".join(hold) + "\n");
    Path("dataset/rl_conv/train.jsonl").write_text("\n".join(train) + "\n");
    print("conv split: train " + str(len(train)) + " / holdout " + str(len(hold)));
}
