"""Split dataset/rl/tasks.jsonl into a FIXED holdout pool + the train candidate pool.

The ladder (rl/run_ladder.jac) needs ONE holdout that never changes rung-to-rung,
so generalization numbers are comparable. We pick `JAC_RL_HOLDOUT_N` (default 15)
tasks spread evenly across source families (sg / lib / gb / graph) by grouping
tasks by family then taking N evenly-strided indices — so the holdout is never all
one family. Everything else is the train candidate pool (`trainpool.jsonl`).

Writes:
  holdout.jsonl   fixed eval pool (eval_rl.jac)        — never trained
  trainpool.jsonl all non-holdout tasks, stable order  — pick_rung.jac slices this
  train.jsonl     == trainpool (the "all" rung default) — trainer / build_sft_gold
  valid.jsonl     == holdout (mlx_lm_lora needs a valid set in --data dir)

# ponytail: even-stride over family-grouped order, not k-fold stratification.
# Exact per-family balance if the family mix ever matters — slice by quota then.
"""

import os;
import json;
import from pathlib { Path }

def family(line: str) -> str {
    # id like "sg_follow_user_act" / "lib_lcg_rand" -> family prefix before first "_"
    rec = json.loads(line);
    meta = json.loads(str(rec["answer"]));
    return str(meta["id"]).split("_")[0];
}

with entry {
    tasks_path = "dataset/rl/tasks.jsonl";
    if not Path(tasks_path).exists() {
        print("MISSING " + tasks_path + " — run: jac run rl/build_tasks.jac");
        import sys;
        sys.exit(1);
    }
    lines: list = [str(l) for l in Path(tasks_path).read_text().split("\n") if l.strip()];
    total = len(lines);
    fam_of: list = [family(str(l)) for l in lines];
    fams_sorted: list = sorted(set(fam_of));

    # family-grouped stable order, then even-stride the holdout across it
    grouped: list = [];
    for fam in fams_sorted {
        for i in range(total) {
            if fam_of[i] == fam { grouped.append(lines[i]); }
        }
    }
    n_hold = int(os.environ.get("JAC_RL_HOLDOUT_N", "15"));
    n_hold = min(n_hold, total - 1) if total > 1 else total;   # leave >=1 to train
    step = total / n_hold if n_hold else total;
    hold_idx: set = set([int(i * step) for i in range(n_hold)]);

    holdout: list = [grouped[i] for i in range(total) if i in hold_idx];
    trainpool: list = [grouped[i] for i in range(total) if i not in hold_idx];

    Path("dataset/rl/holdout.jsonl").write_text("\n".join(holdout) + ("\n" if holdout else ""));
    Path("dataset/rl/trainpool.jsonl").write_text("\n".join(trainpool) + ("\n" if trainpool else ""));
    Path("dataset/rl/train.jsonl").write_text("\n".join(trainpool) + ("\n" if trainpool else ""));
    Path("dataset/rl/valid.jsonl").write_text("\n".join(holdout) + ("\n" if holdout else ""));

    print("=== splits (fixed holdout) ===");
    print("  tasks:          " + str(total));
    print("  holdout.jsonl:  " + str(len(holdout)) + "   (fixed, never trained)");
    print("  trainpool.jsonl:" + str(len(trainpool)) + "   (pick_rung slices this)");
    for fam in fams_sorted {
        c = len([h for h in holdout if family(str(h)) == fam]);
        if c > 0 { print("    holdout " + str(fam) + ": " + str(c)); }
    }
}
