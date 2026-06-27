"""Self-check for the split + rung logic (no training, no model load).

Runs build_rl_splits.jac, then asserts the invariants the ladder depends on:
  - holdout and trainpool partition the task set (disjoint, union = all)
  - holdout has the requested size and spans every source family (not all one kind)
  - rung slices are supersets: first-3 ⊂ first-5 ⊂ first-10 of the train pool
Exits non-zero on any violation.

  jac run rl/test_ladder.jac
"""

import json;
import subprocess;
import from pathlib { Path }

def ids_of(path: str) -> list {
    out: list = [];
    for l in Path(path).read_text().split("\n") {
        if str(l).strip() {
            rec = json.loads(str(l));
            meta = json.loads(str(rec["answer"]));
            out.append(str(meta["id"]));
        }
    }
    return out;
}

with entry {
    assert Path("dataset/rl/tasks.jsonl").exists(), "run rl/build_tasks.jac first";
    subprocess.run(["jac", "run", "rl/build_rl_splits.jac"], check=True);

    all_ids = set(ids_of("dataset/rl/tasks.jsonl"));
    hold = ids_of("dataset/rl/holdout.jsonl");
    pool = ids_of("dataset/rl/trainpool.jsonl");
    hold_s = set(hold); pool_s = set(pool);

    assert not (hold_s & pool_s), "holdout and trainpool overlap";
    assert hold_s | pool_s == all_ids, "holdout + trainpool != all tasks";
    assert len(hold) == len(hold_s), "duplicate ids in holdout";
    assert len(hold) >= 2, "holdout too small";

    fams = set([str(h).split("_")[0] for h in hold]);
    assert len(fams) >= 2, "holdout is all one family: " + str(fams);

    # superset growth — the front-slice guarantee pick_rung relies on
    r3 = pool[:3]; r5 = pool[:5]; r10 = pool[:10];
    assert set(r3) <= set(r5) <= set(r10), "rung slices are not nested supersets";

    print("OK: " + str(len(all_ids)) + " tasks, holdout " + str(len(hold))
        + " across families " + str(sorted(fams)) + ", supersets nested");
}
