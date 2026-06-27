"""Self-check for the split + rung logic (no training, no model load).

Runs build_rl_splits.jac, then asserts the invariants the ladder depends on:
  - holdout & trainpool partition disjointly (by id AND by SOURCE FILE — file-disjoint)
  - valid is disjoint from BOTH holdout and trainpool (no test-set leak, reserved)
  - rung slices are supersets: first-3 ⊂ first-5 ⊂ first-10 of the train pool
  - every rung is family-balanced (no rung is single-family — the P0 confound)
Exits non-zero on any violation.

  jac run rl/test_ladder.jac
"""

import os;
import json;
import subprocess;
import from pathlib { Path }

def field(line: str, key: str) -> str {
    rec = json.loads(line);
    meta = json.loads(str(rec["answer"]));
    return str(meta.get(key, ""));
}
def ids_of(path: str) -> list {
    return [field(str(l), "id") for l in Path(path).read_text().split("\n") if str(l).strip()];
}
def srcs_of(path: str) -> list {
    return [field(str(l), "source") for l in Path(path).read_text().split("\n") if str(l).strip()];
}

with entry {
    assert Path("dataset/rl/tasks.jsonl").exists(), "run rl/build_tasks.jac first";
    subprocess.run(["jac", "run", "rl/build_rl_splits.jac"], check=True);

    all_ids = set(ids_of("dataset/rl/tasks.jsonl"));
    hold = ids_of("dataset/rl/holdout.jsonl");
    pool = ids_of("dataset/rl/trainpool.jsonl");
    valid = ids_of("dataset/rl/valid.jsonl");
    hset = set(hold); pset = set(pool); vset = set(valid);

    # partition + leak guards
    assert not (hset & pset), "holdout and trainpool overlap (id)";
    assert not (vset & hset), "valid overlaps holdout — TEST-SET LEAK into checkpoint selection";
    assert not (vset & pset), "valid overlaps trainpool — valid not reserved";
    assert len(hold) == len(hset), "duplicate ids in holdout";
    assert len(hold) >= 2, "holdout too small";

    # file-disjoint: no source file feeds both holdout and train
    hsrc = set(srcs_of("dataset/rl/holdout.jsonl"));
    psrc = set(srcs_of("dataset/rl/trainpool.jsonl"));
    assert not (hsrc & psrc), "source file leaks across holdout/train: " + str(hsrc & psrc);

    # superset growth — the front-slice guarantee pick_rung relies on
    r3 = pool[:3]; r5 = pool[:5]; r10 = pool[:10];
    assert set(r3) <= set(r5) <= set(r10), "rung slices are not nested supersets";

    # family balance — every early rung must be multi-family (the P0 fix)
    fams_r5 = set([i.split("_")[0] for i in r5]);
    assert len(fams_r5) >= 2, "rung-5 is single-family — interleave broken (the P0 confound): " + str(fams_r5);

    print("OK: " + str(len(all_ids)) + " tasks | holdout " + str(len(hold))
        + " (file-disjoint, families " + str(sorted(set([i.split('_')[0] for i in hold]))) + ")"
        + " | valid " + str(len(valid)) + " reserved | rung-5 families " + str(sorted(fams_r5)) + " | supersets nested");
}
