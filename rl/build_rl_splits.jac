"""Split dataset/rl/tasks.jsonl into a sound train / holdout / valid partition.

Three problems in the old stride split this fixes:
  1. FILE-DISJOINT holdout — the old split strided by family (id prefix), so the
     same source .jac fed BOTH train and holdout (structural leakage: a held-out
     walker shares its skeleton with a trained sibling). Now whole source FILES are
     held out; a held-out file contributes nothing to train.
  2. FAMILY-INTERLEAVED trainpool — the old pool was family-ordered, so pick_rung's
     front-slice gave rungs 1..20 ZERO social_graph tasks while the holdout was 47%
     social_graph: the "more tasks" curve was really a "which family" curve. The
     pool is now round-robin interleaved, so every prefix-N is family-balanced AND
     still a strict superset of the previous rung.
  3. valid != holdout — the old split wrote valid.jsonl == holdout.jsonl, so the
     GRPO/SFT trainer selected checkpoints on the TEST set. valid is now carved from
     the trainpool tail, never from holdout.

Plus a decontam pass: drop any holdout task whose gold body is >=0.85 difflib-similar
to ANY train body (catches cross-file near-duplicates a file split alone misses).

Env: JAC_RL_HOLDOUT_N (15) JAC_RL_VALID_N (4) JAC_RL_DECONTAM (0.85).

# ponytail: greedy whole-file holdout + round-robin interleave + difflib decontam.
# On the current sg-dominated 56-task corpus this yields a SMALL, sg-light holdout
# (social_graph is one 32-task file > the per-file cap, so it stays in train) — that
# is expected until the corpus is mined wider; the WARN lines below flag it.
"""

import os;
import json;
import difflib;
import from pathlib { Path }

def meta_of(line: str) -> dict {
    rec = json.loads(line);
    return json.loads(str(rec["answer"]));
}
def family_of(line: str) -> str { return str(meta_of(line)["id"]).split("_")[0]; }
def source_of(line: str) -> str { return str(meta_of(line).get("source", "unknown")); }
def refbody_of(line: str) -> str {
    p = "dataset/rl/refbodies/" + str(meta_of(line)["id"]) + ".txt";
    return Path(p).read_text().strip() if os.path.exists(p) else "";
}

with entry {
    tasks_path = "dataset/rl/tasks.jsonl";
    if not Path(tasks_path).exists() {
        print("MISSING " + tasks_path + " — run: jac run rl/build_tasks.jac");
        import sys;
        sys.exit(1);
    }
    n_hold = int(os.environ.get("JAC_RL_HOLDOUT_N", "15"));
    n_valid = int(os.environ.get("JAC_RL_VALID_N", "4"));
    thresh = float(os.environ.get("JAC_RL_DECONTAM", "0.85"));
    lines: list = [str(l) for l in Path(tasks_path).read_text().split("\n") if l.strip()];
    total = len(lines);

    # --- group by source file ---
    by_src: dict = {};
    for ln in lines {
        s = source_of(ln);
        grp: list = by_src.get(s, []);
        grp.append(ln);
        by_src[s] = grp;
    }
    cap = max(1, int(n_hold * 0.5) + 1);   # no single file may exceed ~half the holdout

    # --- file-disjoint holdout: add WHOLE small files until ~n_hold ---
    sized: list = [(len(by_src[s]), str(s)) for s in by_src.keys()];
    sized.sort();                           # small files first -> big idiom files stay in train
    hold_files: list = [];
    held = 0;
    for pair in sized {
        (sz, s) = (pair[0], pair[1]);
        if sz > cap { continue; }            # too big to hold whole -> trains
        if held >= n_hold { break; }
        hold_files.append(s);
        held += sz;
    }
    holdout_raw: list = [];
    for s in hold_files { holdout_raw += by_src[s]; }
    trainpool_all: list = [ln for ln in lines if source_of(ln) not in hold_files];

    # --- decontam: drop holdout near-duplicates of any train body ---
    train_bodies: list = [refbody_of(ln) for ln in trainpool_all];
    holdout: list = [];
    dropped = 0;
    for ln in holdout_raw {
        hb = refbody_of(ln);
        dup = False;
        for tb in train_bodies {
            if difflib.SequenceMatcher(None, hb, tb).ratio() >= thresh { dup = True; break; }
        }
        if dup { dropped += 1; } else { holdout.append(ln); }
    }

    # --- family-interleave the trainpool (round-robin) so prefix-N is balanced ---
    fam_groups: dict = {};
    for ln in trainpool_all {
        f = family_of(ln);
        g: list = fam_groups.get(f, []);
        g.append(ln);
        fam_groups[f] = g;
    }
    fam_keys: list = sorted(fam_groups.keys());
    interleaved: list = [];
    idx = 0;
    while len(interleaved) < len(trainpool_all) {
        for fk in fam_keys {
            g2: list = fam_groups[fk];
            if idx < len(g2) { interleaved.append(g2[idx]); }
        }
        idx += 1;
    }

    # --- carve valid from the trainpool TAIL (never the holdout) ---
    n_valid = min(n_valid, max(0, len(interleaved) - 1));
    cut = len(interleaved) - n_valid;
    valid: list = interleaved[cut:] if n_valid else [];
    trainpool: list = interleaved[:cut];

    def dump(p: str, rows: list) {
        Path(p).write_text("\n".join(rows) + ("\n" if rows else ""));
    }
    dump("dataset/rl/holdout.jsonl", holdout);
    dump("dataset/rl/trainpool.jsonl", trainpool);
    dump("dataset/rl/train.jsonl", trainpool);   # default rung = "all"
    dump("dataset/rl/valid.jsonl", valid);       # trainer internal val — DISJOINT from holdout

    # --- report + guards ---
    hf: dict = {}; hs: dict = {};
    for ln in holdout {
        hf[family_of(ln)] = hf.get(family_of(ln), 0) + 1;
        hs[source_of(ln)] = hs.get(source_of(ln), 0) + 1;
    }
    print("=== splits (file-disjoint holdout, interleaved pool) ===");
    print("  tasks:          " + str(total));
    print("  holdout:        " + str(len(holdout)) + "   files=" + json.dumps(sorted(hold_files)));
    print("    by family:    " + json.dumps(hf));
    print("    by source:    " + json.dumps(hs));
    print("  decontam dropped:" + str(dropped) + "   (>= " + str(thresh) + " difflib to a train body)");
    print("  trainpool:      " + str(len(trainpool)) + "   (pick_rung slices this)");
    print("  valid:          " + str(len(valid)) + "   (trainer val; disjoint from holdout)");
    if len(holdout) < n_hold {
        print("  WARN: holdout " + str(len(holdout)) + " < target " + str(n_hold) + " — corpus too small/concentrated; mine more this_is_jac files.");
    }
    if len(hf) < 2 {
        print("  WARN: holdout spans <2 families — generalization signal weak. Mine wider before trusting it.");
    }
}
