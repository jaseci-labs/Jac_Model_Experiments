"""Compact projections of the pending batch files, for the dispatched agents.

WHY. prepare_batches.jac writes a fully self-describing pending record: the
whole authoring `prompt` (about 1.4 KB of identical boilerplate on EVERY row)
plus every field that boilerplate was built from. That is correct for the
handoff contract -- a pending file is meant to stand alone -- but it makes the
SFT track 19 MB across 20 shards, and an agent that has to read 950 KB before
writing its first instruction wastes most of its context on repeated boilerplate.

This script writes ONE compact line per item carrying only what an author
actually needs to write the instruction:

    {batch_id, register, func_name, params, return_type, docstring, jac_code}

The boilerplate rules live once in the agent's own dispatch prompt instead of
being repeated 339 times on disk. Roughly halves the bytes an agent must read,
and lets it work in bounded slices via Read(offset=, limit=) since every record
is exactly one line.

This is a READ-CONVENIENCE artifact only. It is derived from the pending files,
never the other way round, and collect.jac still joins responses against the
REAL pending files by batch_id -- so a stale or missing compact file cannot
corrupt the release, it can only inconvenience an agent.

Run:
    jac run model-experiments/07-nitin-ds-new-sft/scripts/make_compact_batches.jac
"""

import os, json;
import from pathlib { Path }

glob EXP_ROOT: str = str(Path(str(__file__)).resolve().parent.parent);
glob RAW_ROOT: str = EXP_ROOT + "/dataset/raw_output";
glob TRACKS: list[str] = ["sft", "holdout_eval", "dpo"];

with entry:__main__ {
    grand: int = 0;
    for track in TRACKS {
        tdir = RAW_ROOT + "/" + track;
        if not os.path.isdir(tdir) {
            print("skip (no dir): " + track);
            continue;
        }
        names = sorted([f for f in os.listdir(tdir) if f.startswith("pending_batch_") and f.endswith(".jsonl")]);
        for name in names {
            rows = [];
            for line in open(tdir + "/" + name) {
                line = line.strip();
                if line {
                    r = json.loads(line);
                    rows.append({
                        "batch_id": r["batch_id"],
                        "register": r.get("register", ""),
                        "func_name": r["func_name"],
                        "params": r["params"],
                        "return_type": r["return_type"],
                        "docstring": r["docstring"],
                        "jac_code": r["jac_code"]
                    });
                }
            }
            out = tdir + "/compact_" + name;
            with open(out, "w") as f {
                for r2 in rows {
                    f.write(json.dumps(r2) + "\n");
                }
            }
            grand += len(rows);
            print("  " + out + "  (" + str(len(rows)) + " items)");
        }
    }
    print("");
    print("total compact items written: " + str(grand));
}
