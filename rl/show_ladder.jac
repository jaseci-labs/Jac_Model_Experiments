"""Phase 3 — read the curve. Pivot results/rl_ladder.jsonl into a markdown table.

run_ladder.jac appends one row per eval, tagged r<N>/<model>/<cond>/<gen|mem>.
This collapses those into one line per (model, rung, condition) showing the
headline numbers side by side: holdout (gen) pass@1 vs the rung's own train (mem)
pass@1, plus near-pass and avg-osim diagnostics. Read the sweet spot off the gen
column — the rung where it plateaus.

  jac run rl/show_ladder.jac                 # default results/rl_ladder.jsonl
  JAC_LADDER_RESULTS=path jac run rl/show_ladder.jac

Paste the output under the matching section of docs/rl/02-results.md.

# ponytail: dynamic JSON-dict pivot -> strict `jac check` can't type it (same as
# eval_rl.jac / build_tasks.jac). Bar is `jac check -p` (parse) + it runs clean.
"""

import os;
import json;
import from pathlib { Path }

glob RUNG_ORDER: list = ["1", "3", "5", "10", "20", "all"];
glob COND_ORDER: list = ["base", "sft", "sft_grpo", "raw_grpo"];

def order_of(item: str, table: list) -> int {
    return table.index(item) if item in table else 99;
}

def pct(rec: dict, field: str) -> str {
    # rec is an eval row (or {} when that kind is missing); fields are 0-100 floats
    if field in rec { return str(rec[field]) + "%"; }
    return "·";
}

with entry {
    path = os.environ.get("JAC_LADDER_RESULTS", "results/rl_ladder.jsonl");
    if not Path(path).exists() {
        print("no rows yet at " + path + " — run the ladder first (JAC_LADDER_GO=1 jac run rl/run_ladder.jac)");
    } else {
        # cells[model|rung|cond] = {"gen": row, "mem": row}
        cells: dict = {};
        for ln in Path(path).read_text().split("\n") {
            if not str(ln).strip() { continue; }
            rec = json.loads(str(ln));
            parts: list = str(rec.get("tag", "")).split("/");
            if len(parts) != 4 { continue; }                 # r<N>/<model>/<cond>/<kind>
            (rung, model, cond, kind) = (str(parts[0]).lstrip("r"), str(parts[1]), str(parts[2]), str(parts[3]));
            key = model + "|" + rung + "|" + cond;
            slot: dict = cells.get(key, {});
            slot[kind] = rec;
            cells[key] = slot;
        }
        if not cells {
            print("found " + path + " but no tagged rows (need JAC_RL_TAG from run_ladder.jac)");
        } else {
            # explicit composite sort key: model, then ladder rung order, then condition order
            sortable: list = [];
            for k in cells.keys() {
                bits: list = str(k).split("|");
                rk = str(order_of(bits[1], RUNG_ORDER)).zfill(2);
                ck = str(order_of(bits[2], COND_ORDER));
                sortable.append((bits[0] + "|" + rk + "|" + ck, str(k)));
            }
            sortable.sort();

            print("| model | rung | cond | gen pass@1 | mem pass@1 | gen near | gen osim |");
            print("|---|---|---|---|---|---|---|");
            for pair in sortable {
                k = pair[1];
                bits = str(k).split("|");
                slot: dict = cells[k];
                g: dict = slot.get("gen", {});
                m: dict = slot.get("mem", {});
                osim = str(g["avg_osim"]) if ("avg_osim" in g) else "·";
                print("| " + bits[0] + " | " + bits[1] + " | " + bits[2] + " | "
                    + pct(g, "pass1_pct") + " | " + pct(m, "pass1_pct") + " | "
                    + pct(g, "near_pct") + " | " + osim + " |");
            }
        }
    }
}
