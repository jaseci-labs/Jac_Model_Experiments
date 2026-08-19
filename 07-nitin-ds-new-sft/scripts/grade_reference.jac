"""Ceiling probe: run the holdout's OWN reference completion through the exact
same gate the model's output is judged by.

`eval_functional.jac` grades a generation by extracting its fenced Jac and
running it as a standalone `snippet.jac` under `jac run`. Nothing in the
harness ever checks whether the row's *reference* answer survives that same
treatment. If it does not, the row cannot be passed by any model, correct or
not -- it is an unpassable item, and its contribution to the failure count is
a property of the eval set, not of the model.

This script measures that ceiling per (category, gate_class). Same
extract_jac, same run_jac, same binary as grade_eval_detail.jac.

    HOLDOUT=<valid.jsonl> OUT=<jsonl> [WORKERS=8] \
      jac run model-experiments/07-nitin-ds-new-sft/scripts/grade_reference.jac
"""

import json;
import os;
import sys;
import time;
import from concurrent.futures { ThreadPoolExecutor }
import from pathlib { Path }
import from typing { cast }

with entry {
    sys.path.insert(0, str(Path(str(__file__)).resolve().parent));
    import from grade_eval_detail { CODE_GATES, extract_jac, run_jac }  # noqa: E402
}

glob WORKERS: int = int(os.environ.get("WORKERS", "8"));


def probe(rec: dict) -> dict {
    gate = rec.get("gate_class");
    out = {
        "id": rec.get("id"),
        "category": rec.get("category"),
        "gate_class": gate,
        "task_type": rec.get("task_type")
    };
    if gate not in CODE_GATES {
        out["graded"] = False;
        return out;
    }
    msgs = rec.get("messages", []);
    ref = str(msgs[1]["content"]) if len(cast(list, msgs)) > 1 else "";
    code = extract_jac(ref);
    (rc, so, se) = run_jac(code);
    out["graded"] = True;
    out["ref_pass"] = rc == 0;
    out["exit_code"] = rc;
    if rc != 0 {
        out["reference"] = ref;
        out["extracted"] = code;
        out["stderr"] = se[-4000:];
    }
    return out;
}


def main() -> int {
    src = Path(os.environ["HOLDOUT"]);
    dst = Path(os.environ["OUT"]);
    rows = [json.loads(l) for l in src.read_text().split("\n") if l.strip()];
    print(f"[ref] {src.name}: {len(rows)} rows, {WORKERS} workers", flush=True);
    t0 = time.time();
    with ThreadPoolExecutor(max_workers=WORKERS) as ex {
        results = list(ex.map(probe, rows));
    }
    dst.parent.mkdir(parents=True, exist_ok=True);
    with dst.open("w") as fh {
        for r in results {
            fh.write(json.dumps(r) + "\n");
        }
    }
    graded = [r for r in results if r["graded"]];
    ok = sum(1 for r in graded if r["ref_pass"]);
    print(
        f"[ref] {dst}: graded={len(graded)} reference-passes={ok} "
        f"({100.0 * ok / max(1, len(graded)):.1f}%) in {time.time() - t0:.0f}s",
        flush=True
    );
    return 0;
}


with entry:__main__ {
    sys.exit(main());
}
