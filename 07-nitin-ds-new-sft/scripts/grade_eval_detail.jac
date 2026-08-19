"""Per-row grading for failure analysis (phase 2 of 2).

Reads a `*.gen.jsonl` produced by `gen_eval_detail.jac` and applies the SAME
gate `eval_functional.jac` applies:

  - rows whose `gate_class` is not in {behavioral, compile_project,
    compile_only} are prose/ungraded -- counted, never silently dropped;
  - every other row: `extract_jac(generated)` (byte-identical logic to the
    harness), write to `snippet.jac` in a fresh tempdir, `jac run` with
    cwd=tempdir, 30s timeout, pass iff exit 0.

The ONE difference from the harness: rows are graded across a thread pool
instead of serially. Each grade is fully isolated (its own tempdir, its own
jac persistent root store -- that isolation is why the harness uses cwd=d in
the first place), so parallelism cannot change any individual verdict; it only
turns ~20 minutes of serial subprocess wall-clock into ~3. The same 12-way
pattern is what `scripts/compile_check.jac` used for the 7,627-file corpus
compile gate.

Unlike the harness, the exit code, stdout and stderr of every FAILING row are
persisted -- that error text is the entire point of this exercise.

    IN=<gen.jsonl> OUT=<graded.jsonl> [WORKERS=8] [JAC_BIN=...] \
      jac run model-experiments/07-nitin-ds-new-sft/scripts/grade_eval_detail.jac
"""

import json;
import os;
import subprocess;
import sys;
import tempfile;
import time;
import from concurrent.futures { ThreadPoolExecutor }
import from pathlib { Path }
import from typing { cast }

glob ROOT: Path = Path(str(__file__)).resolve().parents[3];
glob CODE_GATES: set[str] = {"behavioral", "compile_project", "compile_only"};
glob JAC_BIN: str = os.environ.get("JAC_BIN", str(ROOT / ".venv/bin/jac"));
glob WORKERS: int = int(os.environ.get("WORKERS", "8"));
glob TIMEOUT: int = int(os.environ.get("TIMEOUT", "30"));


"""Byte-identical to eval_functional.jac's extract_jac."""
def extract_jac(output: str) -> str {
    body = output;
    if "```jac" in output {
        body = output.split("```jac", 1)[1].split("```", 1)[0];
    } elif "```" in output {
        body = output.split("```", 1)[1].rsplit("```", 1)[0];
    }
    return body.strip();
}


"""Byte-identical to eval_functional.jac's run_jac, bar the binary path."""
def run_jac(jac_code: str, timeout: int = TIMEOUT) -> tuple[int, str, str] {
    with tempfile.TemporaryDirectory() as d {
        f = Path(cast(str, d)) / "snippet.jac";
        f.write_text(jac_code);
        try {
            p = subprocess.run(
                [JAC_BIN, "run", str(f)],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cast(str, d)
            );
            return (p.returncode, p.stdout, p.stderr);
        } except (subprocess.SubprocessError, OSError) as e {
            return (124, "", str(e));
        }
    }
}


def grade(rec: dict) -> dict {
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
    code = extract_jac(cast(str, rec.get("generated", "")));
    (rc, so, se) = run_jac(code);
    out["graded"] = True;
    out["pass"] = rc == 0;
    out["exit_code"] = rc;
    out["extracted_len"] = len(code);
    out["gen_len"] = len(cast(str, rec.get("generated", "")));
    out["fenced"] = "```" in rec.get("generated", "");
    if rc != 0 {
        out["prompt"] = rec.get("prompt", "");
        out["generated"] = rec.get("generated", "");
        out["extracted"] = code;
        out["reference"] = rec.get("reference", "");
        out["stdout"] = so[-4000:];
        out["stderr"] = se[-6000:];
    }
    return out;
}


def main() -> int {
    src = Path(os.environ["IN"]);
    dst = Path(os.environ["OUT"]);
    rows = [json.loads(l) for l in src.read_text().split("\n") if l.strip()];
    print(f"[grade] {src.name}: {len(rows)} rows, {WORKERS} workers, jac={JAC_BIN}", flush=True);

    t0 = time.time();
    results: list = [None] * len(rows);
    done = 0;
    with ThreadPoolExecutor(max_workers=WORKERS) as ex {
        futs = {ex.submit(grade, r): i for (i, r) in enumerate(rows)};
        for fut in futs {
        }
        for (fut, i) in futs.items() {
            results[i] = fut.result();
            done += 1;
            if done % 100 == 0 {
                el = time.time() - t0;
                print(f"[grade] {done}/{len(rows)}  {el:.0f}s", flush=True);
            }
        }
    }

    dst.parent.mkdir(parents=True, exist_ok=True);
    with dst.open("w") as fh {
        for r in results {
            fh.write(json.dumps(r) + "\n");
        }
    }

    graded = [r for r in results if r["graded"]];
    passed = sum(1 for r in graded if r["pass"]);
    print(
        f"[grade] {dst}: graded={len(graded)} ungraded={len(results) - len(graded)} "
        f"pass={passed} ({100.0 * passed / max(1, len(graded)):.1f}%) "
        f"in {time.time() - t0:.0f}s",
        flush=True
    );
    return 0;
}


with entry:__main__ {
    sys.exit(main());
}
