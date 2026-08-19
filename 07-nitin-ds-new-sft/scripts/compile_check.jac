import os, json, shutil, subprocess, sys, tempfile;
import from concurrent.futures { ThreadPoolExecutor }
glob JAC = "/Users/ayush/.local/bin/jac";

# CORPUS SHAPE CHANGE vs 06 (documented in CONTEXT_BRIEF.md section 1).
# 06's corpus was a directory of loose files: data/jac_outputs/*.jac.
# 07's corpus is ONE JSONL: data/py2jac_dataset_idiomatic.jsonl, 9,367 records
# of {id:int, entrypoint:str, source:str, jac:str}.
#
# To keep every downstream stage byte-identical to 06's (pipeline.jac keys off
# i["file"], sorts with int(file[:-4]), and builds origin from the file name),
# each record is given the PSEUDO-FILENAME f"{id}.jac". Ids are ints and unique
# (verified: 9367 unique ids over 9367 rows), so the numeric sort and the id
# join both behave exactly as they did over real filenames. Nothing downstream
# had to change.
glob SRC = os.path.expanduser("~/repos/jac-data-gen/data/py2jac_dataset_idiomatic.jsonl");
glob WORKDIR = "/tmp/nitin_new_triage/work";

with entry:__main__ {
    os.makedirs(WORKDIR, exist_ok=True);
    records = {};
    for line in open(SRC) {
        line = line.strip();
        if line {
            r = json.loads(line);
            records[str(r["id"]) + ".jac"] = r["jac"];
        }
    }
    files = [str(i) + ".jac" for i in sorted([int(k[:-4]) for k in records.keys()])];
    OUT = "/tmp/nitin_new_triage/compile_results.jsonl";
    done = set();
    if os.path.exists(OUT) {
        for l in open(OUT) {
            try { done.add(json.loads(l)["file"]); }
            except Exception {  }
        }
    }
    todo = [f for f in files if f not in done];
    print("total", len(files), "todo", len(todo), flush=True);
    import threading;
    lock = threading.Lock();
    fh = open(OUT, "a");
    n = [0];
}

def work(b: str) -> None {
    d = tempfile.mkdtemp(prefix="jc_", dir=WORKDIR);
    try {
        p = os.path.join(d, "snippet.jac");
        with open(p, "w") as sf { sf.write(records[b]); }  # jac:ignore[E1053]
        try {
            r = subprocess.run([JAC, "run", "snippet.jac"], capture_output=True, text=True, timeout=60, cwd=d);
            (rc, err, out) = (r.returncode, r.stderr, r.stdout);
        } except subprocess.TimeoutExpired {
            (rc, err, out) = (124, "TIMEOUT 60s", "");
        } except Exception as e {
            (rc, err, out) = (125, str(e), "");
        }
    } finally {
        shutil.rmtree(d, ignore_errors=True);
    }
    rec = {"file": b, "rc": rc, "stderr": err[-1200:], "stdout": out[:200]};
    with lock {
        fh.write(json.dumps(rec) + "\n"); fh.flush();
        n[0] += 1;
        if n[0] % 250 == 0 { print("done", n[0], flush=True); }
    }
}

with entry:__main__ {
    with ThreadPoolExecutor(max_workers=12) as ex {
        list(ex.map(work, todo));
    }
    fh.close();
    print("ALL DONE", flush=True);
}
