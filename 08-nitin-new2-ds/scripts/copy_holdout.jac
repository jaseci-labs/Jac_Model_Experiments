"""Stage 0b for 08-nitin-new2-ds: import 07's frozen holdout VERBATIM and
independently re-verify that `dataset/candidate_pool.jsonl` does not leak it.

WHY VERBATIM. 07 (`../../07-nitin-ds-new-sft`) carved an 855-row holdout with
seed 42 and froze it before any generation. 08 trains a different (much larger,
seven-corpus) dataset and must be scored on the SAME held-out functions for the
07-vs-08 comparison to mean anything. So the holdout is COPIED, not re-carved:

  nitin_holdout.jsonl       raw carve  (jac_code + provenance, 855 rows)
  nitin_holdout_eval.jsonl  chat form  (messages, 855 rows) -- what eval reads

WHY THE LEAK CHECK IS REPEATED HERE. pipeline.jac already gates every source
against the holdout (numeric id + normalized-target hash) and asserts zero leaks
before writing. That assertion runs against the in-memory pool. This script
re-runs the same check against the pool AS WRITTEN TO DISK, from the holdout
files AS COPIED INTO 08 -- so a truncated write, a partial re-run, or a stale
candidate_pool.jsonl cannot pass silently. It is a cheap independent witness,
not a duplicate of the pipeline's own gate.

THREE COLLISION SURFACES ARE CHECKED (a pool row leaks if ANY hits):
  1. numeric HF-row id      -- only meaningful for the composer / step4_work /
                               step4_dpo sources, which share that id namespace
  2. normalized target text -- norm_code() of `jac` (and of `jac_rejected` for
                               DPO pairs), i.e. comments stripped + whitespace
                               collapsed, the same normalization dedup2.jac uses
  3. holdout PROMPT text    -- the eval file's user turn; catches a row that
                               reached the same holdout task from a different
                               corpus with a differently-formatted target

Rows that collide are DROPPED and candidate_pool.jsonl is rewritten (the prior
file is kept as candidate_pool.jsonl.preleak). Expected drop count: 0.
"""
import os, re, json, shutil, hashlib, collections;

# repo-relative: this file lives at <repo>/model-experiments/08-nitin-new2-ds/scripts/
glob EXP: str = os.path.dirname(os.path.dirname(os.path.abspath(str(__file__))));
glob EXT: str = os.path.dirname(os.path.dirname(EXP));   # repo root
glob OUTD: str = EXP + "/dataset";
# 07 is deleted; its holdout already lives in OUTD (byte-identical, verified).
# Set HOLDOUT_SRC_DIR to re-import from elsewhere; default = OUTD (copy is skipped).
glob SRC_DIR: str = os.environ.get("HOLDOUT_SRC_DIR", OUTD);
glob HOLDOUT_FILES: list[str] = ["nitin_holdout.jsonl", "nitin_holdout_eval.jsonl"];

def jl(path: str) -> list {
    out = [];
    for line in open(path) {
        s = line.strip();
        if s { out.append(json.loads(s)); }
    }
    return out;
}

# identical to pipeline.jac's norm_code (dedup2.jac code_tokens, joined)
def norm_code(code: any) -> str {
    if not code { return ""; }
    parts = [];
    for line in str(code).split("\n") {
        stripped = line.split("#", 1)[0];
        if stripped.strip() != "" { parts.append(" ".join(stripped.split())); }
    }
    return " ".join(parts);
}

def chash(s: str) -> str {
    return hashlib.sha1(s.encode()).hexdigest();
}

def norm_text(s: any) -> str {
    return " ".join(str(s).split()).lower();
}

glob FENCE_RE: any = re.compile(r"^\s*```(?:jac)?\s*\n(.*?)\n?\s*```\s*$", re.S);  # jac:ignore[E1054]

def strip_fences(text: any) -> str {
    m = FENCE_RE.match(str(text));
    return m.group(1) if m else str(text);  # jac:ignore[E1002]
}

def as_int(v: any) -> int {
    try { return int(v); } except Exception { return -1; }
}

with entry:__main__ {
    # ---------- 1. copy verbatim ----------
    for nm in HOLDOUT_FILES {
        src = os.path.join(SRC_DIR, nm);
        dst = os.path.join(OUTD, nm);
        if os.path.abspath(src) != os.path.abspath(dst) { shutil.copyfile(src, dst); }
        same = open(src, "rb").read() == open(dst, "rb").read();
        print("copied %-26s %8d bytes  byte-identical=%s" % (nm, os.path.getsize(dst), same));
        if not same { raise RuntimeError("copy of " + nm + " is not byte-identical"); }
    }

    # ---------- 2. build the collision surfaces ----------
    hold = jl(OUTD + "/nitin_holdout.jsonl");
    hold_eval = jl(OUTD + "/nitin_holdout_eval.jsonl");
    hold_ids = set();
    hold_hashes = set();
    hold_prompts = set();
    for r in hold {
        hold_ids.add(int(str(r["source_file"]).split("#id=")[1]));
        hold_hashes.add(chash(norm_code(r["jac_code"])));
    }
    for r in hold_eval {
        msgs = r["messages"];
        hold_prompts.add(norm_text(msgs[0]["content"]));
        hold_hashes.add(chash(norm_code(strip_fences(msgs[-1]["content"]))));
    }
    print("holdout rows: %d raw / %d eval | ids %d | target hashes %d | prompts %d"
          % (len(hold), len(hold_eval), len(hold_ids), len(hold_hashes), len(hold_prompts)));

    # ---------- 3. re-check the pool as written to disk ----------
    pool = jl(OUTD + "/candidate_pool.jsonl");
    NUMERIC_SOURCES = ["composer", "step4_work", "step4_dpo"];
    hits = [];
    reasons = collections.Counter();
    keep = [];
    for p in pool {
        why = [];
        if chash(norm_code(p["jac"])) in hold_hashes { why.append("target_content"); }
        if p["jac_rejected"] and chash(norm_code(p["jac_rejected"])) in hold_hashes {
            why.append("rejected_content");
        }
        if norm_text(p["prompt"]) in hold_prompts { why.append("prompt_text"); }
        if p["source_type"] in NUMERIC_SOURCES {
            n = as_int(str(p["origin"]).split("#id=")[1]);
            if n in hold_ids { why.append("numeric_id"); }
        }
        if why {
            hits.append([p["id"], p["source_type"], ",".join(why)]);
            for w in why { reasons[w] += 1; }
        } else {
            keep.append(p);
        }
    }

    print("");
    print("pool rows in            : %d" % len(pool));
    print("holdout collisions found: %d  %s" % (len(hits), dict(reasons)));
    for h in hits[:20] { print("   LEAK", h); }

    if hits {
        shutil.copyfile(OUTD + "/candidate_pool.jsonl", OUTD + "/candidate_pool.jsonl.preleak");
        with open(OUTD + "/candidate_pool.jsonl", "w") as f {
            for p in keep { f.write(json.dumps(p) + "\n"); }
        }
        print("rewrote candidate_pool.jsonl: %d rows (old kept as .preleak)" % len(keep));
    } else {
        print("candidate_pool.jsonl UNCHANGED -- zero holdout overlap on all four surfaces.");
    }
    print("pool rows out           : %d" % len(keep));
    by_type = collections.Counter(p["source_type"] for p in keep);
    print("by source_type          :", dict(by_type));
}
