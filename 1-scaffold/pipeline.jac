"""Stage 0 for 08-nitin-new2-ds: merge SEVEN of Nitin's corpora into one
unified candidate pool.

WHAT IS DIFFERENT FROM 07. 07 (`../../07-nitin-ds-new-sft/scripts/pipeline.jac`)
consumed exactly ONE file -- data/py2jac_dataset_idiomatic.jsonl -- and its whole
job was triage (compile gate, quality gate, dedup, decontam, holdout carve) of a
homogeneous flat-function corpus. 08 consumes seven heterogeneous corpora at a
NEWER pin and its job is MERGE + GATE:

  1 osp          data/osp_dataset_pass.jsonl                 chat, test-verified
  2 js2jac       scripts/js2jac_dataset/js2jac_dataset.jsonl  JS/TS -> Jac
  3 step4_dpo    data/step4{b,c,d,e}/dataset_{sonnet,composer}.jsonl (seam disagreements)
  4 step4_work   data/step4{b,c,d,e,f}/work/*.json            Python -> Jac (floor targets)
  5 composer     data/composer_dataset.jsonl                  Python -> Jac (idiomatic targets)
  6 farm         data/farm_dataset.jsonl                      node archetype + CRUD walkers
  7 farm_handler data/farm_handler_dataset.jsonl              same, handler variant

TWO GATES 06 AND 07 DID NOT HAVE, BOTH MANDATORY HERE:

  A. EVAL DENYLIST. evals/function/v1/denylist_ids.txt (1,404 cluster-expanded
     MultiPL-T source ids) seals the function-eval suite. 06 and 07 were built
     before it existed and trained on it -- their numbers are contaminated. Every
     source that shares the numeric HF-row id namespace (composer, step4 pair
     files, step4 work/*.json) is filtered against it here. See report for the
     namespace analysis of the three string-id sources.

  B. QUALITY GRADIENT. docs/PY2JAC_QUALITY_GRADIENT.md documents that the
     py2jac idiomatic corpus has a positional quality gradient: the designed
     mutation gate (_MUTATION_GATE=0.80) was silently dropped in the scale-up, so
     late/recovery rows survived on weak hidden-test suites (first-half median 21
     hidden tests vs second-half 14; mean 29.1 vs 19.7). We re-derive the missing
     signal by counting `test "..."` blocks in work/*.json and gating on it.
     Cutoff TEST_CUTOFF below; reasoning in docs/spec.md.

HOLDOUT. 07's frozen 855-row holdout is REUSED VERBATIM (copied by
`copy_holdout.jac`) so 07 and 08 are comparable. Every candidate whose numeric
id or normalized target text collides with it is dropped here -- and all 855
holdout rows DO appear in composer_dataset.jsonl, so this gate is load-bearing,
not ceremonial.
"""
import os, re, json, glob, random, hashlib, subprocess, collections;

# repo-relative: this file lives at <repo>/1-scaffold/; the experiment dir is
# $EXP (relative to the repo root, or absolute), default experiments/08-nitin-new2-ds
glob EXT: str = os.path.dirname(os.path.dirname(os.path.abspath(str(__file__))));   # repo root
glob EXP: str = os.path.join(EXT, os.environ.get("EXP", "experiments/08-nitin-new2-ds"));
glob OUTD: str = EXP + "/dataset";
# 07 is deleted; 08 carries its own verbatim copy of 07's frozen holdout.
glob HOLDOUT_DIR: str = OUTD;
glob STATS: str = EXP + "/docs/funnel";   # funnel stats, kept with the experiment

glob CORPUS: str = "jac-data-gen";
glob COMMIT: str = "c95b7563c16851010396170c269f1fdc5327ef15";
glob CLONE: str = os.path.expanduser("~/repos/jac-data-gen");
glob WORKTREE: str = os.path.expanduser("~/repos/.jdg-worktrees/c95b7563");

# --- gate constants -------------------------------------------------------
# Hidden-test-count floor for every work/*.json-derived row. 18 is the midpoint
# of the two medians PY2JAC_QUALITY_GRADIENT.md reports (first half 21, second
# half 14), rounded up. Well above the eval builder's own min_tests=5
# (evals/function/v1/manifest.json selection.min_tests).
glob TEST_CUTOFF: int = 18;
# js2jac license safety: keep permissive + no-assertion, drop copyleft + unknown.
glob PERMISSIVE: list = ["MIT", "APACHE-2.0", "BSD-2-CLAUSE", "UNLICENSE", "CC0-1.0", "NOASSERTION"];
glob PAIR_DIRS: list = ["step4b", "step4c", "step4d", "step4e"];
glob DATASET_VERSION: str = "nitin-new2-ds-v1.0.0";
glob RUN_TAG: str = "nitin-new2";

# --- helpers --------------------------------------------------------------
# Resolve a checkout pinned at COMMIT without disturbing the user's own.
# Order: $JDG_ROOT, then a detached worktree of the clone at
# ~/repos/.jdg-worktrees/<commit8> (created on demand). The user's working
# checkout of ~/repos/jac-data-gen is NEVER checked out, reset, or stashed.
def corpus_root() -> str {
    env = os.environ.get("JDG_ROOT", "");
    croot = env if env else WORKTREE;
    if not os.path.exists(os.path.join(croot, ".git")) {
        os.makedirs(os.path.dirname(WORKTREE), exist_ok=True);
        subprocess.run(["git", "-C", CLONE, "worktree", "add", "--detach", WORKTREE, COMMIT], check=True);
        croot = WORKTREE;
    }
    got = subprocess.run(["git", "-C", croot, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip();
    if got != COMMIT {
        raise RuntimeError("corpus at " + croot + " is at " + got + ", expected " + COMMIT);
    }
    return croot;
}

def jl(path: str) -> list {
    out = [];
    for line in open(path) {
        s = line.strip();
        if s { out.append(json.loads(s)); }
    }
    return out;
}

# dedup2.jac code_tokens, joined: strip #-comments, collapse whitespace.
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

glob FENCE_RE: any = re.compile(r"^\s*```(?:jac)?\s*\n(.*?)\n?\s*```\s*$", re.S);  # jac:ignore[E1054]

# osp assistant turns are ```jac ... ``` wrapped; the SFT target is not.
def strip_fences(text: any) -> str {
    m = FENCE_RE.match(str(text));
    return m.group(1) if m else str(text);  # jac:ignore[E1002]
}

glob TEST_RE: any = re.compile(r'^\s*test\s+"', re.M);  # jac:ignore[E1054]

# Re-derive the hidden-test count PY2JAC_QUALITY_GRADIENT.md measures.
def count_tests(blocks: any) -> int {
    if not blocks { return 0; }
    return len(TEST_RE.findall(str(blocks)));  # jac:ignore[E1053]
}

# Numeric HF-row id, or -1 when the id is not in that namespace at all.
def as_int(v: any) -> int {
    try { return int(v); } except Exception { return -1; }
}

# --- farm prompt synthesis ------------------------------------------------
glob NODE_RE: any = re.compile(r"node\s+(\w+)\s*\{([^}]*)\}", re.S);  # jac:ignore[E1054]
glob HAS_RE: any = re.compile(r"has\s+(\w+)\s*:\s*([^;=]+?)\s*(?:=\s*([^;]+?)\s*)?;");
glob EDGE_RE: any = re.compile(r"edge\s+(\w+)\s*:\s*(\w+)\s*(<?--+>?)\s*(\w+)");

# Regex-lift the node/edge shape out of the target so the prompt can state
# the schema in prose. Without this the prompt is under-specified: `archetype`
# carries the WHOLE module's node+edge set, which the manifest does not
# describe, and a prompt that omits it would train the model to invent
# unrelated archetypes.
def describe_schema(archetype: str) -> list {
    lines = [];
    nodes = [];
    for (name, body) in NODE_RE.findall(archetype) {
        fields = [];
        for (fname, ftype, fdef) in HAS_RE.findall(body) {
            t = " ".join(ftype.split());  # jac:ignore[E1032,E1054]
            fields.append(fname + " (" + t + ", default " + " ".join(fdef.split()) + ")" if fdef else fname + " (" + t + ")");  # jac:ignore[E1032,E1054,E1055]
        }
        nodes.append("- `" + name + "`: " + (", ".join(fields) if fields else "no fields"));  # jac:ignore[E1055]
    }
    if nodes {
        lines.append("Node archetypes to define:");
        lines += nodes;
    }
    edges = [];
    for (ename, a, arrow, b) in EDGE_RE.findall(archetype) {
        edges.append("- `" + ename + "`: " + a + " " + arrow + " " + b);  # jac:ignore[E1055]
    }
    if edges {
        lines.append("");
        lines.append("Edge archetypes to define:");
        lines += edges;
    }
    return lines;
}

glob CRUD_TEXT: dict = {
    "create": "`%s` -- walk from the root node, create a new `%s` from the walker's own fields, and report the new node.",
    "list":   "`%s` -- walk from the root node and report every `%s` node hanging off it.",
    "update": "`%s` -- walk to a `%s` node, overwrite the fields supplied on the walker, and report it.",
    "delete": "`%s` -- walk to a `%s` node, delete it, and report a confirmation.",
};

# FARM PROMPT TEMPLATE (documented verbatim in docs/spec.md).
# The corpora carry no prompt field, so one is synthesized from `manifest`,
# `node`, `source_ref` and a regex lift of the archetype's schema. Register is
# spec-style (same register 07 used for a third of its SFT rows).
def farm_prompt(rec: dict) -> str {
    nname = str(rec["node"]);
    man = rec["manifest"];
    ref = rec.get("source_ref") or {};
    repo = str(ref.get("repo", "an application"));
    path = str(ref.get("path", ""));
    head = "Write a single Jac module for the data layer of `" + repo + "`";
    if path { head = head + " (ported from `" + path + "`)"; }
    lines = [head + "."];
    lines.append("");
    lines += describe_schema(str(rec["archetype"]));
    lines.append("");
    ops = [];
    for op in ["create", "list", "update", "delete"] {
        if op in man {
            w = str((man[op] or {}).get("walker", op + "_" + nname));
            ops.append("- " + (CRUD_TEXT[op] % (w, nname)));  # jac:ignore[E1055]
        }
    }
    tag = str(man.get("tag_field", ""));
    intro = "Then add public CRUD walkers for the `" + nname + "` node";
    if tag { intro = intro + " (its identifying field is `" + tag + "`)"; }
    lines.append(intro + ":");
    lines += ops;
    lines.append("");
    lines.append("Output the archetypes first, then the walkers, as one Jac module.");
    return "\n".join(lines);
}

# --- unified record -------------------------------------------------------
def mk(pid: str, stype: str, kind: str, prompt: str, jac: str, rejected: any,
       lang: any, src_text: any, signals: dict, rel_path: str, orig_id: any) -> dict {
    return {
        "id": pid,
        "source_type": stype,
        "task_kind": kind,
        "prompt": prompt,
        "jac": jac,
        "jac_rejected": rejected,
        "origin_source_lang": lang,
        "origin_source_text": src_text,
        "quality_signals": signals,
        "origin": CORPUS + ":" + COMMIT + ":" + rel_path + "#id=" + str(orig_id),
    };
}

with entry:__main__ {
    os.makedirs(STATS, exist_ok=True);
    ROOT = corpus_root();
    print("corpus root:", ROOT, flush=True);

    # ---------- eval denylist (gate A) ----------
    DENY_PATH = os.path.join(ROOT, "evals/function/v1/denylist_ids.txt");
    deny = set();
    for line in open(DENY_PATH) {
        s = line.strip();
        if s { deny.add(int(s)); }
    }
    deny_manifest = json.load(open(os.path.join(ROOT, "evals/function/v1/manifest.json")));
    print("denylist ids:", len(deny), "eval status:", deny_manifest.get("status"), flush=True);

    # ---------- 07 holdout (reused verbatim) ----------
    hold = jl(HOLDOUT_DIR + "/nitin_holdout.jsonl");
    hold_ids = set();
    hold_hashes = set();
    for r in hold {
        hold_ids.add(int(str(r["source_file"]).split("#id=")[1]));
        hold_hashes.add(chash(norm_code(r["jac_code"])));
    }
    print("holdout rows:", len(hold), "ids:", len(hold_ids), "hashes:", len(hold_hashes), flush=True);

    funnel = [];
    rej_sft = [];
    rej_dpo = [];
    pool = [];
}

def reject(bucket: list, pid: str, stype: str, reason: str, detail: str) -> None {
    bucket.append({"pool_id": pid, "source_type": stype, "reject_reason": reason, "detail": detail[:300]});
}

def note(stage: str, delta: int, tail: str) -> None {
    funnel.append([stage, delta, tail]);
}

# ==========================================================================
# 1 osp -- data/osp_dataset_pass.jsonl (chat, 100% test_pass)
# ==========================================================================
with entry:__main__ {
    REL = "data/osp_dataset_pass.jsonl";
    rows = jl(os.path.join(ROOT, REL));
    note("S1  osp rows at pin", len(rows), "");
    n_int = len([r for r in rows if as_int(r["id"]) >= 0]);
    note("S1  osp ids parsable as numeric HF ids (denylist namespace)", n_int, "0 expected");
    kept = 0;
    for r in rows {
        oid = str(r["id"]);
        pid = "osp_" + oid;
        msgs = r["messages"];
        prompt = str(msgs[0]["content"]);
        jac = strip_fences(msgs[-1]["content"]);
        if not jac.strip() {
            reject(rej_sft, pid, "osp", "empty_target", "assistant turn empty after fence strip"); continue;
        }
        if chash(norm_code(jac)) in hold_hashes {
            reject(rej_sft, pid, "osp", "holdout_content_collision", oid); continue;
        }
        pool.append(mk(pid, "osp", "completion", prompt, jac, None, None, None,
            {
                "tier": "verified",
                "test_pass": r.get("test_pass"), "test_verdict": r.get("test_verdict"),
                "test_detail": r.get("test_detail"), "compiler_pass": r.get("compiler_pass"),
                "gate_class": r.get("gate_class"), "upstream_category": r.get("category"),
                "upstream_task_type": r.get("task_type"), "complexity": r.get("complexity"),
                "upstream_run_tag": r.get("run_tag"),
            }, REL, oid));
        kept += 1;
    }
    note("S1  osp -> pool", kept, "%d rejected" % (len(rows) - kept));
}

# ==========================================================================
# 2 js2jac -- scripts/js2jac_dataset/js2jac_dataset.jsonl (JS/TS -> Jac)
# ==========================================================================
with entry:__main__ {
    REL = "scripts/js2jac_dataset/js2jac_dataset.jsonl";
    rows = jl(os.path.join(ROOT, REL));
    note("S2  js2jac rows at pin", len(rows), "");
    n_int = len([r for r in rows if as_int(r["id"]) >= 0]);
    note("S2  js2jac ids parsable as numeric HF ids (denylist namespace)", n_int, "0 expected");
    seen_ids = set();
    seen_hash = set();
    (n_null, n_lic, n_dup, n_hold) = (0, 0, 0, 0);
    kept = 0;
    lic_hist = collections.Counter();
    for r in rows {
        oid = str(r["id"]);
        pid = "js2jac_" + oid;
        lic_hist[str(r.get("spdx") or "")] += 1;
        if not r.get("js") or not r.get("path") or not r.get("jac") {
            n_null += 1; reject(rej_sft, pid, "js2jac", "null_source_field", "js/path/jac missing"); continue;
        }
        spdx = str(r.get("spdx") or "").upper();
        if spdx not in PERMISSIVE {
            n_lic += 1; reject(rej_sft, pid, "js2jac", "license_not_permissive", spdx if spdx else "<blank>"); continue;
        }
        h = chash(norm_code(r["js"]) + "\x00" + norm_code(r["jac"]));
        if oid in seen_ids or h in seen_hash {
            n_dup += 1; reject(rej_sft, pid, "js2jac", "duplicate_id_or_content", oid); continue;
        }
        if chash(norm_code(r["jac"])) in hold_hashes {
            n_hold += 1; reject(rej_sft, pid, "js2jac", "holdout_content_collision", oid); continue;
        }
        seen_ids.add(oid); seen_hash.add(h);
        src = str(r["source"]);
        tier = "floor" if "floor" in src else ("idiomized" if "idiom" in src else "cleaned");
        pool.append(mk(pid, "js2jac", "translation",
            "Translate this JavaScript/TypeScript to idiomatic Jac:\n\n" + str(r["js"]),
            str(r["jac"]), None, "js", str(r["js"]),
            {
                "tier": tier, "spdx": r.get("spdx"), "status_in": r.get("status_in"),
                "js2jac_source": src, "upstream_repo": r.get("repo"),
                "upstream_path": r.get("path"), "upstream_commit": r.get("commit"),
                "chunk": r.get("chunk"),
            }, REL, oid));
        kept += 1;
    }
    note("S2  js2jac null js/path/jac dropped", -n_null, "");
    note("S2  js2jac non-permissive spdx dropped", -n_lic, str(sorted(lic_hist.items())));
    note("S2  js2jac duplicate id/content dropped", -n_dup, "");
    note("S2  js2jac holdout collision dropped", -n_hold, "");
    note("S2  js2jac -> pool", kept, "");
}

# ==========================================================================
# shared py2jac-family load: work/*.json + composer_dataset.jsonl
# ==========================================================================
with entry:__main__ {
    work = {};
    for f in sorted(glob.glob(os.path.join(ROOT, "data/step4*/work/*.json"))) {
        r = json.load(open(f));
        r["_dir"] = f.split("/data/")[1].split("/work/")[0];
        r["_rel"] = "data/" + r["_dir"] + "/work/" + os.path.basename(f);  # jac:ignore[E1055]
        r["_tc"] = count_tests(r.get("test_blocks"));
        work[int(r["id"])] = r;
    }
    composer = {};
    for r in jl(os.path.join(ROOT, "data/composer_dataset.jsonl")) {
        composer[int(r["id"])] = r;
    }
    note("S0  step4*/work/*.json loaded", len(work), "");
    note("S0  composer_dataset.jsonl rows", len(composer), "");
}

# ==========================================================================
# 3 step4_dpo -- seam disagreements (sonnet vs composer) = ready-made pairs
# ==========================================================================
with entry:__main__ {
    seen = set();
    (n_deny, n_hold, n_same, n_dup) = (0, 0, 0, 0);
    n_common = 0;
    kept = 0;
    for d in PAIR_DIRS {
        S = {};
        for r in jl(os.path.join(ROOT, "data/" + d + "/dataset_sonnet.jsonl")) { S[str(r["id"])] = r; }  # jac:ignore[E1053,E1054,E1055]
        C = {};
        for r in jl(os.path.join(ROOT, "data/" + d + "/dataset_composer.jsonl")) { C[str(r["id"])] = r; }  # jac:ignore[E1053,E1054,E1055]
        common = sorted(set(S.keys()) & set(C.keys()), key=lambda k: any : int(k));
        n_common += len(common);
        for k in common {
            (a, b) = (S[k], C[k]);
            if a["source"] == b["source"] { continue; }
            nid = int(k);
            pid = "step4dpo_" + k;  # jac:ignore[E1001]
            chosen = a if a["source"] == "idiomatic" else b;
            rejd   = b if a["source"] == "idiomatic" else a;
            if nid in deny {
                n_deny += 1; reject(rej_dpo, pid, "step4_dpo", "eval_denylist", k); continue;  # jac:ignore[E1053]
            }
            if nid in hold_ids or chash(norm_code(chosen["jac"])) in hold_hashes {
                n_hold += 1; reject(rej_dpo, pid, "step4_dpo", "holdout_leak", k); continue;  # jac:ignore[E1053]
            }
            if norm_code(chosen["jac"]) == norm_code(rejd["jac"]) {
                n_same += 1; reject(rej_dpo, pid, "step4_dpo", "chosen_equals_rejected", k); continue;  # jac:ignore[E1053]
            }
            h = chash(norm_code(chosen["jac"]) + "\x00" + norm_code(rejd["jac"]));
            if pid in seen or h in seen {
                n_dup += 1; reject(rej_dpo, pid, "step4_dpo", "duplicate_pair", k); continue;  # jac:ignore[E1053]
            }
            seen.add(pid); seen.add(h);
            w = work.get(nid);
            py = str(w["python"]) if w else None;
            prompt = ("Translate this Python function to idiomatic Jac:\n\n" + py) if py
                     else ("Write an idiomatic Jac implementation of `" + str(chosen["entrypoint"]) + "`.");
            pool.append(mk(pid, "step4_dpo", "pair_dpo", prompt, str(chosen["jac"]), str(rejd["jac"]),
                "python" if py else None, py,
                {
                    "tier": "idiomatic_vs_floor",
                    "step4_dir": d, "entrypoint": chosen["entrypoint"],
                    "chosen_seam": "sonnet" if chosen is a else "composer",
                    "rejected_seam": "composer" if chosen is a else "sonnet",
                    "test_count": (w["_tc"] if w else None),
                    "coverage": (w.get("coverage") if w else None),
                }, "data/" + d + "/dataset_{sonnet,composer}.jsonl", k));  # jac:ignore[E1053,E1055]
            kept += 1;
        }
    }
    note("S3  step4 sonnet/composer ids joined", n_common, "over " + str(len(PAIR_DIRS)) + " step4 dirs");
    note("S3  step4 seam-disagreement pairs found", kept + n_deny + n_hold + n_same + n_dup, "");
    note("S3  step4_dpo eval-denylist dropped", -n_deny, "");
    note("S3  step4_dpo holdout leak dropped", -n_hold, "");
    note("S3  step4_dpo chosen==rejected dropped", -n_same, "");
    note("S3  step4_dpo duplicate pair dropped", -n_dup, "");
    note("S3  step4_dpo -> pool", kept, "");
}

# ==========================================================================
# 4+5 step4_work / composer -- ONE pass over work ids, split by target quality.
#
# The two sources the brief lists separately are the same join seen from two
# ends, so they are computed together and made DISJOINT to avoid double-counting:
#   composer    = work id whose composer row is source=="idiomatic"  (target = that jac)
#   step4_work  = work id whose composer row is source=="floor"      (target = floor_fn)
# Both are gated on the SAME re-derived hidden-test count (gate B).
# composer idiomatic rows with NO work entry have no computable test signal and
# are excluded wholesale (counted below) -- per PY2JAC_QUALITY_GRADIENT.md those
# are exactly the rec_9000..rec_17000 recovery rows, the weak tail.
# ==========================================================================
with entry:__main__ {
    (n_deny, n_hold, n_low, n_dup) = (0, 0, 0, 0);
    (kept_comp, kept_work) = (0, 0);
    seen = set();
    tc_hist = collections.Counter();
    for nid in sorted(work.keys()) {
        w = work[nid];  # jac:ignore[E1001]
        c = composer.get(nid);
        idiomatic = bool(c) and c["source"] == "idiomatic";
        stype = "composer" if idiomatic else "step4_work";
        pid = ("composer_" + str(nid)) if idiomatic else ("step4work_" + str(nid));
        rel = "data/composer_dataset.jsonl" if idiomatic else w["_rel"];
        target = str(c["jac"]) if idiomatic else str(w["floor_fn"]);
        if nid in deny {
            n_deny += 1; reject(rej_sft, pid, stype, "eval_denylist", str(nid)); continue;
        }
        if nid in hold_ids or chash(norm_code(target)) in hold_hashes {
            n_hold += 1; reject(rej_sft, pid, stype, "holdout_leak", str(nid)); continue;
        }
        tc = w["_tc"];
        tc_hist[tc] += 1;
        if tc < TEST_CUTOFF {
            n_low += 1;
            reject(rej_sft, pid, stype, "below_test_count_cutoff", "test_count=" + str(tc) + " < " + str(TEST_CUTOFF));
            continue;
        }
        if not target.strip() or not str(w["python"]).strip() {
            reject(rej_sft, pid, stype, "empty_source_or_target", str(nid)); continue;
        }
        h = chash(norm_code(w["python"]) + "\x00" + norm_code(target));
        if h in seen {
            n_dup += 1; reject(rej_sft, pid, stype, "duplicate_content", str(nid)); continue;
        }
        seen.add(h);
        pool.append(mk(pid, stype, "translation",
            "Translate this Python function to idiomatic Jac:\n\n" + str(w["python"]),
            target, None, "python", str(w["python"]),
            {
                "tier": "idiomatic" if idiomatic else "floor",
                "test_count": tc, "test_cutoff": TEST_CUTOFF,
                "coverage": w.get("coverage"), "stage": w.get("stage"),  # jac:ignore[E1099]
                "entrypoint": w.get("entrypoint"), "step4_dir": w["_dir"],  # jac:ignore[E1099]
                "work_ref": w["_rel"],
                "composer_source": (c["source"] if c else None),
            }, rel, nid));  # jac:ignore[E1053]
        if idiomatic { kept_comp += 1; } else { kept_work += 1; }
    }
    # composer idiomatic rows with no work entry: no test signal, excluded.
    no_signal = [nid for (nid, c) in composer.items() if c["source"] == "idiomatic" and nid not in work];
    for nid in sorted(no_signal) {
        reject(rej_sft, "composer_" + str(nid), "composer", "no_work_join_no_test_signal", str(nid));
    }
    note("S4  work ids considered (sources 4+5)", len(work), "");
    note("S4  eval-denylist dropped", -n_deny, "");
    note("S4  holdout leak dropped", -n_hold, "");
    note("S4  below test-count cutoff (>=%d) dropped" % TEST_CUTOFF, -n_low, "");
    note("S4  duplicate content dropped", -n_dup, "");
    note("S5  composer (idiomatic target) -> pool", kept_comp, "");
    note("S4  step4_work (floor target) -> pool", kept_work, "");
    note("S5  composer idiomatic rows with NO work join excluded (no test signal)", -len(no_signal), "");
}

# ==========================================================================
# 6+7 farm / farm_handler -- node archetype + CRUD walkers, round-trip gated
# ==========================================================================
with entry:__main__ {
    for (nm, stype, prefix) in [("data/farm_dataset.jsonl", "farm", "farm_"),
                                ("data/farm_handler_dataset.jsonl", "farm_handler", "farmh_")] {
        rows = jl(os.path.join(ROOT, nm));
        note("S6  " + stype + " rows at pin", len(rows), "");
        n_int = len([r for r in rows if as_int(r["id"]) >= 0]);
        note("S6  " + stype + " ids parsable as numeric HF ids (denylist namespace)", n_int, "0 expected");
        seen = set();
        kept = 0;
        for r in rows {
            oid = str(r["id"]);
            pid = prefix + oid;
            target = str(r["archetype"]).rstrip() + "\n\n" + str(r["walkers"]).strip();
            gate = r.get("gate") or {};
            if str(gate.get("reason")) != "round-trip passed" {
                reject(rej_sft, pid, stype, "gate_not_round_trip_passed", json.dumps(gate)); continue;
            }
            h = chash(norm_code(target));
            if h in seen {
                reject(rej_sft, pid, stype, "duplicate_content", oid); continue;
            }
            if h in hold_hashes {
                reject(rej_sft, pid, stype, "holdout_content_collision", oid); continue;
            }
            seen.add(h);
            man = r["manifest"];
            pool.append(mk(pid, stype, "completion", farm_prompt(r), target, None, None, None,  # jac:ignore[E1053]
                {
                    "tier": "verified",
                    "gate_stage": gate.get("stage"), "gate_reason": gate.get("reason"),
                    "node": r.get("node"), "tag_field": man.get("tag_field"),  # jac:ignore[E1032]
                    "crud_ops": sorted([o for o in ["create", "list", "update", "delete"] if o in man]),
                    "source_repo": (r.get("source_ref") or {}).get("repo"),
                    "source_path": (r.get("source_ref") or {}).get("path"),
                }, nm, oid));
            kept += 1;
        }
        note("S6  " + stype + " -> pool", kept, "%d rejected" % (len(rows) - kept));
    }
}

# ==========================================================================
# global invariants + write
# ==========================================================================
# S7 GLOBAL cross-source dedup. Per-source dedup is not enough here: the seven
# corpora overlap. Measured at this pin, before this stage:
#   * farm_handler_dataset.jsonl is a CONTENT SUBSET of farm_dataset.jsonl --
#     all 983 of its ids appear in farm_dataset, 940 byte-identical. Left alone
#     it would double-count ~917 rows.
#   * ~107 osp rows share a target with another osp row.
# Dedup is scoped WITHIN task_kind on purpose: a step4_dpo chosen side legitimately
# equals a composer SFT target (583 cases) -- they are different training signals
# written to different files (dpo_train.jsonl vs sft_train.jsonl), so collapsing
# across kinds would silently delete DPO pairs.
# Priority = strength of the verification behind the row; the strongest copy of a
# duplicated target survives.
glob DEDUP_PRIORITY: dict = {
    "osp": 0, "farm": 1, "farm_handler": 2, "composer": 3,
    "step4_dpo": 4, "step4_work": 5, "js2jac": 6,
};

with entry:__main__ {
    ordered = sorted(pool, key=lambda p: any : (DEDUP_PRIORITY[p["source_type"]], p["id"]));  # jac:ignore[E1054]
    seen = set();
    keep = [];
    n_gdup = 0;
    for p in ordered {
        k = p["task_kind"] + "\x00" + norm_code(p["prompt"]) + "\x00" + norm_code(p["jac"]);  # jac:ignore[E1040,E1055]
        h = chash(k);  # jac:ignore[E1053]
        if h in seen {
            n_gdup += 1;
            reject(rej_dpo if p["task_kind"] == "pair_dpo" else rej_sft,  # jac:ignore[E1040]
                   p["id"], p["source_type"], "global_duplicate_content", p["id"]);  # jac:ignore[E1040,E1053]
            continue;
        }
        seen.add(h);
        keep.append(p);
    }
    pool = sorted(keep, key=lambda p: any : (p["source_type"], p["id"]));  # jac:ignore[E1054]
    note("S7  global cross-source duplicate dropped (within task_kind)", -n_gdup, "%d survive" % len(pool));
}

with entry:__main__ {
    ids = [p["id"] for p in pool];
    dupe_ids = [k for (k, v) in collections.Counter(ids).items() if v > 1];
    if dupe_ids {
        raise RuntimeError("duplicate pool ids: " + str(dupe_ids[:10]));
    }
    # final leakage assertion: nothing in the pool may match the frozen holdout
    leaks = [];
    for p in pool {
        if chash(norm_code(p["jac"])) in hold_hashes { leaks.append([p["id"], "chosen"]); }
        if p["jac_rejected"] and chash(norm_code(p["jac_rejected"])) in hold_hashes { leaks.append([p["id"], "rejected"]); }
        if p["source_type"] in ["composer", "step4_work", "step4_dpo"] {
            n = as_int(str(p["origin"]).split("#id=")[1]);
            if n in hold_ids { leaks.append([p["id"], "numeric_id"]); }
            if n in deny { leaks.append([p["id"], "denylist"]); }
        }
    }
    if leaks {
        raise RuntimeError("HOLDOUT/DENYLIST LEAK in candidate pool: " + str(leaks[:10]));
    }
    note("S7  candidate pool total", len(pool), "0 holdout/denylist leaks");

    os.makedirs(OUTD, exist_ok=True);
    os.makedirs(OUTD + "/rejected/sft", exist_ok=True);
    os.makedirs(OUTD + "/rejected/dpo", exist_ok=True);
    with open(OUTD + "/candidate_pool.jsonl", "w") as f {
        for p in pool { f.write(json.dumps(p) + "\n"); }
    }
    with open(OUTD + "/rejected/sft/rejected.jsonl", "w") as f {
        for r in rej_sft { f.write(json.dumps(r) + "\n"); }
    }
    with open(OUTD + "/rejected/dpo/rejected.jsonl", "w") as f {
        for r in rej_dpo { f.write(json.dumps(r) + "\n"); }
    }
    by_type = collections.Counter(p["source_type"] for p in pool);
    by_kind = collections.Counter(p["task_kind"] for p in pool);
    by_tier = collections.Counter(str((p["quality_signals"] or {}).get("tier")) for p in pool);
    json.dump({
        "commit": COMMIT, "corpus_root": ROOT, "test_cutoff": TEST_CUTOFF,
        "denylist_n": len(deny), "eval_status": deny_manifest.get("status"),
        "holdout_ids": len(hold_ids), "funnel": funnel,
        "by_source_type": dict(by_type), "by_task_kind": dict(by_kind), "by_tier": dict(by_tier),
        "reject_sft": dict(collections.Counter(r["reject_reason"] for r in rej_sft)),
        "reject_dpo": dict(collections.Counter(r["reject_reason"] for r in rej_dpo)),
        "test_count_hist": dict(tc_hist),
        "pool_n": len(pool),
    }, open(STATS + "/stats.json", "w"), indent=1);

    for (a, b, c) in funnel { print("%-64s %+8d  %s" % (a, b, c[:60])); }
    print("");
    print("by source_type:", dict(by_type));
    print("by task_kind  :", dict(by_kind));
    print("by tier       :", dict(by_tier));
    print("rejected sft  :", dict(collections.Counter(r["reject_reason"] for r in rej_sft)));
    print("rejected dpo  :", dict(collections.Counter(r["reject_reason"] for r in rej_dpo)));
    print("candidate_pool:", len(pool), "-> " + OUTD + "/candidate_pool.jsonl");
}
