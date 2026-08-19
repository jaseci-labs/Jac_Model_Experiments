"""Corpus triage for Nitin's NEW corpus -> 07-nitin-ds-new-sft.
Logic is byte-for-byte 06-nitin-ds-sft's pipeline.py; only the corpus
coordinates, the scratch dir, and the version/run tags differ.
Shingle machinery ported verbatim in spirit from
model-experiments/01-sft-dpo/sft_dpo/jacgen2/{decontam_v2,dedup2}.jac
(SHINGLE_N=14, THRESHOLD=0.5, code_tokens = strip #-comments, whitespace split).
"""
import os, re, json, glob, hashlib, random, collections;

glob EXT = "/Volumes/ExtremePro/JaseciLabs/jac_model_studio";
# RESOLVED 2026-08-18. Same repo as 06 (chess10kp/jac-data-gen), NEWER commit:
# 06 pinned 7c25aff3 ("data", 7,627 loose .jac files under data/jac_outputs/).
# 07 pins 11fa3f45 ("Add idiomatic py2jac corpus (9,367 records)"), which adds
# data/py2jac_dataset_idiomatic.jsonl -- a JSONL, not loose files. See
# CONTEXT_BRIEF.md section 1 for the shape analysis. Cloned to the INTERNAL disk
# (~/repos) because the external /Volumes/ExtremePro drive measured ~4.3 MB/s.
glob CORPUS = "jac-data-gen";
glob SRC = os.path.expanduser("~/repos/jac-data-gen/data/py2jac_dataset_idiomatic.jsonl");
glob SRC_REL = "data/py2jac_dataset_idiomatic.jsonl";   # for `origin` strings
glob COMMIT = "11fa3f45a0a349337ae4c355708a7e4974b54a36";
glob SHINGLE_N = 14;          # decontam_v2.jac glob SHINGLE_N
glob THRESHOLD = 0.5;         # decontam_v2.jac glob THRESHOLD
glob RELAX_N = 10;            # relaxed (identifier-only) shingle length, our addition
glob MIN_BODY_TOKENS = 12;    # quality floor, see report
glob MIN_BODY_LINES = 3;

def code_tokens(code: any) -> list {                      # dedup2.jac code_tokens
    lines = [];
    for line in code.split("\n") {
        stripped = line.split("#", 1)[0];
        if stripped.strip() != "" {
            lines.append(stripped);
        }
    }
    return " ".join(lines).split();
}

def ident_tokens(code: any) -> list {                     # relaxed: identifiers+numbers only
    return re.findall(r"[A-Za-z_]\w*|\d+", code);  # jac:ignore[E1002,E1054]
}

def shingles(toks: list, n: int) -> list {
    if len(toks) < n { return []; }
    return [" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1)];
}

def overlap(sh: list, ref: dict) -> float {
    if not sh { return 0.0; }
    return sum(1 for x in sh if x in ref) / len(sh);
}

# ---------- load corpus ----------
with entry:__main__ {
    # CORPUS SHAPE CHANGE vs 06: one JSONL of {id, entrypoint, source, jac}
    # instead of a directory of .jac files. Each record gets the pseudo-filename
    # f"{id}.jac" so every stage below -- which keys off i["file"], sorts with
    # int(file[:-4]), and joins against compile_check's output -- is byte-for-byte
    # 06's logic over the same key space. Ids are unique ints (9367/9367 verified).
    corpus = {};
    for line in open(SRC) {
        line = line.strip();
        if line {
            r = json.loads(line);
            corpus[str(r["id"]) + ".jac"] = r["jac"];
        }
    }
    files = [str(i) + ".jac" for i in sorted([int(k[:-4]) for k in corpus.keys()])];
    comp = {};
    for l in open("/tmp/nitin_new_triage/compile_results.jsonl") {
        r = json.loads(l); comp[r["file"]] = r;
    }

    DOC_RE = re.compile(r'\s*"""(.*?)"""', re.S);  # jac:ignore[E1054]
    items = [];
    for f in files {
        b = f;
        text = corpus[f];
        m = DOC_RE.match(text);  # jac:ignore[E1032]
        doc = m.group(1).strip() if m else "";  # jac:ignore[E1032]
        body = text[m.end():] if m else text;  # jac:ignore[E1032]
        # top-level defs only (col 0); allow generics `def pad[T](...)`.
        names = re.findall(r'^def\s+([A-Za-z_]\w*)\s*(?:\[[^\]]*\])?\s*\(', text, re.M);  # jac:ignore[E1054]
        sigm = re.search(r'^def\s+[A-Za-z_]\w*\s*(?:\[[^\]]*\])?\s*\((.*?)\)\s*->\s*([^\{]+)\{', text, re.M | re.S);  # jac:ignore[E1054]
        c = comp.get(b, {"rc": 999, "stderr": "NOT RUN"});
        items.append(dict(
            file=b, path=f, text=text, doc=doc, body=body,
            func_name=names[0] if names else "", n_defs=len(names),  # jac:ignore[E1053]
            params=(sigm.group(1).strip() if sigm else ""),  # jac:ignore[E1032]
            ret=(sigm.group(2).strip() if sigm else ""),  # jac:ignore[E1032]
            line_count=len(text.rstrip("\n").split("\n")),  # jac:ignore[E1053]
            code_lines=len([l for l in body.split("\n") if l.strip()]),  # jac:ignore[E1032]
            body_toks=len(code_tokens(body)),
            rc=c["rc"], err=c.get("stderr", "")[-400:],
        ));
    }
    N0 = len(items);
    funnel = [("S0  corpus files at %s@%s" % (CORPUS, COMMIT[:8]), N0, "")];

    # ---------- S1 compile gate ----------
    fail = [i for i in items if i["rc"] != 0];
    items = [i for i in items if i["rc"] == 0];
    funnel.append(("S1  compile FAIL dropped (`jac run` exit != 0)", -len(fail), "%d survive" % len(items)));

    # ---------- S2 quality filter ----------
    q_drop = collections.OrderedDict([("no_docstring", []), ("trivial_body", []), ("no_def", []), ("multi_def", [])]);
    keep = [];
    for i in items {
        if not i["func_name"] {
            q_drop["no_def"].append(i); continue;
        }
        if i["n_defs"] > 1 {
            q_drop["multi_def"].append(i); continue;
        }
        if not i["doc"] {
            q_drop["no_docstring"].append(i); continue;
        }
        if i["body_toks"] < MIN_BODY_TOKENS or i["code_lines"] < MIN_BODY_LINES {
            q_drop["trivial_body"].append(i); continue;
        }
        keep.append(i);
    }
    for (k, v) in q_drop.items() {
        funnel.append(("S2  quality drop: %s" % k, -len(v), ""));  # jac:ignore[E1053]
    }
    items = keep;
    funnel.append(("S2  quality-filter survivors", len(items), ""));

    # ---------- S3 exact dedup ----------
    seen = {};
    exact_dups = [];
    keep = [];
    for i in items {
        h = hashlib.sha1(" ".join(code_tokens(i["text"])).encode()).hexdigest();
        i["norm_hash"] = h;
        if h in seen {
            exact_dups.append((i["file"], seen[h])); continue;
        }
        seen[h] = i["file"]; keep.append(i);
    }
    items = keep;
    funnel.append(("S3  exact-duplicate (normalized token hash) dropped", -len(exact_dups), "%d survive" % len(items)));

    # ---------- S4 near-dup, greedy shingle (decontam_v2 algorithm) ----------
    ref = {};                # shingle -> file that first introduced it
    near_dups = [];
    keep = [];
    no_shingle = 0;
    for i in items {
        sh = shingles(code_tokens(i["text"]), SHINGLE_N);
        if not sh {
            no_shingle += 1;
            keep.append(i);
            continue;
        }
        hits = [ref[x] for x in sh if x in ref];
        if len(hits) / len(sh) >= THRESHOLD {
            rep = collections.Counter(hits).most_common(1)[0][0];
            near_dups.append((i["file"], rep, round(len(hits) / len(sh), 3), i["func_name"]));  # jac:ignore[E1054]
            continue;
        }
        for x in sh { ref.setdefault(x, i["file"]); }
        keep.append(i);
    }
    items = keep;
    funnel.append(("S4  near-duplicate dropped (n=%d shingle, overlap>=%.2f)" % (SHINGLE_N, THRESHOLD), -len(near_dups), "%d survive" % len(items)));
}

# ---------- S5 decontamination ----------
def load_jsonl(p: any) -> list {
    out = [];
    if not os.path.exists(p) { return out; }  # jac:ignore[E1054]
    for line in open(p) {  # jac:ignore[E1054]
        line = line.strip();  # jac:ignore[E1032]
        if line { out.append(json.loads(line)); }  # jac:ignore[E1053]
    }
    return out;
}

def rec_text(rec: any) -> str {
    parts = [];
    if "messages" in rec {
        for m in rec["messages"] { parts.append(str(m.get("content", ""))); }
    }
    for k in ("prompt", "chosen", "rejected", "python") {
        if k in rec { parts.append(str(rec[k])); }
    }
    return "\n".join(parts);
}

# [TODO: decide whether 06's holdout (model-experiments/06-nitin-ds-sft/dataset/
#  nitin_holdout.jsonl) should be added as a 5th reference set. The new corpus is a
#  successor to 06's, so overlap is plausible -- but it is only a *required* decontam
#  target if 07 also reports numbers on 06's holdout. Left OFF by default so the
#  triage logic stays mechanically identical to 06's.]
with entry:__main__ {
    REFS = [
      ("sft_valid",  EXT + "/model-experiments/04-cpt-sft/sft_fresh_probe/dataset/sft/valid.jsonl"),
      ("dpo_valid",  EXT + "/model-experiments/04-cpt-sft/sft_fresh_probe/dataset/dpo/valid.jsonl"),
      ("eh_conversion", EXT + "/model-experiments/01-sft-dpo/dataset/eval_holdout/conversion.jsonl"),
      ("eh_graph_conversion", EXT + "/model-experiments/01-sft-dpo/dataset/eval_holdout/graph_conversion.jsonl"),
    ];
    (strict_ref, relax_ref) = ({}, {});
    ref_stats = [];
    ref_func_names = collections.defaultdict(set);
    ref_ident_sets = collections.defaultdict(list);
    for (name, p) in REFS {
        recs = load_jsonl(p);
        (s0, r0) = (len(strict_ref), len(relax_ref));
        for rec in recs {
            t = rec_text(rec);
            for x in shingles(code_tokens(t), SHINGLE_N) { strict_ref[x] = name; }
            for x in shingles(ident_tokens(t), RELAX_N) { relax_ref[x] = name; }
            fn = rec.get("func_name");
            if fn {
                ref_func_names[fn].add(name);
                ref_ident_sets[fn].append((name, set(ident_tokens(str(rec.get("python", ""))))));
            }
        }
        ref_stats.append((name, p, len(recs), len(strict_ref) - s0, len(relax_ref) - r0));
    }

    contam = [];
    keep = [];
    for i in items {
        ct = code_tokens(i["text"]); it = ident_tokens(i["text"]);
        s_sh = shingles(ct, SHINGLE_N); r_sh = shingles(it, RELAX_N);
        s_ov = overlap(s_sh, strict_ref); r_ov = overlap(r_sh, relax_ref);
        if s_ov >= THRESHOLD or r_ov >= THRESHOLD {
            srcs = set([strict_ref[x] for x in s_sh if x in strict_ref] + [relax_ref[x] for x in r_sh if x in relax_ref]);
            contam.append((i["file"], i["func_name"], round(s_ov, 3), round(r_ov, 3), sorted(srcs)));  # jac:ignore[E1054]
            continue;
        }
        keep.append(i);
    }
    items = keep;
    funnel.append(("S5  decontamination dropped (vs 4 holdout files)", -len(contam), "%d survive" % len(items)));

    # S5b func-name collision audit + conservative extra drop at Jaccard>=0.5
    FN_JACCARD = 0.5;
    fn_collide = []; keep = [];
    for i in items {
        if i["func_name"] in ref_func_names {
            best = 0.0; who = "";
            mine = set(ident_tokens(i["text"]));
            for (name, s) in ref_ident_sets.get(i["func_name"], []) {  # jac:ignore[E1022]
                if not s { continue; }
                j = len(mine & s) / len(mine | s);  # jac:ignore[E1055,E1053]
                if j > best { (best, who) = (j, name); }
            }
            fn_collide.append((i["file"], i["func_name"], round(best, 3), who));  # jac:ignore[E1054]
            if best >= FN_JACCARD { continue; }
        }
        keep.append(i);
    }
    fn_dropped = [x for x in fn_collide if x[2] >= FN_JACCARD];
    items = keep;
    funnel.append(("S5b func-name collision w/ holdout, ident-Jaccard>=%.2f dropped" % FN_JACCARD, -len(fn_dropped), "%d survive" % len(items)));

    # ---------- S6 holdout carve ----------
    random.seed(42);
    pool = sorted(items, key=lambda x: any : int(x["file"][:-4]));
    idx = list(range(len(pool)));
    random.shuffle(idx);
    # HOLDOUT B IS A FIXED 855 -- a deliberate constant, not a ratio.
    # 06 matched holdout B to holdout A's 855 rows at carve time
    # ("Same 855-row size (matched deliberately at holdout-carve time)",
    # 06/docs/reports/2026-08-final-comparison.md section 2) so the two columns
    # are the same size and the cross-phase comparison stays like-for-like.
    # Carry it forward UNCHANGED regardless of this corpus's pool size.
    #
    # 06's fallback silently shrank to len(pool)//5 on a small pool. That is
    # exactly the "silent cap" this project forbids: it would hand back a
    # smaller holdout with no signal that cross-comparability had been broken.
    # Fail loudly instead -- an underpowered holdout is a decision for a human,
    # not something a script should quietly choose.
    HOLD_N = 855;
    MIN_POOL_FOR_HOLDOUT = 4000;
    if len(pool) < MIN_POOL_FOR_HOLDOUT {
        raise RuntimeError(
            "clean pool is only " + str(len(pool)) + " rows, below the "
            + str(MIN_POOL_FOR_HOLDOUT) + "-row floor needed to carve the fixed "
            + "855-row holdout B and still leave a usable training pool. "
            + "REFUSING to silently carve a smaller holdout -- 855 is a constant "
            + "matched to holdout A and to 06 for cross-comparability. "
            + "Escalate: either accept a smaller/underpowered holdout explicitly "
            + "(and recompute the detectable-effect floor, spec.md section 7.3), "
            + "or loosen the triage gates and re-run."
        );
    }
    hold_idx = set(idx[:HOLD_N]);
    holdout = [pool[k] for k in sorted(hold_idx)];
    cand    = [pool[k] for k in range(len(pool)) if k not in hold_idx];

    # verify holdout vs candidate pool independence with the same machinery
    cref = {};
    for i in cand {
        for x in shingles(code_tokens(i["text"]), SHINGLE_N) { cref[x] = i["file"]; }  # jac:ignore[E1040]
    }
    leaks = [];
    for i in holdout {
        sh = shingles(code_tokens(i["text"]), SHINGLE_N);  # jac:ignore[E1040]
        ov = overlap(sh, cref);
        if ov >= THRESHOLD { leaks.append((i["file"], round(ov, 3))); }  # jac:ignore[E1040,E1054]
    }
    funnel.append(("S6  frozen Nitin-new holdout carved (seed=42)", HOLD_N, "%d leak-check failures" % len(leaks)));
    funnel.append(("S6  candidate training pool", len(cand), ""));
}

# ---------- write ----------
def to_rec(i: any, split: any) -> dict {
    sid = hashlib.sha1((CORPUS + ":" + COMMIT + ":" + i["file"]).encode()).hexdigest()[:12];  # jac:ignore[E1032,E1055]
    return dict(
        id="nitinnew_%s__%s" % (sid, i["file"][:-4]),
        seed_id=sid,
        split=split,
        source_file=SRC_REL + "#id=" + i["file"][:-4],  # jac:ignore[E1055]
        origin="%s:%s:%s#id=%s" % (CORPUS, COMMIT, SRC_REL, i["file"][:-4]),
        func_name=i["func_name"], params=i["params"], return_type=i["ret"],
        docstring=i["doc"], jac_code=i["text"],
        line_count=i["line_count"], code_lines=i["code_lines"], token_count=i["body_toks"],
        compile_pass=True, compile_cmd="jac run (jaclang 0.16.1, exit 0)",
        dedup_pass=True, decontam_pass=True,
        category="conversion", task_type="python_to_jac_function",
        dataset_version="nitin-new-ds-v1.0.0", run_tag="nitin-new",
    );
}

with entry:__main__ {
    OUTD = EXT + "/model-experiments/07-nitin-ds-new-sft/dataset";
    os.makedirs(OUTD, exist_ok=True);
    with open(OUTD + "/nitin_holdout.jsonl", "w") as f {
        for i in holdout { f.write(json.dumps(to_rec(i, "holdout")) + "\n"); }
    }
    with open(OUTD + "/candidate_pool.jsonl", "w") as f {
        for i in cand { f.write(json.dumps(to_rec(i, "candidate")) + "\n"); }
    }

    stats = dict(
        commit=COMMIT, n0=N0, funnel=funnel,
        compile_fail=[(i["file"], i["rc"], i["err"]) for i in fail],
        q_drop={k: [(i["file"], i["func_name"], i["body_toks"], i["code_lines"]) for i in v] for (k, v) in q_drop.items()},
        exact_dups=exact_dups, near_dups=near_dups, no_shingle=no_shingle,
        ref_stats=ref_stats, contam=contam, fn_collide=fn_collide, fn_dropped=fn_dropped, leaks=leaks,
        holdout_n=len(holdout), cand_n=len(cand),
    );
    json.dump(stats, open("/tmp/nitin_new_triage/stats.json", "w"), indent=1);
    for (a, b, c) in funnel { print("%-62s %+7d  %s" % (a, b, c)); }
    print("\nno-shingle (<%d tokens, dedup inapplicable) kept: %d" % (SHINGLE_N, no_shingle));
    print("compile fails: %d | exact dups: %d | near dups: %d | contam: %d | fn collisions: %d | holdout leaks: %d"
          % (len(fail), len(exact_dups), len(near_dups), len(contam), len(fn_collide), len(leaks)));
}
