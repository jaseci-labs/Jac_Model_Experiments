"""Corpus triage for jac-data-gen -> 06-nitin-ds-sft.
Shingle machinery ported verbatim in spirit from
model-experiments/01-sft-dpo/sft_dpo/jacgen2/{decontam_v2,dedup2}.jac
(SHINGLE_N=14, THRESHOLD=0.5, code_tokens = strip #-comments, whitespace split).
"""
import os, re, json, glob, hashlib, random, collections

EXT = "/Volumes/ExtremePro/JaseciLabs/jac_model_studio"
SRC = os.path.expanduser("~/repos/jac-data-gen/data/jac_outputs")
COMMIT = "7c25aff3110f526eec59e0123ffe6c0c152cce91"
SHINGLE_N = 14          # decontam_v2.jac glob SHINGLE_N
THRESHOLD = 0.5         # decontam_v2.jac glob THRESHOLD
RELAX_N = 10            # relaxed (identifier-only) shingle length, our addition
MIN_BODY_TOKENS = 12    # quality floor, see report
MIN_BODY_LINES = 3

def code_tokens(code):                      # dedup2.jac code_tokens
    lines = []
    for line in code.split("\n"):
        stripped = line.split("#", 1)[0]
        if stripped.strip() != "":
            lines.append(stripped)
    return " ".join(lines).split()

def ident_tokens(code):                     # relaxed: identifiers+numbers only
    return re.findall(r"[A-Za-z_]\w*|\d+", code)

def shingles(toks, n):
    if len(toks) < n: return []
    return [" ".join(toks[i:i+n]) for i in range(len(toks)-n+1)]

def overlap(sh, ref):
    if not sh: return 0.0
    return sum(1 for x in sh if x in ref) / len(sh)

# ---------- load corpus ----------
files = sorted(glob.glob(SRC + "/*.jac"), key=lambda p: int(os.path.basename(p)[:-4]))
comp = {}
for l in open("/tmp/nitin_triage/compile_results.jsonl"):
    r = json.loads(l); comp[r["file"]] = r

DOC_RE = re.compile(r'\s*"""(.*?)"""', re.S)
items = []
for f in files:
    b = os.path.basename(f)
    text = open(f, encoding="utf-8", errors="replace").read()
    m = DOC_RE.match(text)
    doc = m.group(1).strip() if m else ""
    body = text[m.end():] if m else text
    # top-level defs only (col 0); allow generics `def pad[T](...)`.
    names = re.findall(r'^def\s+([A-Za-z_]\w*)\s*(?:\[[^\]]*\])?\s*\(', text, re.M)
    sigm = re.search(r'^def\s+[A-Za-z_]\w*\s*(?:\[[^\]]*\])?\s*\((.*?)\)\s*->\s*([^\{]+)\{', text, re.M | re.S)
    c = comp.get(b, {"rc": 999, "stderr": "NOT RUN"})
    items.append(dict(
        file=b, path=f, text=text, doc=doc, body=body,
        func_name=names[0] if names else "", n_defs=len(names),
        params=(sigm.group(1).strip() if sigm else ""),
        ret=(sigm.group(2).strip() if sigm else ""),
        line_count=len(text.rstrip("\n").split("\n")),
        code_lines=len([l for l in body.split("\n") if l.strip()]),
        body_toks=len(code_tokens(body)),
        rc=c["rc"], err=c.get("stderr", "")[-400:],
    ))
N0 = len(items)
funnel = [("S0  corpus files at jac-data-gen@%s" % COMMIT[:8], N0, "")]

# ---------- S1 compile gate ----------
fail = [i for i in items if i["rc"] != 0]
items = [i for i in items if i["rc"] == 0]
funnel.append(("S1  compile FAIL dropped (`jac run` exit != 0)", -len(fail), "%d survive" % len(items)))

# ---------- S2 quality filter ----------
q_drop = collections.OrderedDict([("no_docstring", []), ("trivial_body", []), ("no_def", []), ("multi_def", [])])
keep = []
for i in items:
    if not i["func_name"]:
        q_drop["no_def"].append(i); continue
    if i["n_defs"] > 1:
        q_drop["multi_def"].append(i); continue
    if not i["doc"]:
        q_drop["no_docstring"].append(i); continue
    if i["body_toks"] < MIN_BODY_TOKENS or i["code_lines"] < MIN_BODY_LINES:
        q_drop["trivial_body"].append(i); continue
    keep.append(i)
for k, v in q_drop.items():
    funnel.append(("S2  quality drop: %s" % k, -len(v), ""))
items = keep
funnel.append(("S2  quality-filter survivors", len(items), ""))

# ---------- S3 exact dedup ----------
seen = {}
exact_dups = []
keep = []
for i in items:
    h = hashlib.sha1(" ".join(code_tokens(i["text"])).encode()).hexdigest()
    i["norm_hash"] = h
    if h in seen:
        exact_dups.append((i["file"], seen[h])); continue
    seen[h] = i["file"]; keep.append(i)
items = keep
funnel.append(("S3  exact-duplicate (normalized token hash) dropped", -len(exact_dups), "%d survive" % len(items)))

# ---------- S4 near-dup, greedy shingle (decontam_v2 algorithm) ----------
ref = {}                # shingle -> file that first introduced it
near_dups = []
keep = []
no_shingle = 0
for i in items:
    sh = shingles(code_tokens(i["text"]), SHINGLE_N)
    if not sh:
        no_shingle += 1
        keep.append(i)
        continue
    hits = [ref[x] for x in sh if x in ref]
    if len(hits) / len(sh) >= THRESHOLD:
        rep = collections.Counter(hits).most_common(1)[0][0]
        near_dups.append((i["file"], rep, round(len(hits)/len(sh), 3), i["func_name"]))
        continue
    for x in sh: ref.setdefault(x, i["file"])
    keep.append(i)
items = keep
funnel.append(("S4  near-duplicate dropped (n=%d shingle, overlap>=%.2f)" % (SHINGLE_N, THRESHOLD), -len(near_dups), "%d survive" % len(items)))

# ---------- S5 decontamination ----------
def load_jsonl(p):
    out = []
    if not os.path.exists(p): return out
    for line in open(p):
        line = line.strip()
        if line: out.append(json.loads(line))
    return out

def rec_text(rec):
    parts = []
    if "messages" in rec:
        for m in rec["messages"]: parts.append(str(m.get("content", "")))
    for k in ("prompt", "chosen", "rejected", "python"):
        if k in rec: parts.append(str(rec[k]))
    return "\n".join(parts)

REFS = [
  ("sft_valid",  EXT + "/model-experiments/04-cpt-sft/sft_fresh_probe/dataset/sft/valid.jsonl"),
  ("dpo_valid",  EXT + "/model-experiments/04-cpt-sft/sft_fresh_probe/dataset/dpo/valid.jsonl"),
  ("eh_conversion", EXT + "/model-experiments/01-sft-dpo/dataset/eval_holdout/conversion.jsonl"),
  ("eh_graph_conversion", EXT + "/model-experiments/01-sft-dpo/dataset/eval_holdout/graph_conversion.jsonl"),
]
strict_ref, relax_ref = {}, {}
ref_stats = []
ref_func_names = collections.defaultdict(set)
ref_ident_sets = collections.defaultdict(list)
for name, p in REFS:
    recs = load_jsonl(p)
    s0, r0 = len(strict_ref), len(relax_ref)
    for rec in recs:
        t = rec_text(rec)
        for x in shingles(code_tokens(t), SHINGLE_N): strict_ref[x] = name
        for x in shingles(ident_tokens(t), RELAX_N): relax_ref[x] = name
        fn = rec.get("func_name")
        if fn:
            ref_func_names[fn].add(name)
            ref_ident_sets[fn].append((name, set(ident_tokens(str(rec.get("python", ""))))))
    ref_stats.append((name, p, len(recs), len(strict_ref)-s0, len(relax_ref)-r0))

contam = []
keep = []
for i in items:
    ct = code_tokens(i["text"]); it = ident_tokens(i["text"])
    s_sh = shingles(ct, SHINGLE_N); r_sh = shingles(it, RELAX_N)
    s_ov = overlap(s_sh, strict_ref); r_ov = overlap(r_sh, relax_ref)
    if s_ov >= THRESHOLD or r_ov >= THRESHOLD:
        srcs = set([strict_ref[x] for x in s_sh if x in strict_ref] + [relax_ref[x] for x in r_sh if x in relax_ref])
        contam.append((i["file"], i["func_name"], round(s_ov,3), round(r_ov,3), sorted(srcs)))
        continue
    keep.append(i)
items = keep
funnel.append(("S5  decontamination dropped (vs 4 holdout files)", -len(contam), "%d survive" % len(items)))

# S5b func-name collision audit + conservative extra drop at Jaccard>=0.5
FN_JACCARD = 0.5
fn_collide = []; keep = []
for i in items:
    if i["func_name"] in ref_func_names:
        best = 0.0; who = ""
        mine = set(ident_tokens(i["text"]))
        for name, s in ref_ident_sets.get(i["func_name"], []):
            if not s: continue
            j = len(mine & s) / len(mine | s)
            if j > best: best, who = j, name
        fn_collide.append((i["file"], i["func_name"], round(best,3), who))
        if best >= FN_JACCARD: continue
    keep.append(i)
fn_dropped = [x for x in fn_collide if x[2] >= FN_JACCARD]
items = keep
funnel.append(("S5b func-name collision w/ holdout, ident-Jaccard>=%.2f dropped" % FN_JACCARD, -len(fn_dropped), "%d survive" % len(items)))

# ---------- S6 holdout carve ----------
random.seed(42)
pool = sorted(items, key=lambda x: int(x["file"][:-4]))
idx = list(range(len(pool)))
random.shuffle(idx)
HOLD_N = 855 if len(pool) >= 4000 else max(1, len(pool)//5)
hold_idx = set(idx[:HOLD_N])
holdout = [pool[k] for k in sorted(hold_idx)]
cand    = [pool[k] for k in range(len(pool)) if k not in hold_idx]

# verify holdout vs candidate pool independence with the same machinery
cref = {}
for i in cand:
    for x in shingles(code_tokens(i["text"]), SHINGLE_N): cref[x] = i["file"]
leaks = []
for i in holdout:
    sh = shingles(code_tokens(i["text"]), SHINGLE_N)
    ov = overlap(sh, cref)
    if ov >= THRESHOLD: leaks.append((i["file"], round(ov,3)))
funnel.append(("S6  frozen Nitin holdout carved (seed=42)", HOLD_N, "%d leak-check failures" % len(leaks)))
funnel.append(("S6  candidate training pool", len(cand), ""))

# ---------- write ----------
def to_rec(i, split):
    sid = hashlib.sha1(("jac-data-gen:" + COMMIT + ":" + i["file"]).encode()).hexdigest()[:12]
    return dict(
        id="nitin_%s__%s" % (sid, i["file"][:-4]),
        seed_id=sid,
        split=split,
        source_file="data/jac_outputs/" + i["file"],
        origin="jac-data-gen:%s:data/jac_outputs/%s" % (COMMIT, i["file"]),
        func_name=i["func_name"], params=i["params"], return_type=i["ret"],
        docstring=i["doc"], jac_code=i["text"],
        line_count=i["line_count"], code_lines=i["code_lines"], token_count=i["body_toks"],
        compile_pass=True, compile_cmd="jac run (jaclang 0.16.1, exit 0)",
        dedup_pass=True, decontam_pass=True,
        category="conversion", task_type="python_to_jac_function",
        dataset_version="nitin-ds-v1.0.0", run_tag="nitin",
    )
OUTD = EXT + "/model-experiments/06-nitin-ds-sft/dataset"
os.makedirs(OUTD, exist_ok=True)
with open(OUTD + "/nitin_holdout.jsonl", "w") as f:
    for i in holdout: f.write(json.dumps(to_rec(i, "holdout")) + "\n")
with open(OUTD + "/candidate_pool.jsonl", "w") as f:
    for i in cand: f.write(json.dumps(to_rec(i, "candidate")) + "\n")

stats = dict(
    commit=COMMIT, n0=N0, funnel=funnel,
    compile_fail=[(i["file"], i["rc"], i["err"]) for i in fail],
    q_drop={k: [(i["file"], i["func_name"], i["body_toks"], i["code_lines"]) for i in v] for k, v in q_drop.items()},
    exact_dups=exact_dups, near_dups=near_dups, no_shingle=no_shingle,
    ref_stats=ref_stats, contam=contam, fn_collide=fn_collide, fn_dropped=fn_dropped, leaks=leaks,
    holdout_n=len(holdout), cand_n=len(cand),
)
json.dump(stats, open("/tmp/nitin_triage/stats.json", "w"), indent=1)
for a, b, c in funnel: print("%-62s %+7d  %s" % (a, b, c))
print("\nno-shingle (<%d tokens, dedup inapplicable) kept: %d" % (SHINGLE_N, no_shingle))
print("compile fails: %d | exact dups: %d | near dups: %d | contam: %d | fn collisions: %d | holdout leaks: %d"
      % (len(fail), len(exact_dups), len(near_dups), len(contam), len(fn_collide), len(leaks)))
