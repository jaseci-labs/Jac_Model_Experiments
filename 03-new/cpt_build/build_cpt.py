#!/usr/bin/env python3
"""Build the CPT dataset per 03-new/docs/cpt-dataset-design.md.

Sources -> raw per-source JSONL -> decontam -> pack (EOS-join, 4096-token
windows, 85/15 split) -> 03-new/dataset/cpt/.

Usage:
  .venv/bin/python3 03-new/cpt_build/build_cpt.py \
      --repos-dir <scratch>/repos --arxiv-dir <scratch>/arxiv/osp
"""
import argparse
import hashlib
import json
import random
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "03-new" / "dataset" / "cpt"
JAC = ROOT / ".venv" / "bin" / "jac"
TOKENIZER_DIR = ROOT / "models" / "qwen-q4"
SEQ_LEN = 4096
VAL_PCT = 15  # 85/15 split per design
SEED = 3

# Jac-bearing repos to mine for the code source (jaseci-labs org).
# Excluded deliberately: jac_ml_studio (this repo: eval holdouts + synthetic
# data), archived-jaseci (jaseci-v1 syntax, outdated), all forks.
CODE_REPOS = [
    "jac", "Agentic-AI", "jac-shadcn", "this_is_jac", "jac-load-test",
    "jac-mcp-playground", "jasketch", "jac-client-playground", "Algo",
    "llvm-slice", "littleX", "jaseci-blogs", "jaseci-studio",
    "agentic-ai-tutorial", "the-jac-workshop", "inr-codelabs",
    "tree-sitter-jac",
]

# RL eval-holdout source files (02-rl-grpo/dataset/rl/holdout*.jsonl) --
# excluded from the code corpus entirely (repo-relative paths in this_is_jac).
HOLDOUT_FILES = {
    "analytics.jac",
    "littlex/frontend.cl.jac",
    "raylib_shim.cl.jac",
    "raylib_shooter/bench.jac",
    "raylib_shooter/web/main.jac",
    "source_lexer.jac",
}

SKIP_DIRS = {"node_modules", ".git", ".venv", "__pycache__", "dist", "build"}

HEADER_RE = re.compile(r"^#{1,6} ", re.M)


def log(msg):
    print(msg, flush=True)


def repo_sha(path: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
    except Exception:
        return "unknown"


def md_chunks(text: str):
    """Semantic chunk: split at markdown header lines, keep the header with
    its section. Leading pre-header content becomes its own chunk."""
    positions = [m.start() for m in HEADER_RE.finditer(text)]
    if not positions:
        return [text.strip()] if text.strip() else []
    chunks = []
    if positions[0] > 0 and text[: positions[0]].strip():
        chunks.append(text[: positions[0]].strip())
    for i, start in enumerate(positions):
        end = positions[i + 1] if i + 1 < len(positions) else len(text)
        seg = text[start:end].strip()
        if seg:
            chunks.append(seg)
    return chunks


def strip_frontmatter(text: str) -> str:
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            return text[end + 4:].lstrip()
    return text


# ---------------------------------------------------------------- sources

def build_docs(repos: Path):
    """Jac docs (jaseci-labs/jac docs/docs) + latest jac-llmdocs release."""
    rows = []
    base = repos / "jac" / "docs" / "docs"
    for f in sorted(base.rglob("*.md")):
        chunks = md_chunks(f.read_text(errors="replace"))
        for ch in chunks:
            first = ch.splitlines()[0] if ch else ""
            rows.append({
                "text": ch,
                "meta": {"source": "jaseci_docs", "type": "official_doc",
                         "upsample_weight": 3,
                         "file": str(f.relative_to(base)),
                         "section": first.lstrip("# ")[:120]},
            })
    llm = repos / "jaseci-llmdocs" / "release" / "jac-llmdocs.md"
    if llm.exists():
        for ch in md_chunks(llm.read_text(errors="replace")):
            first = ch.splitlines()[0] if ch else ""
            rows.append({
                "text": ch,
                "meta": {"source": "jaseci_docs", "type": "llm_doc",
                         "upsample_weight": 1,
                         "file": "jaseci-llmdocs/release/jac-llmdocs.md",
                         "section": first.lstrip("# ")[:120]},
            })
    return rows


TEX_STRIP = [
    (re.compile(r"(?<!\\)%.*$", re.M), ""),                     # comments
    (re.compile(r"~?\\cite[pt]?\*?(\[[^\]]*\])*\{[^}]*\}"), ""),
    (re.compile(r"~?\\(auto)?ref\{[^}]*\}"), ""),
    (re.compile(r"\\label\{[^}]*\}"), ""),
    (re.compile(r"\\begin\{(figure|table)\*?\}.*?\\end\{\1\*?\}",
                re.S), lambda m: "\n".join(
                    re.findall(r"\\caption\{([^}]*)\}", m.group(0)))),
]


def build_paper(arxiv: Path):
    """OSP paper: main.tex \\input order, cleaned, split by \\section."""
    main = (arxiv / "main.tex").read_text(errors="replace")
    order = re.findall(r"\\input\{([^}]+)\}", main)
    rows = []
    for name in order:
        f = arxiv / (name if name.endswith(".tex") else name + ".tex")
        if not f.exists():
            continue
        text = f.read_text(errors="replace")
        for pat, rep in TEX_STRIP:
            text = pat.sub(rep, text)
        # split at \section boundaries, keep the \section{...} line
        parts = re.split(r"(?=\\(?:sub)?section\{)", text)
        for part in parts:
            part = part.strip()
            if len(part) < 40:
                continue
            m = re.match(r"\\(?:sub)?section\{([^}]*)\}", part)
            rows.append({
                "text": part,
                "meta": {"source": "osp_paper", "type": "paper_section",
                         "upsample_weight": 1, "file": f.name,
                         "section": (m.group(1) if m else "")[:120]},
            })
    lst = arxiv / "littlex.jac"
    if lst.exists():
        rows.append({
            "text": "# path: osp-paper/littlex.jac\n" + lst.read_text(errors="replace"),
            "meta": {"source": "osp_paper", "type": "paper_listing",
                     "upsample_weight": 1, "file": "littlex.jac",
                     "section": "littleX listing"},
        })
    return rows


def build_blogs(repos: Path):
    base = repos / "jaseci-blogs" / "docs"
    rows = []
    for f in sorted(base.rglob("*.md")):
        text = strip_frontmatter(f.read_text(errors="replace"))
        slug = f.stem
        for ch in md_chunks(text):
            first = ch.splitlines()[0] if ch else ""
            rows.append({
                "text": ch,
                "meta": {"source": "blog", "type": "blog_post",
                         "upsample_weight": 1,
                         "url": f"https://blogs.jaseci.org/blog/{slug}",
                         "file": str(f.relative_to(base)),
                         "section": first.lstrip("# ")[:120]},
            })
    return rows


def gate_files(files):
    """Syntax-gate .jac files with `jac check -p` (parse-only); returns
    pass-set. Full type-check is the WRONG gate for CPT raw text: the
    v0.16.x checker false-positives on real working client-style code
    (E1030/E1032 on JS interop) and standalone .impl/.cl files, failing
    ~100% of real repos. Parse-only keeps outdated/broken syntax out --
    which is what CPT actually needs. 'PASSED' in output is the signal
    (exit code unreliable; missing files still print '0 errors')."""
    def check(f):
        try:
            r = subprocess.run([str(JAC), "check", "-p", str(f)],
                               capture_output=True, text=True, timeout=60)
            return f, " PASSED" in (r.stdout + r.stderr)
        except Exception:
            return f, False
    passed = set()
    with ThreadPoolExecutor(max_workers=8) as ex:
        for i, (f, ok) in enumerate(ex.map(check, files)):
            if ok:
                passed.add(f)
            if (i + 1) % 200 == 0:
                log(f"  gate: {i+1}/{len(files)} checked, {len(passed)} pass")
    return passed


def topo_order(files, repo_root: Path):
    """Order files so imported modules come before importers (Kahn,
    min-in-degree first, cycle-tolerant). Import edges resolved by matching
    the imported module name to another file's stem within the repo."""
    stems = {}
    for f in files:
        stems.setdefault(f.name.split(".")[0], []).append(f)
    imports_re = re.compile(
        r"^\s*(?:import|include)\s+(?:from\s+)?([\w.]+)", re.M)
    deps = {f: set() for f in files}
    for f in files:
        try:
            text = f.read_text(errors="replace")
        except Exception:
            continue
        for mod in imports_re.findall(text):
            leaf = mod.split(".")[-1]
            for cand in stems.get(leaf, []):
                if cand != f:
                    deps[f].add(cand)
    order, placed = [], set()
    pending = sorted(files, key=lambda f: (len(deps[f]), str(f)))
    while pending:
        progress = False
        rest = []
        for f in pending:
            if deps[f] <= placed:
                order.append(f); placed.add(f); progress = True
            else:
                rest.append(f)
        if not progress:  # cycle: place the least-blocked node
            f = rest.pop(0)
            order.append(f); placed.add(f)
        pending = rest
    return order


def build_code(repos: Path):
    rows, gate_stats = [], {}
    for repo in CODE_REPOS:
        rroot = repos / repo
        if not rroot.exists():
            continue
        all_jac = [f for f in rroot.rglob("*.jac")
                   if not (SKIP_DIRS & set(p.name for p in f.parents))]
        if repo == "jac":
            # jac/tests/** and jac/jaclang/** are compiler test fixtures:
            # standalone-unresolvable imports + intentionally-broken snippets
            # (incl. ungatable .test.jac). Keep only examples/ and docs/ code.
            all_jac = [f for f in all_jac
                       if str(f.relative_to(rroot)).startswith(("jac/examples/", "docs/"))]
        if repo == "this_is_jac":
            all_jac = [f for f in all_jac
                       if str(f.relative_to(rroot)) not in HOLDOUT_FILES]
        log(f"  {repo}: {len(all_jac)} files (parse-gating all)")
        passed = gate_files(all_jac)
        gate_stats[repo] = {"files": len(all_jac), "passed": len(passed)}
        keep = sorted(passed, key=str)
        if not keep:
            continue
        for f in topo_order(keep, rroot):
            try:
                body = f.read_text(errors="replace")
            except Exception:
                continue
            if not body.strip():
                continue
            rel = f.relative_to(rroot)
            rows.append({
                "text": f"# path: {repo}/{rel}\n{body}",
                "meta": {"source": "code", "type": "repo_file",
                         "upsample_weight": 1, "repo": f"jaseci-labs/{repo}",
                         "path": str(rel)},
            })
    return rows, gate_stats


def build_rehearsal(target_tokens: int, tok):
    """General-code CF-rehearsal slice: python files from
    codeparrot/codeparrot-clean-valid (public, ungated), sampled
    deterministically until target_tokens is reached.
    (bigcode/the-stack-smol was the design pick but is HF-gated.)"""
    from datasets import load_dataset
    ds = load_dataset("codeparrot/codeparrot-clean-valid", split="train")
    idx = list(range(len(ds)))
    random.Random(SEED).shuffle(idx)
    rows, total = [], 0
    for i in idx:
        ex = ds[i]
        text = ex["content"]
        if not text.strip() or len(text) > 200_000:
            continue
        n = len(tok.encode(text))
        rows.append({
            "text": text,
            "meta": {"source": "rehearsal", "type": "general_code",
                     "upsample_weight": 1, "lang": "python",
                     "stack_repo": ex.get("repo_name", ""),
                     "license": ex.get("license", "")},
        })
        total += n
        if total >= target_tokens:
            break
    return rows, total


# ------------------------------------------------------------- decontam

def shingles(text: str, n=14):
    words = re.findall(r"\S+", text)
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def load_holdout_shingles():
    """14-gram shingle sets for every RL holdout item (prompt + refbody)."""
    items = []
    rl = ROOT / "02-rl-grpo" / "dataset" / "rl"
    ids = set()
    for name in ["holdout.jsonl", "holdout_big.jsonl", "holdout_clean.jsonl"]:
        f = rl / name
        if not f.exists():
            continue
        for line in f.read_text().splitlines():
            d = json.loads(line)
            a = json.loads(d["answer"])
            ids.add(a.get("id", ""))
            s = shingles(d["prompt"] + " " + a.get("expected_output", ""))
            if s:
                items.append(s)
    for tid in ids:
        rb = rl / "refbodies" / f"{tid}.txt"
        if rb.exists():
            s = shingles(rb.read_text(errors="replace"))
            if s:
                items.append(s)
    return items


def decontam(rows, holdout_sets):
    """Drop any row containing >=50% of a holdout item's shingles
    (containment, not symmetric Jaccard -- rows are much larger than
    holdout items, so containment is the meaningful direction)."""
    kept, dropped = [], []
    for row in rows:
        rs = shingles(row["text"])
        hit = False
        if rs:
            for hs in holdout_sets:
                if len(hs & rs) / len(hs) >= 0.5:
                    hit = True
                    break
        (dropped if hit else kept).append(row)
    return kept, [r["meta"] for r in dropped]


# ----------------------------------------------------------------- pack

def pack_source(rows, tok, eos_id, seed=SEED):
    """Upsample (by weight) -> shuffle documents -> EOS-join token stream ->
    cut SEQ_LEN windows. Documents here = raw rows grouped by file."""
    docs = {}
    for r in rows:
        key = r["meta"].get("file") or r["meta"].get("path") or r["meta"].get("url", "?")
        docs.setdefault(key, {"chunks": [], "w": r["meta"]["upsample_weight"]})
        docs[key]["chunks"].append(r["text"])
    doc_list = []
    for key, d in docs.items():
        doc_list.extend([("\n\n".join(d["chunks"]))] * d["w"])
    random.Random(seed).shuffle(doc_list)
    stream = []
    for d in doc_list:
        stream.extend(tok.encode(d))
        stream.append(eos_id)
    windows = [stream[i:i + SEQ_LEN] for i in range(0, len(stream), SEQ_LEN)]
    if windows and len(windows[-1]) < 256:  # drop tiny tail window
        windows.pop()
    return windows, len(stream)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repos-dir", required=True, type=Path)
    ap.add_argument("--arxiv-dir", required=True, type=Path)
    ap.add_argument("--skip-rehearsal", action="store_true")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
    eos = tok.eos_token
    eos_id = tok.eos_token_id
    log(f"tokenizer: {TOKENIZER_DIR.name}, eos={eos!r} (id {eos_id})")

    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"seq_len": SEQ_LEN, "split": "85/15", "seed": SEED,
                "tokenizer": str(TOKENIZER_DIR.relative_to(ROOT)),
                "eos_token": eos,
                "fim": "skipped: Qwen3-Coder-30B-A3B-Instruct tokenizer has no FIM special tokens; downstream hole-fill is chat-format, not FIM",
                "repos": {}, "sources": {}}
    for repo in CODE_REPOS + ["jaseci-llmdocs"]:
        p = args.repos_dir / repo
        if p.exists():
            manifest["repos"][repo] = repo_sha(p)

    log("== building sources ==")
    sources = {
        "docs": build_docs(args.repos_dir),
        "osp_paper": build_paper(args.arxiv_dir),
        "blogs": build_blogs(args.repos_dir),
    }
    code_rows, gate_stats = build_code(args.repos_dir)
    sources["code"] = code_rows
    manifest["code_gate"] = gate_stats

    log("== decontam ==")
    holdout_sets = load_holdout_shingles()
    log(f"  {len(holdout_sets)} holdout shingle sets")
    drops = {}
    for name in sources:
        sources[name], dropped = decontam(sources[name], holdout_sets)
        drops[name] = dropped
        if dropped:
            log(f"  {name}: dropped {len(dropped)}")
    manifest["decontam"] = {k: len(v) for k, v in drops.items()}
    manifest["decontam_dropped"] = drops
    manifest["holdout_files_excluded"] = sorted(HOLDOUT_FILES)

    log("== writing raw per-source jsonl ==")
    for name, rows in sources.items():
        d = OUT / name
        d.mkdir(exist_ok=True)
        with open(d / "raw.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        log(f"  {name}: {len(rows)} rows")

    log("== packing ==")
    packed = {}
    jac_tokens = 0
    for name, rows in sources.items():
        windows, ntok = pack_source(rows, tok, eos_id)
        packed[name] = windows
        jac_tokens += ntok
        manifest["sources"][name] = {"rows": len(rows), "tokens": ntok,
                                     "windows": len(windows)}
        log(f"  {name}: {ntok} tokens -> {len(windows)} windows")

    if not args.skip_rehearsal:
        # rehearsal at ~20% of TOTAL tokens => 0.25 * jac tokens
        target = jac_tokens // 4
        log(f"== rehearsal (target {target} tokens) ==")
        try:
            rrows, rtok = build_rehearsal(target, tok)
            d = OUT / "rehearsal"
            d.mkdir(exist_ok=True)
            with open(d / "raw.jsonl", "w") as f:
                for r in rrows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            windows, ntok = pack_source(rrows, tok, eos_id)
            packed["rehearsal"] = windows
            manifest["sources"]["rehearsal"] = {"rows": len(rrows),
                                                "tokens": ntok,
                                                "windows": len(windows)}
            log(f"  rehearsal: {ntok} tokens -> {len(windows)} windows")
        except Exception as e:
            log(f"  rehearsal FAILED (non-fatal): {e}")
            manifest["sources"]["rehearsal"] = {"error": str(e)[:300]}

    log("== split 85/15 + write packed ==")
    (OUT / "packed").mkdir(exist_ok=True)
    train_f = open(OUT / "packed" / "train.jsonl", "w")
    val_f = open(OUT / "packed" / "valid.jsonl", "w")
    counts = {"train": 0, "val": 0}
    rng = random.Random(SEED)
    order = []
    for name, windows in packed.items():
        for i, w in enumerate(windows):
            split = "val" if i % 20 < (VAL_PCT // 5) else "train"
            order.append((name, i, split, w))
    rng.shuffle(order)
    for name, i, split, w in order:
        line = json.dumps(
            {"text": tok.decode(w),
             "meta": {"source": name, "window": i}}, ensure_ascii=False)
        (train_f if split == "train" else val_f).write(line + "\n")
        counts["train" if split == "train" else "val"] += 1
    train_f.close(); val_f.close()
    manifest["packed"] = counts

    with open(OUT / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    log(f"== done == train {counts['train']} / val {counts['val']} windows")
    log(f"output: {OUT}")


if __name__ == "__main__":
    sys.exit(main())
