"""
seed_pool.jac — local seed pool builder for tiers 1, 2, 3, 5 (all local,
no LLM calls). Tier 4 (multi-org GitHub scrape) is Task 6, separate.

Produces:
  model-experiments/04-cpt-sft/dataset/shared/seed_pool.jsonl
  model-experiments/04-cpt-sft/dataset/shared/doc_chunks.jsonl

Source locations (see header comment in each gather_tier* function for the
"locate the jac-mcp bundle" search trail):
  - Tier 1 (jac-mcp example apps): jac_mcp is a uv-tool PyPI install here
    (not editable), so it does NOT bundle example apps on disk — its
    resources.impl.jac lazily fetches them from the jaseci-labs/jaseci
    GitHub repo at runtime (`_fetch_example_github`). This machine happens
    to have a local clone of that exact upstream repo as a sibling project
    (`/Volumes/ExtremePro/JaseciLabs/jaseci`, origin =
    github.com/jaseci-labs/jaseci) — same source jac-mcp's GitHub fallback
    targets, just already on disk. IMPORTANT: that sibling clone's checked
    -out working tree is NOT read directly (it sits on a non-default
    branch that has diverged from origin/main). Instead, `gather_tier1`
    fetches `origin/main`, resolves its exact commit SHA, and reads
    `jac/examples/**` out of a `git archive` of that pinned SHA — so tier
    1's content is reproducible and traceable to a specific upstream
    commit regardless of what the sibling repo's working tree happens to
    have checked out. Every tier-1 seed's `origin` field embeds that SHA
    (`example:jaseci@<sha8>:<relpath>`). If `origin/main` can't be
    resolved (offline, etc.), it falls back to whatever commit/branch is
    actually checked out, records that SHA+branch instead, and prints a
    loud WARNING since that's not the canonical source.
  - Tier 2 (jac-mcp docs): jac_mcp DOES bundle docs on disk even in the
    PyPI install: `content/docs/*.md` (reference + tutorials) and
    `content/{pitfalls,patterns,understand}.md` (guide pages), found under
    the uv tool's own site-packages
    (~/.local/share/uv/tools/jaclang/lib/python3.13/site-packages/jac_mcp/).
    The remaining `jac://guide/*` skill pages (jac-core-cheatsheet,
    jac-walker-patterns, etc.) are served via jaclang.cli.guide_store from
    this repo's own venv (.venv/lib/python3.14/site-packages/jaclang/cli/skills/*.md)
    — included too since spec.md 8.2 asks for every jac://guide/* page.
  - Tier 3 (jaclang's own repo): this repo's own venv install,
    .venv/lib/python3.14/site-packages/jaclang/**/*.jac.
  - Tier 5 (this repo's own app code): jms/**/*.jac +
    model-experiments/**/*.jac, excluding /dataset/ and jacgen2/ itself.

Run:
    JAC_RUN_TAG=fresh /Volumes/ExtremePro/JaseciLabs/jac_model_studio/.venv/bin/jac \
        run model-experiments/01-sft-dpo/sft_dpo/jacgen2/seed_pool.jac
(seed_pool.jsonl itself is NOT run-tag-scoped per spec.md 8.6 — JAC_RUN_TAG
is set here only because it's this project's usual invocation convention;
nothing in this file actually reads it.)
"""

import json;
import subprocess;
import tempfile;
import hashlib;
import from pathlib { Path }
import from gate { run_jac_local }
import from mdfence { extract_jac_fences, chunk_doc }
import from dedup2 { rouge_l }

glob JAC_BIN: str = "/Volumes/ExtremePro/JaseciLabs/jac_model_studio/.venv/bin/jac";
glob REPO_ROOT: str = "/Volumes/ExtremePro/JaseciLabs/jac_model_studio";
glob JACLANG_PKG: str = REPO_ROOT + "/.venv/lib/python3.14/site-packages/jaclang";
glob JACLANG_GUIDE_SKILLS: str = JACLANG_PKG + "/cli/skills";
glob JAC_MCP_UV_CONTENT: str = "/Users/ayush/.local/share/uv/tools/jaclang/lib/python3.13/site-packages/jac_mcp/content";
glob JASECI_REPO: str = "/Volumes/ExtremePro/JaseciLabs/jaseci";
glob JASECI_EXAMPLES_SUBPATH: str = "jac/examples";
glob OUT_DIR: str = REPO_ROOT + "/model-experiments/04-cpt-sft/dataset/shared";
glob SEED_POOL_PATH: str = OUT_DIR + "/seed_pool.jsonl";
glob DOC_CHUNKS_PATH: str = OUT_DIR + "/doc_chunks.jsonl";
glob TIER3_MAX_LINES: int = 200;
glob NEAR_DUP_THRESHOLD: float = 0.85;

"""Simple substring->task_type keyword map for domain-hint assignment
(ponytail-appropriate: not exhaustive, defaults to core_language_basics)."""
glob DOMAIN_KEYWORDS: dict[str, str] = {
    "walker": "walker_traversal",
    "endpoint": "sv_endpoint_authoring",
    "node": "node_edge_definition",
    "edge": "node_edge_definition",
    "byllm": "llm_delegated_function",
    "by llm": "llm_delegated_function",
    "by_llm": "llm_delegated_function",
    "auth": "sv_auth_gating",
    "multi-user": "sv_multi_user_sharing",
    "multi_user": "sv_multi_user_sharing",
    ".cl.jac": "cl_component_authoring",
    "component": "cl_component_authoring",
    "route": "cl_routing",
    "routing": "cl_routing",
    "styl": "cl_styling",
    "npm": "npm_integration",
    "native": "native_compile_subset",
    "packaging": "packaging_cli",
    "shadcn": "shadcn_component_composition",
    "scaffold": "scaffold_bootstrap",
    "test": "test_authoring",
    "concurrency": "concurrency_async",
    "async": "concurrency_async",
    "python-integration": "python_interop",
    "python_interop": "python_interop",
    "typed": "typed_signatures",
    "generic": "typed_signatures",
    "impl": "impl_file_split",
    "persistence": "sv_persistence_queries",
    "graph": "graph_query_patterns",
    "query": "graph_query_patterns"
};


def assign_domains(hint: str) -> list[str] {
    lowered: str = hint.lower();
    domains: list[str] = [];
    for kw in DOMAIN_KEYWORDS {
        mapped: str = DOMAIN_KEYWORDS[kw];
        if kw in lowered and mapped not in domains {
            domains.append(mapped);
        }
    }
    if not domains {
        domains.append("core_language_basics");
    }
    return domains;
}


def make_seed_id(origin: str, code: str) -> str {
    return hashlib.sha1((origin + "|" + code).encode("utf-8")).hexdigest()[:12];
}


def check_parse_file(filepath: str) -> bool {
    try {
        result = subprocess.run(
            [JAC_BIN, "check", "-p", filepath], capture_output=True, text=True, timeout=60
        );
        return " PASSED" in (result.stdout + result.stderr);
    } except (subprocess.SubprocessError, OSError) {
        return False;
    }
}


def check_parse_code(code: str) -> bool {
    with tempfile.TemporaryDirectory() as d {
        f: Path = Path(str(d)) / "snippet.jac";
        f.write_text(code);
        return check_parse_file(str(f));
    }
}


"""Attempt to gate a candidate: behavioral (runs clean, pin stdout),
compile_only (parses but doesn't run standalone), or None (drop).
`already_parsed=True` skips the redundant check_parse_code call for tier-3
candidates, which were already parse-gated on disk before becoming
candidates."""
def gate_candidate(code: str, already_parsed: bool = False) -> dict | None {
    (rc, out, err) = run_jac_local(code);
    if rc == 0 {
        return {"gate_class": "behavioral", "expected_output": out.strip()};
    }
    if already_parsed or check_parse_code(code) {
        return {"gate_class": "compile_only", "expected_output": None};
    }
    return None;
}


"""Resolve which commit of the jaseci sibling repo tier 1 should read from,
without ever mutating that repo's working tree (no checkout/reset/pull —
fetch/rev-parse/branch/archive only, all read-only-ish).

Preferred path: fetch `origin/main` and pin to its exact SHA — this is what
a real jac-mcp GitHub fallback would target, regardless of what branch the
sibling clone's working tree happens to have checked out.

Fallback (only if the fetch or rev-parse fails, e.g. offline): read
whatever commit is currently checked out, but record its SHA and branch
name and print a loud WARNING, since that is NOT guaranteed to be the
default branch and may have diverged from origin/main.

Returns (sha, provenance_label, used_pinned_main)."""
def resolve_tier1_ref() -> tuple[str, str, bool] {
    fetch_proc = subprocess.run(
        ["git", "-C", JASECI_REPO, "fetch", "origin", "main"],
        capture_output=True, timeout=120
    );
    if fetch_proc.returncode == 0 {
        rev_proc = subprocess.run(
            ["git", "-C", JASECI_REPO, "rev-parse", "origin/main"],
            capture_output=True, text=True, timeout=30
        );
        if rev_proc.returncode == 0 and rev_proc.stdout.strip() {
            sha: str = rev_proc.stdout.strip();
            print("tier 1: pinned jac-mcp examples to jaseci origin/main @ " + sha[:8]);
            return (sha, "jaseci@" + sha[:8], True);
        }
    }

    head_proc = subprocess.run(
        ["git", "-C", JASECI_REPO, "rev-parse", "HEAD"],
        capture_output=True, text=True, timeout=30
    );
    if head_proc.returncode != 0 or not head_proc.stdout.strip() {
        raise RuntimeError(
            "tier 1 hard-fail: could not fetch/resolve origin/main NOR determine HEAD "
            + "commit of the jaseci sibling repo at " + JASECI_REPO
            + " -- cannot establish a reproducible source ref for tier 1 (jac-mcp examples)."
        );
    }
    branch_proc = subprocess.run(
        ["git", "-C", JASECI_REPO, "branch", "--show-current"],
        capture_output=True, text=True, timeout=30
    );
    sha: str = head_proc.stdout.strip();
    branch: str = branch_proc.stdout.strip() if branch_proc.returncode == 0 else "UNKNOWN";
    if branch not in ["main", "master"] {
        print(
            "WARNING: tier 1 could not fetch/resolve origin/main from the jaseci sibling "
            + "repo at " + JASECI_REPO + " -- falling back to whatever is currently checked "
            + "out there: branch '" + branch + "' @ " + sha[:8] + ". This is NOT the default "
            + "branch a real jac-mcp fetch would target and may have diverged from "
            + "origin/main. Recorded in every tier-1 seed's origin field for traceability."
        );
    }
    return (sha, "jaseci@" + sha[:8] + "@" + branch, False);
}


"""Tier 1 — jac-mcp example apps (sourced from a pinned commit of the local
jaseci-labs/jaseci sibling clone, see module docstring and
`resolve_tier1_ref`). Whole-file seeds; skip /ui/ shadcn boilerplate."""
def gather_tier1() -> list[dict] {
    candidates: list[dict] = [];
    repo_path: Path = Path(JASECI_REPO);
    if not repo_path.exists() {
        raise RuntimeError(
            "tier 1 hard-fail: expected the jaseci sibling clone (jac-mcp example-apps "
            + "source) at " + JASECI_REPO + " but this path does not exist. Escalate -- "
            + "do not silently skip tier 1."
        );
    }

    (sha, label, pinned_main) = resolve_tier1_ref();

    archive_proc = subprocess.run(
        ["git", "-C", JASECI_REPO, "archive", sha, "--", JASECI_EXAMPLES_SUBPATH],
        capture_output=True
    );
    if archive_proc.returncode != 0 {
        raise RuntimeError(
            "tier 1 hard-fail: `git archive " + sha + " -- " + JASECI_EXAMPLES_SUBPATH
            + "` failed in " + JASECI_REPO + ": "
            + archive_proc.stderr.decode("utf-8", errors="ignore")
        );
    }

    with tempfile.TemporaryDirectory(prefix="jacgen2_tier1_") as tmp_dir {
        tar_proc = subprocess.run(["tar", "-x", "-C", str(tmp_dir)], input=archive_proc.stdout);
        if tar_proc.returncode != 0 {
            raise RuntimeError(
                "tier 1 hard-fail: failed to extract git archive of "
                + JASECI_REPO + "@" + sha + " into scratch dir " + str(tmp_dir)
            );
        }
        base: Path = Path(str(tmp_dir)) / JASECI_EXAMPLES_SUBPATH;
        if not base.exists() {
            raise RuntimeError(
                "tier 1 hard-fail: expected " + JASECI_EXAMPLES_SUBPATH + "/ not found in "
                + "the archive of " + JASECI_REPO + "@" + sha + " (extracted to "
                + str(base) + "). Escalate -- do not silently skip tier 1."
            );
        }
        for app_dir in sorted(base.iterdir()) {
            if not app_dir.is_dir() or app_dir.name.startswith(".") {
                continue;
            }
            for f in sorted(app_dir.rglob("*.jac")) {
                rel: str = str(f.relative_to(base));
                if "/ui/" in ("/" + rel) {
                    continue;
                }
                try {
                    code: str = f.read_text(encoding="utf-8", errors="ignore");
                } except Exception {
                    continue;
                }
                candidates.append({
                    "origin": "example:" + label + ":" + rel,
                    "code": code,
                    "hint": rel,
                    "doc_chunk": None,
                    "already_parsed": False
                });
            }
        }
    }
    return candidates;
}


"""Tier 2 — jac-mcp docs. Returns (seed_candidates, doc_chunk_records)."""
def gather_tier2() -> tuple[list[dict], list[dict]] {
    seed_candidates: list[dict] = [];
    chunk_records: list[dict] = [];
    doc_files: list[Path] = [];

    mcp_content_base: Path = Path(JAC_MCP_UV_CONTENT);
    if not mcp_content_base.exists() {
        raise RuntimeError(
            "tier 2 hard-fail: expected jac-mcp content bundle at " + JAC_MCP_UV_CONTENT
            + " but this path does not exist. Escalate -- do not silently skip tier 2."
        );
    }
    skills_dir: Path = Path(JACLANG_GUIDE_SKILLS);
    if not skills_dir.exists() {
        raise RuntimeError(
            "tier 2 hard-fail: expected jaclang guide-skills bundle at "
            + JACLANG_GUIDE_SKILLS + " but this path does not exist. Escalate -- do not "
            + "silently skip tier 2."
        );
    }
    uv_docs_dir: Path = mcp_content_base / "docs";
    if not uv_docs_dir.exists() {
        raise RuntimeError(
            "tier 2 hard-fail: expected jac-mcp docs dir at " + str(uv_docs_dir)
            + " but this path does not exist. Escalate -- do not silently skip tier 2."
        );
    }
    for f in sorted(uv_docs_dir.glob("*.md")) {
        doc_files.append(f);
    }
    for name in ["pitfalls.md", "patterns.md", "understand.md"] {
        f: Path = mcp_content_base / name;
        if f.exists() {
            doc_files.append(f);
        }
    }
    for f in sorted(skills_dir.glob("*.md")) {
        doc_files.append(f);
    }

    for f in doc_files {
        try {
            text: str = f.read_text(encoding="utf-8", errors="ignore");
        } except Exception {
            continue;
        }
        page: str = f.stem;

        chunks: list[dict] = chunk_doc(text);
        heading_to_text: dict[str, str] = {};
        for ch in chunks {
            h: str = str(ch["heading"]);
            if h not in heading_to_text {
                heading_to_text[h] = str(ch["text"]);
            }
            chunk_id: str = hashlib.sha1((page + "|" + h).encode("utf-8")).hexdigest()[:12];
            chunk_records.append({
                "chunk_id": chunk_id,
                "source_page": page,
                "heading": h,
                "text": ch["text"]
            });
        }

        fences: list[dict] = extract_jac_fences(text);
        for fe in fences {
            content: str = str(fe["content"]);
            if len(content.split("\n")) < 3 {
                continue;
            }
            heading: str = str(fe["heading"]);
            doc_chunk: str | None = heading_to_text.get(heading, None);
            seed_candidates.append({
                "origin": "doc:" + page + "#" + heading,
                "code": content,
                "hint": page + " " + heading,
                "doc_chunk": doc_chunk,
                "already_parsed": False
            });
        }
    }
    return (seed_candidates, chunk_records);
}


"""Tier 3 — jaclang's own repo (this project's venv install). Pre-filters
by line count (<=200) then `jac check -p` PASS on disk before a file
becomes a candidate at all — drops intentionally-broken compiler fixtures
without ever building the run_jac_local overhead for them."""
def gather_tier3() -> list[dict] {
    candidates: list[dict] = [];
    base: Path = Path(JACLANG_PKG);
    if not base.exists() {
        raise RuntimeError(
            "tier 3 hard-fail: expected jaclang package install at " + JACLANG_PKG
            + " but this path does not exist. Escalate -- do not silently skip tier 3."
        );
    }
    for f in sorted(base.rglob("*.jac")) {
        try {
            code: str = f.read_text(encoding="utf-8", errors="ignore");
        } except Exception {
            continue;
        }
        if len(code.split("\n")) > TIER3_MAX_LINES {
            continue;
        }
        if check_parse_file(str(f)) {
            rel: str = str(f.relative_to(base));
            candidates.append({
                "origin": "jaclang:" + rel,
                "code": code,
                "hint": rel,
                "doc_chunk": None,
                "already_parsed": True
            });
        }
    }
    return candidates;
}


"""Tier 5 — this repo's own app code. Whole-file seeds (v1 simplification —
no archetype-level splitting, see task brief). Excludes /dataset/ paths and
jacgen2/ itself (don't seed from this own pipeline's code)."""
def gather_tier5() -> list[dict] {
    candidates: list[dict] = [];
    roots: list[str] = [REPO_ROOT + "/jms", REPO_ROOT + "/model-experiments"];
    seen_paths: dict[str, bool] = {};
    for root_dir in roots {
        base: Path = Path(root_dir);
        if not base.exists() {
            continue;
        }
        for f in sorted(base.rglob("*.jac")) {
            full: str = str(f);
            if full in seen_paths {
                continue;
            }
            seen_paths[full] = True;
            if "/dataset/" in full or "/jacgen2/" in full {
                continue;
            }
            try {
                code: str = f.read_text(encoding="utf-8", errors="ignore");
            } except Exception {
                continue;
            }
            rel: str = str(f.relative_to(Path(REPO_ROOT)));
            candidates.append({
                "origin": "self:" + rel,
                "code": code,
                "hint": rel,
                "doc_chunk": None,
                "already_parsed": False
            });
        }
    }
    return candidates;
}


with entry {
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True);

    tier1: list[dict] = gather_tier1();
    (tier2, doc_chunk_records) = gather_tier2();
    tier3: list[dict] = gather_tier3();
    tier5: list[dict] = gather_tier5();

    tier_candidates: dict[int, list[dict]] = {1: tier1, 2: tier2, 3: tier3, 5: tier5};
    candidate_counts: dict[int, int] = {1: len(tier1), 2: len(tier2), 3: len(tier3), 5: len(tier5)};

    all_seeds: list[dict] = [];
    gate_dropped: dict[int, int] = {1: 0, 2: 0, 3: 0, 5: 0};

    for tier in [1, 2, 3, 5] {
        cands: list[dict] = tier_candidates[tier];
        print("tier " + str(tier) + ": gathered " + str(len(cands)) + " raw candidates, gating...");
        for c in cands {
            gate: dict | None = gate_candidate(str(c["code"]), bool(c["already_parsed"]));
            if gate is None {
                gate_dropped[tier] += 1;
                continue;
            }
            hint: str = str(c["hint"]) + " " + str(c["code"])[:500];
            seed_id: str = make_seed_id(str(c["origin"]), str(c["code"]));
            rec: dict = {
                "seed_id": seed_id,
                "seed_tier": tier,
                "origin": c["origin"],
                "jac_code": c["code"],
                "gate_class": gate["gate_class"],
                "expected_output": gate["expected_output"],
                "domains": assign_domains(hint),
                "doc_chunk": c.get("doc_chunk", None)
            };
            all_seeds.append(rec);
        }
    }

    # exact-sha1 dedup
    seen_ids: dict[str, bool] = {};
    exact_dup_dropped: dict[int, int] = {1: 0, 2: 0, 3: 0, 5: 0};
    stage1: list[dict] = [];
    for rec in all_seeds {
        sid: str = str(rec["seed_id"]);
        if sid in seen_ids {
            exact_dup_dropped[int(rec["seed_tier"])] += 1;
            continue;
        }
        seen_ids[sid] = True;
        stage1.append(rec);
    }

    # bucketed near-dup pass: (len(code)//200, code[:20]) buckets, rouge_l >= 0.85, keep first-seen
    buckets: dict[str, list[dict]] = {};
    near_dup_dropped: dict[int, int] = {1: 0, 2: 0, 3: 0, 5: 0};
    final_seeds: list[dict] = [];
    for rec in stage1 {
        code: str = str(rec["jac_code"]);
        bucket_key: str = str(len(code) // 200) + "|" + code[:20];
        if bucket_key not in buckets {
            buckets[bucket_key] = [];
        }
        is_dup: bool = False;
        for kept in buckets[bucket_key] {
            if rouge_l(code, str(kept["jac_code"])) >= NEAR_DUP_THRESHOLD {
                is_dup = True;
                break;
            }
        }
        if is_dup {
            near_dup_dropped[int(rec["seed_tier"])] += 1;
            continue;
        }
        buckets[bucket_key].append(rec);
        final_seeds.append(rec);
    }

    with open(SEED_POOL_PATH, "w") as out_f {
        for rec in final_seeds {
            out_f.write(json.dumps(rec) + "\n");
        }
    }
    with open(DOC_CHUNKS_PATH, "w") as out_f {
        for ch in doc_chunk_records {
            out_f.write(json.dumps(ch) + "\n");
        }
    }

    accepted_by_tier: dict[int, int] = {1: 0, 2: 0, 3: 0, 5: 0};
    for rec in final_seeds {
        accepted_by_tier[int(rec["seed_tier"])] += 1;
    }

    print("");
    print("=== seed pool summary ===");
    total_accepted: int = 0;
    total_dropped: int = 0;
    for tier in [1, 2, 3, 5] {
        accepted: int = accepted_by_tier[tier];
        dropped: int = gate_dropped[tier] + exact_dup_dropped[tier] + near_dup_dropped[tier];
        total_accepted += accepted;
        total_dropped += dropped;
        print(
            "tier " + str(tier) + ": " + str(accepted) + " accepted, " + str(dropped) + " dropped"
            + " (candidates=" + str(candidate_counts[tier])
            + ", gate_dropped=" + str(gate_dropped[tier])
            + ", exact_dup=" + str(exact_dup_dropped[tier])
            + ", near_dup=" + str(near_dup_dropped[tier]) + ")"
        );
    }
    print("TOTAL: " + str(total_accepted) + " accepted, " + str(total_dropped) + " dropped");
    print("doc_chunks: " + str(len(doc_chunk_records)) + " chunks written");
    print("wrote " + SEED_POOL_PATH);
    print("wrote " + DOC_CHUNKS_PATH);
}
