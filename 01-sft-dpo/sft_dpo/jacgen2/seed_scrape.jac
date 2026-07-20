"""
seed_scrape.jac — tier 4 seed source: shallow-clone a fixed list of public
jaseci-labs GitHub repos, parse-gate every .jac file inside, and emit
candidates for gen_seed_review.jac's Fable review pass.

Produces:
  <JAC_SCRAPE_DIR>/<repo>/            -- shallow clones (scratch, not committed)
  model-experiments/04-cpt-sft/dataset/shared/scrape_manifest.json
  model-experiments/04-cpt-sft/dataset/shared/tier4_candidates.jsonl

JAC_SCRAPE_DIR must be set by the caller (e.g. a scratchpad dir) -- no
hardcoded default, so a stray run never clones repos inside the project.

Repo list source: repos.json (same directory) by default. Override with
JAC_REPOS_PATH to point at a smaller/manual repo-list file (used by
check_seed_review.jac's end-to-end smoke run against a single small repo,
without touching the committed repos.json).

The "jac" repo carries compiler-internal fixtures under jac/tests/** and
jac/jaclang/** -- excluded, same convention CPT's own corpus build
(03-cpt-only/cpt_build/build_cpt.py CODE_REPOS) already applies to that repo.

Run:
    JAC_SCRAPE_DIR=/path/to/scratch JAC_RUN_TAG=fresh \
        /Volumes/ExtremePro/JaseciLabs/jac_model_studio/.venv/bin/jac \
        run model-experiments/01-sft-dpo/sft_dpo/jacgen2/seed_scrape.jac
"""

import os;
import json;
import subprocess;
import hashlib;
import from pathlib { Path }
import from datetime { datetime }

glob JAC_BIN: str = "/Volumes/ExtremePro/JaseciLabs/jac_model_studio/.venv/bin/jac";
glob REPO_ROOT: str = "/Volumes/ExtremePro/JaseciLabs/jac_model_studio";
glob OUT_DIR: str = REPO_ROOT + "/model-experiments/04-cpt-sft/dataset/shared";
glob MANIFEST_PATH: str = OUT_DIR + "/scrape_manifest.json";
glob CANDIDATES_PATH: str = OUT_DIR + "/tier4_candidates.jsonl";


def repos_spec_path() -> Path {
    override: str | None = os.environ.get("JAC_REPOS_PATH");
    if override {
        return Path(override);
    }
    return Path(__file__).parent / "repos.json";
}


def scrape_dir() -> Path {
    val: str | None = os.environ.get("JAC_SCRAPE_DIR");
    if not val {
        raise ValueError(
            "JAC_SCRAPE_DIR must be set to a scratch directory -- no hardcoded "
            + "default, so a stray run never clones repos inside the project."
        );
    }
    return Path(val);
}


def clone_repo(org: str, repo: str, dest: Path) -> None {
    if (dest / ".git").exists() {
        print("  " + repo + ": already cloned at " + str(dest) + ", skipping clone");
        return;
    }
    dest.parent.mkdir(parents=True, exist_ok=True);
    url: str = "https://github.com/" + org + "/" + repo + ".git";
    print("  " + repo + ": cloning " + url + " -> " + str(dest));
    proc = subprocess.run(
        ["git", "clone", "--depth", "1", url, str(dest)],
        capture_output=True, text=True, timeout=300
    );
    if proc.returncode != 0 {
        raise RuntimeError("git clone failed for " + repo + ": " + proc.stderr.strip()[:500]);
    }
}


def repo_sha(dest: Path) -> str {
    proc = subprocess.run(
        ["git", "-C", str(dest), "rev-parse", "HEAD"],
        capture_output=True, text=True, timeout=30
    );
    if proc.returncode != 0 {
        raise RuntimeError("git rev-parse HEAD failed for " + str(dest) + ": " + proc.stderr.strip());
    }
    return proc.stdout.strip();
}


def is_excluded(repo: str, rel: str) -> bool {
    if repo == "jac" and (rel.startswith("jac/tests/") or rel.startswith("jac/jaclang/")) {
        return True;
    }
    return False;
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


def make_candidate_id(repo: str, sha: str, path: str) -> str {
    return hashlib.sha1((repo + "@" + sha + ":" + path).encode("utf-8")).hexdigest()[:12];
}


def scan_repo(repo: str, repo_dir: Path, sha: str, out_f: any) -> dict {
    files: list[Path] = sorted(repo_dir.rglob("*.jac"));
    scanned: int = 0;
    excluded: int = 0;
    accepted: int = 0;
    rejected: int = 0;
    for fp in files {
        rel: str = str(fp.relative_to(repo_dir));
        if is_excluded(repo, rel) {
            excluded += 1;
            continue;
        }
        scanned += 1;
        if not check_parse_file(str(fp)) {
            rejected += 1;
            continue;
        }
        try {
            code: str = fp.read_text(encoding="utf-8", errors="ignore");
        } except Exception {
            rejected += 1;
            continue;
        }
        cand: dict = {
            "candidate_id": make_candidate_id(repo, sha, rel),
            "repo": repo,
            "sha": sha,
            "path": rel,
            "jac_code": code
        };
        out_f.write(json.dumps(cand) + "\n");
        accepted += 1;
    }
    return {"scanned": scanned, "excluded": excluded, "accepted": accepted, "rejected": rejected};
}


with entry {
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True);
    spec_path: Path = repos_spec_path();
    if not spec_path.exists() {
        raise RuntimeError("repo spec not found at " + str(spec_path));
    }
    with open(str(spec_path), "r") as spec_fh {
        spec: dict = json.load(spec_fh);
    }
    org: str = str(spec["org"]);
    repos: list[str] = list(spec["repos"]);
    base_dir: Path = scrape_dir();
    base_dir.mkdir(parents=True, exist_ok=True);

    manifest_shas: dict[str, str] = {};
    repo_dirs: dict[str, Path] = {};

    print("=== seed_scrape: cloning " + str(len(repos)) + " repos from " + org + " into " + str(base_dir) + " ===");
    for repo in repos {
        clone_dest: Path = base_dir / repo;
        clone_repo(org, repo, clone_dest);
        clone_sha: str = repo_sha(clone_dest);
        manifest_shas[repo] = clone_sha;
        repo_dirs[repo] = clone_dest;
        print("  " + repo + ": @ " + clone_sha[:8]);
    }

    with open(MANIFEST_PATH, "w") as manifest_fh {
        json.dump({"repos": manifest_shas, "cloned_at": datetime.now().isoformat()}, manifest_fh, indent=2);
    }
    print("wrote " + MANIFEST_PATH);

    grand_scanned: int = 0;
    grand_excluded: int = 0;
    grand_accepted: int = 0;
    grand_rejected: int = 0;

    with open(CANDIDATES_PATH, "w") as out_f {
        for repo in repos {
            stats: dict = scan_repo(repo, repo_dirs[repo], manifest_shas[repo], out_f);
            print(
                "  " + repo + ": " + str(stats["scanned"]) + " files scanned, "
                + str(stats["accepted"]) + " passed parse gate, "
                + str(stats["rejected"]) + " failed parse gate, "
                + str(stats["excluded"]) + " excluded (compiler-internal)"
            );
            grand_scanned += int(stats["scanned"]);
            grand_excluded += int(stats["excluded"]);
            grand_accepted += int(stats["accepted"]);
            grand_rejected += int(stats["rejected"]);
        }
    }

    print("");
    print("=== seed_scrape summary ===");
    print("repos cloned: " + str(len(repos)));
    print("files excluded (compiler-internal dirs): " + str(grand_excluded));
    print("files scanned: " + str(grand_scanned));
    print("parse-gate accepted: " + str(grand_accepted));
    print("parse-gate rejected: " + str(grand_rejected));
    print("wrote " + str(grand_accepted) + " candidates -> " + CANDIDATES_PATH);
}
