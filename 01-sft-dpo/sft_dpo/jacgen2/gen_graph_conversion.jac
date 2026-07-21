"""
gen_graph_conversion.jac -- grows the `conversion` category's graph tier
(spec.md section 4.1, `python_to_jac_graph` scale-up), built on the same
prompt-batch handoff protocol as gen_code_gen.jac (batch.jac).

DIFFERENT from every other jacgen2 generator: this one generates BOTH sides
(Python AND Jac) from scratch, forward-authored from a hardcoded scenario
bank -- not transpiled from a seed pool. The existing 31-example graph tier
(model-experiments/01-sft-dpo/sft_dpo/jacgen/graph_data/train.json, frozen,
untouched by this file) proves the problem-shape/style bar; the scenarios
below are new shapes chosen to widen coverage (trees, DAGs, general graphs,
weighted vs unweighted) rather than near-duplicate it.

Also different: run-tag treatment. Per spec.md section 4.1, this generator
runs ONCE during the fresh build (not per-run-tag like code_gen/debug), and
its accepted output goes to the SHARED, non-run-tag-scoped
dataset/shared/graph_conversion_growth.jsonl -- folding into the same
snapshotted conversion slice both run-tags consume, preserving conversion's
no-variance-between-builds property. require_run_tag() is still called at
entry for consistency/safety (the plan expects this run once during the
fresh build), but it does not affect the output path.

Two phases, selected by env var JAC_BATCH_PHASE:

  prepare -- for each scenario (capped at JAC_PILOT_LIMIT), writes one
    pending batch item whose prompt asks for BOTH a Python and a Jac
    solution to the same forward-authored problem, sharing one test input.
    model="opus" per spec.md section 4.1 (python_to_jac_graph is
    Opus-assigned). Writes pending_batch.jsonl and stops -- no LLM call
    happens in this phase.

  collect -- reads responses_batch.jsonl (written by an external
    orchestrator that dispatched Opus against pending_batch.jsonl), parses
    the PYTHON:/JAC: dual response, runs BOTH solutions (python3 subprocess
    for the Python side, gate.run_jac_local for the Jac side), and accepts
    only if both exit 0 and their stdout matches exactly (the dual-language
    behavioral gate) -- then persists via gate.make_example_v2 to the
    shared growth file.

Run (from repo root):
    JAC_RUN_TAG=fresh JAC_BATCH_PHASE=prepare JAC_PILOT_LIMIT=10 \
        /Volumes/ExtremePro/JaseciLabs/jac_model_studio/.venv/bin/jac \
        run model-experiments/01-sft-dpo/sft_dpo/jacgen2/gen_graph_conversion.jac
    JAC_RUN_TAG=fresh JAC_BATCH_PHASE=collect \
        /Volumes/ExtremePro/JaseciLabs/jac_model_studio/.venv/bin/jac \
        run model-experiments/01-sft-dpo/sft_dpo/jacgen2/gen_graph_conversion.jac
"""

import os;
import json;
import hashlib;
import subprocess;
import tempfile;
import from pathlib { Path }
import from common { require_run_tag, append_jsonl, existing_keys, example_key }
import from gate { run_jac_local, make_example_v2 }
import from batch { make_batch_id, write_pending_batch, read_response_batch, pending_path, responses_path }
import from dedup2 { rouge_l }

glob CATEGORY: str = "conversion";
glob TASK_TYPE: str = "python_to_jac_graph";
glob PENDING_CATEGORY: str = "graph_conversion";
glob GENERATOR_MODEL_ID: str = "claude-opus-4-8";

glob GROWTH_PATH: str = "model-experiments/04-cpt-sft/dataset/shared/graph_conversion_growth.jsonl";
glob REJECTED_PATH: str = "model-experiments/04-cpt-sft/dataset/shared/graph_conversion_rejected.jsonl";

"""The existing (frozen) 31-example graph tier this new growth must be
near-dup-checked against, per spec.md section 4.1's Dedup bullet."""
glob EXISTING_TIER31_PATH: str = "model-experiments/01-sft-dpo/sft_dpo/jacgen/graph_data/train.json";
glob NEAR_DUP_THRESHOLD: float = 0.85;

"""New graph/tree/relational problem shapes, invented to widen coverage past
the existing 31-example tier (which is all reachable-count / leaf-count /
chain-sum / edge-weight-sum variants). Spans trees, DAGs, general graphs,
weighted and unweighted."""
glob SCENARIOS: list[str] = [
    "compute the depth of every node in a binary tree",
    "find all nodes reachable within N hops of a given start node",
    "detect whether a directed graph contains a cycle",
    "compute the in-degree and out-degree of every node in a directed graph",
    "topologically sort the nodes of a DAG",
    "find the shortest path by hop-count between two named nodes",
    "merge two binary trees by summing values at overlapping nodes",
    "find the lowest common ancestor of two nodes in a binary tree",
    "count the number of leaf nodes at each depth level of a tree",
    "invert (mirror) a binary tree left-to-right",
    "compute the diameter of a tree (longest path between any two nodes)",
    "find the connected components of an undirected graph",
    "find all root nodes (no incoming edges) in a DAG",
    "compute the longest weighted path through a DAG",
    "serialize a tree into level-order (breadth-first) lists"
];


def build_prompt(scenario_description: str) -> str {
    return (
        "Design a graph/tree-shaped programming problem based on this scenario: "
        + scenario_description
        + "\n\nWrite TWO solutions to the SAME problem, using the SAME test input, that "
        + "must produce IDENTICAL output:\n"
        + "1. A Python solution using a plain graph representation (dict-of-lists "
        + "adjacency, or a simple class with lists for children/edges) -- this "
        + "represents \"the naive/Python-shaped way.\"\n"
        + "2. An IDIOMATIC Jac solution using true graph-spatial primitives: `node` "
        + "archetypes with typed `has` fields, `edge` archetypes where the relationship "
        + "has its own meaning, and a `walker` that traverses the structure to compute "
        + "the answer -- NOT a Python-style function operating on a dict. This is the "
        + "whole point of this dataset: teaching graph-native OSP style, not a "
        + "transpiled one-to-one port.\n\n"
        + "Both solutions must build the SAME test graph/tree structure and both must "
        + "print the SAME final answer to stdout when run, so their outputs can be "
        + "diffed for equivalence.\n\n"
        + "Reply in EXACTLY this format:\n"
        + "PYTHON:\n"
        + "```python\n"
        + "<complete, runnable python solution, with a __main__ block that builds the "
        + "test structure and prints the answer>\n"
        + "```\n"
        + "JAC:\n"
        + "```jac\n"
        + "<complete, runnable jac solution, with a \"with entry\" block that builds the "
        + "same test structure and prints the answer, using node/edge/walker>\n"
        + "```"
    );
}


def scenario_seed_id(scenario_description: str) -> str {
    digest: str = hashlib.md5(scenario_description.encode()).hexdigest()[:12];
    return "graphconv_" + digest;
}


def run_prepare() {
    tag: str = require_run_tag();
    limit: int = int(os.environ.get("JAC_PILOT_LIMIT", "10"));

    items: list[dict] = [];
    capped: int = 0;
    for scenario in SCENARIOS {
        if len(items) >= limit {
            capped += 1;
            continue;
        }
        seed_id: str = scenario_seed_id(scenario);
        prompt: str = build_prompt(scenario);
        batch_id: str = make_batch_id(CATEGORY, TASK_TYPE, seed_id, 0);
        items.append({
            "batch_id": batch_id,
            "model": "opus",
            "prompt": prompt,
            "category": CATEGORY,
            "task_type": TASK_TYPE,
            "variant_idx": 0,
            "seed_id": seed_id,
            "seed_tier": 0,
            "scenario": scenario
        });
    }

    out_path: str = pending_path(tag, PENDING_CATEGORY);
    write_result: dict = write_pending_batch(out_path, items);

    print("");
    print("=== gen_graph_conversion prepare summary ===");
    print("scenario bank size: " + str(len(SCENARIOS)));
    print("candidate items this run: " + str(len(items)) + " (JAC_PILOT_LIMIT=" + str(limit) + ")");
    print("skipped due to cap: " + str(capped));
    print("newly written to pending batch: " + str(write_result["written"]));
    print("skipped as already-present in pending batch: " + str(write_result["skipped"]));
    print("wrote " + out_path);
}


"""Tolerant extraction of a fenced code block. Tries the language-tagged
fence first (e.g. ```python), falls back to the first bare ``` fence if the
model omitted the language tag. Returns "" if no fence is found at all."""
def extract_fenced(s: str, lang: str) -> str {
    body: str = "";
    marker: str = "```" + lang;
    if marker in s {
        after: str = s.split(marker, 1)[1];
        if "```" in after {
            body = after.split("```", 1)[0];
        }
    } elif "```" in s {
        after2: str = s.split("```", 1)[1];
        if "```" in after2 {
            body = after2.split("```", 1)[0];
        }
    }
    return body.strip();
}


"""Parses a PYTHON:/JAC: dual response into its two code blocks. Tolerant of
minor formatting drift but rejects with a specific reason on structural
failure (missing marker, missing/empty fenced block)."""
def parse_dual_response(text: str) -> dict {
    if "PYTHON:" not in text {
        return {"ok": False, "reason": "missing PYTHON: marker", "python_code": "", "jac_code": ""};
    }
    if "JAC:" not in text {
        return {"ok": False, "reason": "missing JAC: marker", "python_code": "", "jac_code": ""};
    }
    halves: list[str] = text.split("JAC:", 1);
    python_half: str = halves[0].split("PYTHON:", 1)[1];
    jac_half: str = halves[1];

    python_code: str = extract_fenced(python_half, "python");
    jac_code: str = extract_fenced(jac_half, "jac");

    if not python_code {
        return {"ok": False, "reason": "could not parse python fenced block", "python_code": "", "jac_code": ""};
    }
    if not jac_code {
        return {"ok": False, "reason": "could not parse jac fenced block", "python_code": "", "jac_code": ""};
    }
    return {"ok": True, "reason": "ok", "python_code": python_code, "jac_code": jac_code};
}


"""Runs a Python solution via the SYSTEM python3 (not the jac venv) in a
throwaway tempdir, mirroring gate.run_jac_local's isolation approach."""
def run_python_local(python_code: str, timeout: int = 15) -> tuple[int, str, str] {
    with tempfile.TemporaryDirectory() as d {
        f = Path(str(d)) / "snippet.py";
        f.write_text(python_code);
        try {
            p = subprocess.run(
                ["python3", str(f)], capture_output=True, text=True, timeout=timeout, cwd=str(d)
            );
            return (p.returncode, p.stdout, p.stderr);
        } except (subprocess.SubprocessError, OSError) as e {
            return (124, "", str(e));
        }
    }
}


"""The dual-language behavioral gate: both the Python and the Jac solution
must exit 0, and their stdout (stripped) must match exactly."""
def dual_language_gate(python_code: str, jac_code: str) -> tuple[bool, str] {
    (py_exit, py_out, py_err) = run_python_local(python_code);
    if py_exit != 0 {
        return (False, "python failed: exit " + str(py_exit) + ": " + py_err.strip()[:200]);
    }
    (jac_exit, jac_out, jac_err) = run_jac_local(jac_code);
    if jac_exit != 0 {
        return (False, "jac failed: exit " + str(jac_exit) + ": " + jac_err.strip()[:200]);
    }
    if py_out.strip() != jac_out.strip() {
        return (False, "outputs differ: python=" + repr(py_out.strip()) + " jac=" + repr(jac_out.strip()));
    }
    return (True, "ok");
}


"""Builds the "existing corpus" of Jac code to near-dup-check new candidates
against: the frozen 31-example tier (train.json's "jac" field) plus every
Jac code block already accepted into GROWTH_PATH in a prior collect run (so
within-this-tier duplicates across multiple collect runs are also caught,
not just against the original 31). Returns a list of (label, jac_code)
pairs so a rejection can name which existing entry it matched."""
def load_existing_jac_corpus() -> list[tuple[str, str]] {
    corpus: list[tuple[str, str]] = [];

    train_path: Path = Path(EXISTING_TIER31_PATH);
    if train_path.exists() {
        with open(str(train_path), "r") as f {
            tier31: list[dict] = json.load(f);
        }
        for entry in tier31 {
            corpus.append(("train31:" + str(entry.get("name", "?")), str(entry.get("jac", ""))));
        }
    }

    if Path(GROWTH_PATH).exists() {
        with open(GROWTH_PATH, "r") as f {
            growth_lines: list[str] = f.readlines();
        }
        for line in growth_lines {
            s: str = line.strip();
            if not s {
                continue;
            }
            rec: dict = json.loads(s);
            for msg in rec.get("messages", []) {
                if msg.get("role") == "assistant" {
                    jac_code: str = extract_fenced(str(msg.get("content", "")), "jac");
                    if jac_code {
                        corpus.append(("growth:" + str(rec.get("id", "?")), jac_code));
                    }
                }
            }
        }
    }

    return corpus;
}


"""Direct loop-and-compare near-dup check (no bucketing -- the corpus is
only ~41 entries at this tier's target scale, so a full pairwise scan is
plenty fast and far more readable). Returns (is_dup, matched_label, score)."""
def near_dup_match(jac_code: str, corpus: list[tuple[str, str]]) -> tuple[bool, str, float] {
    for (label, existing_code) in corpus {
        score: float = rouge_l(jac_code, existing_code);
        if score >= NEAR_DUP_THRESHOLD {
            return (True, label, score);
        }
    }
    return (False, "", 0.0);
}


def run_collect() {
    tag: str = require_run_tag();
    p_path: str = pending_path(tag, PENDING_CATEGORY);
    r_path: str = responses_path(tag, PENDING_CATEGORY);

    if not Path(p_path).exists() {
        print("ERROR: pending batch not found at " + p_path + " -- run prepare mode first.");
        return;
    }

    responses: dict = read_response_batch(r_path);
    if not responses {
        print("no responses found yet at " + r_path + ".");
        print("this is normal between prepare and the orchestrator's generation pass --");
        print("nothing to collect. Re-run collect mode once " + r_path + " exists.");
        return;
    }

    pending_by_id: dict = {};
    with open(p_path, "r") as f {
        lines: list[str] = f.readlines();
    }
    for line in lines {
        s: str = line.strip();
        if s {
            rec = json.loads(s);
            pending_by_id[rec["batch_id"]] = rec;
        }
    }

    already_persisted: dict = existing_keys(GROWTH_PATH);
    existing_corpus: list[tuple[str, str]] = load_existing_jac_corpus();
    print("near-dup corpus loaded: " + str(len(existing_corpus)) + " existing Jac examples (31-tier + prior growth)");

    processed: int = 0;
    accepted: int = 0;
    already_persisted_skipped: int = 0;
    reason_counts: dict[str, int] = {};

    for batch_id in responses {
        processed += 1;
        resp: dict = dict(responses[batch_id]);
        ctx = pending_by_id.get(batch_id, None);

        if ctx is None {
            reason: str = "no matching pending context for batch_id";
            reason_counts[reason] = reason_counts.get(reason, 0) + 1;
            append_jsonl(REJECTED_PATH, {"batch_id": batch_id, "category": CATEGORY, "reason": reason});
            continue;
        }

        if not bool(resp["ok"]) {
            reason: str = "generation failed (ok:false)";
            reason_counts[reason] = reason_counts.get(reason, 0) + 1;
            append_jsonl(REJECTED_PATH, {
                "batch_id": batch_id, "category": CATEGORY, "seed_id": ctx["seed_id"],
                "scenario": ctx["scenario"], "reason": reason
            });
            continue;
        }

        parsed: dict = parse_dual_response(str(resp["text"]));
        if not bool(parsed["ok"]) {
            reason: str = "parse failed: " + str(parsed["reason"]);
            reason_counts[reason] = reason_counts.get(reason, 0) + 1;
            append_jsonl(REJECTED_PATH, {
                "batch_id": batch_id, "category": CATEGORY, "seed_id": ctx["seed_id"],
                "scenario": ctx["scenario"], "reason": reason
            });
            continue;
        }

        python_code: str = str(parsed["python_code"]);
        jac_code: str = str(parsed["jac_code"]);

        (gate_ok, gate_reason) = dual_language_gate(python_code, jac_code);
        if not gate_ok {
            reason: str = "gate failed: " + gate_reason;
            reason_counts[reason] = reason_counts.get(reason, 0) + 1;
            append_jsonl(REJECTED_PATH, {
                "batch_id": batch_id, "category": CATEGORY, "seed_id": ctx["seed_id"],
                "scenario": ctx["scenario"], "reason": reason
            });
            continue;
        }

        candidate_id: str = example_key(str(ctx["seed_id"]), TASK_TYPE, 0);
        if candidate_id in already_persisted {
            already_persisted_skipped += 1;
            continue;
        }

        (is_dup, dup_label, dup_score) = near_dup_match(jac_code, existing_corpus);
        if is_dup {
            reason: str = "near-dup of " + dup_label + " (rouge_l=" + str(round(dup_score, 3)) + ")";
            reason_counts[reason] = reason_counts.get(reason, 0) + 1;
            append_jsonl(REJECTED_PATH, {
                "batch_id": batch_id, "category": CATEGORY, "seed_id": ctx["seed_id"],
                "scenario": ctx["scenario"], "reason": reason
            });
            continue;
        }

        # Matches the existing conversion category's record shape --
        # jacgen/writer.jac's make_sft_example user-turn wording, verbatim.
        user: str = "Convert this Python to idiomatic Jac:\n\n```python\n" + python_code.strip() + "\n```";
        messages: list[dict] = [
            {"role": "user", "content": user},
            {"role": "assistant", "content": "```jac\n" + jac_code.strip() + "\n```"}
        ];

        rec: dict = make_example_v2(
            category=CATEGORY,
            task_type=TASK_TYPE,
            complexity="medium",
            messages=messages,
            gate_class="behavioral",
            compiler_pass=True,
            test_pass=True,
            seed_id=str(ctx["seed_id"]),
            seed_tier=0,
            variant_idx=0,
            generator="opus-api",
            generator_model_id=GENERATOR_MODEL_ID,
            extra_meta={
                "scenario": ctx["scenario"],
                "generator_model_id_note": "placeholder -- no models.json-style pinning "
                    + "exists yet for the Agent-dispatch (Opus/Fable subagent) path"
            }
        );

        append_jsonl(GROWTH_PATH, rec);
        already_persisted[rec["id"]] = True;
        existing_corpus.append(("growth:" + str(rec["id"]), jac_code));
        accepted += 1;
    }

    rejected_count: int = processed - accepted - already_persisted_skipped;
    print("");
    print("=== gen_graph_conversion collect summary ===");
    print("responses processed: " + str(processed));
    print("accepted (newly written): " + str(accepted));
    print("skipped as already-persisted: " + str(already_persisted_skipped));
    print("rejected: " + str(rejected_count));
    for reason in reason_counts {
        print("  - " + reason + ": " + str(reason_counts[reason]));
    }
    print("wrote accepted examples to " + GROWTH_PATH);
}


with entry:__main__ {
    phase: str | None = os.environ.get("JAC_BATCH_PHASE");
    if phase == "prepare" {
        run_prepare();
    } elif phase == "collect" {
        run_collect();
    } else {
        raise ValueError(
            "JAC_BATCH_PHASE must be set to 'prepare' or 'collect', got: " + str(phase)
        );
    }
}
