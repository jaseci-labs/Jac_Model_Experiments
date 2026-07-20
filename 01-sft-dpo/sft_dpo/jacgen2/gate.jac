"""
Behavioral/compile gates and the v2 record builder for the jacgen2 pipeline.

Re-runs generated Jac through the real compiler/runtime (single-file and
multi-file), then assembles the spec.md (04-cpt-sft) section 6 metadata
schema around a gate result.

NOTE on run_jac_local: this duplicates jacgen/writer.jac's `run_jac` rather
than importing it. A real cross-directory import (`..jacgen.writer` or
`jacgen.writer`) type-checks under `jac check` but fails at runtime here
("attempted relative import with no known parent package") -- `jac run`
executes each entry script without real Python package context, and
"01-sft-dpo" isn't a valid Python identifier for an absolute dotted import
either. Task 3 hit and solved this identical problem by inlining; same move
here rather than fighting the import system again.

NOTE on the subprocess jac path: bare "jac" does NOT resolve to this repo's
venv even when invoked from a script that was itself launched via the
correct absolute `.venv/bin/jac` -- verified empirically (subprocess.run(["jac",
"--version"]) from inside a jac-run script reported `which jac` ==
"/Users/ayush/.local/bin/jac", a stale unrelated venv, with different byllm/
plugin versions than this repo's). The subprocess call below hardcodes the
same absolute path this whole environment requires.
"""

import json;
import subprocess;
import tempfile;
import from pathlib { Path }
import from datetime { datetime }
import from common { require_run_tag, example_key }

glob JAC_BIN: str = "/Volumes/ExtremePro/JaseciLabs/jac_model_studio/.venv/bin/jac";

glob PROMPT_VER: str = "prompt-v2.0";
glob CONTEXT_BUNDLE_VER: str = "jac-context-v2";
glob VALIDATOR_VER: str = "validator-v2.0";
glob DATASET_VER: str = "jac-synth-v2.0.0";


def run_jac_local(jac_code: str, timeout: int = 30) -> tuple[int, str, str] {
    with tempfile.TemporaryDirectory() as d {
        f = Path(str(d)) / "snippet.jac";
        f.write_text(jac_code);
        try {
            # cwd=d, same isolation as jacgen/writer.jac's run_jac -- scopes
            # jac's persistent root store to this tempdir so unrelated calls
            # don't leak state through a shared basename-keyed db.
            p = subprocess.run(
                [JAC_BIN, "run", str(f)], capture_output=True, text=True, timeout=timeout, cwd=str(d)
            );
            return (p.returncode, p.stdout, p.stderr);
        } except (subprocess.SubprocessError, OSError) as e {
            return (124, "", str(e));
        }
    }
}


def run_jac_project(files: dict, main_file: str, timeout: int = 60) -> tuple[int, str, str] {
    with tempfile.TemporaryDirectory() as d {
        dir_path: Path = Path(str(d));
        for name in files {
            fp: Path = dir_path / str(name);
            fp.parent.mkdir(parents=True, exist_ok=True);
            fp.write_text(str(files[name]));
        }
        try {
            p = subprocess.run(
                [JAC_BIN, "run", main_file], capture_output=True, text=True, timeout=timeout, cwd=str(d)
            );
            return (p.returncode, p.stdout, p.stderr);
        } except (subprocess.SubprocessError, OSError) as e {
            return (124, "", str(e));
        }
    }
}


def behavioral_gate(jac_code: str, expected_output: str) -> tuple[bool, str] {
    (code, out, err) = run_jac_local(jac_code);
    if code != 0 {
        return (False, "compile/run failed exit " + str(code) + ": " + err.strip()[:200]);
    }
    if out.strip() != expected_output.strip() {
        return (False, "output " + repr(out.strip()) + " != expected " + repr(expected_output.strip()));
    }
    return (True, "ok");
}


def make_example_v2(
    category: str,
    task_type: str,
    complexity: str,
    messages: list,
    gate_class: str,
    compiler_pass: bool,
    test_pass: any,
    seed_id: str,
    seed_tier: int,
    variant_idx: int,
    generator: str,
    generator_model_id: str,
    extra_meta: dict = {}
) -> dict {
    tag: str = require_run_tag();
    rec: dict = {
        "id": example_key(seed_id, task_type, variant_idx),
        "category": category,
        "task_type": task_type,
        "complexity": complexity,
        "compiler_pass": compiler_pass,
        "test_pass": test_pass,
        "manually_reviewed": False,
        "generator": generator,
        "generator_model_id": generator_model_id,
        "gate_class": gate_class,
        "variant_idx": variant_idx,
        "generation_date": datetime.now().isoformat(),
        "source_prompt_version": PROMPT_VER,
        "context_bundle_version": CONTEXT_BUNDLE_VER,
        "validator_version": VALIDATOR_VER,
        "dataset_version": DATASET_VER,
        "run_tag": tag,
        "seed_id": seed_id,
        "seed_tier": seed_tier,
        "messages": messages
    };
    for k in extra_meta {
        rec[k] = extra_meta[k];
    }
    return rec;
}


def extract_payload(rec: dict, category: str) -> str {
    msgs: list = list(rec["messages"]);
    if category == "trajectory" {
        parts: list = [];
        for m in msgs {
            entry: dict = dict(m);
            if str(entry["role"]) == "assistant" {
                parts.append(str(entry["content"]));
            }
        }
        return "\n".join(parts);
    }

    last: dict = dict(msgs[-1]);
    content: str = str(last["content"]);
    if "```jac" in content {
        return content.split("```jac", 1)[1].rsplit("```", 1)[0].strip();
    }
    return content.strip();
}
