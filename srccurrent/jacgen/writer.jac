"""Persist and re-validate generated Jac conversion examples.

Generation itself is agentic (Claude + jac-mcp). This module only persists the
results and re-runs them through the Jac compiler/runtime so every stored line
is provably compiler-clean and behaviorally correct.
"""

import json;
import subprocess;
import tempfile;
import from pathlib { Path }


def make_sft_example(python_code: str, jac_code: str, difficulty: str,
                     expected_output: str, source: str = "generated_seed",
                     generator: str = "claude-code-mcp") -> dict {
    user: str = "Convert this Python to idiomatic Jac:\n\n```python\n" + python_code.strip() + "\n```";
    assistant: str = "```jac\n" + jac_code.strip() + "\n```";
    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant}
        ],
        "meta": {
            "category": "conversion",
            "difficulty": difficulty,
            "compiler_pass": True,
            "behavioral_pass": True,
            "expected_output": expected_output,
            "generator": generator,
            "source": source
        }
    };
}


def append_jsonl(path: str, rec: dict) {
    p = Path(path);
    p.parent.mkdir(parents=True, exist_ok=True);
    with open(path, "a") as f {
        f.write(json.dumps(rec) + "\n");
    }
}


def run_jac(jac_code: str, timeout: int = 30) -> tuple[int, str, str] {
    with tempfile.TemporaryDirectory() as d {
        f = Path(str(d)) / "snippet.jac";
        f.write_text(jac_code);
        try {
            p = subprocess.run(
                ["jac", "run", str(f)], capture_output=True, text=True, timeout=timeout
            );
            return (p.returncode, p.stdout, p.stderr);
        } except (subprocess.SubprocessError, OSError) as e {
            return (124, "", str(e));
        }
    }
}


def extract_jac(rec: dict) -> str {
    msgs: list = list(rec["messages"]);
    assistant: dict = dict(msgs[1]);
    content: str = str(assistant["content"]);
    return content.split("```jac", 1)[1].rsplit("```", 1)[0].strip();
}


def revalidate_example(rec: dict) -> tuple[bool, str] {
    jac: str = extract_jac(rec);
    (code, out, err) = run_jac(jac);
    if code != 0 {
        return (False, "compile/run failed exit " + str(code) + ": " + err.strip()[:200]);
    }
    meta: dict = dict(rec["meta"]);
    expected: str = str(meta["expected_output"]);
    if out.strip() != expected.strip() {
        return (False, "output " + repr(out.strip()) + " != expected " + repr(expected.strip()));
    }
    return (True, "ok");
}
