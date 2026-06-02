"""Conversion-probe eval harness: score a model on the decontaminated holdout.

For each holdout task: generate Jac from the Python prompt, extract the code,
run it on the recorded behavioral test_cases and compare outputs. Reports
run rate (compiles+executes) and cross-compiled test pass rate. The gate is
`jac run`, not `jac check` — the strict type-checker rejects untyped-but-runnable
code, which is not what "the conversion works" means.

Model-agnostic via `generate`:
  - GEN_MODE="py2jac": smoke stand-in (jac py2jac of the source) -- NO LLM,
    proves the extract->run->score loop today.
  - GEN_MODE="mlx": real model via `python -m mlx_lm.generate` (needs mlx-lm +
    a model; scaffold, not run here).

Run from the repo root:
    jac run srccurrent/jacgen/eval_probe.jac
"""

import json;
import subprocess;
import os;
import from corpus { transpile, sanitize_transpile }
import from writer { run_jac }
import from pathlib { Path }
import from typing { Any }

glob HOLDOUT: str = "dataset/eval_holdout/conversion.jsonl";
glob GEN_MODE: str = "py2jac";   # default; override with env JAC_EVAL_MODE = py2jac | mlx
glob MLX_MODEL: str = "";        # default; override with env JAC_EVAL_MODEL = <fused model path>


def eval_mode() -> str {
    return os.environ.get("JAC_EVAL_MODE", GEN_MODE);
}


def generate(prompt: str, python: str) -> str {
    if eval_mode() == "py2jac" {
        return sanitize_transpile(transpile(python));
    }
    model: str = os.environ.get("JAC_EVAL_MODEL", MLX_MODEL);
    proc = subprocess.run(
        ["python", "-m", "mlx_lm.generate", "--model", model,
         "--prompt", prompt, "--temp", "0.0", "--max-tokens", "512"],
        capture_output=True, text=True, timeout=120
    );
    return proc.stdout;
}


def extract_jac(output: str) -> str {
    body: str = output;
    if "```jac" in output {
        body = output.split("```jac", 1)[1].split("```", 1)[0];
    } elif "```" in output {
        body = output.split("```", 1)[1].rsplit("```", 1)[0];
    }
    body = body.split("with entry", 1)[0];
    return body.strip();
}


def build_program(jac_code: str, func: str, cases: list) -> tuple[str, str] {
    calls: str = "";
    expected: list[str] = [];
    for c in cases {
        inp = c["input"];
        calls += "    print(" + func + "(" + repr(inp) + "));\n";
        expected.append(str(c["output"]));
    }
    prog: str = jac_code + "\n\nwith entry {\n" + calls + "}\n";
    return (prog, "\n".join(expected));
}


with entry {
    p = Path(HOLDOUT);
    total: int = 0;
    comp: int = 0;
    passed: int = 0;

    for line in p.read_text().split("\n") {
        if line.strip() == "" { continue; }
        rec: dict[str, Any] = json.loads(line);
        func: str = rec["func_name"];
        python: str = rec["python"];
        prompt: str = rec["prompt"];
        cases: list = rec["test_cases"];
        total += 1;

        out: str = generate(prompt, python);
        jac: str = extract_jac(out);
        (prog, expected) = build_program(jac, func, cases);
        (code, sout, serr) = run_jac(prog);
        # gate on RUN (execution), not `jac check`: the strict type-checker rejects
        # untyped-but-runnable code, which is not what "the conversion works" means.
        if code == 0 { comp += 1; }
        if code == 0 and sout.strip() == expected.strip() { passed += 1; }
    }

    crate: int = comp * 100 // total;
    prate: int = passed * 100 // total;
    print("=== conversion probe (" + eval_mode() + ") on " + str(total) + " holdout tasks ===");
    print("runs (compiles+executes): " + str(crate) + "%  (" + str(comp) + "/" + str(total) + ")");
    print("cross-compiled test pass: " + str(prate) + "%  (" + str(passed) + "/" + str(total) + ")");
}
