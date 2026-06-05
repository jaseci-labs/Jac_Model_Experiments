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
import re;
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
    cmd: list = ["mlx_lm.generate", "--model", model,
                 "--prompt", prompt, "--temp", "0.0", "--max-tokens", "512"];
    adapter: str = os.environ.get("JAC_EVAL_ADAPTER", "");
    if adapter != "" {
        # eval a checkpoint adapter directly -- no fuse needed per checkpoint
        cmd = cmd + ["--adapter-path", adapter];
    }
    try {
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180);
        if proc.returncode != 0 { return ""; }
        return proc.stdout;
    } except (subprocess.SubprocessError, OSError) {
        return "";
    }
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


def detect_func(jac_code: str, fallback: str) -> str {
    # call the function the model actually defined, even if it renamed it
    m = re.search("def (\\w+)\\s*\\(", jac_code);
    if m is None { return fallback; }
    return m.group(1);
}


# --- mlx_lm.generate token/throughput stats (verbose stdout footer) ---
def gen_tokens(out: str) -> int {
    m = re.search("Generation: ([0-9]+) tokens", out);
    if m is None { return 0; }
    return int(str(m.group(1)));
}


def gen_tps(out: str) -> float {
    m = re.search("Generation: [0-9]+ tokens, ([0-9.]+) tokens-per-sec", out);
    if m is None { return 0.0; }
    return float(str(m.group(1)));
}


def strip_stats(out: str) -> str {
    # drop the mlx verbose footer ("==========\nPrompt..Generation..Peak..") before parsing jac
    if "tokens-per-sec" not in out { return out; }
    return out.rsplit("==========", 1)[0];
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
    gen_tok_total: int = 0;       # tokens the model generated across all tasks
    correct_tokens: int = 0;      # tokens spent on the tasks that passed
    tps_sum: float = 0.0;         # sum of per-task generation tokens/sec
    tps_n: int = 0;
    limit: int = int(os.environ.get("JAC_EVAL_LIMIT", "0"));  # 0 = all; e.g. 50 during training

    for line in p.read_text().split("\n") {
        if line.strip() == "" { continue; }
        if limit > 0 and total >= limit { break; }
        rec: dict = dict(json.loads(line));
        func: str = str(rec["func_name"]);
        python: str = str(rec["python"]);
        prompt: str = str(rec["prompt"]);
        cases: list = list(rec["test_cases"]);
        total += 1;

        out: str = generate(prompt, python);
        gtok: int = gen_tokens(out);
        gtps: float = gen_tps(out);
        gen_tok_total += gtok;
        if gtps > 0.0 { tps_sum += gtps; tps_n += 1; }

        jac: str = extract_jac(strip_stats(out));
        name: str = detect_func(jac, func);
        (prog, expected) = build_program(jac, name, cases);
        (code, sout, serr) = run_jac(prog);
        # gate on RUN (execution), not `jac check`: the strict type-checker rejects
        # untyped-but-runnable code, which is not what "the conversion works" means.
        if code == 0 { comp += 1; }
        if code == 0 and sout.strip() == expected.strip() {
            passed += 1;
            correct_tokens += gtok;
        }
    }

    if total == 0 {
        print("=== conversion probe: no tasks found at " + HOLDOUT + " ===");
    } else {
        crate: int = comp * 100 // total;
        prate: int = passed * 100 // total;
        etps: int = 0;
        if tps_n > 0 { etps = int(tps_sum / float(tps_n)); }
        ttc: int = 0;
        if passed > 0 { ttc = correct_tokens // passed; }

        print("=== conversion probe (" + eval_mode() + ") on " + str(total) + " holdout tasks ===");
        print("runs (compiles+executes): " + str(crate) + "%  (" + str(comp) + "/" + str(total) + ")");
        print("cross-compiled test pass: " + str(prate) + "%  (" + str(passed) + "/" + str(total) + ")");
        print("generation:               " + str(gen_tok_total) + " tokens total, ~" + str(etps) + " tok/s avg");
        print("tokens-to-correct:        " + str(ttc) + " tokens / correct conversion");

        # append a metrics line for the learning curve (set by run_probe.sh per checkpoint)
        metrics_out: str = os.environ.get("JAC_EVAL_METRICS_OUT", "");
        if metrics_out != "" {
            step: int = int(os.environ.get("JAC_EVAL_STEP", "0"));
            row: dict = {"step": step, "total": total, "runs": comp, "test_pass": passed,
                         "runs_pct": crate, "test_pass_pct": prate,
                         "gen_tokens": gen_tok_total, "eval_tps": etps, "tokens_to_correct": ttc};
            with open(metrics_out, "a") as f { f.write(json.dumps(row) + "\n"); }
        }
    }
}
