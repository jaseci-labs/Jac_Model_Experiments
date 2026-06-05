"""Conversion-probe eval harness: score a model on the decontaminated holdout.

For each holdout task: generate Jac from the Python prompt, extract the code,
run it on the recorded behavioral test_cases, and compare outputs. Reports
run rate (compiles+executes), cross-compiled test pass rate, and token
efficiency (generation tokens, eval tokens/sec, tokens-to-correct). The gate is
`jac run`, not `jac check` — the strict type-checker rejects untyped-but-runnable
code, which is not what "the conversion works" means.

Model-agnostic via the mode (env JAC_EVAL_MODE):
  - "py2jac": smoke stand-in (jac py2jac of the source) -- NO LLM, proves the loop.
  - "mlx": real model loaded ONCE in-process (mlx_lm) -- adapter via JAC_EVAL_ADAPTER.

    JAC_EVAL_MODE=mlx JAC_EVAL_MODEL=models/x jac run srccurrent/jacgen/eval_probe.jac
"""

import json;
import os;
import re;
import from corpus { transpile, sanitize_transpile }
import from writer { run_jac }
import from pathlib { Path }
import from typing { Any }

glob HOLDOUT: str = "dataset/eval_holdout/conversion.jsonl";
glob GEN_MODE: str = "py2jac";   # default; override with env JAC_EVAL_MODE = py2jac | mlx
glob MLX_MODEL: str = "";        # default; override with env JAC_EVAL_MODEL = <model path>


def eval_mode() -> str {
    return os.environ.get("JAC_EVAL_MODE", GEN_MODE);
}


def load_model(mp: str, adapter: str) -> tuple {
    import mlx_lm;   # lazy: top-level import crashes the jac type-checker on mlx types
    if adapter != "" {
        return mlx_lm.load(mp, adapter_path=adapter);  # jac:ignore[E1032]
    }
    return mlx_lm.load(mp);  # jac:ignore[E1032]
}


def gen_mlx(model: Any, tok: Any, prompt: str) -> tuple {
    # one in-process generation against an already-loaded model (no reload per task)
    import mlx_lm;
    msgs: list = [{"role": "user", "content": prompt}];
    ptoks = tok.apply_chat_template(msgs, add_generation_prompt=True);
    text: str = "";
    gtok: int = 0;
    gtps: float = 0.0;
    for resp in mlx_lm.stream_generate(model, tok, ptoks, max_tokens=512) {  # jac:ignore[E1053,E1032]
        text += str(resp.text);
        gtok = int(resp.generation_tokens);
        gtps = float(resp.generation_tps);
    }
    return (text, gtok, gtps);
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
    m = re.search("def (\\w+)\\s*\\(", jac_code);
    if m is None { return fallback; }
    return m.group(1);
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
    p = Path(os.environ.get("JAC_HOLDOUT", HOLDOUT));   # override for the graph holdout
    mode: str = eval_mode();

    # load the model ONCE for the whole eval (the speed fix) -- adapter optional
    model: Any = None;
    tok: Any = None;
    if mode == "mlx" {
        mp: str = os.environ.get("JAC_EVAL_MODEL", MLX_MODEL);
        adapter: str = os.environ.get("JAC_EVAL_ADAPTER", "");
        (model, tok) = load_model(mp, adapter);
    }

    total: int = 0;
    comp: int = 0;
    passed: int = 0;
    gen_tok_total: int = 0;
    correct_tokens: int = 0;
    tps_sum: float = 0.0;
    tps_n: int = 0;
    limit: int = int(os.environ.get("JAC_EVAL_LIMIT", "0"));  # 0 = all; e.g. 20 during training

    for line in p.read_text().split("\n") {
        if line.strip() == "" { continue; }
        if limit > 0 and total >= limit { break; }
        rec: dict = dict(json.loads(line));
        func: str = str(rec["func_name"]);
        python: str = str(rec["python"]);
        prompt: str = str(rec["prompt"]);
        cases: list = list(rec["test_cases"]);
        total += 1;

        out: str = "";
        gtok: int = 0;
        gtps: float = 0.0;
        if mode == "py2jac" {
            out = sanitize_transpile(transpile(python));
        } else {
            res = gen_mlx(model, tok, prompt);
            out = str(res[0]);
            gtok = int(res[1]);
            gtps = float(res[2]);
        }
        gen_tok_total += gtok;
        if gtps > 0.0 { tps_sum += gtps; tps_n += 1; }

        jac: str = extract_jac(out);
        name: str = detect_func(jac, func);
        (prog, expected) = build_program(jac, name, cases);
        (code, sout, serr) = run_jac(prog);
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

        print("=== conversion probe (" + mode + ") on " + str(total) + " holdout tasks ===");
        print("runs (compiles+executes): " + str(crate) + "%  (" + str(comp) + "/" + str(total) + ")");
        print("cross-compiled test pass: " + str(prate) + "%  (" + str(passed) + "/" + str(total) + ")");
        print("generation:               " + str(gen_tok_total) + " tokens total, ~" + str(etps) + " tok/s avg");
        print("tokens-to-correct:        " + str(ttc) + " tokens / correct conversion");

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
