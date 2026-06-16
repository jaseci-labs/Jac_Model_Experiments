"""Idiom-quality judge: split a model's correct Jac into idiomatic vs Python-shaped.

The behavioral probe (eval_probe.jac) answers "does the Jac run and match outputs?"
It does NOT answer "is the Jac idiomatic, or just Python-shaped Jac that happens to
run?" — which matters because the training mix is 1:3 idiom:transpile, so a model can
score high on behavior by learning to transpile.

This judge measures idiom two ways, per holdout task, with NO extra model:
  1. divergence-from-transpile: rouge_l(model_output, py2jac(python)). The mechanical
     transpile IS the Python-shaped baseline. HIGH similarity => the model reproduced
     the transpile (Python-shaped). LOW similarity + still runs => it rewrote the code
     into something different/cleaner (idiomatic-divergent).
  2. graph-spatial construct usage: how many Jac-native constructs (walker/node/edge/
     ability/with-entry/visit/spawn/edge-ops/has-typed...) the output uses. Informational
     — simple functions legitimately need only `def` + `with entry`.

Buckets each RUNNING output as `python_shaped` (sim >= SIM_THRESH) or `idiomatic`
(sim < SIM_THRESH); non-running outputs are `broken`. Reports the split + averages.

    JAC_EVAL_MODE=mlx JAC_EVAL_MODEL=models/x JAC_EVAL_ADAPTER=adapters/y \
        jac run srccurrent/jacgen/idiom_eval.jac

Env: JAC_EVAL_MODE (py2jac|mlx), JAC_EVAL_MODEL, JAC_EVAL_ADAPTER, JAC_EVAL_LIMIT,
     JAC_IDIOM_SIM (sim threshold, default 0.7), JAC_IDIOM_OUT (metrics jsonl).
The py2jac mode is a sanity check: it should bucket ~100% python_shaped (sim 1.0).
"""

import json;
import os;
import re;
import from corpus { transpile, sanitize_transpile }
import from writer { run_jac }
import from dedup { rouge_l }
import from pathlib { Path }
import from typing { Any }

glob HOLDOUT: str = "dataset/eval_holdout/conversion.jsonl";

# Jac-native / graph-spatial constructs. Presence => the output leans on Jac idiom
# rather than plain imperative Python-in-Jac.
glob IDIOM_MARKERS: list[str] = [
    "walker ", "node ", "edge ", "with entry", "with exit", "can ",
    "visit ", "spawn", "disengage", "-->", "<--", "++>", "<++",
    " obj ", "has ", "glob ", "report ", " root", " here"
];


def eval_mode() -> str {
    return os.environ.get("JAC_EVAL_MODE", "py2jac");
}


def load_model(mp: str, adapter: str) -> tuple {
    import mlx_lm;   # lazy: top-level import crashes the jac type-checker on mlx types
    if adapter != "" {
        return mlx_lm.load(mp, adapter_path=adapter);  # jac:ignore[E1032]
    }
    return mlx_lm.load(mp);  # jac:ignore[E1032]
}


def gen_mlx(model: Any, tok: Any, prompt: str) -> str {
    import mlx_lm;
    msgs: list = [{"role": "user", "content": prompt}];
    ptoks = tok.apply_chat_template(msgs, add_generation_prompt=True);
    text: str = "";
    for resp in mlx_lm.stream_generate(model, tok, ptoks, max_tokens=512) {  # jac:ignore[E1053,E1032]
        text += str(resp.text);
    }
    return text;
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


def idiom_features(jac_code: str) -> int {
    n: int = 0;
    for marker in IDIOM_MARKERS {
        if marker in jac_code { n += 1; }
    }
    return n;
}


with entry {
    p = Path(os.environ.get("JAC_HOLDOUT", HOLDOUT));   # override for the graph holdout
    mode: str = eval_mode();
    sim_thresh: float = float(os.environ.get("JAC_IDIOM_SIM", "0.7"));
    limit: int = int(os.environ.get("JAC_EVAL_LIMIT", "0"));

    model: Any = None;
    tok: Any = None;
    if mode == "mlx" {
        mp: str = os.environ.get("JAC_EVAL_MODEL", "");
        adapter: str = os.environ.get("JAC_EVAL_ADAPTER", "");
        (model, tok) = load_model(mp, adapter);
    }

    total: int = 0;
    runs: int = 0;
    idiomatic: int = 0;
    python_shaped: int = 0;
    sim_sum: float = 0.0;
    feat_sum: int = 0;

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
        if mode == "py2jac" {
            out = sanitize_transpile(transpile(python));
        } else {
            out = gen_mlx(model, tok, prompt);
        }

        jac: str = extract_jac(out);
        name: str = detect_func(jac, func);
        (prog, expected) = build_program(jac, name, cases);
        (code, sout, serr) = run_jac(prog);
        ran: bool = (code == 0 and sout.strip() == expected.strip());
        if not ran { continue; }
        runs += 1;

        baseline: str = sanitize_transpile(transpile(python));   # Python-shaped reference
        sim: float = rouge_l(jac, baseline);
        feats: int = idiom_features(jac);
        sim_sum += sim;
        feat_sum += feats;
        if sim >= sim_thresh { python_shaped += 1; } else { idiomatic += 1; }
    }

    if total == 0 {
        print("=== idiom judge: no tasks at " + HOLDOUT + " ===");
    } else {
        runs_pct: int = runs * 100 // total;
        idiom_pct: int = 0;
        pyshape_pct: int = 0;
        avg_sim: float = 0.0;
        avg_feat: float = 0.0;
        if runs > 0 {
            idiom_pct = idiomatic * 100 // runs;
            pyshape_pct = python_shaped * 100 // runs;
            avg_sim = sim_sum / float(runs);
            avg_feat = float(feat_sum) / float(runs);
        }
        print("=== idiom judge (" + mode + ") on " + str(total) + " holdout tasks ===");
        print("correct (runs+matches):   " + str(runs_pct) + "%  (" + str(runs) + "/" + str(total) + ")");
        print("  of those -> idiomatic:  " + str(idiom_pct) + "%  (" + str(idiomatic) + ")  [diverged from transpile, sim<" + str(sim_thresh) + "]");
        print("  of those -> py-shaped:  " + str(pyshape_pct) + "%  (" + str(python_shaped) + ")  [reproduced the transpile, sim>=" + str(sim_thresh) + "]");
        print("avg transpile-similarity: " + str(round(avg_sim, 3)) + "  (1.0 = identical to py2jac; lower = more rewritten)");
        print("avg idiom-construct count:" + str(round(avg_feat, 2)) + "  (graph-spatial/Jac-native markers per output)");

        out_path: str = os.environ.get("JAC_IDIOM_OUT", "");
        if out_path != "" {
            row: dict = {"mode": mode, "total": total, "runs": runs,
                         "idiomatic": idiomatic, "python_shaped": python_shaped,
                         "avg_sim": round(avg_sim, 4), "avg_feat": round(avg_feat, 3)};
            with open(out_path, "a") as f { f.write(json.dumps(row) + "\n"); }
        }
    }
}
