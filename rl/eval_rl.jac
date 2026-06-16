"""Score a model on the RL holdout (dataset/rl/holdout.jsonl).

Loads the MLX model ONCE (eval_probe.jac pattern), generates a completion per
held-out task, splices it into the task template, runs it isolated (`jac run`,
cwd=tmp so the persistent root graph never accumulates), and reports
run% / behavior-pass% / avg idiom-construct count. Pass the GRPO adapter via
JAC_EVAL_ADAPTER to measure the post-RL model; omit it for the base.

  JAC_EVAL_MODEL=models/qwen-q4 JAC_EVAL_ADAPTER=adapters/qwen3-grpo jac run rl/eval_rl.jac

Env: JAC_EVAL_MODEL (required), JAC_EVAL_ADAPTER (""), JAC_RL_HOLDOUT
     (dataset/rl/holdout.jsonl), JAC_RL_OUT (append a metrics JSON line),
     JAC_EVAL_LIMIT (0=all). Lazy-imports mlx_lm -> parse-check only (-p).
"""

import os;
import re;
import json;
import tempfile;
import subprocess;
import from pathlib { Path }

glob IDIOM_MARKERS: list[str] = [
    "walker ", "node ", "edge ", "with entry", "with exit", "can ",
    "visit ", "spawn", "disengage", "-->", "<--", "++>", "<++",
    " obj ", "has ", "report ", " root", " here"
];

def load_model(mp: str, adapter: str) -> tuple {
    import mlx_lm;
    if adapter {
        return mlx_lm.load(mp, adapter_path=adapter);
    }
    return mlx_lm.load(mp);
}

def gen_mlx(model: any, tok: any, prompt: str) -> str {
    import mlx_lm;
    msgs = [{"role": "user", "content": prompt}];
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False);
    out = "";
    for resp in mlx_lm.stream_generate(model, tok, text, max_tokens=512) {
        out += resp.text;
    }
    return out;
}

def extract_jac(output: str) -> str {
    if "```jac" in output {
        part = output.split("```jac", 1)[1];
        return part.split("```", 1)[0].strip();
    }
    if "```" in output {
        part = output.split("```", 1)[1];
        return part.split("```", 1)[0].strip();
    }
    return output.strip();
}

def run_jac(code: str, timeout: int = 20) -> tuple[int, str, str] {
    with tempfile.TemporaryDirectory() as d {
        f = Path(str(d)) / "snippet.jac";
        f.write_text(code);
        try {
            p = subprocess.run(
                ["jac", "run", str(f)], capture_output=True, text=True, timeout=timeout, cwd=str(d)
            );
            return (p.returncode, p.stdout, p.stderr);
        } except (subprocess.SubprocessError, OSError) as e {
            return (124, "", str(e));
        }
    }
}

def idiom_count(body: str) -> int {
    n = 0;
    for m in IDIOM_MARKERS {
        n += body.count(m);
    }
    return n;
}

with entry {
    mp: str = os.environ.get("JAC_EVAL_MODEL", "");
    adapter: str = os.environ.get("JAC_EVAL_ADAPTER", "");
    holdout: str = os.environ.get("JAC_RL_HOLDOUT", "dataset/rl/holdout.jsonl");
    limit: int = int(os.environ.get("JAC_EVAL_LIMIT", "0"));
    if not mp {
        print("set JAC_EVAL_MODEL");
    } else {
        (model, tok) = load_model(mp, adapter);
        lines = [l for l in Path(holdout).read_text().split("\n") if l.strip()];
        if limit > 0 {
            lines = lines[:limit];
        }
        total = 0; runs = 0; passed = 0; idiom_sum = 0;
        for ln in lines {
            rec = json.loads(ln);
            meta = json.loads(rec["answer"]);
            tid = str(meta["id"]);
            expected = str(meta["expected_output"]);
            tmpl_path = "dataset/rl/templates/" + tid + ".jac";
            if not Path(tmpl_path).exists() {
                continue;
            }
            template = Path(tmpl_path).read_text();
            total += 1;
            gen = gen_mlx(model, tok, str(rec["prompt"]));
            body = extract_jac(gen);
            spliced = template.replace("__HOLE__", body);
            (code, out, err) = run_jac(spliced, int(meta.get("timeout", 20)));
            if code == 0 {
                runs += 1;
                idiom_sum += idiom_count(body);
                if out.strip() == expected.strip() {
                    passed += 1;
                }
            }
            print(tid + ": run=" + str(code == 0) + " pass=" + str(code == 0 and out.strip() == expected.strip()));
        }
        run_pct = (100.0 * runs / total) if total else 0.0;
        pass_pct = (100.0 * passed / total) if total else 0.0;
        avg_idiom = (idiom_sum / runs) if runs else 0.0;
        print("=== RL holdout eval ===");
        print("  model:   " + mp + ("  +adapter " + adapter if adapter else "  (base)"));
        print("  tasks:   " + str(total));
        print("  runs:    " + str(runs) + "  (" + str(round(run_pct, 1)) + "%)");
        print("  pass:    " + str(passed) + "  (" + str(round(pass_pct, 1)) + "%)");
        print("  idiom:   " + str(round(avg_idiom, 2)) + " constructs/correct");
        out_path: str = os.environ.get("JAC_RL_OUT", "");
        if out_path {
            with open(out_path, "a") as fh {
                fh.write(json.dumps({
                    "model": mp, "adapter": adapter, "tasks": total,
                    "runs": runs, "pass": passed,
                    "run_pct": round(run_pct, 2), "pass_pct": round(pass_pct, 2),
                    "avg_idiom": round(avg_idiom, 2)
                }) + "\n");
            }
        }
    }
}
