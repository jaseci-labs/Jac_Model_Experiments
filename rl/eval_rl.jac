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
import difflib;
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

def gen_mlx_sampled(model: any, tok: any, prompt: str, temp: float) -> str {
    # Temperature-sampled generation for pass@k (greedy can't show RL distribution
    # shifts; sampled reveals whether ANY of k tries passes).
    import mlx_lm;
    import from mlx_lm.sample_utils { make_sampler }
    msgs = [{"role": "user", "content": prompt}];
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False);
    sampler = make_sampler(temp=temp);
    out = "";
    for resp in mlx_lm.stream_generate(model, tok, text, max_tokens=512, sampler=sampler) {
        out += resp.text;
    }
    return out;
}

def passes(template: str, gen: str, expected: str, timeout: int) -> bool {
    body = extract_jac(gen);
    (code, out, err) = run_jac(template.replace("__HOLE__", body), timeout);
    return code == 0 and out.strip() == expected.strip();
}

def unwrap_unit(body: str) -> str {
    # Model usually emits the whole unit (`can X { ... }`) but the template hole
    # is INSIDE the unit -> nesting breaks it. Return the inner block when the
    # completion is one wrapped unit; leave bare statement bodies untouched.
    s = body.strip();
    if re.match(r"^(can|def|walker|node|obj|edge|with)\b", s) and "{" in s and "}" in s {
        return s[s.index("{") + 1 : s.rindex("}")].strip();
    }
    return s;
}

def extract_jac(output: str) -> str {
    if "```jac" in output {
        part = output.split("```jac", 1)[1];
        return unwrap_unit(part.split("```", 1)[0]);
    }
    if "```" in output {
        part = output.split("```", 1)[1];
        return unwrap_unit(part.split("```", 1)[0]);
    }
    return unwrap_unit(output);
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
        k: int = int(os.environ.get("JAC_EVAL_K", "1"));        # 1 = greedy only; >1 = sampled pass@k
        temp: float = float(os.environ.get("JAC_EVAL_TEMP", "1.0"));
        lines = [l for l in Path(holdout).read_text().split("\n") if l.strip()];
        if limit > 0 {
            lines = lines[:limit];
        }
        total = 0; runs = 0; passed = 0; passed_k = 0; idiom_sum = 0;
        osim_sum = 0.0; near_pass = 0;   # continuous progress metrics (detect partial gains)
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
            timeout = int(meta.get("timeout", 20));
            total += 1;
            # pass@1 (greedy) + continuous metrics
            gen = gen_mlx(model, tok, str(rec["prompt"]));
            body = extract_jac(gen);
            (code, out, err) = run_jac(template.replace("__HOLE__", body), timeout);
            osim = 0.0; p1 = False;
            if code == 0 {
                runs += 1;
                idiom_sum += idiom_count(body);
                osim = difflib.SequenceMatcher(None, out.strip(), expected.strip()).ratio();
                osim_sum += osim;
                if osim >= 0.9 { near_pass += 1; }
                if out.strip() == expected.strip() { passed += 1; p1 = True; }
            }
            # pass@k (sampled): any of k tries passes (greedy counts as one try)
            kp = p1;
            if k > 1 and not kp {
                for j in range(k) {
                    if passes(template, gen_mlx_sampled(model, tok, str(rec["prompt"]), temp), expected, timeout) {
                        kp = True;
                        break;
                    }
                }
            }
            if kp { passed_k += 1; }
            print(tid + ": pass@1=" + str(p1) + " pass@" + str(k) + "=" + str(kp) + " osim=" + str(round(osim, 2)));
        }
        run_pct = (100.0 * runs / total) if total else 0.0;
        pass_pct = (100.0 * passed / total) if total else 0.0;
        passk_pct = (100.0 * passed_k / total) if total else 0.0;
        near_pct = (100.0 * near_pass / total) if total else 0.0;
        avg_idiom = (idiom_sum / runs) if runs else 0.0;
        avg_osim = (osim_sum / total) if total else 0.0;   # over ALL tasks (0 for non-running)
        print("=== RL holdout eval ===");
        print("  model:    " + mp + ("  +adapter " + adapter if adapter else "  (base)"));
        print("  tasks:    " + str(total) + "   (k=" + str(k) + " temp=" + str(temp) + ")");
        print("  runs:     " + str(runs) + "  (" + str(round(run_pct, 1)) + "%)   compiles+executes");
        print("  pass@1:   " + str(passed) + "  (" + str(round(pass_pct, 1)) + "%)   greedy exact match");
        print("  pass@" + str(k) + ":   " + str(passed_k) + "  (" + str(round(passk_pct, 1)) + "%)   any of k sampled");
        print("  near-pass:" + str(near_pass) + "  (" + str(round(near_pct, 1)) + "%)   output-sim >= 0.9");
        print("  avg-osim: " + str(round(avg_osim, 3)) + "   continuous output similarity (all tasks)");
        print("  idiom:    " + str(round(avg_idiom, 2)) + " constructs/correct");
        out_path: str = os.environ.get("JAC_RL_OUT", "");
        if out_path {
            with open(out_path, "a") as fh {
                fh.write(json.dumps({
                    "model": mp, "adapter": adapter, "tasks": total, "k": k,
                    "runs": runs, "pass1": passed, "passk": passed_k, "near_pass": near_pass,
                    "run_pct": round(run_pct, 2), "pass1_pct": round(pass_pct, 2),
                    "passk_pct": round(passk_pct, 2), "near_pct": round(near_pct, 2),
                    "avg_osim": round(avg_osim, 4), "avg_idiom": round(avg_idiom, 2)
                }) + "\n");
            }
        }
    }
}
