#!/usr/bin/env python3
"""Ad hoc head-to-head: base qwen-q4 vs fused qwen-cpt-v1 on the same prompts.

Not a substitute for the real Checkpoint 1 (semantic MCQ + trust check) --
that's still unbuilt. This is a fast, real, compiler-verified sanity check on
whether CPT-v1 actually changed generation behavior, run because "does it
work" deserves an empirical answer, not just a loss curve.
"""
import gc
import json
import re
import subprocess
import sys
from pathlib import Path

from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

ROOT = Path(__file__).resolve().parents[2]
JAC = ROOT / ".venv" / "bin" / "jac"

PROMPTS = [
    {
        "id": "syntax_fn",
        "kind": "code",
        "prompt": "Write a Jac function `add(a: int, b: int) -> int` that returns their sum.",
    },
    {
        "id": "osp_walker",
        "kind": "code",
        "prompt": "Write a Jac walker named `CountNodes` that starts at root, visits every "
                  "node connected via outgoing edges, and prints the total count of nodes visited.",
    },
    {
        "id": "py2jac",
        "kind": "code",
        "prompt": "Convert this Python function to idiomatic Jac:\n```python\ndef is_even(n):\n    return n % 2 == 0\n```",
    },
    {
        "id": "concept_node_edge",
        "kind": "concept",
        "prompt": "In Jac's Object-Spatial Programming model, what is the difference between a `node` and an `edge`? Answer in 2-3 sentences.",
    },
    {
        "id": "concept_by_llm",
        "kind": "concept",
        "prompt": "What does the `by llm()` construct do in Jac? Answer in 2-3 sentences.",
    },
]


def extract_code(text):
    m = re.search(r"```(?:jac)?\n(.*?)```", text, re.S)
    return m.group(1).strip() if m else text.strip()


def jac_check(code):
    tmp = ROOT / "03-new" / "cpt_train" / "_h2h_tmp.jac"
    tmp.write_text(code)
    try:
        r = subprocess.run([str(JAC), "check", "-p", str(tmp)],
                            capture_output=True, text=True, timeout=30)
        out = r.stdout + r.stderr
        return "PASSED" in out
    except Exception as e:
        return False
    finally:
        tmp.unlink(missing_ok=True)


def run_model(model_id, path):
    print(f"\n{'='*70}\nLOADING {model_id} ({path})\n{'='*70}", flush=True)
    model, tok = load(str(ROOT / path))
    sampler = make_sampler(temp=0.2, top_p=0.9)
    results = []
    for p in PROMPTS:
        msgs = [{"role": "user", "content": p["prompt"]}]
        ptoks = tok.apply_chat_template(msgs, add_generation_prompt=True)
        text = generate(model, tok, ptoks, max_tokens=400, sampler=sampler, verbose=False)
        row = {"id": p["id"], "kind": p["kind"], "output": text}
        if p["kind"] == "code":
            code = extract_code(text)
            row["code"] = code
            row["compiles"] = jac_check(code)
        results.append(row)
        print(f"  [{p['id']}] done ({len(text)} chars)" +
              (f" compiles={row.get('compiles')}" if p["kind"] == "code" else ""), flush=True)
    del model, tok
    gc.collect()
    return results


def main():
    base = run_model("qwen-q4", "models/qwen-q4")
    cpt = run_model("qwen-cpt-v1", "models/qwen-cpt-v1-fused-q4")

    out = {"base": base, "cpt": cpt}
    outp = ROOT / "03-new" / "results" / "cpt-v1" / "headtohead.json"
    outp.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {outp}")

    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    for b, c in zip(base, cpt):
        line = f"{b['id']:20s} kind={b['kind']:8s}"
        if b["kind"] == "code":
            line += f" base_compiles={b['compiles']!s:5s} cpt_compiles={c['compiles']!s:5s}"
        print(line)


if __name__ == "__main__":
    sys.exit(main())
