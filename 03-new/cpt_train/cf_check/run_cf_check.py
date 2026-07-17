#!/usr/bin/env python3
"""CF regression check: base qwen-q4 vs fused qwen-cpt-v1 on 16 general Python
coding tasks (tasks.py), NOT Jac -- tests whether CPT hurt general coding
ability, per design.md's CF guard. Exact-output graded, executed in a
subprocess with a timeout (never trust generated code in-process).
"""
import gc
import json
import re
import subprocess
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from tasks import TASKS

from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

ROOT = Path(__file__).resolve().parents[3]

GRADE_TEMPLATE = """
{code}

import sys, json
_tests = {tests!r}
_entry = {entry!r}
_fn = globals()[_entry]
_results = []
for args, expected in _tests:
    try:
        got = _fn(*args)
        _results.append(got == expected)
    except Exception as e:
        _results.append(False)
print(json.dumps(_results))
"""


def extract_code(text):
    m = re.search(r"```(?:python)?\n(.*?)```", text, re.S)
    return m.group(1).strip() if m else text.strip()


def grade(code, entry_point, tests):
    script = GRADE_TEMPLATE.format(code=code, tests=tests, entry=entry_point)
    tmp = ROOT / "03-new" / "cpt_train" / "cf_check" / "_grade_tmp.py"
    tmp.write_text(script)
    try:
        r = subprocess.run([sys.executable, str(tmp)], capture_output=True,
                            text=True, timeout=10)
        if r.returncode != 0:
            return False, 0, len(tests), r.stderr[-200:]
        results = json.loads(r.stdout.strip().splitlines()[-1])
        npass = sum(results)
        return npass == len(tests), npass, len(tests), ""
    except Exception as e:
        return False, 0, len(tests), str(e)
    finally:
        tmp.unlink(missing_ok=True)


def run_model(model_id, path, adapter_path=None):
    print(f"\n{'='*70}\nLOADING {model_id} ({path})"
          f"{' + adapter ' + adapter_path if adapter_path else ''}\n{'='*70}", flush=True)
    load_kwargs = {"adapter_path": str(ROOT / adapter_path)} if adapter_path else {}
    model, tok = load(str(ROOT / path), **load_kwargs)
    sampler = make_sampler(temp=0.2, top_p=0.9)
    results = []
    for t in TASKS:
        msgs = [{"role": "user", "content": t["prompt"] + "\n\nRespond with only the Python function, no explanation."}]
        ptoks = tok.apply_chat_template(msgs, add_generation_prompt=True)
        text = generate(model, tok, ptoks, max_tokens=300, sampler=sampler, verbose=False)
        code = extract_code(text)
        ok, npass, ntotal, err = grade(code, t["entry_point"], t["tests"])
        results.append({"id": t["id"], "output": text, "code": code,
                        "pass": ok, "npass": npass, "ntotal": ntotal, "err": err})
        print(f"  [{t['id']}] {npass}/{ntotal} {'PASS' if ok else 'FAIL'}" +
              (f" ({err})" if err else ""), flush=True)
    del model, tok
    gc.collect()
    return results


def main():
    base = run_model("qwen-q4", "models/qwen-q4")
    cpt = run_model("qwen-cpt-v1", "models/qwen-cpt-v1-fused-q4")

    out = {"base": base, "cpt": cpt}
    outp = ROOT / "03-new" / "results" / "cpt-v1" / "cf_check.json"
    outp.write_text(json.dumps(out, indent=2))

    base_pass = sum(r["pass"] for r in base)
    cpt_pass = sum(r["pass"] for r in cpt)
    n = len(TASKS)
    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    print(f"base qwen-q4:    {base_pass}/{n} ({100*base_pass/n:.0f}%)")
    print(f"cpt-v1:          {cpt_pass}/{n} ({100*cpt_pass/n:.0f}%)")
    print(f"delta:           {cpt_pass - base_pass:+d} tasks")
    for b, c in zip(base, cpt):
        if b["pass"] != c["pass"]:
            print(f"  DIFF {b['id']}: base={b['pass']} cpt={c['pass']}")
    print(f"\nwrote {outp}")


if __name__ == "__main__":
    sys.exit(main())
