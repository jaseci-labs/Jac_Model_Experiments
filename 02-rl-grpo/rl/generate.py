"""Best-of-k Jac generator (the deployable artifact).

Given a task (by id, or a raw prompt), sample k completions from the SFT model,
extract the Jac, and return the FIRST that the jac compiler accepts (compiles, and
runs if a template is available). The compiler is the verifier — no gold answer
needed. On the holdout this reaches ~78% vs ~55% greedy (see RL_FINDINGS.md).

  # by task id (splices into its template + runs — strongest verification)
  python3 02-rl-grpo/rl/generate.py --id lib_one_dp
  # by raw prompt (compile-check only)
  python3 02-rl-grpo/rl/generate.py --prompt "Write a Jac function that ..."
Env: JAC_GEN_MODEL (default the SFT rung-20 model), K (8), TEMP (0.8).
"""
import os, re, sys, json, argparse, tempfile, subprocess
import mlx_lm
from mlx_lm.sample_utils import make_sampler

DEFAULT_MODEL = "models/jac-qwen3coder-r20-rft-q4"
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
JAC = os.path.join(ROOT, ".venv", "bin", "jac")
JAC = JAC if os.path.exists(JAC) else "jac"

# --- extraction (mirrors the FIXED 02-rl-grpo/rl/eval_rl.jac: target-unit brace-match) ---
def hole_name(t):
    m = re.search(r"(?:def|can|walker|node|obj|edge)\s+(\w+)\b[^{]*\{\s*__HOLE__", t); return m.group(1) if m else ""
def unit_body(code, name):
    if not name: return ""
    m = re.search(r"(?:def|can|walker|node|obj|edge)\s+" + re.escape(name) + r"\b[^{]*\{", code)
    if not m: return ""
    st = m.end() - 1; d = 0; i = st
    while i < len(code):
        c = code[i]
        if c == "{": d += 1
        elif c == "}":
            d -= 1
            if d == 0: return code[st + 1:i].strip()
        i += 1
    return ""
def unwrap(s):
    s = s.strip()
    if re.match(r"^(can|def|walker|node|obj|edge|with)\b", s) and "{" in s and "}" in s:
        return s[s.index("{") + 1:s.rindex("}")].strip()
    return s
def extract(o, template=""):
    code = o
    if "```jac" in o: code = o.split("```jac", 1)[1].split("```", 1)[0]
    elif "```" in o: code = o.split("```", 1)[1].split("```", 1)[0]
    b = unit_body(code, hole_name(template)) if template else ""
    return b if b else unwrap(code)

def jac_ok(code, run):  # (rc, stdout); rc==0 means compiles (and runs if run=True)
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, "g.jac"); open(f, "w").write(code)
        cmd = [JAC, "run", f] if run else [JAC, "check", "-p", f]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=25, cwd=d)
            return p.returncode, p.stdout
        except Exception:
            return 124, ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id"); ap.add_argument("--prompt")
    a = ap.parse_args()
    mp = os.environ.get("JAC_GEN_MODEL", DEFAULT_MODEL)
    K = int(os.environ.get("K", "8")); TEMP = float(os.environ.get("TEMP", "0.8"))
    template = ""
    if a.id:
        tasks = {json.loads(json.loads(l)["answer"])["id"]: json.loads(l)
                 for l in open(os.path.join(ROOT, "02-rl-grpo/dataset/rl/tasks.jsonl")) if l.strip()}
        rec = tasks[a.id]; prompt = rec["prompt"]
        tp = os.path.join(ROOT, "02-rl-grpo/dataset/rl/templates", a.id + ".jac")
        template = open(tp).read() if os.path.exists(tp) else ""
    elif a.prompt:
        prompt = a.prompt
    else:
        prompt = sys.stdin.read()
    model, tok = mlx_lm.load(mp)
    sampler = make_sampler(temp=TEMP)
    text = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)
    best = None
    for j in range(K):
        o = ""
        for r in mlx_lm.stream_generate(model, tok, text, max_tokens=512, sampler=sampler): o += r.text
        body = extract(o, template)
        code = template.replace("__HOLE__", body) if template else body
        rc, out = jac_ok(code, run=bool(template))
        if rc == 0:
            print("# verified by jac (%s) on sample %d/%d\n" % ("run" if template else "check", j + 1, K))
            print(body)
            if template: print("\n# --- stdout ---\n" + out.strip(), file=sys.stderr)
            return
        if best is None: best = body
    print("# WARNING: no sample passed the compiler in %d tries — returning best-effort\n" % K)
    print(best or "")

if __name__ == "__main__":
    main()
