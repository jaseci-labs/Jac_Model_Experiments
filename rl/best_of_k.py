"""Best-of-k DEPLOY accuracy: sample k, pick the first completion that compiles+runs
(compiler-verified, NO answer peek), score it. This is the honest deployable number
(vs pass@k which cheats by checking against the gold). Reuses the FIXED body-aware
extractor. Env: JAC_EVAL_MODEL, JAC_RL_HOLDOUT, K(8), TEMP(0.8)."""
import json, os, re, sys, tempfile, subprocess
import mlx_lm
from mlx_lm.sample_utils import make_sampler
MP=os.environ["JAC_EVAL_MODEL"]; H=os.environ.get("JAC_RL_HOLDOUT","dataset/rl/holdout.jsonl")
AD=os.environ.get("JAC_EVAL_ADAPTER",""); K=int(os.environ.get("K","8")); TEMP=float(os.environ.get("TEMP","0.8"))
model,tok=mlx_lm.load(MP, adapter_path=AD) if AD else mlx_lm.load(MP)
def gen(prompt,temp):
    text=tok.apply_chat_template([{"role":"user","content":prompt}],add_generation_prompt=True,tokenize=False)
    s=make_sampler(temp=temp); o=""
    for r in mlx_lm.stream_generate(model,tok,text,max_tokens=512,sampler=s): o+=r.text
    return o
def hole_name(t):
    m=re.search(r"(?:def|can|walker|node|obj|edge)\s+(\w+)\b[^{]*\{\s*__HOLE__",t); return m.group(1) if m else ""
def unit_body(code,name):
    if not name: return ""
    m=re.search(r"(?:def|can|walker|node|obj|edge)\s+"+re.escape(name)+r"\b[^{]*\{",code)
    if not m: return ""
    st=m.end()-1; d=0; i=st
    while i<len(code):
        c=code[i]
        if c=="{": d+=1
        elif c=="}":
            d-=1
            if d==0: return code[st+1:i].strip()
        i+=1
    return ""
def unwrap(s):
    s=s.strip()
    if re.match(r"^(can|def|walker|node|obj|edge|with)\b",s) and "{" in s and "}" in s: return s[s.index("{")+1:s.rindex("}")].strip()
    return s
def extract(o,t):
    code=o
    if "```jac" in o: code=o.split("```jac",1)[1].split("```",1)[0]
    elif "```" in o: code=o.split("```",1)[1].split("```",1)[0]
    b=unit_body(code,hole_name(t)); return b if b else unwrap(code)
def run(spliced,timeout):
    with tempfile.TemporaryDirectory() as d:
        f=os.path.join(d,"s.jac"); open(f,"w").write(spliced)
        try:
            p=subprocess.run(["jac","run",f],capture_output=True,text=True,timeout=timeout,cwd=d); return p.returncode,p.stdout
        except: return 124,""
tasks=[json.loads(l) for l in open(H) if l.strip()]
deploy=oracle=covered=greedy=0; N=len(tasks)
for rec in tasks:
    m=json.loads(rec["answer"]); tid=m["id"]; exp=m["expected_output"].strip(); to=int(m.get("timeout",20))
    tmpl=open("dataset/rl/templates/%s.jac"%tid).read()
    # greedy (k=1, temp 0) for reference
    gb=extract(gen(rec["prompt"],0.0),tmpl); gc,go=run(tmpl.replace("__HOLE__",gb),to)
    if gc==0 and go.strip()==exp: greedy+=1
    # k samples
    picked=None; any_correct=False; any_running=False
    for _ in range(K):
        b=extract(gen(rec["prompt"],TEMP),tmpl); c,out=run(tmpl.replace("__HOLE__",b),to)
        if c==0:
            any_running=True
            if picked is None: picked=(out.strip()==exp)   # deploy: FIRST running one, no peek
            if out.strip()==exp: any_correct=True
    if any_running: covered+=1
    if picked: deploy+=1
    if any_correct: oracle+=1
    print("%-26s greedy=%s deploy=%s oracle=%s"%(tid, gc==0 and go.strip()==exp, picked, any_correct),flush=True)
print("\n=== %s  (K=%d temp=%.1f, n=%d) ==="%(MP.split('/')[-1],K,TEMP,N))
print("greedy pass@1          : %d/%d (%.1f%%)"%(greedy,N,100*greedy/N))
print("best-of-k DEPLOY (pick 1st running, no peek): %d/%d (%.1f%%)"%(deploy,N,100*deploy/N))
print("  coverage (>=1 running available)          : %d/%d (%.1f%%)"%(covered,N,100*covered/N))
print("oracle pass@k (any sample correct)          : %d/%d (%.1f%%)"%(oracle,N,100*oracle/N))
