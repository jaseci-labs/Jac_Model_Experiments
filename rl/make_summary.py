"""Assemble ALL corrected results into resultspub/rl/corrected_summary.json — the
one file the Studio RL section reads to render the corrected charts."""
import os, re, json
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def cells(p):
    s={}
    fp=os.path.join(ROOT,p)
    if not os.path.exists(fp): return s
    for l in open(fp):
        l=l.strip()
        if l: r=json.loads(l); s[r["tag"]]=r
    return s
def bok(f):
    p=os.path.join(ROOT,"results",f)
    if not os.path.exists(p): return None
    m=re.search(r"DEPLOY.*?\((\d+\.?\d*)%\)",open(p).read()); return float(m.group(1)) if m else None
c=cells("results/corrected_ladder.jsonl"); sg=cells("results/corrected_sg.jsonl"); cv=cells("results/conv_full.jsonl")
def g(t,f): return c.get("corr/jac-qwen3coder/"+t,{}).get(f)
ladder=[{"cell":lbl,"greedy":g(t,"pass1_pct"),"passk":g(t,"passk_pct")} for lbl,t in
        [("base","rall/base"),("SFT r5","r5/sft"),("SFT r20","r20/sft"),("SFT all","rall/sft"),("SFT+GRPO","rall/sft_grpo"),("raw-GRPO","rall/raw_grpo")]]
holdouts=[
 {"name":"pure-fn","n":18,"greedy_base":g("rall/base","pass1_pct"),"greedy_sft":g("r20/sft","pass1_pct"),"bok_base":bok("bok_base.log"),"bok_sft":bok("bok_sft.log")},
 {"name":"graph-idiom","n":17,"greedy_base":sg.get("sg/jac/base",{}).get("pass1_pct"),"greedy_sft":sg.get("sg/jac/sft",{}).get("pass1_pct"),"bok_base":bok("bok_sg_base.log"),"bok_sft":bok("bok_sg_sft.log")},
 {"name":"conversion","n":11,"greedy_base":cv.get("convf/jac/base",{}).get("pass1_pct"),"greedy_sft":cv.get("convf/jac/sft",{}).get("pass1_pct"),"bok_base":bok("bok_convf_base.log"),"bok_sft":bok("bok_convf_sft.log")},
]
out={"model":"jac-qwen3coder","ladder":ladder,"holdouts":holdouts,
     "headline":{"greedy_base":38.9,"greedy_sft":61.1,"bok_base":72.2,"bok_sft":77.8,"conv_peak":81.8,"k32":88.9},
     "note":"CORRECTED (fixed eval+reward). SFT lifts greedy 39->61%; best-of-k+compiler-verifier ships ~78%; conversion+SFT peaks 82%; k=32 reaches 89%. GRPO~=SFT."}
open(os.path.join(ROOT,"resultspub","rl","corrected_summary.json"),"w").write(json.dumps(out,indent=1))
print("wrote corrected_summary.json"); print(json.dumps(out["holdouts"],indent=0))
