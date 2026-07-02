"""Assemble ALL corrected results into resultspub/rl/corrected_summary.json — the
one file the Studio RL section + make_graphs.py read."""
import os, re, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def cells(p):
    s = {}
    fp = os.path.join(ROOT, p)
    if not os.path.exists(fp):
        return s
    for l in open(fp):
        l = l.strip()
        if l:
            r = json.loads(l); s[r["tag"]] = r
    return s


def bok(f):
    p = os.path.join(ROOT, "results", f)
    if not os.path.exists(p):
        return None
    m = re.search(r"DEPLOY.*?\((\d+\.?\d*)%\)", open(p).read())
    return round(float(m.group(1)), 1) if m else None


c = cells("results/corrected_ladder.jsonl")
sg = cells("results/corrected_sg.jsonl")
cv = cells("results/conv_full.jsonl")
cl = cells("results/clean_measure.jsonl")
bg = cells("results/big_holdout.jsonl")
def g(t, f): return c.get("corr/jac-qwen3coder/" + t, {}).get(f)

ladder = [{"cell": lbl, "greedy": g(t, "pass1_pct"), "passk": g(t, "passk_pct")} for lbl, t in
          [("base", "rall/base"), ("SFT r5", "r5/sft"), ("SFT r20", "r20/sft"),
           ("SFT all", "rall/sft"), ("SFT+GRPO", "rall/sft_grpo"), ("raw-GRPO", "rall/raw_grpo")]]


def hd(name, n, gb, gs, bb, bs):
    return {"name": name, "n": n, "greedy_base": gb, "greedy_sft": gs, "bok_base": bb, "bok_sft": bs}


holdouts = [
    hd("pure-fn", 18, g("rall/base", "pass1_pct"), g("r20/sft", "pass1_pct"), bok("bok_base.log"), bok("bok_sft.log")),
    hd("graph", 17, sg.get("sg/jac/base", {}).get("pass1_pct"), sg.get("sg/jac/sft", {}).get("pass1_pct"), bok("bok_sg_base.log"), bok("bok_sg_sft.log")),
    hd("conversion", 11, cv.get("convf/jac/base", {}).get("pass1_pct"), cv.get("convf/jac/sft", {}).get("pass1_pct"), bok("bok_convf_base.log"), bok("bok_convf_sft.log")),
    hd("clean", 16, cl.get("clean/base", {}).get("pass1_pct"), cl.get("clean/sft", {}).get("pass1_pct"), bok("bok_clean_base.log"), bok("bok_clean_sft.log")),
    hd("big+fresh", 32, bg.get("big/base", {}).get("pass1_pct"), bg.get("big/sft", {}).get("pass1_pct"), bok("bok_big_base.log"), bok("bok_big_sft.log")),
]

# the story arc: how the headline number moved as bugs were fixed + levers added
journey = [
    {"stage": "v1 (broken eval)", "acc": 11.1},
    {"stage": "eval fixed · base greedy", "acc": 38.9},
    {"stage": "+ SFT · greedy", "acc": 61.1},
    {"stage": "+ best-of-k", "acc": 77.8},
    {"stage": "+ drop junk tasks", "acc": 93.8},
]
# sampling budget vs accuracy (jac-qwen3coder, pure-fn)
kscale = [{"k": "greedy", "acc": 38.9}, {"k": "k=8", "acc": 72.2}, {"k": "k=32", "acc": 88.9}]

out = {"model": "jac-qwen3coder", "ladder": ladder, "holdouts": holdouts,
       "journey": journey, "kscale": kscale,
       "headline": {"greedy_base": 38.9, "greedy_sft": 61.1, "bok_base": 72.2, "bok_sft": 77.8, "clean_ceiling": 93.8, "conv_peak": 81.8, "k32": 88.9},
       "note": "CORRECTED (fixed eval+reward). SFT lifts greedy 39->61%; best-of-k+compiler-verifier ships ~78% (94% on clean tasks); conversion+SFT peaks 82%; k=32 reaches 89%. GRPO~=SFT. SFT lift holds at n=32."}
open(os.path.join(ROOT, "resultspub", "rl", "corrected_summary.json"), "w").write(json.dumps(out, indent=1))
print("wrote corrected_summary.json —", len(holdouts), "holdouts,", len(journey), "journey stages")
