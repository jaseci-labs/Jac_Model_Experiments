"""Structural-variety graph tasks: typed/weighted EDGES + linked-list CHAINS.

The first 24 tasks share one skeleton (build-adj + accumulator-walker over nodes).
These add genuinely different idiom:
 - weighted typed edges: `edge Road { has w: int; }`, `+>:Road(w=..):+>`,
   iterate `[edge here -->]`, read `e.w`. Input adj is {node: [[target, weight], ...]}.
 - linked-list chains: input is a flat list -> a chain of nodes wired `++>` -> walker.

Expected outputs are derived by EXECUTING the Python reference; every Jac is validated
on every case via `jac run`. Results are MERGED into train.json / holdout.json (appended
to the existing 24/10), then graph_seeds.jac / graph_holdout.jac pick them up.

    python3 srccurrent/jacgen/graph_data/gen_graph_struct.py
"""
import json, subprocess, tempfile, os, sys

JAC = ".venv/bin/jac"
HERE = os.path.dirname(__file__)

# weighted-edge input pool: adj = {node: [[target, weight], ...]}
POOL_W = [
    ({"a": [["b", 5], ["c", 3]], "b": [["c", 2]], "c": []}, "a"),
    ({"a": [["b", 10]], "b": [["c", 1]], "c": [["d", 4]], "d": []}, "a"),
    ({"r": [["x", 2], ["y", 8], ["z", 6]], "x": [], "y": [], "z": []}, "r"),
    ({"a": []}, "a"),
]
# chain input pool: a flat list
POOL_C = [[3, 1, 4, 1, 5], [10, 20, 30], [-2, 7, -1], [4], []]

TASKS = []
def add(name, kind, py, jac, thr=None):
    TASKS.append((name, kind, py.strip("\n"), jac.strip("\n"), thr))

# ---- weighted-edge builder (shared jac scaffold) ----
def wjac(node_extra, walker, ret):
    needs_k = "self.k" in walker[2]
    thr_line = ("\n    thr: int = int(spec[\"threshold\"]);" if needs_k else "")
    return f'''
edge Road {{ has w: int; }}
node WC {{ has name: str; has seen: bool = False; }}

walker {walker[0]} {{
    {walker[1]}
    can step with WC entry {{
        if not here.seen {{
            here.seen = True;
            {walker[2]}
            visit [-->];
        }}
    }}
}}

def {walker[3]}(spec: dict) -> int {{
    adj: dict = dict(spec["adj"]); start: str = str(spec["start"]);{thr_line}
    nodes: dict = {{}};
    for nm in adj.keys() {{ nodes[nm] = WC(name=str(nm)); }}
    for (s, ts) in adj.items() {{ for pr in list(ts) {{ tg = str(pr[0]); if tg not in nodes {{ nodes[tg] = WC(name=tg); }} }} }}
    for (s, ts) in adj.items() {{ for pr in list(ts) {{ nodes[s] +>:Road(w=int(pr[1])):+> nodes[str(pr[0])]; }} }}
    w = nodes[start] spawn {walker[0]}({("k=thr" if "self.k" in walker[2] else "")});
    return {ret};
}}
'''

PY_W = '''
def {name}(spec: dict) -> int:
    """{doc}"""
    adj = spec["adj"]; start = spec["start"]{kline}
    seen = set(); stack = [start]; acc = {init}
    while stack:
        n = stack.pop()
        if n in seen: continue
        seen.add(n)
        for tgt, w in adj.get(n, []):
            {body}
            stack.append(tgt)
    return acc
'''

# weighted tasks: (name, doc, init, py_body(uses w,k,acc), jac walker tuple, thr)
add("sum_edge_weights", "wsum",
    PY_W.format(name="sum_edge_weights", doc="Sum the weights of all edges reachable from start.",
                kline="", init="0", body="acc += w"),
    wjac(None, ("SumW", "has total: int = 0;", "for e in [edge here -->] { self.total += e.w; }", "sum_edge_weights"), "w.total"))
add("count_edges", "wcount",
    PY_W.format(name="count_edges", doc="Count the edges reachable from start.",
                kline="", init="0", body="acc += 1"),
    wjac(None, ("CntE", "has count: int = 0;", "for e in [edge here -->] { self.count += 1; }", "count_edges"), "w.count"))
add("max_edge_weight", "wmax",
    PY_W.format(name="max_edge_weight", doc="Return the maximum edge weight reachable (0 if none).",
                kline="", init="0", body="acc = w if w > acc else acc"),
    wjac(None, ("MaxE", "has best: int = 0;", "for e in [edge here -->] { if e.w > self.best { self.best = e.w; } }", "max_edge_weight"), "w.best"))
add("count_heavy_edges", "wheavy",
    PY_W.format(name="count_heavy_edges", doc="Count edges whose weight exceeds the threshold.",
                kline="; thr = spec[\"threshold\"]", init="0", body="acc += 1 if w > thr else 0"),
    wjac(None, ("Heavy", "has k: int;\n    has count: int = 0;", "for e in [edge here -->] { if e.w > self.k { self.count += 1; } }", "count_heavy_edges"), "w.count"),
    thr=4)

# ---- chain builder ----
def cjac(walker, ret):
    return f'''
node Cell {{ has val: int; has seen: bool = False; }}

walker {walker[0]} {{
    {walker[1]}
    can step with Cell entry {{
        if not here.seen {{ here.seen = True; {walker[2]} visit [-->]; }}
    }}
}}

def {walker[3]}(spec: dict) -> int {{
    items: list = list(spec["items"]);
    if len(items) == 0 {{ return {walker[4]}; }}
    cells: list = [];
    for v in items {{ cells.append(Cell(val=int(v))); }}
    i: int = 0;
    while i < len(cells) - 1 {{ cells[i] ++> cells[i + 1]; i += 1; }}
    w = cells[0] spawn {walker[0]}();
    return {ret};
}}
'''
PY_C = '''
def {name}(spec: dict) -> int:
    """{doc}"""
    items = spec["items"]
    {body}
'''
add("sum_chain", "c", PY_C.format(name="sum_chain", doc="Sum the values in the chain.", body="return sum(items) if items else 0"),
    cjac(("SumC", "has total: int = 0;", "self.total += here.val;", "sum_chain", "0"), "w.total"))
add("max_chain", "c", PY_C.format(name="max_chain", doc="Return the maximum value in the chain (0 if empty).", body="return max(items) if items else 0"),
    cjac(("MaxC", "has best: int = -2147483648;", "if here.val > self.best { self.best = here.val; }", "max_chain", "0"), "w.best"))
add("count_chain", "c", PY_C.format(name="count_chain", doc="Count the elements in the chain.", body="return len(items)"),
    cjac(("CntC", "has count: int = 0;", "self.count += 1;", "count_chain", "0"), "w.count"))
add("count_positive_chain", "c", PY_C.format(name="count_positive_chain", doc="Count positive values in the chain.", body="return sum(1 for v in items if v > 0)"),
    cjac(("PosC", "has count: int = 0;", "if here.val > 0 { self.count += 1; }", "count_positive_chain", "0"), "w.count"))
add("last_chain", "c", PY_C.format(name="last_chain", doc="Return the last value in the chain (0 if empty).", body="return items[-1] if items else 0"),
    cjac(("LastC", "has last: int = 0;", "self.last = here.val;", "last_chain", "0"), "w.last"))
add("min_chain", "c", PY_C.format(name="min_chain", doc="Return the minimum value in the chain (0 if empty).", body="return min(items) if items else 0"),
    cjac(("MinC", "has best: int = 2147483647;", "if here.val < self.best { self.best = here.val; }", "min_chain", "0"), "w.best"))

HOLDOUT_NAMES = {"count_heavy_edges", "last_chain", "min_chain"}  # disjoint-ish from train skeletons

def make_cases(name, kind, py, thr):
    ns = {}
    exec(py, {"sum": sum, "max": max, "min": min, "len": len, "abs": abs}, ns)
    fn = ns[name]
    cases = []
    if kind.startswith("w"):
        for adj, st in POOL_W:
            spec = {"adj": adj, "start": st}
            if thr is not None: spec["threshold"] = thr
            cases.append({"input": spec, "output": str(fn(spec))})
    else:
        for items in POOL_C:
            spec = {"items": items}
            cases.append({"input": spec, "output": str(fn(spec))})
    return cases

def run_case(jac, func, inp):
    prog = jac + "\n\nwith entry {\n    print(" + func + "(" + repr(inp) + "));\n}\n"
    with tempfile.NamedTemporaryFile("w", suffix=".jac", delete=False, dir="/tmp") as f:
        f.write(prog); p = f.name
    try:
        r = subprocess.run([JAC, "run", p], capture_output=True, text=True, timeout=30)
        return r.returncode, r.stdout.strip(), r.stderr.strip()
    finally:
        os.unlink(p)

train_new, hold_new, ok = [], [], True
for (name, kind, py, jac, thr) in TASKS:
    cases = make_cases(name, kind, py, thr)
    good = True
    for c in cases:
        rc, out, err = run_case(jac, name, c["input"])
        if not (rc == 0 and out == c["output"]):
            ok = False; good = False
            print(f"FAIL {name} in={c['input']} exp={c['output']} got={out!r} rc={rc}")
            if err: print("   ", err.splitlines()[-1] if err.splitlines() else err)
            break
    if good:
        task = {"name": name, "difficulty": "composed", "py": py, "jac": jac, "cases": cases}
        (hold_new if name in HOLDOUT_NAMES else train_new).append(task)
        print(f"OK   {name} ({len(cases)} cases)")

if not ok:
    print("\nFAILURES — not merging"); sys.exit(1)

# merge (append, dedup by name)
for fn, extra in [("train.json", train_new), ("holdout.json", hold_new)]:
    path = os.path.join(HERE, fn)
    existing = json.load(open(path))
    have = {t["name"] for t in existing}
    merged = existing + [t for t in extra if t["name"] not in have]
    json.dump(merged, open(path, "w"), indent=1)
    print(f"{fn}: {len(existing)} -> {len(merged)}")
