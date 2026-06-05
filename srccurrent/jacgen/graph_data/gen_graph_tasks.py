"""Scale the graph idiom tier: generate diverse, validated graph-conversion tasks.

Each task: a single-dict-arg `def` over a graph. Python = dict+stack traversal;
idiomatic Jac = build nodes/edges + spawn an accumulator walker. We auto-derive the
expected outputs by EXECUTING the Python reference on a fixed graph pool (no hand
errors), then VALIDATE the Jac produces the same via `jac run`. Emits train.json +
holdout.json (disjoint predicate sets). Re-run after editing SPECS:

    python3 srccurrent/jacgen/graph_data/gen_graph_tasks.py

Diversity note: tasks share the build+walker skeleton on purpose — that skeleton IS
the graph idiom the model must learn; the predicate/aggregation is what varies.
"""
import json, subprocess, tempfile, os, sys, textwrap

JAC = ".venv/bin/jac"
HERE = os.path.dirname(__file__)

# fixed graph pool: (adj, vals) — varied shapes (branch, chain, deep, wide)
POOL = [
    ({"a": ["b", "c"], "b": ["d"], "c": ["d"], "d": []},
     {"a": 5, "b": 8, "c": 2, "d": 10}),
    ({"a": ["b"], "b": ["c"], "c": ["d"], "d": []},
     {"a": 3, "b": 6, "c": 9, "d": 12}),
    ({"r": ["x", "y", "z"], "x": [], "y": ["w"], "z": [], "w": []},
     {"r": 1, "x": 2, "y": 3, "z": 4, "w": 7}),
    ({"a": ["b", "c"], "b": [], "c": []},
     {"a": -3, "b": 4, "c": -1}),
    ({"n": []}, {"n": 0}),
]
STARTS = ["a", "a", "r", "a", "n"]

# Each spec: name, doc, kind, and the per-node Python predicate/expr + Jac equivalent.
# kind: 'count'(count where pred), 'sum'(sum of expr), 'max'/'min'/'product'/'range'/
#       'leaves'(count nodes with no children), 'branches'(>=2 children), 'reach'(count).
# pred/expr use `v` (python) and `here.val` (jac).  needs_thr -> param k from spec.thr
SPECS_TRAIN = [
    ("count_reachable", "Count nodes reachable from start.", "reach", None, None, None),
    ("sum_reachable", "Sum the values of all nodes reachable from start.", "sum", "v", "here.val", None),
    ("max_reachable", "Return the maximum node value reachable from start.", "max", None, None, None),
    ("min_reachable", "Return the minimum node value reachable from start.", "min", None, None, None),
    ("product_reachable", "Multiply the values of all nodes reachable from start.", "product", None, None, None),
    ("range_reachable", "Return max minus min over reachable node values.", "range", None, None, None),
    ("count_leaves", "Count reachable nodes that have no children.", "leaves", None, None, None),
    ("count_branches", "Count reachable nodes with two or more children.", "branches", None, None, None),
    ("count_even", "Count reachable nodes whose value is even.", "count", "v % 2 == 0", "here.val % 2 == 0", None),
    ("count_odd", "Count reachable nodes whose value is odd.", "count", "v % 2 == 1", "here.val % 2 == 1", None),
    ("count_positive", "Count reachable nodes with a positive value.", "count", "v > 0", "here.val > 0", None),
    ("count_negative", "Count reachable nodes with a negative value.", "count", "v < 0", "here.val < 0", None),
    ("count_zero", "Count reachable nodes whose value is zero.", "count", "v == 0", "here.val == 0", None),
    ("count_div3", "Count reachable nodes whose value is divisible by 3.", "count", "v % 3 == 0", "here.val % 3 == 0", None),
    ("count_above", "Count reachable nodes whose value exceeds the threshold.", "count", "v > k", "here.val > self.k", 5),
    ("count_below", "Count reachable nodes whose value is below the threshold.", "count", "v < k", "here.val < self.k", 5),
    ("sum_even", "Sum the values of reachable nodes whose value is even.", "sum", "v if v % 2 == 0 else 0", "(here.val if here.val % 2 == 0 else 0)", None),
    ("sum_positive", "Sum the positive values among reachable nodes.", "sum", "v if v > 0 else 0", "(here.val if here.val > 0 else 0)", None),
    ("sum_squares", "Sum the squares of reachable node values.", "sum", "v * v", "here.val * here.val", None),
    ("sum_above", "Sum reachable node values that exceed the threshold.", "sum", "v if v > k else 0", "(here.val if here.val > self.k else 0)", 4),
    ("count_abs_above", "Count reachable nodes whose absolute value exceeds the threshold.", "count", "abs(v) > k", "(here.val if here.val >= 0 else -here.val) > self.k", 5),
    ("max_abs", "Return the maximum absolute node value reachable.", "maxabs", None, None, None),
    ("count_leaf_or_branch", "Count reachable nodes that are leaves or branches (not single-child).", "leafbranch", None, None, None),
    ("sum_leaf_values", "Sum the values of reachable leaf nodes (no children).", "sumleaf", None, None, None),
]
SPECS_HOLDOUT = [
    ("count_ge_threshold", "Count reachable nodes whose value is >= the threshold.", "count", "v >= k", "here.val >= self.k", 6),
    ("count_le_threshold", "Count reachable nodes whose value is <= the threshold.", "count", "v <= k", "here.val <= self.k", 4),
    ("sum_odd", "Sum the values of reachable nodes whose value is odd.", "sum", "v if v % 2 == 1 else 0", "(here.val if here.val % 2 == 1 else 0)", None),
    ("count_div2or3", "Count reachable nodes divisible by 2 or 3.", "count", "v % 2 == 0 or v % 3 == 0", "(here.val % 2 == 0 or here.val % 3 == 0)", None),
    ("sum_abs", "Sum the absolute values of reachable nodes.", "sum", "abs(v)", "(here.val if here.val >= 0 else -here.val)", None),
    ("count_in_band", "Count reachable nodes with value strictly between 0 and the threshold.", "count", "0 < v < k", "(here.val > 0 and here.val < self.k)", 9),
    ("max_even", "Return the maximum even value reachable (or -1 if none).", "maxeven", None, None, None),
    ("count_single_child", "Count reachable nodes that have exactly one child.", "single", None, None, None),
    ("product_positive", "Multiply the positive values among reachable nodes (1 if none).", "prodpos", None, None, None),
    ("count_gt_double_threshold", "Count reachable nodes whose value exceeds twice the threshold.", "count", "v > 2 * k", "here.val > 2 * self.k", 3),
]

PY_TMPL = {
"reach": '''
def {name}(spec: dict) -> int:
    """{doc}"""
    adj = spec["adj"]; start = spec["start"]
    seen = set(); stack = [start]
    while stack:
        n = stack.pop()
        if n in seen: continue
        seen.add(n)
        stack.extend(adj.get(n, []))
    return len(seen)
''',
"count": '''
def {name}(spec: dict) -> int:
    """{doc}"""
    adj = spec["adj"]; vals = spec["vals"]; start = spec["start"]{kline_py}
    seen = set(); stack = [start]; acc = 0
    while stack:
        n = stack.pop()
        if n in seen: continue
        seen.add(n); v = vals[n]
        if {pred}: acc += 1
        stack.extend(adj.get(n, []))
    return acc
''',
"sum": '''
def {name}(spec: dict) -> int:
    """{doc}"""
    adj = spec["adj"]; vals = spec["vals"]; start = spec["start"]{kline_py}
    seen = set(); stack = [start]; acc = 0
    while stack:
        n = stack.pop()
        if n in seen: continue
        seen.add(n); v = vals[n]
        acc += {expr}
        stack.extend(adj.get(n, []))
    return acc
''',
"max": '''
def {name}(spec: dict) -> int:
    """{doc}"""
    adj = spec["adj"]; vals = spec["vals"]; start = spec["start"]
    seen = set(); stack = [start]; best = None
    while stack:
        n = stack.pop()
        if n in seen: continue
        seen.add(n)
        if best is None or vals[n] > best: best = vals[n]
        stack.extend(adj.get(n, []))
    return best
''',
"min": '''
def {name}(spec: dict) -> int:
    """{doc}"""
    adj = spec["adj"]; vals = spec["vals"]; start = spec["start"]
    seen = set(); stack = [start]; best = None
    while stack:
        n = stack.pop()
        if n in seen: continue
        seen.add(n)
        if best is None or vals[n] < best: best = vals[n]
        stack.extend(adj.get(n, []))
    return best
''',
"product": '''
def {name}(spec: dict) -> int:
    """{doc}"""
    adj = spec["adj"]; vals = spec["vals"]; start = spec["start"]
    seen = set(); stack = [start]; p = 1
    while stack:
        n = stack.pop()
        if n in seen: continue
        seen.add(n); p *= vals[n]
        stack.extend(adj.get(n, []))
    return p
''',
"range": '''
def {name}(spec: dict) -> int:
    """{doc}"""
    adj = spec["adj"]; vals = spec["vals"]; start = spec["start"]
    seen = set(); stack = [start]; lo = None; hi = None
    while stack:
        n = stack.pop()
        if n in seen: continue
        seen.add(n); v = vals[n]
        if lo is None or v < lo: lo = v
        if hi is None or v > hi: hi = v
        stack.extend(adj.get(n, []))
    return hi - lo
''',
"maxabs": '''
def {name}(spec: dict) -> int:
    """{doc}"""
    adj = spec["adj"]; vals = spec["vals"]; start = spec["start"]
    seen = set(); stack = [start]; best = 0
    while stack:
        n = stack.pop()
        if n in seen: continue
        seen.add(n); a = abs(vals[n])
        if a > best: best = a
        stack.extend(adj.get(n, []))
    return best
''',
"maxeven": '''
def {name}(spec: dict) -> int:
    """{doc}"""
    adj = spec["adj"]; vals = spec["vals"]; start = spec["start"]
    seen = set(); stack = [start]; best = -1
    while stack:
        n = stack.pop()
        if n in seen: continue
        seen.add(n)
        if vals[n] % 2 == 0 and vals[n] > best: best = vals[n]
        stack.extend(adj.get(n, []))
    return best
''',
"leaves": '''
def {name}(spec: dict) -> int:
    """{doc}"""
    adj = spec["adj"]; start = spec["start"]
    seen = set(); stack = [start]; acc = 0
    while stack:
        n = stack.pop()
        if n in seen: continue
        seen.add(n); kids = adj.get(n, [])
        if len(kids) == 0: acc += 1
        stack.extend(kids)
    return acc
''',
"branches": '''
def {name}(spec: dict) -> int:
    """{doc}"""
    adj = spec["adj"]; start = spec["start"]
    seen = set(); stack = [start]; acc = 0
    while stack:
        n = stack.pop()
        if n in seen: continue
        seen.add(n); kids = adj.get(n, [])
        if len(kids) >= 2: acc += 1
        stack.extend(kids)
    return acc
''',
"single": '''
def {name}(spec: dict) -> int:
    """{doc}"""
    adj = spec["adj"]; start = spec["start"]
    seen = set(); stack = [start]; acc = 0
    while stack:
        n = stack.pop()
        if n in seen: continue
        seen.add(n); kids = adj.get(n, [])
        if len(kids) == 1: acc += 1
        stack.extend(kids)
    return acc
''',
"leafbranch": '''
def {name}(spec: dict) -> int:
    """{doc}"""
    adj = spec["adj"]; start = spec["start"]
    seen = set(); stack = [start]; acc = 0
    while stack:
        n = stack.pop()
        if n in seen: continue
        seen.add(n); kids = adj.get(n, [])
        if len(kids) != 1: acc += 1
        stack.extend(kids)
    return acc
''',
"sumleaf": '''
def {name}(spec: dict) -> int:
    """{doc}"""
    adj = spec["adj"]; vals = spec["vals"]; start = spec["start"]
    seen = set(); stack = [start]; acc = 0
    while stack:
        n = stack.pop()
        if n in seen: continue
        seen.add(n)
        if len(adj.get(n, [])) == 0: acc += vals[n]
        stack.extend(adj.get(n, []))
    return acc
''',
"prodpos": '''
def {name}(spec: dict) -> int:
    """{doc}"""
    adj = spec["adj"]; vals = spec["vals"]; start = spec["start"]
    seen = set(); stack = [start]; p = 1
    while stack:
        n = stack.pop()
        if n in seen: continue
        seen.add(n)
        if vals[n] > 0: p *= vals[n]
        stack.extend(adj.get(n, []))
    return p
''',
}

def jac_for(name, kind, jac_pred, jac_expr, thr):
    """Build the idiomatic Jac (node + walker + def builder) for a task kind."""
    val_field = "has val: int; " if kind not in ("reach", "leaves", "branches", "single", "leafbranch") else ""
    kids_field = "has nkids: int; " if kind in ("leaves", "branches", "single", "leafbranch") else ""
    has_k = thr is not None
    # walker state + per-node body
    if kind == "reach":
        state = "has count: int = 0;"; body = "self.count += 1;"
    elif kind == "count":
        state = ("has k: int;\n    " if has_k else "") + "has count: int = 0;"
        body = f"if {jac_pred} {{ self.count += 1; }}"
    elif kind == "sum":
        state = ("has k: int;\n    " if has_k else "") + "has total: int = 0;"
        body = f"self.total += {jac_expr};"
    elif kind == "max":
        state = "has best: int = -2147483648;"; body = "if here.val > self.best { self.best = here.val; }"
    elif kind == "min":
        state = "has best: int = 2147483647;"; body = "if here.val < self.best { self.best = here.val; }"
    elif kind == "maxabs":
        state = "has best: int = 0;"; body = "if (here.val if here.val >= 0 else -here.val) > self.best { self.best = (here.val if here.val >= 0 else -here.val); }"
    elif kind == "maxeven":
        state = "has best: int = -1;"; body = "if here.val % 2 == 0 and here.val > self.best { self.best = here.val; }"
    elif kind == "product":
        state = "has result: int = 1;"; body = "self.result *= here.val;"
    elif kind == "prodpos":
        state = "has result: int = 1;"; body = "if here.val > 0 { self.result *= here.val; }"
    elif kind == "range":
        state = "has lo: int = 2147483647;\n    has hi: int = -2147483648;"
        body = "if here.val < self.lo { self.lo = here.val; } if here.val > self.hi { self.hi = here.val; }"
    elif kind == "sumleaf":
        state = "has total: int = 0;"; body = "if here.nkids == 0 { self.total += here.val; }"
        kids_field = "has nkids: int; "; val_field = "has val: int; "
    elif kind in ("leaves", "branches", "single", "leafbranch"):
        state = "has count: int = 0;"
        cond = {"leaves": "here.nkids == 0", "branches": "here.nkids >= 2",
                "single": "here.nkids == 1", "leafbranch": "here.nkids != 1"}[kind]
        body = f"if {cond} {{ self.count += 1; }}"
    else:
        raise ValueError(kind)

    ret = {"reach": "w.count", "count": "w.count", "sum": "w.total",
           "max": "w.best", "min": "w.best", "maxabs": "w.best",
           "maxeven": "w.best", "product": "w.result", "prodpos": "w.result",
           "range": "w.hi - w.lo", "sumleaf": "w.total",
           "leaves": "w.count", "branches": "w.count", "single": "w.count",
           "leafbranch": "w.count"}[kind]

    needs_vals = val_field != ""
    needs_kids = kids_field != ""
    Nn = "GN_" + name
    Wn = "W_" + name
    # node ctor args
    ctor_extra = ""
    if needs_vals: ctor_extra += ", val=int(vals[NAMEEXPR])"
    if needs_kids: ctor_extra += ", nkids=NK"
    def ctor(name_expr, nk_expr, val_name):
        s = f"{Nn}(name=str({name_expr}), seen=False"
        if needs_vals: s += f", val=int(vals[{val_name}])"
        if needs_kids: s += f", nkids={nk_expr}"
        return s + ")"

    spawn = f"nodes[start] spawn {Wn}(" + (f"k=thr" if has_k else "") + ")"
    vals_line = "    vals: dict = dict(spec[\"vals\"]);\n" if needs_vals else ""
    thr_line = "    thr: int = int(spec[\"threshold\"]);\n" if has_k else ""

    jac = f'''
node {Nn} {{ has name: str; {val_field}{kids_field}has seen: bool = False; }}

walker {Wn} {{
    {state}
    can step with {Nn} entry {{
        if not here.seen {{ here.seen = True; {body} visit [-->]; }}
    }}
}}

def {name}(spec: dict) -> int {{
    adj: dict = dict(spec["adj"]); start: str = str(spec["start"]);
{vals_line}{thr_line}    nodes: dict = {{}};
    for nm in adj.keys() {{ nodes[nm] = {ctor("nm", "len(list(adj[nm]))", "nm")}; }}
    for (s, ts) in adj.items() {{ for t in list(ts) {{ if t not in nodes {{ nodes[t] = {ctor("t", "0", "t")}; }} }} }}
    for (s, ts) in adj.items() {{ for t in list(ts) {{ nodes[s] ++> nodes[t]; }} }}
    w = {spawn};
    return {ret};
}}
'''
    return jac.strip("\n")

def build_py(name, doc, kind, pyside, thr):
    """pyside = the Python predicate/expression (uses `v` and maybe `k`)."""
    tmpl = PY_TMPL[kind]
    kline = '; k = spec["threshold"]' if thr is not None else ""
    return tmpl.format(name=name, doc=doc, pred=(pyside or ""), expr=(pyside or ""),
                       kline_py=kline).strip("\n")

def make_cases(name, kind, thr):
    """Run the python ref to derive expected outputs over the graph pool."""
    py = build_py(name, "d", kind, SPECMAP[name].get("pred"), thr)
    ns = {}
    exec(py, {"abs": abs, "len": len, "set": set}, ns)
    fn = ns[name]
    cases = []
    for (adj, vals), st in zip(POOL, STARTS):
        spec = {"adj": adj, "start": st}
        if kind not in ("reach", "leaves", "branches", "single", "leafbranch"):
            spec["vals"] = vals
        if thr is not None:
            spec["threshold"] = thr
        out = fn(spec)
        cases.append({"input": spec, "output": str(out)})
    return cases

# index specs for make_cases
SPECMAP = {}
for (n, d, k, pr, ex, th) in SPECS_TRAIN + SPECS_HOLDOUT:
    SPECMAP[n] = {"doc": d, "kind": k, "pred": pr, "expr": ex, "thr": th}

def run_case(jac_code, func, inp):
    prog = jac_code + "\n\nwith entry {\n    print(" + func + "(" + repr(inp) + "));\n}\n"
    with tempfile.NamedTemporaryFile("w", suffix=".jac", delete=False, dir="/tmp") as f:
        f.write(prog); p = f.name
    try:
        r = subprocess.run([JAC, "run", p], capture_output=True, text=True, timeout=30)
        return r.returncode, r.stdout.strip(), r.stderr.strip()
    finally:
        os.unlink(p)

def build_tasks(specs):
    tasks = []; ok = True
    for (name, doc, kind, pred, expr, thr) in specs:
        py = build_py(name, doc, kind, pred, thr)        # pred = python side (uses v,k)
        jac = jac_for(name, kind, expr, expr, thr)       # expr = jac side (uses here.val, self.k)
        cases = make_cases(name, kind, thr)
        for c in cases:
            rc, out, err = run_case(jac, name, c["input"])
            if not (rc == 0 and out == c["output"]):
                ok = False
                print(f"FAIL {name}  in={c['input']}  exp={c['output']} got={out!r} rc={rc}")
                if err: print("   ", err.splitlines()[-1] if err.splitlines() else err)
                break
        else:
            tasks.append({"name": name, "difficulty": "composed", "py": py, "jac": jac, "cases": cases})
            print(f"OK   {name} ({len(cases)} cases)")
            continue
    return tasks, ok

print("=== TRAIN ==="); train, ok1 = build_tasks(SPECS_TRAIN)
print("=== HOLDOUT ==="); hold, ok2 = build_tasks(SPECS_HOLDOUT)
if not (ok1 and ok2):
    print("\nFAILURES — not emitting"); sys.exit(1)
json.dump(train, open(os.path.join(HERE, "train.json"), "w"), indent=1)
json.dump(hold, open(os.path.join(HERE, "holdout.json"), "w"), indent=1)
print(f"\nemitted train.json ({len(train)})  holdout.json ({len(hold)})")
