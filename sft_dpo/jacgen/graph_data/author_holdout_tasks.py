"""Author + validate 6 graph holdout tasks (DISJOINT task types from training),
then emit committed train.json (the 8) + holdout.json (these 6)."""
import json, subprocess, tempfile, os, sys

JAC = ".venv/bin/jac"
HOLD = []
def task(name, difficulty, py, jac, cases):
    HOLD.append({"name": name, "difficulty": difficulty, "py": py.strip("\n"),
                 "jac": jac.strip("\n"), "cases": cases})

# H1. min reachable value
task("min_reachable", "composed",
'''
def min_reachable(spec: dict) -> int:
    """Return the minimum node value reachable from start."""
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
'''
node MnN { has name: str; has val: int; has seen: bool = False; }

walker Min {
    has best: int = 2147483647;
    can step with MnN entry {
        if not here.seen { here.seen = True; if here.val < self.best { self.best = here.val; } visit [-->]; }
    }
}

def min_reachable(spec: dict) -> int {
    adj: dict = dict(spec["adj"]); vals: dict = dict(spec["vals"]); start: str = str(spec["start"]);
    nodes: dict = {};
    for name in adj.keys() { nodes[name] = MnN(name=str(name), val=int(vals[name])); }
    for (s, ts) in adj.items() { for t in list(ts) { if t not in nodes { nodes[t] = MnN(name=str(t), val=int(vals[t])); } } }
    for (s, ts) in adj.items() { for t in list(ts) { nodes[s] ++> nodes[t]; } }
    w = nodes[start] spawn Min();
    return w.best;
}
''',
[{"input": {"adj": {"a": ["b","c"], "b": [], "c": []}, "vals": {"a": 7, "b": 2, "c": 9}, "start": "a"}, "output": "2"},
 {"input": {"adj": {"a": ["b"], "b": []}, "vals": {"a": 4, "b": 8}, "start": "a"}, "output": "4"}])

# H2. count reachable nodes with odd value
task("count_odd_reachable", "composed",
'''
def count_odd_reachable(spec: dict) -> int:
    """Count reachable nodes whose value is odd."""
    adj = spec["adj"]; vals = spec["vals"]; start = spec["start"]
    seen = set(); stack = [start]; cnt = 0
    while stack:
        n = stack.pop()
        if n in seen: continue
        seen.add(n)
        if vals[n] % 2 == 1: cnt += 1
        stack.extend(adj.get(n, []))
    return cnt
''',
'''
node OdN { has name: str; has val: int; has seen: bool = False; }

walker Odd {
    has count: int = 0;
    can step with OdN entry {
        if not here.seen { here.seen = True; if here.val % 2 == 1 { self.count += 1; } visit [-->]; }
    }
}

def count_odd_reachable(spec: dict) -> int {
    adj: dict = dict(spec["adj"]); vals: dict = dict(spec["vals"]); start: str = str(spec["start"]);
    nodes: dict = {};
    for name in adj.keys() { nodes[name] = OdN(name=str(name), val=int(vals[name])); }
    for (s, ts) in adj.items() { for t in list(ts) { if t not in nodes { nodes[t] = OdN(name=str(t), val=int(vals[t])); } } }
    for (s, ts) in adj.items() { for t in list(ts) { nodes[s] ++> nodes[t]; } }
    w = nodes[start] spawn Odd();
    return w.count;
}
''',
[{"input": {"adj": {"a": ["b","c"], "b": ["d"], "c": [], "d": []}, "vals": {"a": 1, "b": 2, "c": 3, "d": 5}, "start": "a"}, "output": "3"},
 {"input": {"adj": {"a": ["b"], "b": []}, "vals": {"a": 2, "b": 4}, "start": "a"}, "output": "0"}])

# H3. sum of values strictly above threshold
task("sum_above", "composed",
'''
def sum_above(spec: dict) -> int:
    """Sum the values of reachable nodes whose value exceeds the threshold."""
    adj = spec["adj"]; vals = spec["vals"]; start = spec["start"]; thr = spec["threshold"]
    seen = set(); stack = [start]; total = 0
    while stack:
        n = stack.pop()
        if n in seen: continue
        seen.add(n)
        if vals[n] > thr: total += vals[n]
        stack.extend(adj.get(n, []))
    return total
''',
'''
node SaN { has name: str; has val: int; has seen: bool = False; }

walker SumAbove {
    has threshold: int;
    has total: int = 0;
    can step with SaN entry {
        if not here.seen { here.seen = True; if here.val > self.threshold { self.total += here.val; } visit [-->]; }
    }
}

def sum_above(spec: dict) -> int {
    adj: dict = dict(spec["adj"]); vals: dict = dict(spec["vals"]);
    start: str = str(spec["start"]); thr: int = int(spec["threshold"]);
    nodes: dict = {};
    for name in adj.keys() { nodes[name] = SaN(name=str(name), val=int(vals[name])); }
    for (s, ts) in adj.items() { for t in list(ts) { if t not in nodes { nodes[t] = SaN(name=str(t), val=int(vals[t])); } } }
    for (s, ts) in adj.items() { for t in list(ts) { nodes[s] ++> nodes[t]; } }
    w = nodes[start] spawn SumAbove(threshold=thr);
    return w.total;
}
''',
[{"input": {"adj": {"a": ["b","c"], "b": [], "c": []}, "vals": {"a": 5, "b": 10, "c": 3}, "start": "a", "threshold": 4}, "output": "15"},
 {"input": {"adj": {"a": ["b"], "b": []}, "vals": {"a": 1, "b": 2}, "start": "a", "threshold": 9}, "output": "0"}])

# H4. count branch nodes (>= 2 children) among reachable
task("count_branches", "composed",
'''
def count_branches(spec: dict) -> int:
    """Count reachable nodes that have two or more children."""
    adj = spec["adj"]; start = spec["start"]
    seen = set(); stack = [start]; cnt = 0
    while stack:
        n = stack.pop()
        if n in seen: continue
        seen.add(n)
        kids = adj.get(n, [])
        if len(kids) >= 2: cnt += 1
        stack.extend(kids)
    return cnt
''',
'''
node BrN { has name: str; has nkids: int; has seen: bool = False; }

walker Branch {
    has count: int = 0;
    can step with BrN entry {
        if not here.seen { here.seen = True; if here.nkids >= 2 { self.count += 1; } visit [-->]; }
    }
}

def count_branches(spec: dict) -> int {
    adj: dict = dict(spec["adj"]); start: str = str(spec["start"]);
    nodes: dict = {};
    for name in adj.keys() { nodes[name] = BrN(name=str(name), nkids=len(list(adj[name]))); }
    for (s, ts) in adj.items() { for t in list(ts) { if t not in nodes { nodes[t] = BrN(name=str(t), nkids=0); } } }
    for (s, ts) in adj.items() { for t in list(ts) { nodes[s] ++> nodes[t]; } }
    w = nodes[start] spawn Branch();
    return w.count;
}
''',
[{"input": {"adj": {"a": ["b","c"], "b": ["d","e"], "c": [], "d": [], "e": []}, "start": "a"}, "output": "2"},
 {"input": {"adj": {"a": ["b"], "b": []}, "start": "a"}, "output": "0"}])

# H5. range (max - min) of reachable values
task("range_reachable", "composed",
'''
def range_reachable(spec: dict) -> int:
    """Return max minus min over the values of reachable nodes."""
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
'''
node RaN { has name: str; has val: int; has seen: bool = False; }

walker Range {
    has lo: int = 2147483647;
    has hi: int = -2147483648;
    can step with RaN entry {
        if not here.seen {
            here.seen = True;
            if here.val < self.lo { self.lo = here.val; }
            if here.val > self.hi { self.hi = here.val; }
            visit [-->];
        }
    }
}

def range_reachable(spec: dict) -> int {
    adj: dict = dict(spec["adj"]); vals: dict = dict(spec["vals"]); start: str = str(spec["start"]);
    nodes: dict = {};
    for name in adj.keys() { nodes[name] = RaN(name=str(name), val=int(vals[name])); }
    for (s, ts) in adj.items() { for t in list(ts) { if t not in nodes { nodes[t] = RaN(name=str(t), val=int(vals[t])); } } }
    for (s, ts) in adj.items() { for t in list(ts) { nodes[s] ++> nodes[t]; } }
    w = nodes[start] spawn Range();
    return w.hi - w.lo;
}
''',
[{"input": {"adj": {"a": ["b","c"], "b": [], "c": []}, "vals": {"a": 5, "b": 1, "c": 9}, "start": "a"}, "output": "8"},
 {"input": {"adj": {"a": ["b"], "b": []}, "vals": {"a": 3, "b": 7}, "start": "a"}, "output": "4"}])

# H6. count reachable nodes with a negative value
task("count_negative", "composed",
'''
def count_negative(spec: dict) -> int:
    """Count reachable nodes whose value is negative."""
    adj = spec["adj"]; vals = spec["vals"]; start = spec["start"]
    seen = set(); stack = [start]; cnt = 0
    while stack:
        n = stack.pop()
        if n in seen: continue
        seen.add(n)
        if vals[n] < 0: cnt += 1
        stack.extend(adj.get(n, []))
    return cnt
''',
'''
node NgN { has name: str; has val: int; has seen: bool = False; }

walker Neg {
    has count: int = 0;
    can step with NgN entry {
        if not here.seen { here.seen = True; if here.val < 0 { self.count += 1; } visit [-->]; }
    }
}

def count_negative(spec: dict) -> int {
    adj: dict = dict(spec["adj"]); vals: dict = dict(spec["vals"]); start: str = str(spec["start"]);
    nodes: dict = {};
    for name in adj.keys() { nodes[name] = NgN(name=str(name), val=int(vals[name])); }
    for (s, ts) in adj.items() { for t in list(ts) { if t not in nodes { nodes[t] = NgN(name=str(t), val=int(vals[t])); } } }
    for (s, ts) in adj.items() { for t in list(ts) { nodes[s] ++> nodes[t]; } }
    w = nodes[start] spawn Neg();
    return w.count;
}
''',
[{"input": {"adj": {"a": ["b","c"], "b": [], "c": []}, "vals": {"a": -1, "b": 2, "c": -3}, "start": "a"}, "output": "2"},
 {"input": {"adj": {"a": ["b"], "b": []}, "vals": {"a": 1, "b": 2}, "start": "a"}, "output": "0"}])


def run_case(jac_code, func, inp):
    prog = jac_code + "\n\nwith entry {\n    print(" + func + "(" + repr(inp) + "));\n}\n"
    with tempfile.NamedTemporaryFile("w", suffix=".jac", delete=False, dir="/tmp") as f:
        f.write(prog); path = f.name
    try:
        r = subprocess.run([JAC, "run", path], capture_output=True, text=True, timeout=30)
        return r.returncode, r.stdout.strip(), r.stderr.strip()
    finally:
        os.unlink(path)

ok = True
for t in HOLD:
    for c in t["cases"]:
        rc, out, err = run_case(t["jac"], t["name"], c["input"])
        if not (rc == 0 and out == c["output"]):
            ok = False
            print(f"FAIL {t['name']} exp={c['output']} got={out!r}")
            if err: print("  ", err.splitlines()[-1] if err.splitlines() else err)
    else:
        print(f"OK   {t['name']} ({len(t['cases'])} cases)")

if not ok:
    sys.exit(1)

os.makedirs("srccurrent/jacgen/graph_data", exist_ok=True)
train = json.load(open("/tmp/graph_tasks.json"))
json.dump(train, open("srccurrent/jacgen/graph_data/train.json", "w"), indent=1)
json.dump(HOLD, open("srccurrent/jacgen/graph_data/holdout.json", "w"), indent=1)
print(f"\nemitted: train.json ({len(train)})  holdout.json ({len(HOLD)})  -> srccurrent/jacgen/graph_data/")
