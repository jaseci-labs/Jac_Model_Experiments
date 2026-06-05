"""Author + validate graph-shaped conversion tasks against .venv jac.

Each task: a single-dict-arg `def` whose idiomatic Jac builds nodes/edges and
spawns a walker (graph-spatial) — diverging hard from any dict-loop transpile.
Validates each idiomatic jac by running it on every test input and checking output.
Emits a JSON the .jac builders consume.
"""
import json, subprocess, tempfile, os, sys

JAC = ".venv/bin/jac"

# Shared idiomatic graph builder snippet pattern: build nodes from adj, wire edges,
# spawn an accumulator walker from `start`. Each task supplies the node/walker + def.
TASKS = []

def task(name, difficulty, py, jac, cases):
    TASKS.append({"name": name, "difficulty": difficulty, "py": py.strip("\n"),
                  "jac": jac.strip("\n"), "cases": cases})

# 1. count reachable nodes (BFS)
task("count_reachable", "composed",
'''
def count_reachable(spec: dict) -> int:
    """Count nodes reachable from start (including start)."""
    adj = spec["adj"]; start = spec["start"]
    seen = set(); stack = [start]
    while stack:
        n = stack.pop()
        if n in seen: continue
        seen.add(n)
        stack.extend(adj.get(n, []))
    return len(seen)
''',
'''
node GN { has name: str; has seen: bool = False; }

walker Count {
    has count: int = 0;
    can step with GN entry {
        if not here.seen { here.seen = True; self.count += 1; visit [-->]; }
    }
}

def count_reachable(spec: dict) -> int {
    adj: dict = dict(spec["adj"]); start: str = str(spec["start"]);
    nodes: dict = {};
    for name in adj.keys() { nodes[name] = GN(name=str(name)); }
    for (s, ts) in adj.items() { for t in list(ts) { if t not in nodes { nodes[t] = GN(name=str(t)); } } }
    for (s, ts) in adj.items() { for t in list(ts) { nodes[s] ++> nodes[t]; } }
    w = nodes[start] spawn Count();
    return w.count;
}
''',
[{"input": {"adj": {"a": ["b","c"], "b": ["d"], "c": ["d"], "d": []}, "start": "a"}, "output": "4"},
 {"input": {"adj": {"x": ["y"], "y": []}, "start": "x"}, "output": "2"},
 {"input": {"adj": {"a": []}, "start": "a"}, "output": "1"}])

# 2. sum of reachable node values
task("sum_reachable", "composed",
'''
def sum_reachable(spec: dict) -> int:
    """Sum the values of all nodes reachable from start."""
    adj = spec["adj"]; vals = spec["vals"]; start = spec["start"]
    seen = set(); stack = [start]; total = 0
    while stack:
        n = stack.pop()
        if n in seen: continue
        seen.add(n); total += vals[n]
        stack.extend(adj.get(n, []))
    return total
''',
'''
node VN { has name: str; has val: int; has seen: bool = False; }

walker Sum {
    has total: int = 0;
    can step with VN entry {
        if not here.seen { here.seen = True; self.total += here.val; visit [-->]; }
    }
}

def sum_reachable(spec: dict) -> int {
    adj: dict = dict(spec["adj"]); vals: dict = dict(spec["vals"]); start: str = str(spec["start"]);
    nodes: dict = {};
    for name in adj.keys() { nodes[name] = VN(name=str(name), val=int(vals[name])); }
    for (s, ts) in adj.items() { for t in list(ts) { if t not in nodes { nodes[t] = VN(name=str(t), val=int(vals[t])); } } }
    for (s, ts) in adj.items() { for t in list(ts) { nodes[s] ++> nodes[t]; } }
    w = nodes[start] spawn Sum();
    return w.total;
}
''',
[{"input": {"adj": {"a": ["b","c"], "b": [], "c": []}, "vals": {"a": 1, "b": 2, "c": 3}, "start": "a"}, "output": "6"},
 {"input": {"adj": {"a": ["b"], "b": ["c"], "c": []}, "vals": {"a": 10, "b": 20, "c": 30}, "start": "b"}, "output": "50"}])

# 3. max value reachable
task("max_reachable", "composed",
'''
def max_reachable(spec: dict) -> int:
    """Return the maximum node value reachable from start."""
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
'''
node MN { has name: str; has val: int; has seen: bool = False; }

walker Max {
    has best: int = -2147483648;
    can step with MN entry {
        if not here.seen { here.seen = True; if here.val > self.best { self.best = here.val; } visit [-->]; }
    }
}

def max_reachable(spec: dict) -> int {
    adj: dict = dict(spec["adj"]); vals: dict = dict(spec["vals"]); start: str = str(spec["start"]);
    nodes: dict = {};
    for name in adj.keys() { nodes[name] = MN(name=str(name), val=int(vals[name])); }
    for (s, ts) in adj.items() { for t in list(ts) { if t not in nodes { nodes[t] = MN(name=str(t), val=int(vals[t])); } } }
    for (s, ts) in adj.items() { for t in list(ts) { nodes[s] ++> nodes[t]; } }
    w = nodes[start] spawn Max();
    return w.best;
}
''',
[{"input": {"adj": {"a": ["b","c"], "b": [], "c": []}, "vals": {"a": 1, "b": 9, "c": 3}, "start": "a"}, "output": "9"},
 {"input": {"adj": {"a": ["b"], "b": []}, "vals": {"a": 5, "b": 2}, "start": "a"}, "output": "5"}])

# 4. count nodes whose value exceeds a threshold
task("count_above", "composed",
'''
def count_above(spec: dict) -> int:
    """Count reachable nodes whose value exceeds the threshold."""
    adj = spec["adj"]; vals = spec["vals"]; start = spec["start"]; thr = spec["threshold"]
    seen = set(); stack = [start]; cnt = 0
    while stack:
        n = stack.pop()
        if n in seen: continue
        seen.add(n)
        if vals[n] > thr: cnt += 1
        stack.extend(adj.get(n, []))
    return cnt
''',
'''
node TN { has name: str; has val: int; has seen: bool = False; }

walker Above {
    has threshold: int;
    has count: int = 0;
    can step with TN entry {
        if not here.seen { here.seen = True; if here.val > self.threshold { self.count += 1; } visit [-->]; }
    }
}

def count_above(spec: dict) -> int {
    adj: dict = dict(spec["adj"]); vals: dict = dict(spec["vals"]);
    start: str = str(spec["start"]); thr: int = int(spec["threshold"]);
    nodes: dict = {};
    for name in adj.keys() { nodes[name] = TN(name=str(name), val=int(vals[name])); }
    for (s, ts) in adj.items() { for t in list(ts) { if t not in nodes { nodes[t] = TN(name=str(t), val=int(vals[t])); } } }
    for (s, ts) in adj.items() { for t in list(ts) { nodes[s] ++> nodes[t]; } }
    w = nodes[start] spawn Above(threshold=thr);
    return w.count;
}
''',
[{"input": {"adj": {"a": ["b","c"], "b": ["d"], "c": [], "d": []}, "vals": {"a": 5, "b": 8, "c": 2, "d": 10}, "start": "a", "threshold": 5}, "output": "2"},
 {"input": {"adj": {"a": ["b"], "b": []}, "vals": {"a": 1, "b": 1}, "start": "a", "threshold": 5}, "output": "0"}])

# 5. is target reachable (membership) -> 1/0
task("path_exists", "composed",
'''
def path_exists(spec: dict) -> int:
    """Return 1 if target is reachable from start, else 0."""
    adj = spec["adj"]; start = spec["start"]; target = spec["target"]
    seen = set(); stack = [start]
    while stack:
        n = stack.pop()
        if n == target: return 1
        if n in seen: continue
        seen.add(n)
        stack.extend(adj.get(n, []))
    return 0
''',
'''
node PN { has name: str; has seen: bool = False; }

walker Find {
    has target: str;
    has found: int = 0;
    can step with PN entry {
        if here.name == self.target { self.found = 1; }
        if not here.seen { here.seen = True; visit [-->]; }
    }
}

def path_exists(spec: dict) -> int {
    adj: dict = dict(spec["adj"]); start: str = str(spec["start"]); target: str = str(spec["target"]);
    nodes: dict = {};
    for name in adj.keys() { nodes[name] = PN(name=str(name)); }
    for (s, ts) in adj.items() { for t in list(ts) { if t not in nodes { nodes[t] = PN(name=str(t)); } } }
    for (s, ts) in adj.items() { for t in list(ts) { nodes[s] ++> nodes[t]; } }
    w = nodes[start] spawn Find(target=target);
    return w.found;
}
''',
[{"input": {"adj": {"a": ["b"], "b": ["c"], "c": []}, "start": "a", "target": "c"}, "output": "1"},
 {"input": {"adj": {"a": ["b"], "b": []}, "start": "a", "target": "z"}, "output": "0"}])

# 6. count leaves (reachable nodes with no outgoing edges)
task("count_leaves", "composed",
'''
def count_leaves(spec: dict) -> int:
    """Count reachable nodes that have no children."""
    adj = spec["adj"]; start = spec["start"]
    seen = set(); stack = [start]; leaves = 0
    while stack:
        n = stack.pop()
        if n in seen: continue
        seen.add(n)
        kids = adj.get(n, [])
        if not kids: leaves += 1
        stack.extend(kids)
    return leaves
''',
'''
node LN { has name: str; has nkids: int; has seen: bool = False; }

walker Leaves {
    has count: int = 0;
    can step with LN entry {
        if not here.seen { here.seen = True; if here.nkids == 0 { self.count += 1; } visit [-->]; }
    }
}

def count_leaves(spec: dict) -> int {
    adj: dict = dict(spec["adj"]); start: str = str(spec["start"]);
    nodes: dict = {};
    for name in adj.keys() { nodes[name] = LN(name=str(name), nkids=len(list(adj[name]))); }
    for (s, ts) in adj.items() { for t in list(ts) { if t not in nodes { nodes[t] = LN(name=str(t), nkids=0); } } }
    for (s, ts) in adj.items() { for t in list(ts) { nodes[s] ++> nodes[t]; } }
    w = nodes[start] spawn Leaves();
    return w.count;
}
''',
[{"input": {"adj": {"a": ["b","c"], "b": [], "c": ["d"], "d": []}, "start": "a"}, "output": "2"},
 {"input": {"adj": {"a": []}, "start": "a"}, "output": "1"}])

# 7. product of reachable values
task("product_reachable", "composed",
'''
def product_reachable(spec: dict) -> int:
    """Multiply the values of all nodes reachable from start."""
    adj = spec["adj"]; vals = spec["vals"]; start = spec["start"]
    seen = set(); stack = [start]; prod = 1
    while stack:
        n = stack.pop()
        if n in seen: continue
        seen.add(n); prod *= vals[n]
        stack.extend(adj.get(n, []))
    return prod
''',
'''
node QN { has name: str; has val: int; has seen: bool = False; }

walker Prod {
    has result: int = 1;
    can step with QN entry {
        if not here.seen { here.seen = True; self.result *= here.val; visit [-->]; }
    }
}

def product_reachable(spec: dict) -> int {
    adj: dict = dict(spec["adj"]); vals: dict = dict(spec["vals"]); start: str = str(spec["start"]);
    nodes: dict = {};
    for name in adj.keys() { nodes[name] = QN(name=str(name), val=int(vals[name])); }
    for (s, ts) in adj.items() { for t in list(ts) { if t not in nodes { nodes[t] = QN(name=str(t), val=int(vals[t])); } } }
    for (s, ts) in adj.items() { for t in list(ts) { nodes[s] ++> nodes[t]; } }
    w = nodes[start] spawn Prod();
    return w.result;
}
''',
[{"input": {"adj": {"a": ["b","c"], "b": [], "c": []}, "vals": {"a": 2, "b": 3, "c": 4}, "start": "a"}, "output": "24"},
 {"input": {"adj": {"a": ["b"], "b": []}, "vals": {"a": 5, "b": 5}, "start": "a"}, "output": "25"}])

# 8. count reachable nodes with an even value
task("count_even_reachable", "composed",
'''
def count_even_reachable(spec: dict) -> int:
    """Count reachable nodes whose value is even."""
    adj = spec["adj"]; vals = spec["vals"]; start = spec["start"]
    seen = set(); stack = [start]; cnt = 0
    while stack:
        n = stack.pop()
        if n in seen: continue
        seen.add(n)
        if vals[n] % 2 == 0: cnt += 1
        stack.extend(adj.get(n, []))
    return cnt
''',
'''
node EN { has name: str; has val: int; has seen: bool = False; }

walker Even {
    has count: int = 0;
    can step with EN entry {
        if not here.seen { here.seen = True; if here.val % 2 == 0 { self.count += 1; } visit [-->]; }
    }
}

def count_even_reachable(spec: dict) -> int {
    adj: dict = dict(spec["adj"]); vals: dict = dict(spec["vals"]); start: str = str(spec["start"]);
    nodes: dict = {};
    for name in adj.keys() { nodes[name] = EN(name=str(name), val=int(vals[name])); }
    for (s, ts) in adj.items() { for t in list(ts) { if t not in nodes { nodes[t] = EN(name=str(t), val=int(vals[t])); } } }
    for (s, ts) in adj.items() { for t in list(ts) { nodes[s] ++> nodes[t]; } }
    w = nodes[start] spawn Even();
    return w.count;
}
''',
[{"input": {"adj": {"a": ["b","c"], "b": ["d"], "c": [], "d": []}, "vals": {"a": 2, "b": 3, "c": 4, "d": 6}, "start": "a"}, "output": "3"},
 {"input": {"adj": {"a": ["b"], "b": []}, "vals": {"a": 1, "b": 3}, "start": "a"}, "output": "0"}])


def run_case(jac_code, func, inp):
    prog = jac_code + "\n\nwith entry {\n    print(" + func + "(" + repr(inp) + "));\n}\n"
    with tempfile.NamedTemporaryFile("w", suffix=".jac", delete=False, dir="/tmp") as f:
        f.write(prog); path = f.name
    try:
        r = subprocess.run([JAC, "run", path], capture_output=True, text=True, timeout=30)
        return r.returncode, r.stdout.strip(), r.stderr.strip()
    finally:
        os.unlink(path)

ok_all = True
for t in TASKS:
    for c in t["cases"]:
        rc, out, err = run_case(t["jac"], t["name"], c["input"])
        passed = (rc == 0 and out == c["output"])
        if not passed:
            ok_all = False
            print(f"FAIL {t['name']}  expected={c['output']}  got={out!r}  rc={rc}")
            if err: print("   stderr:", err.splitlines()[-1] if err.splitlines() else err)
    else:
        print(f"OK   {t['name']}  ({len(t['cases'])} cases)")

if ok_all:
    json.dump(TASKS, open("/tmp/graph_tasks.json", "w"), indent=1)
    print(f"\nALL {len(TASKS)} TASKS GREEN -> /tmp/graph_tasks.json")
else:
    print("\nFIX FAILURES BEFORE EMITTING")
    sys.exit(1)
