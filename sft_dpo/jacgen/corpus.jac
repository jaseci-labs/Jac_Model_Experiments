"""Shared corpus helpers: fetch, filter, python-case extraction, transpile.

Used by the automated conversion generator (scale_conversion.jac). The corpus
is `Vezora/Tested-22k-Python-Alpaca`, pulled via the HuggingFace datasets-server
rows API (curl + json; urllib is untyped under the Jac checker). py_cases runs a
candidate on a small input pool and keeps up to `max_cases` inputs that yield
DISTINCT outputs, so the behavioral gate uses varied, non-degenerate test cases.
"""

import json;
import inspect;
import builtins;
import re;
import subprocess;
import tempfile;
import signal;
import from pathlib { Path }
import from typing { Any }


def _on_timeout(_signum: int, _frame: object) {
    raise TimeoutError("exec timeout");
}

glob DATASET: str = "Vezora/Tested-22k-Python-Alpaca";
glob ALLOWED_IMPORTS: list[str] = ["math", "re", "string", "functools", "collections", "itertools"];
glob BANNED: list[str] = ["np.", "pd.", "yield", "open(", "input(", "@", "class ", "async ", "lambda", "print(", "self"];
glob BUILTINS: list[str] = ["set", "list", "dict", "tuple", "type", "str", "int", "float",
                            "bool", "range", "len", "sum", "min", "max", "sorted", "map",
                            "filter", "zip", "enumerate", "abs", "round", "reversed"];
glob INPUTS: list[Any] = ["hello", "Racecar", "abcabc", "level", "Hello, World!", "step on no pets",
                     5, 10, 2, 0, 7, 12, 1, 100,
                     [1, 2, 3, 4], [3, 1, 2], [5, 5, 5], ["a", "b", "a"],
                     [10, 20, 30], [2, 4, 6, 8], [4, 2, 7, 1, 9], 3.5];


def fetch_page(offset: int, length: int) -> list[dict] {
    # Fetch one page; retry on empty/non-JSON (throttle/transient). Returns [] on give-up.
    url: str = "https://datasets-server.huggingface.co/rows?dataset=" + DATASET
        + "&config=default&split=train&offset=" + str(offset) + "&length=" + str(length);
    tries: int = 0;
    while tries < 4 {
        tries += 1;
        p = subprocess.run(
            ["curl", "-s", "--retry", "2", "--retry-delay", "1", "-H", "User-Agent: jacgen", url],
            capture_output=True, text=True, timeout=60
        );
        body: str = p.stdout.strip();
        if body == "" { continue; }
        try {
            data: dict = dict(json.loads(body));
        } except json.JSONDecodeError {
            continue;
        }
        rows: list = list(data["rows"]);
        out: list[dict] = [];
        for r in rows { row: dict = dict(r["row"]); out.append(row); }
        return out;
    }
    return [];
}


def fetch_rows(scan: int) -> list[dict] {
    out: list[dict] = [];
    off: int = 0;
    while len(out) < scan {
        page: list[dict] = fetch_page(off, min(100, scan - len(out)));
        if len(page) == 0 { break; }
        for row in page { out.append(row); }
        off += 100;
    }
    return out;
}


def first_func(code: str) -> str {
    lines: list[str] = code.split("\n");
    start: int = -1;
    for (i, l) in enumerate(lines) {
        if l.startswith("def ") { start = i; break; }
    }
    if start < 0 { return ""; }
    blk: list[str] = [lines[start]];
    for l in lines[start + 1:] {
        if l.strip() == "" or l.startswith(" ") or l.startswith("\t") { blk.append(l); }
        else { break; }
    }
    return "\n".join(blk).rstrip();
}


def is_clean(code: str) -> bool {
    head: str = code.split("\n")[0];
    if "*" in head { return False; }
    for b in BANNED { if b in code { return False; } }
    imports: list = list(re.findall("(?:^\\s*import |^\\s*from )(\\w+)", code, re.M));  # jac:ignore[E1054]
    for m in imports { if m not in ALLOWED_IMPORTS { return False; } }
    if re.search("\\breturn\\s+\\S", code) is None { return False; }
    nlines: int = len(code.split("\n"));
    return nlines > 1 and nlines <= 14;
}


def py_cases(code: str, max_cases: int = 3) -> dict {
    ns: dict = {};
    try {
        exec(code, {"__builtins__": builtins}, ns);
    } except Exception {  # jac:ignore[W2052] mined code can raise anything
        return {"ok": False};
    }
    funcs: list = [];
    names: list[str] = [];
    for (k, v) in ns.items() {
        if callable(v) and not k.startswith("__") {
            funcs.append(v);
            names.append(str(k));
        }
    }
    if len(funcs) == 0 { return {"ok": False}; }
    fn = funcs[-1];
    name: str = names[-1];
    try {
        if len(inspect.signature(fn).parameters) != 1 { return {"ok": False}; }  # jac:ignore[E1053]
    } except (ValueError, TypeError) {
        return {"ok": False};
    }
    inputs: list = [];
    outputs: list = [];
    seen_out: list[str] = [];
    signal.signal(signal.SIGALRM, _on_timeout);  # jac:ignore[E1053]
    for a in INPUTS {
        if len(inputs) >= max_cases { break; }
        signal.alarm(2);
        try {
            o = fn(a);
            signal.alarm(0);
        } except TimeoutError {
            signal.alarm(0);
            break;
        } except Exception {  # jac:ignore[W2052] mined code can raise anything
            signal.alarm(0);
            continue;
        }
        if o is None { continue; }
        if not isinstance(o, (int, float, str, bool, list, dict)) { continue; }
        key: str = repr(o);
        if key in seen_out { continue; }
        seen_out.append(key);
        inputs.append(a);
        outputs.append(o);
    }
    if len(inputs) == 0 { return {"ok": False}; }
    return {"ok": True, "name": name, "inputs": inputs, "outputs": outputs};
}


def transpile(py_src: str) -> str {
    with tempfile.TemporaryDirectory() as d {
        fp = Path(str(d)) / "f.py";
        fp.write_text(py_src);
        try {
            r = subprocess.run(["jac", "py2jac", str(fp)], capture_output=True, text=True, timeout=30);
        } except (subprocess.SubprocessError, OSError) {
            return "";
        }
        if r.returncode != 0 { return ""; }
        return r.stdout;
    }
}


def sanitize_transpile(code: str) -> str {
    # py2jac over-escapes builtins as `set(), `list() etc.; strip the backtick so
    # the transpiled code compiles.
    out: str = code;
    for b in BUILTINS { out = out.replace("`" + b, b); }
    return out;
}


def normalize(code: str) -> str {
    out: str = "";
    for line in code.split("\n") {
        s: str = line.split("#", 1)[0].strip();
        if s != "" { out += s + " "; }
    }
    return out;
}
