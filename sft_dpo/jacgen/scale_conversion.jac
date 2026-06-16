"""Automated Python -> Jac conversion generator (MultiPL-T style, volume tier).

Mine real functions from an online corpus, mechanically transpile to Jac via
`jac py2jac`, then keep only those whose Jac output behaviorally matches Python
across multiple distinct test cases. Exact-body dedup keeps it O(n) at scale.
The idiomatic hand/mined-crafted core lives in seed_conversion.jac (32 examples);
this is the volume tier that scales toward the conversion-probe target.

Run from the repo root:
    jac run srccurrent/jacgen/scale_conversion.jac
"""

import from corpus { fetch_page, first_func, is_clean, py_cases, transpile, normalize }
import from writer { make_sft_example, append_jsonl, run_jac }
import from pathlib { Path }
import from typing { Any }

glob OUT_PATH: str = "dataset/conversion/sft_auto.jsonl";
glob SOURCE: str = "Vezora/Tested-22k-Python-Alpaca";
glob TARGET: int = 1500;
glob SCAN_CAP: int = 15000;


def build_program(name: str, jac_fn: str, inputs: list) -> str {
    calls: str = "";
    for inp in inputs { calls += "    print(" + name + "(" + repr(inp) + "));\n"; }
    return jac_fn + "\n\nwith entry {\n" + calls + "}\n";
}


def difficulty_of(jac_fn: str) -> str {
    if len(jac_fn.split("\n")) <= 8 { return "atomic"; }
    return "idiomatic";
}


with entry {
    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True);
    open(OUT_PATH, "w").close();

    print("generating up to " + str(TARGET) + " conversion examples from " + SOURCE + " ...");
    seen: dict[str, int] = {};
    kept: int = 0;
    cand: int = 0;
    tp_fail: int = 0;
    behave_fail: int = 0;
    offset: int = 0;
    empty_pages: int = 0;

    while kept < TARGET and offset < SCAN_CAP {
        rows: list[dict] = fetch_page(offset, 100);
        offset += 100;
        if len(rows) == 0 {
            empty_pages += 1;
            if empty_pages >= 3 { print("3 consecutive empty pages, stopping"); break; }
            continue;
        }
        empty_pages = 0;

        for r in rows {
            if kept >= TARGET { break; }
            output: str = str(r["output"]);
            src: str = first_func(output);
            if src == "" or not is_clean(src) { continue; }
            norm: str = normalize(src);
            if norm in seen { continue; }
            seen[norm] = 1;

            pc: dict = py_cases(src, 3);
            ok: bool = bool(pc["ok"]);
            if not ok { continue; }
            cand += 1;
            name: str = str(pc["name"]);
            inputs: list = list(pc["inputs"]);
            outputs: list = list(pc["outputs"]);

            jac_fn: str = transpile(src);
            if jac_fn == "" { tp_fail += 1; continue; }

            prog: str = build_program(name, jac_fn, inputs);
            expected: str = "\n".join([str(o) for o in outputs]);
            (code, out, err) = run_jac(prog);
            if code != 0 or out.strip() != expected.strip() { behave_fail += 1; continue; }

            diff: str = difficulty_of(jac_fn);
            ex: dict = make_sft_example(src, prog, diff, expected, SOURCE, "jac-py2jac");
            append_jsonl(OUT_PATH, ex);
            kept += 1;
            if kept % 100 == 0 {
                print("kept " + str(kept) + "/" + str(TARGET) + "  (scanned offset " + str(offset)
                    + ", candidates " + str(cand) + ", transpile_fail " + str(tp_fail)
                    + ", behave_fail " + str(behave_fail) + ")");
            }
        }
    }

    print("DONE: kept " + str(kept) + " -> " + OUT_PATH);
    print("  scanned to offset " + str(offset) + ", unique candidates " + str(cand)
        + ", transpile_fail " + str(tp_fail) + ", behave_fail " + str(behave_fail));
}
