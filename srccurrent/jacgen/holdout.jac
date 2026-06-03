"""Build the decontaminated conversion eval holdout (automated, no hand-authoring).

A conversion eval task is {python prompt, behavioral test_cases}. The model
later converts the Python to Jac; the harness runs its Jac on each input and
checks the output. Cases come from corpus.py_cases (multiple distinct outputs).
Mines a corpus slice DISJOINT from training (training used offsets 0-8200; this
starts at 12000) and drops anything that 14-gram-overlaps the training set.

Run from the repo root:
    jac run srccurrent/jacgen/holdout.jac
"""

import json;
import from corpus { fetch_page, first_func, is_clean, py_cases }
import from decontam { build_training_shingles, is_contaminated }
import from pathlib { Path }
import from typing { Any }

glob TRAIN_FILES: list[str] = ["dataset/conversion/sft.jsonl", "dataset/conversion/sft_auto.jsonl"];
glob OUT_PATH: str = "dataset/eval_holdout/conversion.jsonl";
glob SOURCE: str = "Vezora/Tested-22k-Python-Alpaca";
glob START_OFFSET: int = 12000;
glob TARGET: int = 150;
glob SCAN_CAP: int = 22000;


def make_prompt(py: str) -> str {
    return "Convert this Python to idiomatic Jac:\n\n```python\n" + py.strip() + "\n```";
}


with entry {
    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True);
    open(OUT_PATH, "w").close();

    print("building training 14-gram shingle set...");
    train: dict = build_training_shingles(TRAIN_FILES);
    print("training shingles: " + str(len(train)));

    seen: dict = {};
    kept: int = 0;
    contam: int = 0;
    offset: int = START_OFFSET;
    empty: int = 0;

    while kept < TARGET and offset < SCAN_CAP {
        rows: list[dict] = fetch_page(offset, 100);
        offset += 100;
        if len(rows) == 0 {
            empty += 1;
            if empty >= 3 { print("3 empty pages, stopping"); break; }
            continue;
        }
        empty = 0;

        for r in rows {
            if kept >= TARGET { break; }
            output: str = str(r["output"]);
            src: str = first_func(output);
            if src == "" or not is_clean(src) { continue; }
            if src in seen { continue; }
            seen[src] = 1;
            if is_contaminated(src, train) { contam += 1; continue; }

            pc: dict = py_cases(src, 3);
            ok: bool = bool(pc["ok"]);
            if not ok { continue; }
            name: str = str(pc["name"]);
            inputs: list = list(pc["inputs"]);
            outputs: list = list(pc["outputs"]);
            cases: list = [];
            for (i, inp) in enumerate(inputs) {
                cases.append({"input": inp, "output": outputs[i]});
            }

            rec: dict = {
                "func_name": name,
                "python": src,
                "prompt": make_prompt(src),
                "test_cases": cases,
                "source": SOURCE
            };
            with open(OUT_PATH, "a") as f { f.write(json.dumps(rec) + "\n"); }
            kept += 1;
            if kept % 50 == 0 {
                print("kept " + str(kept) + "/" + str(TARGET) + "  (offset " + str(offset)
                    + ", contaminated-skipped " + str(contam) + ")");
            }
        }
    }
    print("DONE holdout: kept " + str(kept) + ", contaminated-skipped " + str(contam)
        + ", scanned to offset " + str(offset) + " -> " + OUT_PATH);
}
