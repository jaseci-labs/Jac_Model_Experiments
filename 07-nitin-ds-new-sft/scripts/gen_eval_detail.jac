"""Per-row generation dump for failure analysis (phase 1 of 2).

Mirrors `04-cpt-sft/sft_cptv2_probe/jacgen/eval_functional.jac`'s generation
path EXACTLY -- same chat-template encoding, same `mlx_lm.batch_generate`
call, same max_tokens=768, same batch_size=32, same file order -- but instead
of grading inline and throwing the text away, it persists one JSONL record per
holdout row:

    {id, category, gate_class, prompt, generated}

Why a copy and not a patch of the shared harness: `eval_functional.jac` is
referenced by name from six eval runner scripts across 04-cpt-sft and
07-nitin-ds-new-sft, all of which parse its stdout shape. Changing it in place
would silently alter the harness that produced every published number in this
project. This file is additive and read-only with respect to that harness.

Prose/ungraded rows ARE generated (they are in the original run too) so batch
composition -- and therefore the float non-determinism of 4-bit batched
matmul that the original harness's docstring warns about -- stays identical
to the run being reproduced. They are recorded with graded=false.

Grading happens in `grade_eval_detail.jac` (phase 2).

    ADAPTER=<path> OUT_PREFIX=<dir/tag> [HOLDOUTS=A,B] \
      jac run model-experiments/07-nitin-ds-new-sft/scripts/gen_eval_detail.jac
"""

import json;
import os;
import sys;
import time;
import from pathlib { Path }
import from typing { cast }

glob ROOT: Path = Path(str(__file__)).resolve().parents[3];

glob HOLDOUTS: dict[str, Path] = {
    "A": ROOT / "model-experiments/04-cpt-sft/sft_fresh_probe/dataset/sft/valid.jsonl",
    "B": ROOT / "model-experiments/07-nitin-ds-new-sft/dataset/nitin_holdout_eval.jsonl"
};

glob BATCH_SIZE: int = int(os.environ.get("BATCH_SIZE", "32"));
glob MAX_TOKENS: int = int(os.environ.get("MAX_TOKENS", "768"));
glob MODEL: str = os.environ.get("MODEL", "models/qwen-q4");


def main() -> int {
    import mlx_lm;

    adapter = os.environ["ADAPTER"];
    out_prefix = os.environ["OUT_PREFIX"];
    which = os.environ.get("HOLDOUTS", "A,B").split(",");

    t0 = time.time();
    print(f"[gen] loading {MODEL} + adapter={adapter or '(none)'}", flush=True);
    # [TODO: jac port blocked -- the Jac type checker crashes on any reference to
    # `mlx_lm.load` ("'NoneType' object has no attribute 'is_instantiable_class'"),
    # and it also rejects tuple-unpacking an `Any`-typed result, so this one
    # statement is kept as verbatim inline Python.]
    model: any;
    tok: any;
    ::py::
    model, tok = (
        mlx_lm.load(MODEL, adapter_path=adapter) if adapter else mlx_lm.load(MODEL)
    )
    ::py::
    print(f"[gen] model loaded in {time.time() - t0:.1f}s", flush=True);

    for key in which {
        key = key.strip();
        holdout = HOLDOUTS[key];
        out_path = Path(f"{out_prefix}_holdout{key}.gen.jsonl");
        out_path.parent.mkdir(parents=True, exist_ok=True);

        rows = [
            json.loads(line)
            for line in holdout.read_text().split("\n")
            if line.strip()
        ];
        print(f"[gen] holdout {key}: {len(rows)} rows -> {out_path}", flush=True);

        th = time.time();
        with out_path.open("w") as fh {
            for start in range(0, len(rows), BATCH_SIZE) {
                batch = rows[start : start + BATCH_SIZE];
                encoded = [];
                for rec in batch {
                    msgs = [{"role": "user", "content": str(rec["messages"][0]["content"])}];
                    encoded.append(
                        list(tok.apply_chat_template(msgs, add_generation_prompt=True))
                    );
                }
                resp = mlx_lm.batch_generate(
                    model, tok, encoded, max_tokens=MAX_TOKENS, verbose=False
                );
                texts = list(resp.texts);
                for (rec, out) in zip(batch, texts) {
                    fh.write(
                        json.dumps(
                            {
                                "id": rec.get("id"),
                                "category": rec.get("category"),
                                "gate_class": rec.get("gate_class"),
                                "task_type": rec.get("task_type"),
                                "prompt": str(rec["messages"][0]["content"]),
                                "reference": (
                                    str(rec["messages"][1]["content"])
                                    if len(cast(list, rec.get("messages", []))) > 1
                                    else ""
                                ),
                                "generated": out
                            }
                        )
                        + "\n"
                    );
                }
                fh.flush();
                done = min(start + BATCH_SIZE, len(rows));
                el = time.time() - th;
                print(
                    f"[gen] {key} {done}/{len(rows)}  {el:.0f}s elapsed, "
                    f"eta {el / done * (len(rows) - done):.0f}s",
                    flush=True
                );
            }
        }
        print(f"[gen] holdout {key} done in {time.time() - th:.0f}s", flush=True);
    }

    print(f"[gen] ALL DONE in {time.time() - t0:.0f}s", flush=True);
    return 0;
}


with entry:__main__ {
    sys.exit(main());
}
