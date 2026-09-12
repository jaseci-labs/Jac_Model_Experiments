"""NaN guard for the SFT training split (Stage 2, after prep_training_dirs.sh).

WHY. Both SFT configs train with `mask_prompt: true` and `max_seq_length: 3072`.
mlx_lm masks the prompt (loss only on the assistant turn) and truncates every
row to max_seq_length. A row whose PROMPT ALONE is >= max_seq_length tokens
therefore has zero loss-bearing tokens after truncation -> 0/0 -> NaN loss,
which poisoned 08's first stock SFT run (run dir set aside, not kept in this repo).
This script drops exactly those rows.

The prompt length is measured the same way mlx_lm.tuner.datasets.ChatDataset
computes its mask offset: apply_chat_template(messages[:-1],
add_generation_prompt=True) with the base model's own tokenizer.

Rewrites $DATA_DIR/{train,valid}.jsonl IN PLACE (no backup; a file is only
rewritten when at least one row is dropped, and kept rows are written
byte-for-byte). Prints dropped counts. Idempotent.

    [DATA_DIR=<exp>/dataset/sft] [MODEL=models/qwen-q4] [MAX_SEQ_LENGTH=3072] \
      .venv/bin/jac run experiments/08-nitin-new2-ds/scaffold/nan_guard.jac

The experiment is the dir this file sits in (<repo>/experiments/<exp>/scaffold/).
"""

import json;
import os;
import from pathlib { Path }

glob EXP: Path = Path(str(__file__)).resolve().parents[1];   # <repo>/experiments/<exp> (this file: <exp>/scaffold/)
glob ROOT: Path = EXP.parents[1];                            # repo root
glob DATA_DIR: Path = Path(os.environ.get("DATA_DIR", str(EXP / "dataset" / "sft")));
glob MODEL: Path = Path(os.environ.get("MODEL", str(ROOT / "models" / "qwen-q4")));
glob MAX_SEQ: int = int(os.environ.get("MAX_SEQ_LENGTH", "3072"));


def prompt_len(tok: any, rec: dict[str, any]) -> int {
    messages: list[dict[str, any]] = list(rec["messages"]);
    add_gen: bool = messages[-1].get("role") == "assistant";
    ids: list[int] = list(tok.apply_chat_template(
        messages[:-1], tools=rec.get("tools", None),
        add_generation_prompt=add_gen, return_dict=False
    ));
    return len(ids);
}


with entry {
    import from mlx_lm.tokenizer_utils { load }

    tok = load(MODEL);
    print("tokenizer: " + str(MODEL) + "  max_seq_length=" + str(MAX_SEQ));
    total_dropped: int = 0;
    for split in ["train", "valid"] {
        p: Path = DATA_DIR / (split + ".jsonl");
        if not p.exists() {
            print("  SKIP " + str(p) + " (missing)");
            continue;
        }
        kept: list[str] = [];
        dropped: list[str] = [];
        for line in p.read_text().split("\n") {
            if line.strip() == "" { continue; }
            rec: dict[str, any] = dict(json.loads(line));
            n: int = prompt_len(tok, rec);
            if n >= MAX_SEQ {
                dropped.append(str(rec.get("id", "?")) + " (" + str(n) + " prompt tokens)");
            } else {
                kept.append(line);
            }
        }
        for d in dropped { print("    drop " + d); }
        if len(dropped) > 0 {
            p.write_text("\n".join(kept) + "\n");
        }
        print("  " + str(p) + ": kept " + str(len(kept)) + ", dropped " + str(len(dropped)));
        total_dropped += len(dropped);
    }
    print("NAN_GUARD: dropped " + str(total_dropped) + " row(s) total");
}
