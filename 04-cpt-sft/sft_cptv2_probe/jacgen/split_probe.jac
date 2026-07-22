"""Fresh 85/15 split for the CPT-v2-SFT/DPO hypothesis probe.

Deliberately ignores the frozen 18-id pilot holdout in dataset/shared/holdouts/
(that stays reserved for the formal 3-arm protocol) -- this is a separate,
one-off quick test, sized properly against the real 9608/826-row datasets.

Deterministic shuffle (seed 42, matches the project's existing lora.yaml
convention) then a straight 85/15 slice, not a stride split -- at ~1441/124
holdout rows a stride split and a shuffled slice are statistically
equivalent, and a slice is simpler to reason about.

Run from the repo root:
    jac run model-experiments/04-cpt-sft/sft_cptv2_probe/jacgen/split_probe.jac
"""

import json;
import random;
import from pathlib { Path }
import from typing { Any }

glob SFT_SRC: str = "model-experiments/04-cpt-sft/dataset/fresh/releases/sft_train.jsonl";
glob DPO_SRC: str = "model-experiments/04-cpt-sft/dataset/fresh/releases/dpo_train.jsonl";
glob OUT_ROOT: str = "model-experiments/04-cpt-sft/sft_cptv2_probe/dataset";
glob SEED: int = 42;
glob VALID_FRAC: float = 0.15;
glob TOKENIZER_MODEL: str = "models/qwen-q4";


def read_lines(path: str) -> list[dict] {
    rows: list[dict] = [];
    p = Path(path);
    for line in p.read_text().split("\n") {
        if line.strip() == "" { continue; }
        rows.append(dict(json.loads(line)));
    }
    return rows;
}


def write_jsonl(path: str, rows: list[dict]) {
    with open(path, "w") as f {
        for r in rows { f.write(json.dumps(r) + "\n"); }
    }
}


def split_rows(rows: list[dict]) -> tuple[list[dict], list[dict]] {
    idx: list[int] = list(range(len(rows)));
    rng = random.Random(SEED);
    rng.shuffle(idx);
    n_valid: int = int(round(len(rows) * VALID_FRAC));
    valid_idx: set = set(idx[:n_valid]);
    train: list[dict] = [];
    valid: list[dict] = [];
    for (i, r) in enumerate(rows) {
        if i in valid_idx { valid.append(r); } else { train.append(r); }
    }
    return (train, valid);
}


def token_stats(texts: list[str], tok: Any) -> dict {
    counts: list[int] = [];
    for t in texts { counts.append(len(tok.encode(t))); }
    counts.sort();
    n: int = len(counts);
    def pct(p: float) -> int {
        if n == 0 { return 0; }
        i: int = int(p * float(n - 1));
        return counts[i];
    }
    mx: int = 0;
    if n > 0 { mx = counts[n - 1]; }
    return {"p50": pct(0.50), "p95": pct(0.95), "p99": pct(0.99), "max": mx, "n": n};
}


with entry {
    Path(OUT_ROOT + "/sft").mkdir(parents=True, exist_ok=True);
    Path(OUT_ROOT + "/dpo").mkdir(parents=True, exist_ok=True);

    # --- SFT split ---
    sft_rows: list[dict] = read_lines(SFT_SRC);
    (sft_train, sft_valid) = split_rows(sft_rows);
    sft_train_stripped: list[dict] = [{"messages": r["messages"]} for r in sft_train];
    write_jsonl(OUT_ROOT + "/sft/train.jsonl", sft_train_stripped);
    write_jsonl(OUT_ROOT + "/sft/valid.jsonl", sft_valid);  # full rows -- doubles as functional-eval holdout
    print("SFT train: " + str(len(sft_train)) + "  valid(holdout): " + str(len(sft_valid)));

    # --- DPO split ---
    dpo_rows: list[dict] = read_lines(DPO_SRC);
    (dpo_train, dpo_valid) = split_rows(dpo_rows);
    dpo_train_stripped: list[dict] = [{"prompt": str(r["prompt"]), "chosen": str(r["chosen"]),
                                        "rejected": str(r["rejected"])} for r in dpo_train];
    dpo_valid_stripped: list[dict] = [{"prompt": str(r["prompt"]), "chosen": str(r["chosen"]),
                                        "rejected": str(r["rejected"])} for r in dpo_valid];
    write_jsonl(OUT_ROOT + "/dpo/train.jsonl", dpo_train_stripped);
    write_jsonl(OUT_ROOT + "/dpo/valid.jsonl", dpo_valid_stripped);
    print("DPO train: " + str(len(dpo_train)) + "  valid: " + str(len(dpo_valid)));

    # --- token-length stats (sizes max_seq_length in Task 2/6's configs) ---
    import mlx_lm;   # lazy import, same convention as eval_probe.jac
    (_model, tok) = mlx_lm.load(TOKENIZER_MODEL);  # jac:ignore[E1032]

    sft_texts: list[str] = [];
    for r in sft_rows {
        joined: str = "";
        for m in r["messages"] { joined += str(m["content"]) + "\n"; }
        sft_texts.append(joined);
    }
    dpo_texts: list[str] = [];
    for r in dpo_rows {
        dpo_texts.append(str(r["prompt"]) + "\n" + str(r["chosen"]));
        dpo_texts.append(str(r["prompt"]) + "\n" + str(r["rejected"]));
    }

    stats: dict = {"sft": token_stats(sft_texts, tok), "dpo": token_stats(dpo_texts, tok)};
    with open(OUT_ROOT + "/dataset_stats.json", "w") as f { f.write(json.dumps(stats, indent=2)); }
    print("token stats -> " + OUT_ROOT + "/dataset_stats.json");
    print(json.dumps(stats, indent=2));
}
