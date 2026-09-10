"""Stage 1 for 08-nitin-new2-ds: turn the gated candidate pool into the two
training releases mlx_lm / mlx_lm_lora actually read.

    dataset/candidate_pool.jsonl        (16,167 gated rows, seven corpora)
        |-- task_kind in {completion, translation}  ->  dataset/sft_train.jsonl
        `-- task_kind == pair_dpo                   ->  dataset/dpo_train.jsonl

07 got these files straight out of `collect.jac`, because 07 GENERATED its
training rows. 08 does not generate anything -- it merges and gates existing
corpora -- so this is the equivalent seam: pool schema in, trainer schema out.
The 85/15 train/valid split of each release is NOT done here; it is done by
`prep_training_dirs.sh` (seed 42, sort-then-shuffle), byte-for-byte the same
splitter 07 and 04-cpt-sft used.

TWO OUTPUT SCHEMAS, MATCHING 07's EXACTLY WHERE IT MATTERS.

  SFT  (`dataset/sft/` is read as mlx_lm chat data, so `messages` is required)
       messages = [{user: prompt}, {assistant: "```jac\\n<target>\\n```"}]
       The assistant turn IS FENCED. 07's 6,781 SFT rows are 100% fenced and
       the eval/grading scripts strip the fence before compiling, so an unfenced
       08 would train the model out of a convention the harness still expects.

  DPO  (`dpo_fixed_train.jac` reads prompt/chosen/rejected and builds the chat
       turns itself, so the code strings must be RAW)
       prompt / chosen / rejected, unfenced -- again matching 07, whose 3,423
       DPO rows are 100% unfenced.

Everything else carried over is provenance: `id`, `source_type`, `origin` and
`quality_signals` come through untouched so any release row can be traced back
to its pool row and from there to `jac-data-gen` at the pinned commit.
`task_type` and `register` are the 07-compatible slicing labels (07 had exactly
one task_type and two registers; 08 has five and three), derived from
source_type only -- no new information, just 07's column names so the two
experiments can be sliced by the same reporting code.
"""
import os, json, collections;

glob EXT: str = "/Volumes/ExtremePro/JaseciLabs/jac_model_studio";
glob EXP: str = EXT + "/model-experiments/08-nitin-new2-ds";
glob OUTD: str = EXP + "/dataset";
glob STATS: str = "/tmp/nitin_new2_triage";

glob DATASET_VERSION: str = "nitin-new2-ds-v1.0.0";
glob RUN_TAG: str = "nitin-new2";
glob DPO_AXIS: str = "idiomaticity";

# 07-compatible slicing labels, derived from source_type alone.
glob TASK_TYPE: dict[str, str] = {
    "osp": "osp_module",
    "farm": "farm_crud_module",
    "farm_handler": "farm_crud_module",
    "js2jac": "js_to_jac",
    "composer": "python_to_jac_function",
    "step4_work": "python_to_jac_function",
    "step4_dpo": "python_to_jac_function",
};
# osp prompts are issue reports (a user asking); farm prompts are synthesized
# specs (see pipeline.jac farm_prompt); the rest are literal translate orders.
glob REGISTER: dict[str, str] = {
    "osp": "user_request",
    "farm": "spec_style",
    "farm_handler": "spec_style",
    "js2jac": "translate_request",
    "composer": "translate_request",
    "step4_work": "translate_request",
    "step4_dpo": "translate_request",
};

def jl(path: str) -> list[dict] {
    out = [];
    for line in open(path) {
        s = line.strip();
        if s { out.append(json.loads(s)); }
    }
    return out;
}

def common(p: dict) -> dict {
    return {
        "id": p["id"],
        "source_type": p["source_type"],
        "task_type": TASK_TYPE[p["source_type"]],
        "tier": (p["quality_signals"] or {}).get("tier"),
        "dataset_version": DATASET_VERSION,
        "run_tag": RUN_TAG,
        "origin": p["origin"],
        "quality_signals": p["quality_signals"],
    };
}

def fence(code: str) -> str {
    return "```jac\n" + str(code).strip() + "\n```";
}

with entry:__main__ {
    pool = jl(OUTD + "/candidate_pool.jsonl");
    print("candidate pool:", len(pool));

    sft = [];
    dpo = [];
    for p in pool {
        base = common(p);
        if p["task_kind"] == "pair_dpo" {
            row = dict(base);
            row["axis"] = DPO_AXIS;
            row["prompt"] = p["prompt"];
            row["chosen"] = str(p["jac"]).strip();
            row["rejected"] = str(p["jac_rejected"]).strip();
            dpo.append(row);
        } else {
            row = dict(base);
            row["register"] = REGISTER[p["source_type"]];
            row["messages"] = [
                {"role": "user", "content": p["prompt"]},
                {"role": "assistant", "content": fence(str(p["jac"]))},
            ];
            sft.append(row);
        }
    }

    # trainer-contract assertions -- prep_training_dirs.sh drops rows that fail
    # these silently, so fail loudly here instead of shipping a short split.
    for r in sft {
        if not r["messages"] or len(list(r["messages"])) != 2 {
            raise RuntimeError("bad messages on " + str(r["id"]));
        }
        if not str(r["messages"][0]["content"]).strip() {
            raise RuntimeError("empty user turn on " + str(r["id"]));
        }
        if not str(r["messages"][1]["content"]).strip() {
            raise RuntimeError("empty assistant turn on " + str(r["id"]));
        }
    }
    for r in dpo {
        for k in ["prompt", "chosen", "rejected"] {
            if not str(r[k]).strip() { raise RuntimeError("empty " + k + " on " + str(r["id"])); }
        }
        if r["chosen"] == r["rejected"] {
            raise RuntimeError("chosen == rejected on " + str(r["id"]));
        }
    }
    ids = [r["id"] for r in sft] + [r["id"] for r in dpo];
    dupes = [k for (k, v) in collections.Counter(ids).items() if v > 1];
    if dupes { raise RuntimeError("duplicate release ids: " + str(dupes[:10])); }
    if len(sft) + len(dpo) != len(pool) {
        raise RuntimeError("release row count does not match the pool");
    }

    with open(OUTD + "/sft_train.jsonl", "w") as f {
        for r in sft { f.write(json.dumps(r) + "\n"); }
    }
    with open(OUTD + "/dpo_train.jsonl", "w") as f {
        for r in dpo { f.write(json.dumps(r) + "\n"); }
    }

    sft_type = collections.Counter(r["source_type"] for r in sft);
    dpo_type = collections.Counter(r["source_type"] for r in dpo);
    os.makedirs(STATS, exist_ok=True);
    json.dump({
        "pool_n": len(pool), "sft_n": len(sft), "dpo_n": len(dpo),
        "sft_by_source_type": dict(sft_type), "dpo_by_source_type": dict(dpo_type),
        "sft_by_register": dict(collections.Counter(r["register"] for r in sft)),
        "sft_by_task_type": dict(collections.Counter(r["task_type"] for r in sft)),
        "sft_by_tier": dict(collections.Counter(str(r["tier"]) for r in sft)),
        "dataset_version": DATASET_VERSION, "run_tag": RUN_TAG, "dpo_axis": DPO_AXIS,
    }, open(STATS + "/release.json", "w"), indent=1);

    print("sft_train.jsonl:", len(sft), dict(sft_type));
    print("dpo_train.jsonl:", len(dpo), dict(dpo_type));
    print("sft by register:", dict(collections.Counter(r["register"] for r in sft)));
    print("sft by tier    :", dict(collections.Counter(str(r["tier"]) for r in sft)));
}
