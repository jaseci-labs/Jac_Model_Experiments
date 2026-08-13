"""
collect.jac -- Phase B ("collect") of the batch-handoff pipeline for
06-nitin-ds-sft. Joins each shard's pending_batch_<track>_<NN>.jsonl with the
responses_batch_<track>_<NN>.jsonl a dispatched Claude Code Agent wrote for it,
gates every pair, and appends survivors to the release file. Every drop is
counted and attributed -- this project's "no silent caps" rule (workflow.md
conventions).

    JAC_BATCH_TRACK=sft  -> dataset/sft_train.jsonl   (rejects: dataset/rejected/sft/)
    JAC_BATCH_TRACK=dpo  -> dataset/dpo_train.jsonl   (rejects: dataset/rejected/dpo/)

Responses file format, one JSON object per line, exactly jacgen2's
batch.jac contract:

    {"batch_id": "<copied verbatim from the pending item>", "text": "<model output>", "ok": true}

SFT gates (workflow.md Stage 1):
  1. ok == true, instruction non-empty
  2. instruction is prose, not code: no ``` fence, no `{`+`;` pair, no long
     verbatim line lifted out of the answer
  3. instruction is not a bare restatement of the function name, and clears a
     minimum length
  4. the assistant turn is byte-identical to the pool's jac_code -- the agent is
     never given a slot to write code, so any divergence means a mis-join or a
     stray edit, and the pair is dropped
  5. compile: the pool row already carries compile_pass == true from the triage
     stage's `jac run` gate, and gate 4 proves the code is that exact byte
     string, so the compile evidence transfers. Set JAC_RECHECK_COMPILE=1 to
     re-run `jac run` on every row anyway (~1.2 s each, so ~2 h for 5474 --
     off by default because it re-proves an already-proven, unmodified
     artifact).

DPO gates (workflow.md Stage 2 -- the dual gate, both mandatory, no flag):
  1. ok == true, response parses into BUG_TYPE / BUGGY / SYMPTOM (or is an
     explicit `SKIP: <reason>`, which is a clean drop, not an error)
  2. chosen (the pool's jac_code) compiles -- `jac run` exit 0
  3. rejected ALSO compiles -- `jac run` exit 0. A rejected side that fails to
     parse teaches "prefer code that compiles" instead of "prefer code that is
     correct", which is the wrong preference signal (spec.md section 3.1)
  4. chosen != rejected, checked both byte-wise AND after comment/whitespace
     normalization, so a whitespace-only "edit" cannot masquerade as a bug
     injection
  5. the function signature survived: `def <func_name>` still present in the
     rejected side

  Stated limitation, per workflow.md Stage 2: this corpus ships no pinned
  expected outputs and these files are bare `def`s with no `with entry`, so
  running them proves compilation and nothing else. That the injected bug
  actually diverges behaviourally is NOT mechanically proven here -- it rests on
  Fable's authoring plus the symptom line. Do not describe these pairs as
  behaviourally verified.

Run (from repo root):

    JAC_BATCH_TRACK=sft /Volumes/ExtremePro/JaseciLabs/jac_model_studio/.venv/bin/jac \
        run model-experiments/06-nitin-ds-sft/scripts/collect.jac

Env:
    JAC_BATCH_TRACK      sft | dpo                       (required)
    JAC_BIN              jac binary used for compile gates
                         (default /Users/ayush/.local/bin/jac -- the triage
                          report's finding 4: the project venv's jac lives on
                          the 4.3 MB/s external drive and a single invocation
                          there exceeded 120 s. Same jaclang 0.16.1, parity
                          verified 10/10 on a stratified sample.)
    JAC_RECHECK_COMPILE  1 to force the optional SFT compile re-check
    JAC_SHARDS           comma-separated shard numbers to collect, e.g. "1,2,3"
                         (default: every shard with a responses file on disk)
    JAC_GENERATOR_MODEL_ID  recorded into generator_model_id (default "unpinned-agent-dispatch")

NOTE: helpers below are inlined from jacgen2's batch.jac / common.jac / gate.jac
rather than imported, for the reason spelled out in prepare_batches.jac's
docstring -- a cross-directory Jac import type-checks but dies at runtime.
"""

import os;
import json;
import subprocess;
import tempfile;
import from pathlib { Path }
import from datetime { datetime }

glob EXP_ROOT: str = str(Path(str(__file__)).resolve().parent.parent);
glob RAW_ROOT: str = EXP_ROOT + "/dataset/raw_output";

glob JAC_BIN: str = os.environ.get("JAC_BIN", "/Users/ayush/.local/bin/jac");

glob CATEGORY: str = "conversion";
glob TASK_TYPE: str = "python_to_jac_function";
glob DPO_AXIS: str = "correctness";

glob PROMPT_VER: str = "nitin-reverse-instruct-v1.0";
glob CONTEXT_BUNDLE_VER: str = "jac-context-v2";
glob VALIDATOR_VER: str = "validator-v2.0";
glob DATASET_VER: str = "nitin-ds-v1.0.0";
glob RUN_TAG: str = "nitin";

glob MIN_INSTRUCTION_WORDS: int = 5;
glob MIN_INSTRUCTION_CHARS: int = 25;


# ---------------------------------------------------------------------------
# Inlined helpers (see docstring)
# ---------------------------------------------------------------------------

def append_jsonl(path: str, rec: dict) {
    p: Path = Path(path);
    p.parent.mkdir(parents=True, exist_ok=True);
    with open(path, "a") as f {
        f.write(json.dumps(rec) + "\n");
    }
}

def read_jsonl(path: str) -> list[dict] {
    rows: list[dict] = [];
    p: Path = Path(path);
    if not p.exists() {
        return rows;
    }
    with open(path, "r") as f {
        lines: list[str] = f.readlines();
    }
    for line in lines {
        s: str = line.strip();
        if s {
            rec = json.loads(s);
            rows.append(dict(rec));
        }
    }
    return rows;
}

def existing_keys(path: str) -> dict {
    keys: dict = {};
    for rec in read_jsonl(path) {
        keys[str(rec["id"])] = True;
    }
    return keys;
}

def read_response_batch(path: str) -> dict {
    responses: dict = {};
    for rec in read_jsonl(path) {
        responses[str(rec["batch_id"])] = {"text": rec.get("text", ""), "ok": rec.get("ok", False)};
    }
    return responses;
}

"""Runs a Jac snippet in its own tempdir. cwd is set to that tempdir -- the same
isolation gate.jac's run_jac_local uses, so jac's persistent root store cannot
leak state between snippets."""
def run_jac_local(jac_code: str, timeout: int = 60) -> tuple[int, str, str] {
    with tempfile.TemporaryDirectory() as d {
        f: Path = Path(str(d)) / "snippet.jac";
        f.write_text(jac_code);
        try {
            p = subprocess.run(
                [JAC_BIN, "run", str(f)], capture_output=True, text=True, timeout=timeout, cwd=str(d)
            );
            return (p.returncode, p.stdout, p.stderr);
        } except (subprocess.SubprocessError, OSError) as e {
            return (124, "", str(e));
        }
    }
}

"""Comment-stripped, whitespace-normalized token stream -- jacgen2's dedup2.jac
`code_tokens`. Used here only to prove chosen and rejected genuinely differ in
code, not merely in spacing."""
def code_tokens(text: str) -> list[str] {
    out: list[str] = [];
    for line in text.split("\n") {
        s: str = line.split("#")[0].strip();
        if s {
            for tok in s.split() {
                out.append(tok);
            }
        }
    }
    return out;
}


# ---------------------------------------------------------------------------
# Shard discovery
# ---------------------------------------------------------------------------

def shard_indices(track: str) -> list[int] {
    override: str | None = os.environ.get("JAC_SHARDS");
    found: list[int] = [];
    if override {
        for part in str(override).split(",") {
            s: str = part.strip();
            if s {
                found.append(int(s));
            }
        }
        return found;
    }
    d: Path = Path(RAW_ROOT + "/" + track);
    if not d.exists() {
        return found;
    }
    for p in sorted(d.glob("responses_batch_" + track + "_*.jsonl")) {
        stem: str = p.name.replace(".jsonl", "");
        found.append(int(stem.split("_")[-1]));
    }
    return sorted(found);
}

def pending_file(track: str, idx: int) -> str {
    return RAW_ROOT + "/" + track + "/pending_batch_" + track + "_" + str(idx).zfill(2) + ".jsonl";
}

def responses_file(track: str, idx: int) -> str {
    return RAW_ROOT + "/" + track + "/responses_batch_" + track + "_" + str(idx).zfill(2) + ".jsonl";
}

def bump(counts: dict[str, int], reason: str) {
    counts[reason] = counts.get(reason, 0) + 1;
}


# ---------------------------------------------------------------------------
# SFT collect
# ---------------------------------------------------------------------------

"""Returns "" when the instruction is acceptable, otherwise the rejection
reason."""
def check_instruction(instruction: str, ctx: dict) -> str {
    if not instruction {
        return "empty instruction text";
    }
    if "```" in instruction {
        return "instruction contains a code fence";
    }
    if "{" in instruction and ";" in instruction {
        return "instruction looks like code (contains both '{' and ';')";
    }
    words: list[str] = instruction.split();
    if len(words) < MIN_INSTRUCTION_WORDS or len(instruction) < MIN_INSTRUCTION_CHARS {
        return "instruction too short (" + str(len(words)) + " words, " + str(len(instruction)) + " chars)";
    }
    fname: str = str(ctx["func_name"]);
    stripped: str = instruction.lower().replace(fname.lower(), " ");
    letters: str = "";
    for ch in stripped {
        if ch.isalpha() {
            letters += ch;
        }
    }
    if len(letters) < 15 {
        return "instruction is a bare restatement of the function name";
    }
    # Verbatim answer-leak check: any substantial line of the target code showing
    # up inside the instruction means the model quoted the answer.
    for line in str(ctx["jac_code"]).split("\n") {
        s: str = line.strip();
        if len(s) >= 20 and s in instruction {
            return "instruction quotes the answer code verbatim";
        }
    }
    return "";
}

def run_collect_sft() {
    release: str = EXP_ROOT + "/dataset/sft_train.jsonl";
    rejected_path: str = EXP_ROOT + "/dataset/rejected/sft/rejected.jsonl";
    model_id: str = os.environ.get("JAC_GENERATOR_MODEL_ID", "unpinned-agent-dispatch");
    recheck: bool = os.environ.get("JAC_RECHECK_COMPILE", "0") == "1";

    already: dict = existing_keys(release);
    shards: list[int] = shard_indices("sft");
    if not shards {
        print("no responses_batch_sft_*.jsonl found under " + RAW_ROOT + "/sft/ -- nothing to collect.");
        print("this is the normal state between prepare and the dispatched agents' pass.");
        return;
    }

    processed: int = 0;
    accepted: int = 0;
    dup_skipped: int = 0;
    reasons: dict[str, int] = {};

    for idx in shards {
        p_path: str = pending_file("sft", idx);
        r_path: str = responses_file("sft", idx);
        pending: dict = {};
        for rec in read_jsonl(p_path) {
            pending[str(rec["batch_id"])] = rec;
        }
        if not pending {
            print("shard " + str(idx).zfill(2) + ": pending file missing/empty at " + p_path + " -- skipped.");
            continue;
        }
        responses: dict = read_response_batch(r_path);
        shard_accepted: int = 0;

        for batch_id in responses {
            processed += 1;
            resp: dict = dict(responses[batch_id]);
            ctx = pending.get(batch_id, None);
            if ctx is None {
                bump(reasons, "no matching pending context for batch_id");
                append_jsonl(rejected_path, {"batch_id": batch_id, "shard": idx,
                    "reason": "no matching pending context for batch_id"});
                continue;
            }
            if not bool(resp["ok"]) {
                bump(reasons, "generation failed (ok:false)");
                append_jsonl(rejected_path, {"batch_id": batch_id, "shard": idx,
                    "pool_id": ctx["pool_id"], "reason": "generation failed (ok:false)"});
                continue;
            }

            instruction: str = str(resp["text"]).strip();
            bad: str = check_instruction(instruction, dict(ctx));
            if bad {
                bump(reasons, bad);
                append_jsonl(rejected_path, {"batch_id": batch_id, "shard": idx,
                    "pool_id": ctx["pool_id"], "reason": bad, "instruction": instruction[:300]});
                continue;
            }

            jac_code: str = str(ctx["jac_code"]);
            if not bool(ctx["compile_pass"]) {
                bump(reasons, "pool row not compile_pass");
                append_jsonl(rejected_path, {"batch_id": batch_id, "shard": idx,
                    "pool_id": ctx["pool_id"], "reason": "pool row not compile_pass"});
                continue;
            }
            if recheck {
                (rc, out, err) = run_jac_local(jac_code);
                if rc != 0 {
                    reason: str = "compile re-check failed exit " + str(rc);
                    bump(reasons, reason);
                    append_jsonl(rejected_path, {"batch_id": batch_id, "shard": idx,
                        "pool_id": ctx["pool_id"], "reason": reason, "stderr": err.strip()[:300]});
                    continue;
                }
            }

            rec_id: str = str(ctx["seed_id"]) + "__" + TASK_TYPE + "__" + str(ctx["variant_idx"]);
            if rec_id in already {
                dup_skipped += 1;
                continue;
            }

            rec: dict = {
                "id": rec_id,
                "category": CATEGORY,
                "task_type": TASK_TYPE,
                "complexity": "simple",
                "compiler_pass": True,
                "test_pass": None,
                "manually_reviewed": False,
                "generator": "opus-api",
                "generator_model_id": model_id,
                "gate_class": "compile_only",
                "variant_idx": int(ctx["variant_idx"]),
                "generation_date": datetime.now().isoformat(),
                "source_prompt_version": PROMPT_VER,
                "context_bundle_version": CONTEXT_BUNDLE_VER,
                "validator_version": VALIDATOR_VER,
                "dataset_version": DATASET_VER,
                "run_tag": RUN_TAG,
                "seed_id": str(ctx["seed_id"]),
                "seed_tier": 0,
                "messages": [
                    {"role": "user", "content": instruction},
                    {"role": "assistant", "content": "```jac\n" + jac_code + "\n```"}
                ],
                "origin": ctx["origin"],
                "register": ctx["register"]
            };
            append_jsonl(release, rec);
            already[rec_id] = True;
            accepted += 1;
            shard_accepted += 1;
        }
        print("  shard " + str(idx).zfill(2) + ": " + str(len(responses)) + " responses, "
            + str(shard_accepted) + " accepted");
    }

    print("");
    print("=== collect sft summary ===");
    print("shards collected: " + str(len(shards)));
    print("responses processed: " + str(processed));
    print("accepted (newly written): " + str(accepted));
    print("skipped as already-persisted: " + str(dup_skipped));
    print("rejected: " + str(processed - accepted - dup_skipped));
    for reason in reasons {
        print("  - " + reason + ": " + str(reasons[reason]));
    }
    print("compile re-check: " + ("ON" if recheck else "OFF (trusting triage `jac run` gate + byte-identity)"));
    print("release: " + release);
    print("rejects: " + rejected_path);
}


# ---------------------------------------------------------------------------
# DPO collect
# ---------------------------------------------------------------------------

"""Parses the BUG_TYPE / BUGGY / SYMPTOM reply shape. Returns {} on anything
malformed rather than raising -- a bad response is a rejected row, not a crash.
A leading `SKIP:` comes back as {"skip": "<reason>"}."""
def parse_dpo_response(text: str) -> dict {
    s: str = text.strip();
    if s.startswith("SKIP:") {
        return {"skip": s[5:].strip()};
    }
    if "BUGGY:" not in s or "SYMPTOM:" not in s {
        return {};
    }
    bug_type: str = "unspecified";
    if s.startswith("BUG_TYPE:") {
        first: str = s.split("\n")[0];
        bug_type = first.split(":", 1)[1].strip();
    }
    after: str = s.split("BUGGY:", 1)[1];
    if "```" not in after {
        return {};
    }
    fenced: str = after.split("```", 1)[1];
    if "```" not in fenced {
        return {};
    }
    body: str = fenced.split("```", 1)[0];
    if body.startswith("jac\n") {
        body = body[4:];
    } elif body.startswith("jac\r\n") {
        body = body[5:];
    }
    buggy: str = body.strip("\n");
    tail: str = s.split("SYMPTOM:", 1)[1];
    symptom: str = tail.strip().split("\n")[0].strip();
    if not buggy.strip() or not symptom {
        return {};
    }
    return {"buggy_code": buggy, "symptom": symptom, "bug_type": bug_type};
}

def run_collect_dpo() {
    release: str = EXP_ROOT + "/dataset/dpo_train.jsonl";
    rejected_path: str = EXP_ROOT + "/dataset/rejected/dpo/rejected.jsonl";

    already: dict = existing_keys(release);
    shards: list[int] = shard_indices("dpo");
    if not shards {
        print("no responses_batch_dpo_*.jsonl found under " + RAW_ROOT + "/dpo/ -- nothing to collect.");
        print("this is the normal state between prepare and the dispatched agents' pass.");
        return;
    }

    processed: int = 0;
    accepted: int = 0;
    dup_skipped: int = 0;
    reasons: dict[str, int] = {};

    for idx in shards {
        p_path: str = pending_file("dpo", idx);
        r_path: str = responses_file("dpo", idx);
        pending: dict = {};
        for rec in read_jsonl(p_path) {
            pending[str(rec["batch_id"])] = rec;
        }
        if not pending {
            print("shard " + str(idx).zfill(2) + ": pending file missing/empty at " + p_path + " -- skipped.");
            continue;
        }
        responses: dict = read_response_batch(r_path);
        shard_accepted: int = 0;

        for batch_id in responses {
            processed += 1;
            resp: dict = dict(responses[batch_id]);
            ctx = pending.get(batch_id, None);
            if ctx is None {
                bump(reasons, "no matching pending context for batch_id");
                append_jsonl(rejected_path, {"batch_id": batch_id, "shard": idx,
                    "reason": "no matching pending context for batch_id"});
                continue;
            }
            if not bool(resp["ok"]) {
                bump(reasons, "generation failed (ok:false)");
                append_jsonl(rejected_path, {"batch_id": batch_id, "shard": idx,
                    "pool_id": ctx["pool_id"], "reason": "generation failed (ok:false)"});
                continue;
            }

            parsed: dict = parse_dpo_response(str(resp["text"]));
            if not parsed {
                bump(reasons, "unparseable response (expected BUG_TYPE/BUGGY/SYMPTOM)");
                append_jsonl(rejected_path, {"batch_id": batch_id, "shard": idx,
                    "pool_id": ctx["pool_id"],
                    "reason": "unparseable response (expected BUG_TYPE/BUGGY/SYMPTOM)"});
                continue;
            }
            if "skip" in parsed {
                bump(reasons, "agent declined (SKIP)");
                append_jsonl(rejected_path, {"batch_id": batch_id, "shard": idx,
                    "pool_id": ctx["pool_id"], "reason": "agent declined (SKIP)",
                    "detail": parsed["skip"]});
                continue;
            }

            chosen: str = str(ctx["jac_code"]);
            rejected_code: str = str(parsed["buggy_code"]);

            if chosen.strip() == rejected_code.strip() {
                bump(reasons, "bug injection silently failed: chosen == rejected (byte-identical)");
                append_jsonl(rejected_path, {"batch_id": batch_id, "shard": idx,
                    "pool_id": ctx["pool_id"],
                    "reason": "bug injection silently failed: chosen == rejected (byte-identical)"});
                continue;
            }
            if code_tokens(chosen) == code_tokens(rejected_code) {
                bump(reasons, "bug injection silently failed: identical after comment/whitespace normalization");
                append_jsonl(rejected_path, {"batch_id": batch_id, "shard": idx,
                    "pool_id": ctx["pool_id"],
                    "reason": "bug injection silently failed: identical after comment/whitespace normalization"});
                continue;
            }
            if ("def " + str(ctx["func_name"])) not in rejected_code {
                bump(reasons, "rejected variant lost the original function signature");
                append_jsonl(rejected_path, {"batch_id": batch_id, "shard": idx,
                    "pool_id": ctx["pool_id"],
                    "reason": "rejected variant lost the original function signature"});
                continue;
            }

            (c_rc, c_out, c_err) = run_jac_local(chosen);
            if c_rc != 0 {
                reason: str = "chosen failed compile gate exit " + str(c_rc);
                bump(reasons, reason);
                append_jsonl(rejected_path, {"batch_id": batch_id, "shard": idx,
                    "pool_id": ctx["pool_id"], "reason": reason, "stderr": c_err.strip()[:300]});
                continue;
            }
            (r_rc, r_out, r_err) = run_jac_local(rejected_code);
            if r_rc != 0 {
                reason: str = "rejected failed compile gate exit " + str(r_rc)
                    + " (bug must be a LOGIC divergence, not a syntax break)";
                bump(reasons, "rejected failed compile gate (syntax break, not a logic bug)");
                append_jsonl(rejected_path, {"batch_id": batch_id, "shard": idx,
                    "pool_id": ctx["pool_id"], "reason": reason, "stderr": r_err.strip()[:300]});
                continue;
            }

            rec_id: str = str(ctx["seed_id"]) + "__" + DPO_AXIS + "__" + str(ctx["variant_idx"]);
            if rec_id in already {
                dup_skipped += 1;
                continue;
            }

            rec: dict = {
                "id": rec_id,
                "axis": DPO_AXIS,
                "prompt": "Write correct Jac code that avoids this bug: " + str(parsed["symptom"]),
                "chosen": chosen,
                "rejected": rejected_code,
                "run_tag": RUN_TAG,
                "generator": "fable-api",
                "generation_date": datetime.now().isoformat(),
                "compiler_pass_chosen": True,
                "compiler_pass_rejected": True,
                "origin": ctx["origin"],
                "bug_type": parsed["bug_type"]
            };
            append_jsonl(release, rec);
            already[rec_id] = True;
            accepted += 1;
            shard_accepted += 1;
        }
        print("  shard " + str(idx).zfill(2) + ": " + str(len(responses)) + " responses, "
            + str(shard_accepted) + " accepted");
    }

    print("");
    print("=== collect dpo summary ===");
    print("shards collected: " + str(len(shards)));
    print("responses processed: " + str(processed));
    print("accepted (newly written): " + str(accepted));
    print("skipped as already-persisted: " + str(dup_skipped));
    print("rejected: " + str(processed - accepted - dup_skipped));
    for reason in reasons {
        print("  - " + reason + ": " + str(reasons[reason]));
    }
    print("dual compile gate: ON (mandatory, chosen AND rejected, jac=" + JAC_BIN + ")");
    print("behavioural divergence: NOT mechanically verified -- no pinned expected outputs");
    print("release: " + release);
    print("rejects: " + rejected_path);
}


with entry:__main__ {
    track: str | None = os.environ.get("JAC_BATCH_TRACK");
    if track == "sft" {
        run_collect_sft();
    } elif track == "dpo" {
        run_collect_dpo();
    } else {
        raise ValueError(
            "JAC_BATCH_TRACK must be set to 'sft' or 'dpo', got: " + str(track)
        );
    }
}
