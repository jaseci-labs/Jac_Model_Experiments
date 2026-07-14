

# Attempt 03 Workflow: CPT → SFT/DPO → GRPO

Operational runbook for the full stack. Architecture/ablation/model/eval design of record is [design.md](design.md); CPT data spec is [cpt-dataset-design.md](cpt-dataset-design.md). SFT/DPO and RL/GRPO redesign specs are follow-ups (not yet written — Phase 4 and Phase 6 below are placeholders until those land).

```mermaid
---
config:
  flowchart:
    nodeSpacing: 55
    rankSpacing: 80
  themeVariables:
    fontSize: 16px
---
flowchart TD
    subgraph P0["Phase 0: Base eval (checkpoint 0)"]
        BASE["Qwen3-Coder-30B-A3B-Instruct\n(unmodified)"]
        MCQ0["Semantic MCQ"]
        JUDGE0["Online-judge\n(compiler/pass@k)"]
        BASE --> MCQ0 & JUDGE0
    end

    subgraph P1["Phase 1: CPT data build (see cpt-dataset-design.md)"]
        direction TB
        SRC_DOCS["Jac docs\njaseci-labs/jaseci docs/docs\nsemantic chunk, upsample 3x"]
        SRC_PAPER["OSP paper\narXiv 2503.15812 (tex)\nsection chunk, strip \\cite/\\ref"]
        SRC_BLOG["Blogs\nblogs.jaseci.org\nSOURCE UNKNOWN - blocking"]
        SRC_CODE["Code\njaseci-labs org repos\nrepo-pack + 50% FIM"]
        SRC_REHEARSE["CF rehearsal\ngeneral code/Python slice\n15-30% of tokens"]
        DECONTAM["Decontaminate vs\nexisting eval holdouts\n(14-gram MinHash, Jaccard>0.5)"]
        PACK["Pack + EOS-join\n(tokenizer.eos_token)"]
        SPLIT1["Split 85/15 train/val\nstratified by source"]
        MANIFEST1["manifest.json\n(counts, weights, decontam drops)"]
        SRC_DOCS & SRC_PAPER & SRC_BLOG & SRC_CODE & SRC_REHEARSE --> DECONTAM --> PACK --> SPLIT1 --> MANIFEST1
    end

    P0 --> P1

    subgraph P2["Phase 2: CPT training"]
        CPT_LORA["LoRA continual pretrain\nnext-token, raw text\nQ4, 48GB M5 Pro"]
        CF_MON["CF monitor:\ngeneral-coding eval\nevery checkpoint"]
        CPT_LORA <--> CF_MON
    end

    MANIFEST1 --> P2

    subgraph P3["Checkpoint 1: +CPT eval"]
        MCQ1["Semantic MCQ\n(picks CPT checkpoint)"]
        TRUST1["Human-sample trust check\n(Hamming sim vs LLM grade)"]
        MCQ1 --> TRUST1
    end

    P2 --> P3

    subgraph P4["Phase 4: SFT/DPO (redesigned data, same recipe)\n[spec not yet written]"]
        SFT_RECIPE["01-sft-dpo recipe, unchanged\n(94% functional / 100% idiom-DPO proven)"]
        DPO_PAIRS["Redesigned DPO pairs:\nsemantic-correct vs\nsubtly-wrong OSP idiom\n(both compile)"]
        SFT_RECIPE --> DPO_PAIRS
    end

    P3 -->|CPT checkpoint accepted| P4

    subgraph P5["Checkpoint 2: +CPT+SFT/DPO eval"]
        MCQ2["Semantic MCQ"]
        JUDGE2["Online-judge\n(compiler/pass@k)"]
        TRUST2["Human-sample trust check"]
        MCQ2 & JUDGE2 --> TRUST2
    end

    P4 --> P5

    subgraph P6["Phase 6: RL/GRPO (redesigned corpus+reward)\n[spec not yet written]"]
        RL_SRC["Multi-project source diversity\n(was 100% this_is_jac)"]
        RL_GRADE["Type-B AST-equivalence\n/ partial-credit grader\n(was exact-stdout all-or-nothing)"]
        RL_BAL["Idiom balance:\nwalker/graph vs plain-fn\n(was 48-57% unholdoutable)"]
        RL_GRPO["GRPO on +CPT+SFT/DPO ckpt"]
        RL_SRC & RL_GRADE & RL_BAL --> RL_GRPO
    end

    P5 -->|SFT/DPO checkpoint accepted| P6

    subgraph P7["Checkpoint 3: +CPT+SFT/DPO+GRPO eval"]
        MCQ3["Semantic MCQ"]
        JUDGE3["Online-judge\n(compiler/pass@k)"]
        TRUST3["Human-sample trust check"]
        MCQ3 & JUDGE3 --> TRUST3
    end

    P6 --> P7

    subgraph P8["Phase 8: Read the ablation"]
        DELTA["4-point delta table:\nbase / +CPT / +CPT+SFT-DPO / +CPT+SFT-DPO+GRPO"]
        HYP["Confirm/kill hypothesis:\ndid CPT move the semantic\nceiling SFT+RL both hit?"]
        DELTA --> HYP
    end

    P0 -.->|checkpoint 0 row| P8
    P3 -.->|checkpoint 1 row| P8
    P5 -.->|checkpoint 2 row| P8
    P7 -.->|checkpoint 3 row| P8

    style P0 fill:#e8eaf6,stroke:#3F51B5
    style P1 fill:#e8f4fd,stroke:#2196F3
    style P2 fill:#fff8e1,stroke:#FFC107
    style P3 fill:#e8eaf6,stroke:#3F51B5
    style P4 fill:#fff3e0,stroke:#FF9800
    style P5 fill:#e8eaf6,stroke:#3F51B5
    style P6 fill:#fce4ec,stroke:#E91E63
    style P7 fill:#e8eaf6,stroke:#3F51B5
    style P8 fill:#c8e6c9,stroke:#4CAF50
```

---

## Phase 0 — Base eval (checkpoint 0)

Run unmodified Qwen3-Coder-30B-A3B-Instruct through both eval tracks (semantic MCQ + online-judge). This is the floor every later checkpoint is read against — same discipline as the RL ladder's rung-0.

## Phase 1 — CPT data build

Full detail in [cpt-dataset-design.md](cpt-dataset-design.md). Summary:

1. **Collect 4 sources**: Jac docs (md, semantic-chunked, 3x upsample), OSP paper (tex, section-chunked), blogs (**source location still unresolved — blocking item**), code (jaseci-labs org repos, repo-packed + 50% FIM).
2. **Add CF-rehearsal slice**: general code/Python data, target 15-30% of total CPT tokens (exact ratio TBD by sweep).
3. **Decontaminate** every source against existing eval holdouts (`02-rl-grpo` RL corpus, `01-sft-dpo` eval holdout) — 14-gram MinHash, Jaccard >0.5 flagged/dropped. Most load-bearing on the code source.
4. **Pack**: concatenate documents/chunks, join with the real tokenizer EOS token, truncate final chunk per pack.
5. **Split** 85/15 train/val, stratified by `meta.source`.
6. **Emit manifest**: per-source counts, upsample weights applied, decontam drops, tokenizer/EOS version.

**Gates everything downstream** — no CPT checkpoint exists until this build is done.

## Phase 2 — CPT training

LoRA continual pretrain (next-token, raw text) on the packed CPT dataset, Q4 on 48GB M5 Pro. Track a general-coding regression eval (CF monitor) at every checkpoint alongside training loss — any drop is a stop signal, matches the CF guard in [design.md](design.md) Stage 1.

**DONE (2026-07-14) — `cpt-v1`.** 2586/2586 iters (3 epochs, 862 train windows), 6h49m, exit 0. Train loss −18.5%, val loss −24.4%, moving together (no overfit signature). Peak mem 30.4GB/48GB. Config: `03-new/cpt_train/config.yaml`. Adapter: `03-new/adapters/cpt-v1/` (gitignored). Fused: `models/qwen-cpt-v1-fused-q4`, registered in Studio chat as **Qwen · CPT-v1**. Launched + monitored live through Studio's 03 CPT → TRAIN tab (`studio/cpt.sv.jac`). Full writeup: [cpt-v1-training-results.md](cpt-v1-training-results.md). **CF regression check (2026-07-14): PASS.** 16-task general-Python coding benchmark (`03-new/cpt_train/cf_check/`), exact-output graded — `qwen-q4` 16/16 vs `qwen-cpt-v1` 16/16, zero regression. See [analysis.md](analysis.md).

## Checkpoint 1 — +CPT eval

Semantic MCQ picks the CPT checkpoint to carry forward (cheapest signal, no compiler in the loop). Human-sample trust check (Hamming similarity vs. LLM-graded result on the same subset) must pass before trusting the MCQ result at scale — per [design.md](design.md)'s eval section.

## Phase 4 — SFT/DPO (redesigned data)

*Spec not yet written — this phase is a placeholder until the SFT/DPO redesign follow-up (`design.md` sequencing item 2) lands.* Planned shape: reuse the `01-sft-dpo` LoRA recipe/hyperparameters unchanged; redesign DPO pair composition to contrast semantically-correct vs. subtly-wrong OSP idiom (both compile) instead of syntax-fix pairs. Runs on top of the accepted CPT checkpoint.

## Checkpoint 2 — +CPT+SFT/DPO eval

Full track: semantic MCQ + online-judge (compiler/pass@k, reusing the `02-rl-grpo` harness) + human trust check.

## Phase 6 — RL/GRPO (redesigned corpus+reward)

*Spec not yet written — placeholder until the RL/GRPO redesign follow-up (`design.md` sequencing item 3) lands.* Planned shape: fix the 3 diagnosed `02-rl-grpo` corpus limits — multi-project source diversity (break the 100% `this_is_jac` ceiling), Type-B AST-equivalence/partial-credit grading (replace exact-stdout all-or-nothing), deliberate walker/graph-idiom balance (break the 48-57%-in-one-file concentration). GRPO runs on the accepted +CPT+SFT/DPO checkpoint. New falsifiable hypothesis, not a re-test of the closed `02-rl-grpo` null.

## Checkpoint 3 — +CPT+SFT/DPO+GRPO eval

Same full track as checkpoint 2.

## Phase 8 — Read the ablation

Build the 4-point delta table (base / +CPT / +CPT+SFT-DPO / +CPT+SFT-DPO+GRPO) across both semantic MCQ and behavioral pass rate. Confirms or kills the core hypothesis: does CPT move the semantic ceiling that both SFT and GRPO were previously bumping into (`design.md`'s Problem section)? If checkpoint 3 shows GRPO still flat above checkpoint 2 even with CPT underneath, that's a cleaner, stronger null than the one already recorded in `02-rl-grpo/RL_FINDINGS.md`.

---

## Checklist

- [x] Blog source location resolved (jaseci-labs/jaseci-blogs)
- [x] General-code rehearsal corpus picked + license-checked (codeparrot-clean-valid)
- [x] jaseci-labs org repo inventory enumerated (17 Jac repos, code_gate in manifest.json)
- [ ] Phase 0 base eval recorded (both tracks)
- [x] Phase 1 CPT dataset built, manifest emitted, decontam clean
- [x] Phase 2 CPT training run complete (2586/2586 iters, train loss -18.5%, val loss -24.4% — see `cpt-v1-training-results.md`). **CF regression check PASS** (16/16 both models, zero delta — `analysis.md`).
- [ ] Checkpoint 1 MCQ + trust check recorded, CPT checkpoint accepted
- [ ] SFT/DPO redesign spec written (gates Phase 4)
- [ ] Phase 4 SFT/DPO run, Checkpoint 2 full eval recorded
- [ ] RL/GRPO redesign spec written (gates Phase 6)
- [ ] Phase 6 GRPO run, Checkpoint 3 full eval recorded
- [ ] Phase 8 ablation table built, hypothesis confirmed/killed
