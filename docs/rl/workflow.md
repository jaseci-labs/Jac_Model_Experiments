# Reinforcement Learning Workflow

```mermaid
---
config:
  flowchart:
    nodeSpacing: 55
    rankSpacing: 75
    padding: 25
  themeVariables:
    fontSize: 18px
---
flowchart TD
    subgraph Corpus["this_is_jac corpus"]
        LIB["Pure-lib units\nsource_lexer / source_index\n(function-shaped)"]
        GRAPH["Graph-spatial units\nsocial_graph / guestbook\n(walkers / nodes / edges)"]
    end

    subgraph Authoring["Task authoring (manual)"]
        DRV["rl/drivers/*.jac\ndeterministic jac-run-able file\nbody wrapped in HOLE sentinels"]
        BT["build_tasks.jac\nrun as-authored -> capture stdout\nisolated cwd (no .jac persistence)"]
        TASKS["dataset/rl/tasks.jsonl\n{prompt, answer}\n+ templates/<id>.jac"]
        SPLIT["build_rl_splits.jac\ncurriculum easy->hard\n+ decontam"]
        SETS["train / valid / holdout .jsonl"]
    end

    LIB --> DRV
    GRAPH --> DRV
    DRV --> BT --> TASKS --> SPLIT --> SETS

    subgraph Models["RL bases (3)"]
        M1["qwen3coder\nQwen3-Coder-30B-A3B (fresh)"]
        M2["jac-qwen3coder\nSFT+DPO best"]
        M3["qwen36\nQwen3.6-27B dense (fresh)"]
    end

    WARM{"jac-trained\nalready?"}
    M1 --> WARM
    M2 --> WARM
    M3 --> WARM
    WARM -->|"no (fresh)"| RFT["Warm-start\nRFT / SFT\n(beat cold-start sparse reward)"]
    WARM -->|"yes"| DIRECT["GRPO direct"]
    RFT --> DIRECT

    subgraph GRPO["GRPO loop  (mlx-lm-lora, local MLX)"]
        POL["Policy (LoRA)\nsample group_size completions"]
        SPL["splice completion into template\n(__HOLE__)"]
        RUN["jac run  (isolated cwd)"]
        REW["reward jac_behavioral\n0.3 compiles + 0.3 runs\n+ 0.3 output_match + 0.1 idiom"]
        ADV["group-relative advantage"]
        UPD["LoRA update"]
        REF["frozen base = reference\n(one weight set, fits 48GB)"]
        POL --> SPL --> RUN --> REW --> ADV --> UPD --> POL
        REF -.KL.-> ADV
    end

    SETS --> POL
    DIRECT --> POL

    UPD --> ADAPT["adapters/<name>-grpo"]

    subgraph Eval["Evaluation"]
        EV["eval_rl.jac on holdout\nload once -> gen -> splice -> jac run"]
        BASE["base score"]
        POST["+grpo score"]
        RES["results/RL_RESULTS.md\nrun% / behavior% / idiom\nbase vs RL, per model"]
        EV --> BASE
        EV --> POST
        BASE --> RES
        POST --> RES
    end

    ADAPT --> EV
    SETS -. holdout .-> EV

    classDef done fill:#dff0d8,stroke:#3c763d,color:#1b3d1b;
    classDef todo fill:#fcf2cc,stroke:#8a6d3b,color:#3d3411;
    class DRV,RFT todo;
    class BT,TASKS,SPLIT,SETS,POL,SPL,RUN,REW,ADV,UPD,REF,ADAPT,EV,BASE,POST,RES done;
```

**Legend.** Green = scaffolding built + validated (branch `rl-phase`). Amber = remaining owner work: author ≥30 drivers (`DRV`) and warm-start the two fresh bases (`RFT`). The GRPO loop itself is proven end-to-end on a real 30B (2-iter smoke: reward loaded, rollouts scored, adapter produced). See [`strat.md`](strat.md) for the full rationale.
