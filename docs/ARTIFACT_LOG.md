# Artifact Log — 2026-07-09 cleanup

Record of every model/adapter that existed before the big cleanup (1.32TB → ~125GB).
Hyperparameters for every adapter: `docs/training_configs/<name>.json` (all 41 preserved, including deleted runs).
Eval numbers: `RL_FINDINGS.md`, `README.md`, `resultspub/rl/corrected_summary.json`, raw logs in `results/`.

## How to recreate anything deleted

- **Base models**: re-download + quantize via mlx (HF repos below).
- **Fused models**: `mlx_lm.fuse --model <base> --adapter-path <adapter>` — base path + dataset recorded in each `training_configs/<name>.json`. All rft LoRA adapters were KEPT, so every ladder rung is re-fusable.
- **GRPO/loser adapters**: weights deleted permanently (deliberate; null result / bakeoff losers). Recreate = retrain with archived config + `rl/run_grpo.sh` / `sft_dpo/run_probe.sh` etc.

## KEPT (~125GB)

| Path | Size | Why |
|---|---|---|
| models/qwen-q4 | 16G | Base Qwen/Qwen3-Coder-30B-A3B-Instruct — re-fusing source for fresh-arm adapters |
| models/jac-qwen3coder-q4 | 16G | SFT+DPO Jac base ("capable" RL starting policy); base for jac-arm adapters; NOT exactly recreatable |
| models/jac-qwen3coder-r20-rft-q4 | 16G | **Best model**: 61.1% greedy pass@1 (corrected ladder peak) |
| models/jac-qwen3coder-rall-rft-q4 | 16G | rall rung; best pass@k on some holdouts |
| adapters/jac-qwen3coder-{r1,r3,r5,r10,r20,rall}-rft | 3.1G ea | Jac-arm SFT ladder LoRAs |
| adapters/qwen3coder-{r1,r3,r5,r10,r20,rall}-rft | 3.1G ea | Fresh-arm control ladder LoRAs |
| adapters/qwen3coder-conv-rft, jac-convfull-rft, jac-sg-r20-rft, jac-big-rft | 3.1G ea | Conversion / social-graph / big-holdout SFT LoRAs |
| adapters/qwen-probe, qwen-dry, qwen-dpo | 7.3G/1.0G/1.1G | Incumbent's SFT/DPO record |

## DELETED — models/ (42 dirs, ~640GB)

Base HF models (re-downloadable):

| Dir(s) | Size | HF repo |
|---|---|---|
| qwen-q8 | 30G | Qwen/Qwen3-Coder-30B-A3B-Instruct (q4 kept) |
| qwen3i-q8 / qwen3i-q4 | 30G/16G | Qwen/Qwen3-30B-A3B-Instruct-2507 |
| gemma-q8 / gemma-q4 | 25G/13G | google gemma (see training_configs/gemma-*.json for exact path) |
| ling-q8 / ling-q4 | 17G/8.8G | inclusionAI/Ling-Coder-lite |
| dscoder-q8 / dscoder-q4 | 16G/8.2G | deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct |
| qwen25c-q8 / qwen25c-q4 | 15G/7.7G | Qwen/Qwen2.5-Coder-14B-Instruct |
| gptoss-q4 | 10G | openai/gpt-oss-20b (MXFP4-broken for finetuning, see bakeoff notes) |

Fused SFT/DPO bakeoff models (losers; eval tables in README.md + resultspub/initmodelchoice/):

| Dir(s) | Size |
|---|---|
| qwen3i-jac-fused-q8/q4 | 30G/16G |
| gemma-jac-fused-q8/q4, gemma-jac-dpo-fused-q8 | 25G/13G/25G |
| ling-jac-fused-q8 | 17G |
| dscoder-jac-fused-q8/q4, dscoder-jac-dpo-fused-q8 | 16G/8.2G/16G |
| qwen25c-jac-fused-q8/q4, qwen25c-jac-dpo-fused-q8 | 15G/7.7G/15G |

Winner's superseded fused (superseded by kept jac-qwen3coder-q4):

| Dir | Size |
|---|---|
| qwen-jac-fused-q8/q4 | 30G/16G |
| qwen-jac-dpo-fused-q8 | 30G |

Ladder/experiment fused (all re-fusable from kept adapters):

| Dir(s) | Size | Corrected-ladder greedy |
|---|---|---|
| jac-qwen3coder-{r1,r3,r5,r10}-rft-q4 | 16G ea | see corrected_ladder.jsonl; r5 = 83.3% pass@8 |
| qwen3coder-{r1,r3,r5,r10,r20,rall}-rft-q4 | 16G ea | fresh-arm control |
| qwen3coder-conv-rft-q4, jac-convfull-rft-q4, jac-sg-r20-rft-q4, jac-big-rft-q4 | 16G ea | conversion/sg/big experiments |

## DELETED — adapters/ (22 dirs, ~540GB)

GRPO (null result — GRPO added 0 over SFT; raw-GRPO == base 38.9%; full analysis in RL_FINDINGS.md):

| Dir | Size |
|---|---|
| jac-qwen3coder-rall-{raw,sg,sgt}-grpo | 59G/59G/60G |
| qwen3coder-rall-{raw,sg,sgt}-grpo | 58G/58G/60G |

(bulk of each = embedded fused model; real LoRA was ~282MB. Weights deleted permanently by explicit decision.)

Bakeoff loser adapters:

| Dir(s) | Size |
|---|---|
| gemma-dpo / gemma-probe / gemma-dry | 48G/9.3G/1.3G |
| dscoder-dpo / dscoder-probe / dscoder-dry | 30G/4.6G/678M |
| qwen25c-dpo / qwen25c-probe / qwen25c-dry | 28G/613M/88M |
| qwen3i-dpo / qwen3i-probe / qwen3i-dry | 1.3G/7.3G/1.0G |
| ling-probe / ling-dry | 4.6G/673M |
| gptoss-probe / gptoss-dry | 3.8G/563M |
