### Function holdout (150 tasks)
| model | base | SFT | DPO | SFT sim | DPO sim | idiom gain |
|---|---|---|---|---|---|---|
| Qwen3-Coder-30B-A3B (incumbent) | 0% | 94% | 93% | 0.968 | 0.959 | +0.008 |
| gpt-oss-20b | 0% | 92% | — | 0.973 | — | — |
| Qwen3-30B-A3B-Instruct | 0% | 95% | 94% | 0.964 | 0.958 | +0.006 |
| DeepSeek-Coder-V2-Lite | 2% | 94% | 94% | 0.968 | 0.967 | +0.001 |
| Ling-Coder-lite | 0% | — | — | — | — | — |
| Qwen2.5-Coder-14B | 0% | 94% | 93% | 0.964 | 0.964 | +0.000 |
| Gemma-3-26B-A4B (prior) | 0% | 93% | 93% | — | 0.957 | — |

### Graph holdout (13 tasks — node/edge/walker idiom)
| model | base | SFT | DPO | SFT sim | DPO sim | idiom gain |
|---|---|---|---|---|---|---|
| Qwen3-Coder-30B-A3B (incumbent) | 0% | 46% | 61% | 0.457 | 0.338 | +0.119 |
| gpt-oss-20b | 0% | 61% | — | 0.210 | — | — |
| Qwen3-30B-A3B-Instruct | 0% | 53% | 53% | 0.558 | 0.223 | +0.335 |
| DeepSeek-Coder-V2-Lite | 0% | 15% | 23% | 0.707 | 0.546 | +0.160 |
| Ling-Coder-lite | — | — | — | — | — | — |
| Qwen2.5-Coder-14B | 0% | 38% | 23% | 0.444 | 0.232 | +0.212 |
| Gemma-3-26B-A4B (prior) | 0% | 15% | 15% | 0.667 | 0.667 | +0.000 |
