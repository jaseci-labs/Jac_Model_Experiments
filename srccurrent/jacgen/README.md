# jacgen — Jac-native conversion data generation

Synthetic **Python → Jac conversion** training data, with every example gated by
the Jac compiler (`jac run` exit 0) **and** a behavioral check (program output
matches expected across multiple distinct test cases). Generation is agentic
(Claude + jac-mcp) for the idiomatic core and automated (transpile + behavioral
gate, MultiPL-T style) for volume. All tooling is written in Jac.

## Modules

| File | Role |
|---|---|
| `writer.jac` | Build SFT examples, append JSONL, `run_jac`, `revalidate_example`. |
| `dedup.jac` | ROUGE-L (LCS) near-duplicate guard on the Jac body. |
| `corpus.jac` | Shared lib: paged corpus fetch (+retry), filter, exec-based test-case extraction (SIGALRM-timeout protected), `jac py2jac` transpile. |
| `mine.jac` | Write an inspectable pool of mined runnable functions to `dataset/source_pool/mined.jsonl`. |
| `seed_conversion.jac` | The idiomatic core: hand/mined-crafted conversion seeds (24 generated + 8 mined) + DPO pairs → `dataset/conversion/sft.jsonl`. |
| `scale_conversion.jac` | Volume tier: mine an online corpus, transpile to Jac, keep only behaviorally-correct → `dataset/conversion/sft_auto.jsonl`. |
| `dataset_stats.jac` | Report dataset composition by difficulty / generator / source. |
| `verify_dataset.jac` | Independent integrity audit: re-run every stored example through `jac run`. |

## Data tiers

- **Idiomatic core** (`sft.jsonl`, `generator: claude-code-mcp`) — true graph-spatial
  Jac (walkers, nodes, edges, obj) where the problem calls for it. The quality bar.
- **Volume tier** (`sft_auto.jsonl`, `generator: jac-py2jac`) — mechanical transpile
  of real corpus functions, compiler + behaviorally validated. Python-shaped Jac;
  breadth and real-world task diversity, not idiom.
- **DPO** (`dpo.jsonl`) — idiomatic (chosen) vs Python-shaped Jac (rejected).

## Source corpus

`Vezora/Tested-22k-Python-Alpaca` via the HuggingFace datasets-server rows API.
CodeSearchNet was rejected — repo-coupled, ~4/600 runnable standalone; Vezora is
~28% runnable. Functions are kept only if self-contained, single-argument,
stdlib-only, and return a value across distinct test inputs.

## Run

```bash
# idiomatic core (+ re-validate)
jac run srccurrent/jacgen/seed_conversion.jac
# mine a pool to inspect
jac run srccurrent/jacgen/mine.jac
# automated volume tier (to TARGET in scale_conversion.jac)
jac run srccurrent/jacgen/scale_conversion.jac
# report + audit
jac run srccurrent/jacgen/dataset_stats.jac
jac run srccurrent/jacgen/verify_dataset.jac
```

Generated data lands in gitignored `dataset/`.
