# qwen2 overnight rebuild — STATUS

Plan: ~/.claude/plans/rebuild-lost-sft-dpo-model-unified-grove.md
Append-only progress log. Read this first after any context loss.

- 2026-07-19 04:10 — Phase 0 start. Disk 3396GB free. check.sh green (22 modules, 39/39 revalidated). Backups → dataset/backups/pre_expansion_20260719/. seed_conversion.jac renamed .DANGEROUS. caffeinate pid 55878, heavy lock claimed (old lock stale builder:dataset_stats pid 0).
- 2026-07-19 04:2x — Phase 0 done: graph seeds restored (sft.jsonl 147), checkpoint commit af43e32. Dead symlinks removed. Phase A: qwen-q8 quantized OK; SFT+DPO fuses running in background.
- Phase 1 done: ingest_batch.jac / dpo_authored.jac / ingest_holdout.jac written, parse-checked, smoke-tested (accept, reject-dup, reject-invented-output, receipt-refuse all verified). Staging dirs created.
- 2026-07-19 ~04:45 — Phase A COMPLETE: qwen-q8 (30G) rebuilt, qwen-jac-fused-q8 + qwen-jac-dpo-fused-q8 re-fused from surviving adapters, both smoke-verified (valid Jac output, 67-69 tok/s, peak 32.5GB). qwen2-q4/q8 APFS clones seeded. OG MODEL RECOVERED.
- Phase 2 in flight: wave 1 (b01-b04 walker themes) Fable subagents authoring.
- Wave 1 (b01-b04) DONE: 56/56 accepted, sft.jsonl 203. Dedup gate fixed category-aware (graph 0.95). Wave 2 (b05-b08: typed edges, edge attrs, hetero nodes, ability composition) launched.
- b05+b06 ingested (28/28), sft.jsonl 231 (committed via external "changes pushed" commit). b07/b08 killed by session API limit (resets 7am Detroit) — respawn scheduled. Remaining: waves 2.5-5 (b07-b20), holdout, DPO gen, splits, training.
