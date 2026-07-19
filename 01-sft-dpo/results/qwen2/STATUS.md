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
- Waves 1-3 (b01-b11) complete: 154 new examples, sft.jsonl 301. Found+fixed real toolchain bug: writer.jac run_jac shared a basename-keyed .jac/data/snippet.db across ALL validation calls (jac persists root state to disk); root-anchored examples leaked state cross-call. Fixed via cwd=tempdir isolation (commit 297c7ba). b01-b10 unaffected (never touch root). b11 re-verified clean after fix (14/14).
- Wave 4 (b12-15: littlex/guestbook-inspired graph_osp + lib_*.py function-tier mining) launched.
- Wave 4 (b12-15) ingested: +51/56 accepted, sft.jsonl 352. Final wave 5 (b16-20: RL-task-pool domains + non-graph obj/typed-collections/error-handling depth) launched -- last SFT batch wave before holdout growth + DPO.
- PHASE 2 COMPLETE: SFT generation done, sft.jsonl 147->420 (+273). Found+fixed 2nd toolchain bug: ingest_batch.jac's python cross-check parsed first "def" in source, false-rejecting multi-helper examples (b20: 6/14 -> 13/14 after fix, commit d1b0c07). Generator breakdown: 116 hand-authored, 31 graph seeds, 273 fable-agent-v1.
- Phase 3 (holdout growth) launched: h01/h02, 20 new graph eval tasks targeting novel aggregation shapes (diameter/topo/bipartite/cycle-length + coloring/component-histogram/path-count/centroid), decontaminated against full 420-example train set.
