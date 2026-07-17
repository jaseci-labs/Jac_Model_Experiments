# CPT-v2 design: docs-dominant corpus, epoch-loop training, dual-track eval vs jac-gpt

2026-07-17. Follow-up to the closed CPT-v1 null (`03-new/docs/cpt-v1-training-results.md`, `analysis.md` — Checkpoint-1 semantic MCQ NULL, base and CPT-v1 byte-identical on 20/20 questions, see memory `project-attempt03-cpt-design`). This doc supersedes an earlier same-topic draft that was lost to a drive I/O fault before it was committed — nothing downstream depended on it, no content is being silently dropped.

Status: **design only.** Nothing in this doc is implemented yet. Scope for this pass is documentation — this spec plus `workflow.md` in this folder. Implementation is gated on your review of both, per the brainstorming→writing-plans handoff.

## 1. Why CPT-v1 nulled, and what this attempt changes

CPT-v1 mixed docs (1.96M tok) with a 17-repo code corpus (992K tok) and 20% general-Python rehearsal (750K tok). Free-form generation vocabulary shifted (real Jac keywords like `has`, `spawn ->`, `here.*` appeared where base invented a fake DSL), but constrained-choice concept recognition (the 20-question MCQ) didn't move at all — same 18/20, same 2 wrong, byte-identical text. Two live, non-exclusive hypotheses:

1. **Corpus dilution** — code + rehearsal at ~45% of tokens diluted the semantic signal docs alone would have given.
2. **Instrument mismatch** — MCQ (constrained choice) may not detect the kind of learning CPT produces; open-ended generation graded against a grounded reference might show movement MCQ can't.

CPT-v2 tests both, plus three new mechanisms this round adds: **(a)** an LLM curation pass over the corpus before packing, **(b)** an epoch-by-epoch checkpoint-and-resume training loop with automated regression gating, to get past the fixed 3-epoch LR-schedule ceiling CPT-v1 hit, **(c)** a second eval track designed to answer "does this beat jac-gpt," which cosine-similarity-to-oracle cannot answer by construction (see §5).

## 2. Corpus: docs-dominant, CF-insured (unchanged decision, restated precisely)

Locked earlier: **docs + small rehearsal slice**, not docs-only, not docs+code. Drop the 17-repo code corpus entirely; keep docs, OSP paper, blogs; shrink rehearsal from CPT-v1's ~20%-of-total to **~10%-of-total**, CF-insurance only, not primary training signal.

| source | CPT-v1 tokens | CPT-v2 tokens | change |
|---|---|---|---|
| docs (jaseci_docs, 3x upsample) + jac-llmdocs | 1.96M | 1.96M | unchanged |
| osp_paper | 35K | 35K | unchanged |
| blogs | 64K | 64K | unchanged |
| code (17-repo) | 992K | **0** | dropped entirely |
| rehearsal (codeparrot-clean-valid) | 750K (~20%) | **~230K (~10%)** | shrunk |
| **total** | **3.80M** | **~2.29M** | |

### 2.1 `build_cpt.py` changes required (not yet implemented)

Current script (`03-new/cpt_build/build_cpt.py`) has no corpus-selection flag — `build_code()` always runs, and the rehearsal target is hardcoded as `jac_tokens // 4` (25% of non-rehearsal tokens, which nets to ~20% of the total once rehearsal is added — this is where CPT-v1's 750K/20% came from). Two new CLI flags:

- `--drop-code` — skip `build_code(args.repos_dir)` entirely; `sources["code"]` never gets built, manifest simply won't have a `code` key. No other pipeline stage needs to change — decontam/pack/split already iterate `sources` generically.
- `--rehearsal-frac F` (default keeps current `0.25` behavior for backward compat with any CPT-v1 rebuild) — replaces the hardcoded `jac_tokens // 4` with `int(jac_tokens * F)`. For CPT-v2, `F = 0.111` (so rehearsal ends up ~10% of `jac_tokens + rehearsal_tokens`, matching the ~230K target: `2.06M * 0.111 ≈ 229K`).

Output path changes too: CPT-v1's dataset lives at `03-new/dataset/cpt/`. CPT-v2 needs its own path (**`03-new/dataset/cpt-v2/`**) so the v1 dataset (and its `manifest.json`, still referenced by `cpt.sv.jac`'s DATA tab) isn't overwritten. `OUT` in `build_cpt.py` becomes a CLI-settable path, not a hardcoded constant.

Everything else in the build (fence-aware `md_chunks`, `split_paragraphs`, decontam via 14-gram containment ≥0.5 against RL holdouts, `pack_source`'s overlap-on-truncation packing) is reused unchanged — those bugs were already found and fixed for CPT-v1's rebuild (v2 fixes, per memory `project-attempt03-cpt-design`), no reason to touch proven logic.

**Honest gap**: 10% rehearsal isn't zero. This is "docs-dominant, CF-insured," not a clean docs-only ablation against CPT-v1's mix — if CPT-v2 moves the needle, corpus dilution *and* the curation pass (§3) are both live explanations; this design can't cleanly isolate which one mattered. A true zero-rehearsal ablation would need its own run and its own CF risk acceptance — out of scope here, noted as a possible follow-up if CPT-v2 succeeds and you want to know which lever actually did it.

## 3. Fable curation pass (new)

Between chunking and packing, a **Fable subagent** (Claude Fable 5, `Agent` tool `model: "fable"`) reads the per-source `raw.jsonl` files (post `md_chunks`/`split_paragraphs`, pre-`pack_source` — i.e. one row per paragraph-chunk, already has `meta.file`/`meta.section`) and produces a curation verdict per chunk:

- **drop** — boilerplate (license headers, empty changelog entries, nav-only fragments), near-duplicate of another chunk already seen (Fable is shown chunks in file order within a source, so true cross-file dedup needs a companion cheap pass — see below).
- **keep** — default.
- **upweight** — genuinely core-concept material (OSP fundamentals, walker/node/edge semantics, spawn/visit mechanics) beyond what the existing 3x docs-wide upsample already gives; a targeted *additional* multiplier on top.

Fable operates in batches (e.g. 50 chunks per subagent call, source-by-source — docs alone is ~8500 rows post-v2-fix, too many for one context window) and returns a structured verdict list: `{chunk_id, verdict, reason}`. **Chunk IDs** need to be stable and cheap to compute before this pass exists — add a `meta.chunk_id` (e.g. `sha1(file + section + text[:80])[:12]`) to every row at build time so Fable's verdicts can be joined back reliably regardless of row ordering.

Near-duplicate detection is **not** Fable's job — LLM judgment on "is this a duplicate of some other chunk 3000 rows ago" is unreliable and expensive at this scale. A cheap pre-pass (same 14-gram shingle machinery already in `build_cpt.py`'s `decontam()`, run chunk-against-chunk within a source instead of chunk-against-holdout) flags near-dup candidate pairs *before* Fable sees them; Fable only judges the flagged candidates (which of a near-dup pair to drop, if either) plus makes keep/upweight/drop calls on content quality for everything else.

Output: `03-new/dataset/cpt-v2/curation.json` — `{chunk_id: {verdict, reason, source: "fable"|"shingle-dedup"}}`. A small deterministic script applies it (`apply_curation.py`, not yet written): drop rows where verdict=drop, multiply `upsample_weight` for upweight rows, everything else passes through unchanged into `pack_source`. Curation is fully auditable — every drop/upweight has a logged reason, re-runnable if you disagree with a verdict (just edit `curation.json` and re-pack, no need to re-run Fable).

**Honest gap**: Fable curation adds a new, unproven variable to this run. If CPT-v2 improves over CPT-v1, part of that could be curation quality rather than corpus composition — same entanglement risk noted in §2.1's honest gap. Mitigated by logging every verdict+reason so a future ablation (curation on vs off) is cheap to run if this matters later.

## 4. Training: epoch-by-epoch checkpoint loop

### 4.1 The actual 3-epoch ceiling (root cause, confirmed by reading `config.yaml`)

CPT-v1's `03-new/cpt_train/config.yaml` sets:

```yaml
batch_size: 1
iters: 2586                      # 862 windows/epoch * 3 epochs
lr_schedule:
  name: cosine_decay
  warmup: 259
  arguments: [1.0e-5, 2586, 1.0e-6]   # decays TO 1e-6 AT step 2586
```

The cosine schedule's decay target is baked to the total iter count. At step 2586 the LR has already decayed to its floor (1e-6) — training "past" 2586 with this same config would continue at a floor LR that's essentially inert, not meaningfully continue the schedule. This is the real mechanism behind "we've only ever gone 3 epochs" — it's not a wall-clock or memory limit (CPT-v1's 6h49m run had no OOM, peaked at 30.4GB/48GB, well under the ceiling), it's that nobody has generated a schedule for more than 3 epochs' worth of steps.

### 4.2 Leg-based config generation (fix)

Structure CPT-v2 training as N **legs**, each one epoch (packed-window count depends on the v2 corpus size — recompute once the v2 build runs; CPT-v1 was 862 train windows/epoch at 3.80M tokens, v2's ~2.29M tokens will produce fewer). Each leg gets its own config file with a schedule computed for a fixed *global* total (not re-derived per leg), so the LR trajectory is one coherent cosine decay across all legs rather than N independent decay-to-floor-and-restart cycles (which would be a very different, worse training dynamic — repeatedly hitting LR floor and jumping back to peak every epoch fights the whole point of cosine decay).

Concretely: decide the **ceiling** epoch count upfront (§4.3 — 12), compute `total_iters = windows_per_epoch * 12` and the *single* cosine schedule for that total once, before leg 1 ever runs. Each leg's config (`config_v2_leg{N}.yaml`) is identical except `iters` (this leg's slice) and `--resume-adapter-file` (previous leg's checkpoint) — the LR schedule config block (`arguments: [1e-5, total_iters, 1e-6]`) stays the same across all legs, and `mlx_lm.lora` resumes the optimizer/schedule position correctly because `--resume-adapter-file` combined with running from the correct step-offset preserves where the cosine curve is (needs verification against `mlx_lm`'s actual resume semantics before implementation — flagged as an implementation-phase risk, not resolved by this doc).

This means the loop is **not** "keep bumping a config's `iters` and re-running from scratch" — it's "pre-commit to a schedule for the ceiling, run legs against slices of that one schedule, and be willing to stop early if a leg's checkpoint fails the gate" (§4.4). If it stops at leg 6 of a 12-epoch schedule, the model has only seen the first half of a schedule designed to decay over 12 epochs — LR at leg 6 will still be relatively high, not yet at floor. That's an accepted tradeoff of the stop-loss design, not a bug: a model stopped early on a 12-epoch schedule got less LR-decay "settling" than one that ran the full schedule, which is one plausible explanation if an early-stopped v2 checkpoint looks noisier than expected.

### 4.3 Stop rule: floor 6, soft target 8, hard ceiling 12

- **Floor: 6 epochs.** No stop-loss halt is honored before leg 6 completes, even if a CF-check dips — log the dip, keep training. Guards against a single noisy/transient CF-check false-halting a run that would have recovered.
- **Target: 8 epochs.** The expected landing zone if nothing forces an early or late stop.
- **Ceiling: 12 epochs.** Hard cap regardless of CF-check results — the schedule (§4.2) is generated for exactly this total, and running past it isn't meaningful (would need a new schedule generated for a new ceiling, i.e. restarting the design decision, not just "keep going").
- **Stop-loss gate (active from leg 7 onward):** after each leg's checkpoint, run the CF-check (16-task harness, `03-new/cpt_train/cf_check/`, exact-output graded, reused unchanged). Score `<16/16` past the floor halts the loop; the last leg that scored `16/16` is kept as the final checkpoint (adapter files aren't deleted — `mlx_lm.lora`'s `save_every: 100` keeps intermediate checkpoints in the adapter dir already, so "keep the last good one" just means "point the fuse step at that checkpoint's step number," not a rollback operation).

Numbers (6/8/12) are what you approved — flagging here as the authoritative source so `workflow.md`'s diagram and any future implementation doesn't drift from this doc.

### 4.4 `cpt.sv.jac` integration (implementation-phase note, not built today)

`start_cpt_training(name, config)` already resumes from the latest `*_adapters.safetensors` checkpoint under `03-new/adapters/<name>/` and takes an explicit `config` path override — this mechanism is directly reusable for legs: each leg is a separate `start_cpt_training("cpt-v2", config="config_v2_leg{N}.yaml")` call. What's missing: `CPT_TOTAL_ITERS["cpt-v2"]` is currently a single static number (`5172`, stub 6-epoch value from before this design existed) — for the leg loop, "total iters so far" needs to track the *cumulative* target through the current leg, not one fixed number for the whole run, so `_cpt_build_status`'s progress bar reads correctly leg-by-leg. This is a real code change (`cpt.sv.jac` + `CptTrain.cl.jac`'s progress display), scoped to the implementation plan, not this doc.

The existing `CPT_CONFIGS`/`CPT_TOTAL_ITERS` stub entries (`config_v2_6epoch.yaml`, `config_v3_rank32.yaml`) predate this design and don't match it (single static 6-epoch config vs. this doc's 12-epoch-schedule/6-legs-minimum loop). They should be replaced, not built-out as originally stubbed — noted so the implementation plan doesn't accidentally build the wrong thing because a stub already existed.

## 5. Monitoring: Sonnet per-leg judgment (advisory, not a gate)

After each leg's CF-check passes (or is floor-overridden), a **Sonnet subagent** reviews and writes one paragraph to a running log (`03-new/results/cpt-v2/leg_reviews.md`, append-only):

- CF-check score for this leg.
- Train/val loss delta vs. the previous leg (parsed from `train.log`, same `metrics.parse_train_log` the Studio TRAIN tab already uses).
- 3-5 sample generations from `eval_headtohead.py`'s existing prompt set, greedy decode, this leg's checkpoint vs. the previous leg's — flagging anything that reads as repetitive/degenerate (a failure mode CF-check's exact-match Python tasks wouldn't catch, since CF-check only exercises code-completion, not free-form fluency).

This is **advisory only** — the hard gate is §4.3's CF-check stop-loss. Sonnet's note doesn't halt or continue the loop; it's there for you to read after the fact, catching soft-signal problems (style collapse, mode collapse into a repeated phrase) that a binary pass/fail exact-match harness structurally can't see. If a leg's Sonnet review ever flags something CF-check missed, that's a signal the CF-check task bank needs expanding — not an automated action.

## 6. Eval: two tracks

Both tracks reuse the same ~100-question bank, generated once (§6.1), scored two different ways (§6.2, §6.3) because they answer two different questions.

### 6.1 Question generation (Fable)

Chunk the **final packed v2 docs corpus** (post-curation, so the questions are grounded in what the model actually trained on) into source paragraphs. A Fable subagent generates 1-2 open-ended semantic questions per chunk (not MCQ — free-form, testing conceptual understanding: "what happens when...", "why would you use X instead of Y", not "define X") → ~100 total after sampling down from the full per-chunk yield. Saved to `03-new/cpt_train/eval_v2/questions.json` as `{id, question, source_chunk_id, source_file}` — the `source_chunk_id` link lets Track B's judge (§6.3) pull the exact ground-truth passage per question without re-searching the corpus.

This question bank is reusable for future CPT attempts (v3, v4, ...) the same way the CPT-v1 MCQ bank already is — score every future checkpoint against the same 100 questions for a real longitudinal comparison.

### 6.2 Track A — convergence (cosine similarity vs jac-gpt oracle)

Unchanged from the earlier locked decisions: local sentence-transformer embedding (`all-mpnet-base-v2` or BGE-small, CPU/MLX, no API cost), cosine similarity between each candidate model's answer and jac-gpt's answer to the same question. Candidates: `qwen-q4` (base), `qwen-cpt-v1`, `qwen-cpt-v2` (this run's accepted checkpoint). Aggregate mean similarity + per-question win-rate vs base.

**What this answers**: did CPT-v2 move the model's free-form answers closer to jac-gpt's grounded answers than CPT-v1 did, and than base does. A real convergence signal here is evidence CPT-v2 fixed (at least in part) the instrument-mismatch hypothesis from §1 — MCQ missed movement that open-ended-vs-grounded-reference comparison catches.

**What this cannot answer**: whether CPT-v2 is *better than* jac-gpt. The metric's maximum is 1.0 = perfect match to jac-gpt's phrasing — by construction, tying jac-gpt is the ceiling, not beating it. This is why Track B exists.

### 6.3 Track B — win/loss (Sonnet blind pairwise vs source docs)

For each question, pull the ground-truth passage via `source_chunk_id` (§6.1). A Sonnet subagent is shown: the question, the ground-truth doc passage, and two anonymized answers (CPT-v2's and jac-gpt's, order randomized per question, labels are "Answer A"/"Answer B" — the judge is never told which system produced which). The judge scores each answer's correctness and completeness against the passage independently, then picks a winner or declares a tie, with a one-line justification per question. Order-randomization + blinding matters here specifically because an un-blinded judge could develop a prior that "the fluent RAG-style answer is probably jac-gpt and probably right" — blinding forces the judgment onto the actual text.

Aggregate: win/loss/tie count for CPT-v2 vs jac-gpt across the ~100 questions. **This is the metric that can show "CPT-v2 beats jac-gpt"** — it's not capped at matching a reference, because the reference for correctness is the docs themselves, not jac-gpt's paraphrase of them.

**Honest gap, stated in §1 terms**: jac-gpt is RAG — it retrieves the actual passage into its own context before answering, so on questions where the docs are unambiguous and directly on-point, jac-gpt has a structural advantage no amount of CPT (a parametric, lossy-memorization method on a rank-16 LoRA over ~2.3M tokens) is likely to fully close. Realistic expectation: CPT-v2 is more likely to win or tie on questions requiring synthesis/inference across multiple doc sections (where retrieval alone doesn't guarantee the right passage(s) got pulled) than on single-passage factual lookups. If Track B shows CPT-v2 losing badly specifically on simple factual-lookup questions, that's expected, not a sign of a broken run.

### 6.4 Acceptance bar

CPT-v2 accepted only if it beats **both** base and CPT-v1 on Track A mean cosine-to-oracle by a real margin (not noise-level) **and** wins or ties Track B against jac-gpt on at least half the ~100 questions. Same not-a-silent-null discipline as CPT-v1's gates — if either track nulls, say so plainly, don't round up.

## 7. jac-gpt oracle setup

Clone `github.com/jaseci-labs/Agentic-AI`, subdirectory `jac-gpt-fullstack`, to `03-new/cpt_train/jac_gpt_oracle/` (gitignore this path — vendored external clone, not project code, add the line when this actually gets cloned). It's self-hostable and scriptable: `jacServer` endpoints (`RagChat`, `QAChat`, `CodingChat`, `interact`, `search_docs`) can be driven programmatically, no UI needed.

**`OPENAI_API_KEY` placement**: a `.env` file at `03-new/cpt_train/jac_gpt_oracle/.env` (the clone's own root) — its `jac.toml` already lists `python-dotenv` as a dependency, so it auto-loads on `jac start main.jac`. Never export it in a shell profile, never commit it. `jac_ml_studio`'s repo-root `.gitignore` already excludes `.env` (verified, first line) — no gitignore edit needed for the key file itself, only for the clone directory as a whole once it exists on disk.

## 8. File/directory layout (planned, not yet created except this docs folder)

```
03-new/
  docs/cpt-v2/
    design.md              <- this file
    workflow.md             <- mermaid diagram + phase runbook
  dataset/cpt-v2/           <- new build output (v1's dataset/cpt/ untouched)
    docs/, osp_paper/, blogs/, rehearsal/   raw.jsonl per source (no code/)
    curation.json           <- Fable + shingle-dedup verdicts
    packed/{train,valid}.jsonl
    manifest.json
  cpt_train/
    config.yaml              <- v1's, untouched
    config_v2_leg{1..12}.yaml  <- new, generated (not hand-written)
    eval_v2/
      questions.json          <- Fable-generated ~100 Q bank
      track_a_results.json
      track_b_results.json
    jac_gpt_oracle/            <- gitignored clone of Agentic-AI/jac-gpt-fullstack
      .env                     <- OPENAI_API_KEY, gitignored dir
  adapters/cpt-v2/            <- LoRA checkpoints, one leg's worth of *_adapters.safetensors accumulating
  results/cpt-v2/
    leg_reviews.md             <- Sonnet per-leg advisory log
    cf_check/, track_a/, track_b/   <- eval outputs, mirrors cpt-v1/ layout
```

## 9. Prereqs before any implementation runs

- `pip install sentence-transformers` (not yet in the project).
- `OPENAI_API_KEY` — you supply/export when we reach the oracle-query step, scoped to the clone's `.env` only.
- Clone + boot `Agentic-AI/jac-gpt-fullstack` locally.
- Studio dev server stopped before any dual/triple-model-load script (base + cpt-v1 + cpt-v2 sequential loads, same 48GB-combined risk as CPT-v1's gate runs).
- `mlx_lm.lora`'s actual resume-with-preserved-schedule-position behavior needs verification (§4.2) before the leg-config generator is built — if resume doesn't preserve schedule position the way assumed, the leg design needs rework before implementation, not after.

## 10. Next

Per the brainstorming→writing-plans handoff: this doc + `workflow.md` go to you for review. Once approved, `writing-plans` produces the implementation plan, in this order (matches the dependency chain above): (1) `build_cpt.py` flags + curation-apply script, (2) v2 corpus build run, (3) Fable curation pass, (4) leg-config generator + `mlx_lm.lora` resume-schedule verification, (5) `cpt.sv.jac` multi-leg support, (6) epoch-loop training run with per-leg CF-check + Sonnet review, (7) jac-gpt oracle clone/boot, (8) Fable question-gen, (9) Track A cosine-sim script, (10) Track B Sonnet-judge script, (11) acceptance readout.
