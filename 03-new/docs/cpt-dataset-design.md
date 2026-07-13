# CPT Dataset Design

*Follow-up spec #1 of `03-new/docs/design.md` (gates everything downstream — no CPT checkpoint until this data exists). Extends `03-new/rui.md`, the source guide this doc is built from.*

## Scope

Dataset preparation for **continual pre-training** (CPT) of Qwen3-Coder-30B-A3B-Instruct on Jac-language material. Raw-text, next-token-prediction format — not instruction/chat format (that's the SFT/DPO stage). Four sources: Jac docs, the OSP paper, blogs, code. Plus a general-code rehearsal slice for catastrophic-forgetting (CF) protection (`03-new/docs/design.md` Stage 1).

## Format

Every source normalizes to the same packed-document JSONL, per `rui.md`:

```json
{"text": "...", "meta": {"source": "jaseci_docs", "type": "official_doc", "upsample_weight": 3}}
```

Packing rule: concatenate whole documents (or semantic chunks within a document) up to the model's max sequence length, joined by the tokenizer's EOS token, truncating the last document in the pack if it overruns:

```
[doc1 tokens] <EOS> [doc2 tokens] <EOS> [doc3 tokens...] <EOS> [doc4 tokens (truncated)]
```

**EOS token**: pull the actual `eos_token` from the Qwen3-Coder-30B-A3B tokenizer config at build time (`tokenizer.eos_token` via `transformers.AutoTokenizer`) — don't hardcode a literal, Qwen tokenizer special tokens have changed across releases.

`meta.source` values used below: `jaseci_docs`, `osp_paper`, `blog`, `code`, `rehearsal`. `meta.type` further discriminates within a source (e.g. `official_doc`, `paper_section`, `blog_post`, `repo_file`). `meta.upsample_weight` controls how many times a document's chunks are duplicated into the packing stream before shuffling (integer ≥1).

## Source 1 — Jac lang docs

- **Location**: `.md` files under [`jaseci-labs/jaseci/docs/docs`](https://github.com/jaseci-labs/jaseci/tree/main/docs/docs).
- **Chunking**: semantic only — split by header tags (`#`, `##`, `###`, ...). Keep markdown annotations (code fences, links, bold) in the chunk text; don't strip formatting.
- **Upsample**: 3x (highest-quality, most curated source — reinforced most).
- **`meta`**: `{"source": "jaseci_docs", "type": "official_doc", "upsample_weight": 3, "file": "<repo-relative path>", "section": "<header text>"}`

**Workflow** (per `rui.md`, unchanged):
1. Read each `.md` file.
2. Split into chunks at header boundaries → conceptually `List[List[str]]` (outer = documents/files, inner = chunks within that file).
3. When packing: concatenate a file's chunks, append EOS, continue into the next file's chunks; truncate the pack's final chunk if it exceeds max sequence length.
4. Emit one JSONL line per pack (see Format above).

## Source 2 — OSP paper

- **Location**: [OSP paper LaTeX source](https://arxiv.org/src/2503.15812) (arXiv source tarball).
- **Chunking**: same semantic approach as docs, but at `\section`/`\subsection` boundaries instead of markdown headers.
- **Cleaning**: strip LaTeX noise that doesn't carry prose signal — `\cite{...}`, `\ref{...}`, `\label{...}`, raw `\begin{figure}...\end{figure}` blocks (keep captions), bibliography/appendix boilerplate. Keep equations and section prose.
- **Upsample**: 1x default (single paper, one document — high upsample would overweight one narrow voice/style vs. the docs corpus). Revisit if CPT eval shows OSP-specific concepts under-learned.
- **`meta`**: `{"source": "osp_paper", "type": "paper_section", "upsample_weight": 1, "section": "<\\section text>"}`

## Source 3 — Blogs

- **Location**: [blogs.jaseci.org/blog](https://blogs.jaseci.org/blog) — **source repo/CMS not yet identified** (`rui.md`: "I haven't found the source of these blog yet"). Before this source can build, someone must find whether the blog is static-site-generated from a git repo (preferred: pull markdown source directly) or is only reachable as rendered HTML (fallback: scrape + convert).
- **Preferred**: markdown source, same chunking/format as docs.
- **Fallback**: HTML → markdown conversion (e.g. `html2text` or `trafilatura`) before chunking; strip nav/footer/sidebar boilerplate that isn't post content.
- **Upsample**: 1x default (unvetted quality vs. official docs — don't overweight until reviewed).
- **`meta`**: `{"source": "blog", "type": "blog_post", "upsample_weight": 1, "url": "<post URL>", "title": "<post title>"}`
- **Blocking open item**: locate the blog's source repo (check `jaseci-labs` org for a `blogs`/`website` repo) or confirm scrape-from-HTML is the only path.

## Source 4 — Code (jaseci-labs org repos)

Per user decision: pull all public Jac source from the `jaseci-labs` GitHub org (jaseci core, jac-lang, example/template repos) — largest real-world idiomatic Jac corpus available, same org as the docs.

- **Collection**: enumerate public repos under `github.com/jaseci-labs`, clone, collect every `.jac` file (and `.impl.jac`/`.test.jac` companions). Exclude generated/vendored code, `node_modules`, build artifacts.
- **Repo-level packing** (per `wholestack/strat.md`'s established convention — reuse, don't reinvent): build a per-repo file-dependency graph from `import`/`include` statements, topologically sort (tolerate cycles), prepend a file-path comment (`# path: <repo>/<relative path>`) to each file, concatenate in dependency order into one packed document per repo (or per logical subproject if a repo is large/monorepo-shaped).
- **Fill-in-the-Middle (FIM)**: apply at the packed-document level — split into prefix/middle/suffix, reorder PSM (`<|fim_start|>prefix<|fim_hole|>suffix<|fim_end|>middle<|eos|>`), at a 50% FIM rate on this source only (DeepSeek-Coder's ablation, already the convention in `wholestack/strat.md`). The other 50% stays left-to-right.
- **Decontamination**: any file (or near-duplicate, MinHash Jaccard >0.85) already present in `01-sft-dpo/dataset/` or `02-rl-grpo/dataset/` training data is fine to keep (CPT and SFT see the same real code — that's expected overlap, not leakage). What must be excluded is anything overlapping the **eval holdout** (see Decontamination below) — check jaseci-labs repos for accidental inclusion of holdout task source files.
- **Upsample**: 1x (code volume from a whole org is already substantial; no need to reinforce further given the docs/paper are the primary semantic-teaching sources).
- **`meta`**: `{"source": "code", "type": "repo_file" | "repo_pack", "upsample_weight": 1, "repo": "<org/repo>", "path": "<relative path>", "fim": true|false}`

## CF-rehearsal slice (general code)

Per `design.md` Stage 1: mix in general code/Python data alongside the Jac corpus to guard against catastrophic forgetting of general coding ability.

- **Source**: a public general-code pretraining slice (e.g. a filtered sample of a permissively-licensed corpus like The Stack or StarCoder's training mix) — needs a concrete pick before build; not resolved by this doc.
- **Ratio**: start at the `wholestack/strat.md` Phase 5 composition-ratio convention as a reference point (code ~70/NL-about-code ~10/etc.) but scoped down to CPT's two-way split — target **general-code rehearsal at 15-30% of total CPT tokens**, Jac material (docs+paper+blogs+code) at 70-85%. Exact ratio is a CPT-run hyperparameter to sweep, not fixed here; track via the CF eval (general-coding benchmark) at each CPT checkpoint and adjust if regression appears.
- **`meta`**: `{"source": "rehearsal", "type": "general_code", "upsample_weight": 1}`

## Decontamination

Reuse the existing convention (`wholestack/strat.md` Phase 1/4, `02-rl-grpo`'s file-disjoint holdout discipline): before packing, run every chunk/document from every source against the existing eval holdouts (`02-rl-grpo`'s RL task corpus, any `01-sft-dpo` eval holdout) with 14-gram MinHash shingles. Flag Jaccard >0.5 overlap for manual review; drop confirmed matches. This matters most for the **code** source (org repos could contain the exact files RL/SFT holdout tasks were mined from) and least for docs/paper/blogs (prose, unlikely to shingle-match code eval tasks, but check anyway — cheap).

## Dataset assembly

- **Layout**: `03-new/dataset/cpt/{docs,osp_paper,blogs,code,rehearsal}/` for per-source intermediate JSONL, packed into `03-new/dataset/cpt/packed/{train,val}.jsonl` for the final CPT input. Matches the `dataset/` convention already used in `01-sft-dpo/` and `02-rl-grpo/`.
- **Split**: train/val only (no test split needed here — CPT isn't graded on held-out packed-doc loss as a primary metric; the semantic MCQ + behavioral judge are the real measurement instruments, per `design.md`). 85/15 train/val, stratified by `meta.source` so val isn't dominated by one source.
- **Manifest**: each build emits a manifest (`03-new/dataset/cpt/manifest.json`) recording per-source document/chunk/token counts, upsample weights applied, decontamination drops, and the tokenizer/EOS-token version used — needed to reproduce or audit a CPT run later.

## Open items (not resolved by this doc)

1. ~~**Blog source location**~~ RESOLVED: `jaseci-labs/jaseci-blogs` repo, markdown under `docs/blog/posts/`.
2. ~~**General-code rehearsal corpus pick**~~ RESOLVED: `codeparrot/codeparrot-clean-valid` (public; the design's `the-stack-smol` pick turned out HF-gated).
3. **Rehearsal mix ratio** — built at ~20%; 15-30% sweep still open.
4. **OSP paper LaTeX cleaning fidelity** — figure/table caption handling not fully specified; revisit once the actual `.tex` source is pulled and inspected.
5. ~~**jaseci-labs org repo inventory**~~ RESOLVED: 17 Jac-bearing repos mined (see manifest).

## Build notes (dataset built 2026-07-13, commit 177a978)

Built to `03-new/dataset/cpt/` by `03-new/cpt_build/build_cpt.py`. **3.84M tokens, 939 windows (790 train / 149 val).** Deviations from spec, all forced by ground truth:

- **Gate is parse-only** (`jac check -p`), not full type-check: the v0.16.x checker false-positives on real working client-style code (E1030/E1032 on JS interop, standalone `.impl.jac`), failing ~100% of real repos. Parse-only keeps broken/outdated syntax out — the actual goal. `inr-codelabs` (0/4, jac-v1 syntax) correctly excluded by it.
- **`jac/tests/**` + `jac/jaclang/**` excluded**: compiler test fixtures — standalone-unresolvable imports and intentionally-broken snippets, not idiomatic teaching material. `jac/examples/**` + `docs/**.jac` kept.
- **FIM skipped**: Qwen3-Coder-30B-A3B-Instruct tokenizer ships no FIM special tokens; downstream hole-fill is chat-format anyway.
- **Decontam uses containment** (≥50% of a holdout item's shingles present in the doc), not symmetric Jaccard — corpus docs are much larger than holdout items, so Jaccard would under-trigger. Caught 7 real leaks: holdout mirrors inside `jac` repo examples (raylib_shooter, littleX frontend) + `this_is_jac/main.jac` shooter dupe.
- **`docs` source includes latest `jaseci-llmdocs` release** (1x weight) alongside official docs (3x).
