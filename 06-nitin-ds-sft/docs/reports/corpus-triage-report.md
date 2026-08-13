# 06-nitin-ds-sft — Corpus Triage Report

**Date:** 2026-08-10
**Scope:** `CONTEXT_BRIEF.md` §1 (validate/compile), §4(b) (carve the new frozen holdout), §6 (dedup + decontaminate).
**Source:** `https://github.com/chess10kp/jac-data-gen`, commit **`7c25aff3110f526eec59e0123ffe6c0c152cce91`** (single commit, message `data`, dated 2026-08-10), path `data/jac_outputs/*.jac`, 7,627 files, 30 MB.
**Result:** **6,329** clean rows survive the full funnel → **855** frozen Nitin holdout + **5,474** training candidate pool. Compile pass rate **92.5%**. Real intra-corpus duplication is **0.22%** — far lower than the brief anticipated.

Artifacts written:

| Path | Rows |
|---|---|
| `model-experiments/06-nitin-ds-sft/dataset/candidate_pool.jsonl` | 5,474 |
| `model-experiments/06-nitin-ds-sft/dataset/nitin_holdout.jsonl` | 855 |
| `model-experiments/06-nitin-ds-sft/docs/reports/corpus-triage-stats.json` | machine-readable drop lists (every dropped filename, with reason) |
| `model-experiments/06-nitin-ds-sft/scripts/compile_check.py` | stage S1 runner |
| `model-experiments/06-nitin-ds-sft/scripts/pipeline.py` | stages S2–S6, reproducible end to end |

---

## 1. The funnel (no silent caps — every drop is enumerated in `corpus-triage-stats.json`)

| Stage | Δ | Survivors | % of S0 |
|---|---:|---:|---:|
| **S0** corpus at `jac-data-gen@7c25aff3` | +7,627 | 7,627 | 100.0% |
| **S1** compile FAIL (`jac run` exit ≠ 0) | −575 | 7,052 | 92.5% |
| **S2a** no docstring | −275 | | |
| **S2b** trivial body (<12 body tokens or <3 code lines) | −419 | | |
| **S2c** no parseable top-level `def` | −5 | | |
| **S2d** >1 top-level `def` | −0 | | |
| **S2** quality-filter survivors | | 6,353 | 83.3% |
| **S3** exact duplicate (normalized token SHA-1) | −1 | 6,352 | 83.3% |
| **S4** near duplicate (n=14 shingle, overlap ≥ 0.50) | −13 | 6,339 | 83.1% |
| **S5** decontamination vs 4 holdout files | −1 | 6,338 | 83.1% |
| **S5b** func-name collision w/ holdout, ident-Jaccard ≥ 0.50 | −9 | **6,329** | **83.0%** |
| **S6** frozen Nitin holdout carved (`random.seed(42)`) | 855 | | |
| **S6** training candidate pool | 5,474 | | |

Total dropped: 1,298 of 7,627 (17.0%). 575 of those (44%) are the compile gate; 699 (54%) are the quality filter; only 24 (1.8%) are dedup + decontam.

---

## 2. S1 — compile validation (7,627 files, all of them)

**Method.** `jac run` exit code, per this project's non-negotiable convention that generated/tooling code is *never* gated on `jac check` (`01-sft-dpo/sft_dpo/jacgen2/dedup_v2.jac` module docstring; `datagen/spec.md` §7). Each file was copied into its own tempdir and run with `cwd` set to that tempdir — the exact isolation pattern in `jacgen2/gate.jac`'s `run_jac_local`, so jac's persistent root store can't leak state between files and the source corpus stays pristine. 12-way thread parallelism, 60 s timeout per file (nothing timed out).

**Safe to run, not just compile:** grep confirms **zero** `with entry` blocks anywhere in the 7,627 files, so `jac run` is a pure parse+load+typecheck of a single `def`; nothing executes. This is why a *run*-based gate is safe at this scale.

**Result: 7,052 pass (92.46%) / 575 fail (7.54%).**

### Failure breakdown — one bug accounts for 98.4% of it

| Count | Error |
|---:|---|
| 566 | `error[E0002]: Missing ';'` |
| 6 | `name '__jac_lambda_1' is not defined` (runtime, at load) |
| 1 | `No module named '_pytest'` (`import from _pytest.config …`) |
| 1 | `error[E0013]: 'to' is a keyword and cannot be used as a parameter name` |
| 1 | `error[E0001]: Expected 'to', got 'while'` |

**The 566 are a single systematic transpiler defect in the upstream generator, not 566 independent problems.** Every one is a Python *implicit-return expression* emitted as a bare expression statement with neither `return` nor `;`:

```jac
def CRRAutilityPP(c: float, gam: float) -> float {
    -gam * (c ** (-gam - 1.0))      # <- no `return`, no `;`
}
```

This is trivially machine-fixable (prefix `return `, suffix `;`) and would recover ~566 files, lifting the compile pass rate from 92.5% to ~99.9%. **I did not fix them** — repairing source files silently changes what "Nitin's dataset" is, which is exactly the thing research question 2 is measuring. Flagging it as the single highest-leverage upstream fix available, and as a recommendation for `chess10kp/jac-data-gen` rather than something to patch downstream.

The 6 `__jac_lambda_1 is not defined` failures are a *jaclang 0.16.1* bug (lambda inside a comprehension/default), not a corpus defect. Noted, not worked around.

### Toolchain parity — checked, not assumed

The project's own `jac` (`/Volumes/…/.venv/bin/jac`, hardcoded as `JAC_BIN` in `gate.jac`) lives on the external drive. A **single** invocation of it exceeded 120 s — consistent with the brief's ~4.3 MB/s measurement — making 7,627 runs (~10 days) impossible there. Bulk work therefore used `/Users/ayush/.local/bin/jac` on internal disk: **1.2 s/file, ~12 files/s wall, ~10 min total.**

Both are **jaclang 0.16.1** (verified via `dist-info`). The internal install carries three extra plugins the project venv lacks (`jac_mcp` 0.1.20, `jac_scale` 0.2.23, `jac_super` 0.1.20), so parity was *empirically verified*, not assumed: a stratified 10-file sample (5 passes, 5 failures, `random.seed(1)`) was re-run through the project venv binary. **10/10 exit codes agreed.** Parity holds.

---

## 3. S2 — quality filter

The brief asks specifically what fraction lack docstrings, because SFT reverse-instruction authoring depends on them.

**Only 288 of 7,627 (3.8%) lack a docstring** — 275 of those survive to S2, so the filter drops 3.9% of compile-passing files. This corpus is **not** docstring-poor; reverse-authoring is comfortably viable. Post-filter docstring quality:

| Metric | Value |
|---|---|
| docstring words: min / p10 / median / p90 / max | 1 / 5 / 10 / 32 / 310 |
| docstrings under 4 words | 302 (4.8%) — thin but usable |
| multi-line docstrings | 2,009 (31.7%) |
| structured (`Args:` / `Returns:` / `:param`) | 459 (7.3%) |

**`trivial_body` (419, 6.6%)** is the only large quality drop. Threshold: <12 body tokens (docstring excluded) *or* <3 non-blank code lines. Typical casualty:

```jac
""" Bytes to int """
def bytes_to_int(byte_array: bytes) -> int {
    return int.from_bytes(byte_array, byteorder='big');
}
```

One stdlib call. There is no meaningful Python→Jac conversion skill to learn here, and as a DPO item there is nowhere to hide a subtle bug (`CONTEXT_BRIEF.md` §2's correctness axis needs a behavioral divergence to exist). The threshold is a judgement call; the full list of 419 filenames is in `corpus-triage-stats.json` if a later pass wants them back.

### A bug in my own filter, found and fixed — flagging it because it nearly cost 172 good rows

The first run of this filter reported `no_def=98` and `multi_def=79`. **Both were wrong**, caused by my own regex, not by the corpus:

- `no_def` was matching `^\s*def\s+NAME\s*\(` and so missed **generic** functions, `def pad[T](data: list[list[T]], …)`. There are 85 such files in the final pool. They are valid Jac and compile clean.
- `multi_def` was counting *nested* helper functions (`def _reformat` indented inside an outer `def`) as second top-level definitions.

Fixed to `^def\s+NAME(\[…\])?\s*\(` anchored at column 0. After the fix: `no_def` = 5, `multi_def` = **0** (no file in the corpus has two top-level `def`s — the brief's "each file = ONE function" claim is exactly right). Net effect of the fix: **+172 rows recovered**. Anyone re-running `pipeline.py` gets the corrected numbers; this paragraph exists so the earlier wrong numbers, if they surface in a scrollback anywhere, are recognisable as stale.

The 5 remaining `no_def` files are genuine oddities kept out on purpose: three use Jac's backtick keyword-escape for the function name (`` def `skip(…) ``, `` def `test(…) ``, `` def `match(…) ``), one has no parameter list at all (`def config_section_data -> str {`), one indents its `def`. They compile, but signature extraction for reverse-authoring would produce garbage on them.

---

## 4. S3/S4 — deduplication within the corpus

**Algorithm — ported, not reinvented,** per brief §6. Straight from `01-sft-dpo/sft_dpo/jacgen2/decontam_v2.jac` and `dedup2.jac`:

- `code_tokens`: strip `#`-comments per line, drop blank lines, whitespace split (`dedup2.jac`).
- `shingles(text, n)`: all contiguous n-token windows, `SHINGLE_N = 14` (`decontam_v2.jac` glob).
- Match test: fraction of my shingles present in the reference set ≥ `THRESHOLD = 0.5` (`decontam_v2.jac` glob).

For intra-corpus dedup this is applied **greedily and in a fixed order** (ascending numeric filename): maintain a growing shingle→first-file index of everything accepted so far; a file whose shingles are ≥50% already in the index is dropped and attributed to its heaviest contributor. This is `decontam_v2.jac`'s `is_contaminated` used as its own reference set, which is also the "exact-hash first, then bucketed, never raw pairwise" policy `dedup_v2.jac`'s own docstring prescribes for exactly this scale. Raw O(n²) ROUGE-L (what `dedup_v2.jac` does today at ~150 records) would be ~20M LCS computations here — the docstring explicitly says don't.

**Every file in the pool produced ≥1 shingle** (`no_shingle = 0`), so the dedup pass is applicable to 100% of the pool — checked, because with a median of only 42 code tokens per file, n=14 was a real risk of silently no-op'ing on short functions.

**Result: 1 exact duplicate, 13 near duplicates.**

Exact: `416684.jac` ≡ `195401.jac` (identical after token normalization).

Near-dup clusters (dropped file → representative kept, overlap, function):

| Dropped | Kept | Overlap | Function |
|---|---|---:|---|
| `271639.jac` | `255437.jac` | 0.945 | `validate_rng_seed` |
| `393926.jac` | `106984.jac` | 0.871 | `my_lcs` |
| `391287.jac` | `169672.jac` | 0.822 | `choose` |
| `333269.jac` | `13851.jac` | 0.818 | `_extract_defines_from_option_list` |
| `124899.jac` | `50524.jac` | 0.810 | `sizeformat` |
| `135543.jac` | `90861.jac` | 0.714 | `word_probabilities` |
| `290629.jac` | `114896.jac` | 0.682 | `sizeof_fmt` |
| `244370.jac` | `84300.jac` | 0.600 | `_twos_complement` |
| `296566.jac` | `69181.jac` | 0.576 | `map_quads` |
| `366591.jac` | `50524.jac` | 0.545 | `sizeof_fmt` |
| `120917.jac` | `19803.jac` | 0.500 | `RPL_STATSUPTIME` |
| `132929.jac` | `19803.jac` | 0.500 | `RPL_TRACECONNECTING` |
| `422981.jac` | `19803.jac` | 0.500 | `RPL_ENDOFNAMES` |

Two real clusters worth naming: **`sizeof_fmt`/`sizeformat`** (the human-readable-bytes function, 3 variants — the single most copy-pasted utility on the Python internet) and the **`RPL_*` IRC numeric-reply handlers** (3 variants, all near-identical boilerplate off the same IRC library).

### The surprising result: this corpus is genuinely NOT redundant

The brief predicted large mined Python corpora "cluster heavily around common utility patterns." **For this corpus that is wrong.** 14 total duplicates out of 6,353 is **0.22%**. I did not trust that, so I swept the parameters:

| shingle n | threshold | near-dups dropped | % of pool |
|---:|---:|---:|---:|
| 14 | 0.50 | 13 | 0.21% |
| 14 | 0.30 | 26 | 0.42% |
| 10 | 0.50 | 20 | 0.32% |
| 8 | 0.50 | 22 | 0.35% |
| 8 | 0.30 | 62 | 1.00% |
| 6 | 0.50 | 42 | 0.68% |
| 5 | 0.40 | 108 | 1.74% |

Even at an aggressively loose n=5/0.40 — which starts flagging merely *similar* functions rather than duplicates — only 1.74% is implicated. Corroborating independent signal: **5,890 distinct function names across 6,329 rows (93.1% unique)**; the most repeated name is `flatten` at 12. The duplication genuinely isn't there. The upstream generator appears to have sampled sparsely from a very large source rather than densely from a small one (see §7). **The convention parameters (n=14, 0.50) are the ones used for the shipped artifacts**; the sweep is reported so nobody has to wonder whether 0.22% was a threshold artifact.

---

## 5. S5 — decontamination against all four holdouts

References built per brief §6, all four files, no exceptions:

| Reference | Rows | Strict shingles (n=14, code tokens) | Relaxed shingles (n=10, identifiers) |
|---|---:|---:|---:|
| `04-cpt-sft/sft_fresh_probe/dataset/sft/valid.jsonl` | 1,428 | 164,163 | 172,525 |
| `04-cpt-sft/sft_fresh_probe/dataset/dpo/valid.jsonl` | 115 | 11,577 | 11,780 |
| `01-sft-dpo/dataset/eval_holdout/conversion.jsonl` | 150 | 4,946 | 5,779 |
| `01-sft-dpo/dataset/eval_holdout/graph_conversion.jsonl` | 27 | 1,947 | 1,906 |

Reference text = every `messages[*].content` plus any `prompt` / `chosen` / `rejected` / `python` field — deliberately a superset of `decontam_v2.jac`'s per-category extraction. At n=14 exact-token shingles a false positive is essentially impossible, so the recall-maximising choice costs nothing.

### The problem the convention's strict shingle cannot see, and what was done about it

`eval_holdout/{conversion,graph_conversion}.jsonl` store **Python** in their `python` field. Nitin's corpus is **Jac**. `decontam_v2.jac`'s own docstring already concedes this ("near-zero overlap against a Python-only reference is the expected, honest result"), but there it was benign — the categories being checked weren't Python-derived. **Here it is not benign:** this corpus *is* transpiled Python, so a genuine shared source function would be real contamination that a Jac-vs-Python 14-gram comparison would score at ~0.0, because Jac welds `;`, `{`, `}` and type annotations into the token stream. The convention as written would report "clean" without having actually looked.

Three checks were run instead of one:

1. **Strict** (convention): n=14 over `code_tokens`. Drop at ≥0.50.
2. **Relaxed** (added): n=10 over identifiers-and-numbers only (`[A-Za-z_]\w*|\d+`), discarding all punctuation, operators and delimiters — which puts Jac and Python on a comparable token stream. Drop at ≥0.50.
3. **Function-name collision audit** (added): the `eval_holdout` schema carries `func_name`; every pool file whose function name matches a holdout name was scored by identifier-set Jaccard against that holdout item's Python.

**Results:**

- **Strict: 0 hits.** Exactly as predicted — and on its own it would have been a false all-clear.
- **Relaxed: 1 hit, dropped.** `355452.jac` (`is_prime`), relaxed overlap **0.635** against `sft_valid` (strict overlap: 0.000). This single row is the entire justification for adding check 2.
- **Function-name collisions: 78 flagged, 9 dropped (S5b).** All 78 are `is_prime`, `factorial` and similar textbook utilities against `eval_holdout/conversion.jsonl`. Top scores: `457240.jac` `factorial` J=0.615, `372503.jac` `is_prime` J=0.560, `456849.jac` `is_prime` J=0.542. 9 sit at J ≥ 0.50; 25 at J ≥ 0.30.

**Judgement call on the 9, stated plainly:** these are almost certainly *not* contamination in the copied-from sense — `is_prime` and `factorial` have one canonical implementation, they are in every model's pretraining corpus a thousand times over, and identifier-set Jaccard is a weak similarity measure that saturates on short functions. Dropping them was chosen anyway: 9 rows out of 6,338 is 0.14% of the pool, and the alternative is defending "the model may have memorised the holdout's `is_prime`" in a report whose entire purpose is a clean stock-vs-spectrum comparison. Cheap insurance, in the same spirit as §5's `--verify-layers` preflight. The other 69 collisions (J < 0.50) were **kept** and are listed in `corpus-triage-stats.json` under `fn_collide`.

---

## 6. S6 — the frozen Nitin holdout

**855 rows**, carved with `random.seed(42)` (matching the project-wide seed) from the 6,329-row clean pool. Remainder → 5,474-row candidate pool.

**Why 855 and not "several hundred":** it is the *exact* graded-item count of holdout (a) (`sft/valid.jsonl` → 855 code-graded items), so the two holdout columns in the brief's 2×2×3 matrix have identical statistical power and the cross-column read is clean. The pool comfortably supports it — the binding consideration was power, not availability. Concretely, for the effect size this experiment is built to detect (the existing fresh-arm gap, stock 69.8% vs spectrum 74.7%, ≈4.9 pp):

- at n=855, unpaired two-proportion z ≈ **2.27** → detectable at α=0.05
- at n=500, the same gap gives z ≈ **1.73** → **not** detectable

A 500-row holdout would have risked the experiment reporting "no significant difference" as an artifact of holdout size. Paired McNemar (which `spectrum-workflow.md` calibrates to, since both arms answer identical items) is strictly more powerful than these unpaired figures, so 855 has margin.

**Independence verification (brief §6's explicit requirement — actually run, not asserted):**

- *Holdout vs candidate pool:* the same n=14/0.50 machinery was re-run with the 5,474 candidates as the reference set. **0 of 855 holdout rows exceed threshold.** This is also guaranteed structurally — S4's greedy pass means every survivor already has <50% shingle overlap with the union of all prior survivors — but it was measured rather than argued.
- *Holdout vs holdout (a):* S5/S5b ran on the whole 6,338-row pool **before** the split, so the holdout inherits the same decontamination as the training pool. No Nitin-holdout item is a near-dup of an (a)-holdout item; the two holdout results are independent.
- *Freeze:* both files are written once, from a seeded shuffle, and `pipeline.py` is deterministic — re-running reproduces byte-identical splits.

### Record schema

No exact precedent existed. `dataset/shared/seed_pool.jsonl` uses `{seed_id, seed_tier, origin, jac_code, …}`; the SFT release (`04-cpt-sft/dataset/fresh/releases/sft_train.jsonl`) uses `{id, category, task_type, …, messages, origin, run_tag, dataset_version}`. These are a *pre-messages* pool, so the schema unions the requested fields with the naming conventions of both, so downstream SFT-pair authoring can lift fields straight across:

```json
{"id": "nitin_8ed37fd232df__100", "seed_id": "8ed37fd232df", "split": "candidate",
 "source_file": "data/jac_outputs/100.jac",
 "origin": "jac-data-gen:7c25aff3110f526eec59e0123ffe6c0c152cce91:data/jac_outputs/100.jac",
 "func_name": "compare_elements",
 "params": "prev_hash_dict: dict[str, str], current_hash_dict: dict[str, str]",
 "return_type": "dict[str, str]", "docstring": "...", "jac_code": "...",
 "line_count": 22, "code_lines": 18, "token_count": 64,
 "compile_pass": true, "compile_cmd": "jac run (jaclang 0.16.1, exit 0)",
 "dedup_pass": true, "decontam_pass": true,
 "category": "conversion", "task_type": "python_to_jac_function",
 "dataset_version": "nitin-ds-v1.0.0", "run_tag": "nitin"}
```

`origin` follows brief §2's mandated `jac-data-gen:<commit>:<path>` format verbatim. `seed_id` is `sha1(origin)[:12]`, matching `seed_pool.jsonl`'s 12-hex-char convention. `category`/`task_type` are pre-set to brief §2's decided values. `params`/`return_type` are pre-extracted because reverse-instruction authoring needs the signature and re-parsing 5,474 files later would be wasted work.

---

## 7. Provenance, licensing, and other things spotted

### The repo carries no metadata at all

`git ls-files` returns nothing but `data/jac_outputs/`. No README, no generator script, no LICENSE, no manifest, no split. One commit, message `data`. **There is no upstream statement of where this Python came from or under what license.** The brief's §1 characterisation is accurate and, if anything, understated.

### Filename IDs are row indices into a ~459k-row source

Filenames are numeric, **min 70, max 459,271**, all distinct, 7,627 of them — i.e. a **1.66% sparse sample** of a source dataset of roughly 459k rows. That sparsity is the direct mechanical explanation for §4's near-zero duplication, and it means **the upstream corpus could be scaled ~60× if more of it transpiles**.

A ~459k-row Python function corpus with docstrings points at the CodeSearchNet-Python family (`code-search-net-python` is ≈457k rows). **Stated as a hypothesis, not a finding** — nothing in the repo confirms it. Worth asking Nitin directly, because the answer determines the license of the training data.

Reassuringly for contamination: `eval_holdout/*.jsonl` declare `source: "Vezora/Tested-22k-Python-Alpaca"` — a 22k-row dataset. Max index 459,271 ≫ 22,000, so the two corpora are almost certainly disjoint in origin. That is consistent with S5's near-zero hit rate being real rather than an artifact of the Jac-vs-Python token gap (which §5 checked separately anyway).

### Licensing: no headers, but unmistakable OSS provenance

**No copyright or license headers survive into any file.** The 4 files matching `copyright|licen[sc]e|SPDX` are functions *about* licensing (`Combine the spdx ids.`, `Lookup package licenses`) — false positives, not headers. Whatever the upstream pipeline did, it stripped or never captured module-level headers.

That is **not** the same as the data being unencumbered. The docstrings leak provenance loudly:

| Signal in final pool | Count |
|---|---:|
| any URL | 152 |
| `stackoverflow.com` | 24 |
| `github.com` | 20 |
| `Author:` / `@author` | 6 |
| `arxiv.org` | 3 |

Examples: `Copied from http://stackoverflow.com/a/3438818/3710392`, `Obtained from: https://github.com/doukremt/distance/blob/master/distance/_simpledists.py on Sept 8 2015`, `paper : https://arxiv.org/abs/1509.02971`. This is mined real-world OSS Python. **Two consequences:**

1. **License status is unknown and unknowable from the repo.** Fine for an internal research replication; would need resolving before any model trained on it is released.
2. **152 files will teach the model to emit URLs in docstrings.** They were left in — stripping them would alter the dataset being evaluated, which is the thing under test. Flagging it as a known artifact of the SFT targets.

### The brief's §1 "not graph-native" claim — independently confirmed

Regex sweep of the final 6,329-row pool for `node`/`edge`/`walker`/`obj`/`can` declarations, `visit`, `spawn`, `-->`, `<--`: **16 hits, all incidental** (variables named `node`, arrows inside strings). Zero genuine OSP constructs, zero `with entry` blocks. The corpus is 100% flat function conversions. Brief §1 is correct and the framing decisions built on it stand.

### Corpus shape (final pool)

| Metric | median | p90 | max |
|---|---:|---:|---:|
| line count (incl. docstring) | 11 | 24 | 101 |
| body tokens (docstring excluded) | 31 | 72 | 248 |

These are **small** functions — a median of 11 lines. Relevant downstream: DPO buggy-variant authoring (brief §2) has little surface area to hide a subtle behavioral bug in a median 31-token body, and the 419 files already cut at S2 were the worst of it. Expect a non-trivial DPO-pair authoring reject rate and budget for it.

---

## 8. Things the next agent should know

1. **The candidate pool is 5,474 rows; the existing fresh-arm SFT trained on 8,100** (`04-cpt-sft/sft_fresh_probe/dataset/sft/train.jsonl`). Research question 2 ("is this dataset truly better") is therefore **confounded by size**, 0.68× the training data. This is not fixable by shrinking the holdout — dropping the holdout to 500 buys 355 rows (→5,829, still 0.72×) and costs the statistical power to answer research question 1 (§6). The honest options are (a) report the confound, or (b) additionally train the existing dataset subsampled to 5,474 as a third arm. **Recommend (a) plus an explicit caveat**, since (b) doubles the training budget. Either way it must appear in the final comparison report — it cannot be left implicit.
2. **Fixing the 566 `Missing ';'` files upstream would take the pool to ~6,000+** (5,474 → ~5,900 candidates) and nearly close the size gap. That is the highest-leverage single action available, and it belongs in `chess10kp/jac-data-gen`, not in a downstream patch.
3. **The corpus is 60× larger than what was sampled** (§7). If more data is wanted, ask for more of the source rather than augmenting these 7,627.
4. **Do not use the project venv's `jac` for bulk work.** It is on the 4.3 MB/s external drive; one invocation exceeds 120 s. Use `/Users/ayush/.local/bin/jac` (same jaclang 0.16.1, parity verified 10/10) and write only final artifacts to the external tree.
5. **`decontam_v2.jac`'s strict shingle is blind to Jac-vs-Python contamination.** It reported 0 hits here; the relaxed identifier-only variant found a real one. Any future decontam of Jac against a Python reference must run both, or it is reporting a false all-clear. Consider back-porting the relaxed check into `decontam_v2.jac`.

## 9. Reproducing

```bash
git -C ~/repos/jac-data-gen log -1 --format=%H     # 7c25aff3110f526eec59e0123ffe6c0c152cce91
python3 model-experiments/06-nitin-ds-sft/scripts/compile_check.py   # ~10 min, writes compile_results.jsonl
python3 model-experiments/06-nitin-ds-sft/scripts/pipeline.py        # ~2 min, writes both JSONLs + stats.json
```

`compile_check.py` is resumable (it skips filenames already present in `compile_results.jsonl`). `pipeline.py` is deterministic (`random.seed(42)`) and idempotent. Both hardcode the fast internal-disk jac and the external-drive output paths.
