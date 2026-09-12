#!/usr/bin/env python3
"""09 data builder: "continue this partial Jac function" pairs + the mixed SFT release.

Stage A (pairs): for every 08 SFT row whose single ```jac answer contains a
single-line-signature `def name(...) ... {` with >= MIN_BODY body lines, pick ONE
such function (seeded). ~25% of rows (seeded) are cut right after the def line's `{`
(no trailing newline; target = "\n" + body + "}" -- the eval's zero-body shape); the rest
are cut after a random 20-60% of the body lines (prefix ends with a newline). Emit
  prompt  = <instruction wording> + "\n\n" + <answer code from file start through the cut line>
  answer  = the rest of that function through its closing brace (raw, NO fences)
Code after the function is dropped, so the answer is exactly "finish this function".
Every pair is checked: prefix + target must equal the original code through the
function's closing brace, and the function slice must equal the original function.

Stage B (mix): dataset/sft/train.jsonl = all train pairs + REPLAY_TRAIN seeded 08 train
rows (unchanged messages); valid likewise with REPLAY_VALID. Seeded shuffle. Only
messages + metadata keys are kept.

Inputs (read only): experiments/08-nitin-new2-ds/dataset/sft/{train,valid}.jsonl
Outputs: <exp>/dataset/completion_pairs_{train,valid}.jsonl, <exp>/dataset/sft/{train,valid}.jsonl

    python3 experiments/09-completion-pairs/scaffold/build_completion_pairs.py      # FORCE=1 to overwrite
"""
import json
import os
import random
import re
import statistics
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]          # <repo>/experiments/<exp>
ROOT = EXP.parents[1]
SRC = Path(os.environ.get("SRC_SFT", ROOT / "experiments" / "08-nitin-new2-ds" / "dataset" / "sft"))
OUT = EXP / "dataset"
SEED = int(os.environ.get("SEED", "42"))
MIN_BODY = 6
CUT_LO, CUT_HI = 0.20, 0.60
MAX_CHARS = 9000          # prompt+target char budget (~2.8k tokens) so max_seq_length 3072 never truncates the target
AFTER_DEF_FRAC = 0.25     # share of rows cut right after the def line's `{` (eval's zero-body shape)
REPLAY_TRAIN, REPLAY_VALID = 1500, 500
FORCE = os.environ.get("FORCE", "0") == "1"

# Our own wordings. Deliberately NOT jac-data-gen's eval prompt sentence.
WORDINGS = [
    "The Jac function below is unfinished; the code stops partway through its body. "
    "Write the rest of the function, starting exactly where the code stops. "
    "Output only the remaining Jac code -- do not repeat anything already shown and do not wrap it in Markdown fences.",
    "Finish this partially written Jac code. Pick up at the exact point where it ends and write "
    "only what is missing to complete the function, including any blocks that are still open. "
    "Do not restate the given lines and do not use ``` fences.",
    "Here is the beginning of a Jac function that was cut off mid-body. Complete it from the exact "
    "stopping point. Reply with just the missing Jac lines (no repetition of the code above, no Markdown code fences).",
    "Complete the Jac code below. It ends in the middle of a function; continue from precisely that "
    "spot until the function is done. Give only the new Jac source that follows, without echoing the "
    "existing code and without Markdown formatting.",
    "This Jac source is truncated inside a function body. Produce the continuation that completes the "
    "function, beginning right after the last line shown. Only the continuation: no copy of the provided "
    "code, no fences.",
    "Continue writing the following Jac function from where it breaks off, closing every open block so "
    "the function is complete. Output the remaining code only; the lines already given must not be "
    "repeated, and do not add ``` markers.",
]
EVAL_SENTENCE = ("Continue the Jac source below. Preserve the existing signature and behavior. "
                 "Return only the missing Jac continuation; do not repeat the visible prefix and do not add Markdown fences.")
assert not any(EVAL_SENTENCE in w or w in EVAL_SENTENCE for w in WORDINGS)

FENCE_RE = re.compile(r"\A```jac\n(.*)\n```\s*\Z", re.S)
# [ \t] not \s: under re.M, ^\s* would swallow preceding blank lines and misplace the def line.
DEF_RE = re.compile(r"^([ \t]*)(?:(?:static|override)[ \t]+)?def(?::\w+)?[ \t]+\w+[ \t]*\(.*\)[^{\n]*\{[ \t]*$", re.M)


def line_states(code: str, start: int):
    """Scan `code` from char offset `start` (the def line's start) tracking brace depth,
    skipping strings ("", '', triple-quoted, prefixed f/r/b) and comments (# ..., #* ... *#).
    Returns (per-line list of (depth_at_eol, in_string_at_eol) starting at the def line,
    index of the line where depth returns to 0) or None if never balanced."""
    out = []
    depth, i, n = 0, start, len(code)
    quote = None           # current string delimiter: ', ", ''' or """
    block_comment = False
    opened = False
    while i < n:
        c = code[i]
        if c == "\n":
            out.append((depth, quote is not None or block_comment))
            i += 1
            continue
        if block_comment:
            if code.startswith("*#", i):
                block_comment = False
                i += 2
            else:
                i += 1
            continue
        if quote:
            if c == "\\":
                i += 2
                continue
            if code.startswith(quote, i):
                i += len(quote)
                quote = None
            else:
                i += 1
            continue
        if code.startswith("#*", i):
            block_comment = True
            i += 2
            continue
        if c == "#":
            j = code.find("\n", i)
            i = n if j < 0 else j
            continue
        if code.startswith('"""', i) or code.startswith("'''", i):
            quote = code[i:i + 3]
            i += 3
            continue
        if c in "\"'":
            quote = c
            i += 1
            continue
        if c == "{":
            depth += 1
            opened = True
        elif c == "}":
            depth -= 1
            if depth < 0:
                return None
            if opened and depth == 0:
                # the rest of this line must be empty for a clean "finish the function" target
                j = code.find("\n", i)
                rest = code[i + 1:] if j < 0 else code[i + 1:j]
                if rest.strip():
                    return None
                out.append((0, False))
                return out, len(out) - 1
        i += 1
    return None


def candidates(code: str):
    """Yield (def_line_idx, close_line_idx, states) for every eligible function."""
    lines = code.split("\n")
    offsets, pos = [], 0
    for ln in lines:
        offsets.append(pos)
        pos += len(ln) + 1
    line_of = {off: k for k, off in enumerate(offsets)}
    for m in DEF_RE.finditer(code):
        d = line_of.get(m.start())
        if d is None:
            continue
        r = line_states(code, m.start())
        if r is None:
            continue
        states, rel_close = r
        close = d + rel_close
        # the closing brace must sit alone at the def's own indent; anything else means the
        # string/comment heuristic mis-scanned (e.g. an apostrophe in JSX text) -- skip
        if lines[close].rstrip() != m.group(1) + "}":
            continue
        if close - d - 1 >= MIN_BODY:
            yield d, close, states


def make_pair(row, rng, stats):
    msgs = row["messages"]
    m = FENCE_RE.match(msgs[-1]["content"])
    if not m:
        stats["no_single_fence"] += 1
        return None
    code = m.group(1)
    cands = list(candidates(code))
    if not cands:
        stats["no_eligible_def"] += 1
        return None
    rng.shuffle(cands)
    lines = code.split("\n")
    after_def_draw = rng.random() < AFTER_DEF_FRAC       # one seeded draw per row
    for d, close, states in cands:
        n_body = close - d - 1
        frac = rng.uniform(CUT_LO, CUT_HI)
        orig_through = "\n".join(lines[:close + 1])
        func = "\n".join(lines[d:close + 1])
        def_off = len("\n".join(lines[:d])) + (1 if d else 0)
        # zero-body-line shape (the eval's): prefix stops at the def line's `{` with NO trailing
        # newline, target = "\n" + body + closing brace. Needs a def line with nothing after `{`.
        after_def = after_def_draw and lines[d].endswith("{")
        if after_def:
            ks = [0]
        else:
            k_lo, k_hi = 1, n_body - 2                # >=1 body line shown, >=2 body lines left
            k0 = min(max(1, round(frac * n_body)), k_hi)
            # nearest k to k0 whose cut line is non-blank and not inside a string/comment
            ks = sorted(range(k_lo, k_hi + 1), key=lambda k: (abs(k - k0), k))
        for k in ks:
            cut = d + k                                # last prefix line (absolute index)
            depth, in_str = states[k]
            if in_str or not lines[cut].strip():
                continue
            if after_def:
                prefix = "\n".join(lines[:d + 1])
                target = "\n" + "\n".join(lines[d + 1:close + 1])
            else:
                prefix = "\n".join(lines[:cut + 1]) + "\n"
                target = "\n".join(lines[cut + 1:close + 1])
            # validation: exact reproduction of the original through the closing brace,
            # and of the function text itself
            joined = prefix + target
            if joined != orig_through or joined[def_off:] != func or not code.startswith(joined):
                stats["validation_fail"] += 1
                return None
            wording = rng.choice(WORDINGS)
            prompt = wording + "\n\n" + prefix
            if len(prompt) + len(target) > MAX_CHARS:
                stats["over_char_budget"] += 1
                break                                  # try the next candidate function
            return {
                "id": row["id"] + "__cont",
                "origin": row.get("origin"),
                "source_type": "completion_pair",
                "parent_source_type": row.get("source_type"),
                "cut_frac": round(k / n_body, 4),
                "cut_after_def_line": after_def,
                "cut_inside_open_block": depth > 1,
                "prefix_lines": cut + 1,
                "target_lines": close - cut,
                "messages": [{"role": "user", "content": prompt},
                             {"role": "assistant", "content": target}],
            }
        else:
            stats["no_valid_cut"] += 1
    return None


def read_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(p, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    mixed = {s: OUT / "sft" / f"{s}.jsonl" for s in ("train", "valid")}
    if not FORCE and any(p.exists() and p.stat().st_size for p in mixed.values()):
        sys.exit(f"refusing: {OUT/'sft'} already populated (FORCE=1 to overwrite; re-run nan_guard after)")
    replay_n = {"train": REPLAY_TRAIN, "valid": REPLAY_VALID}
    for split in ("train", "valid"):
        rows = read_jsonl(SRC / f"{split}.jsonl")
        rng = random.Random(f"{SEED}:{split}:pairs")
        stats = {k: 0 for k in ("no_single_fence", "no_eligible_def", "no_valid_cut",
                                "over_char_budget", "validation_fail")}
        pairs = [p for p in (make_pair(r, rng, stats) for r in rows) if p]
        write_jsonl(OUT / f"completion_pairs_{split}.jsonl", pairs)
        n = len(pairs)
        inside = sum(p["cut_inside_open_block"] for p in pairs)
        after = sum(p["cut_after_def_line"] for p in pairs)
        print(f"[{split}] rows scanned {len(rows)}  pairs {n}  "
              f"inside-open-block {inside} ({100*inside/max(n,1):.1f}%)  "
              f"after-def-line {after} ({100*after/max(n,1):.1f}%)  "
              f"median prefix lines {statistics.median(p['prefix_lines'] for p in pairs)}  "
              f"median target lines {statistics.median(p['target_lines'] for p in pairs)}  "
              f"median cut_frac {statistics.median(p['cut_frac'] for p in pairs):.3f}")
        print(f"    skipped: {stats}")
        by_parent = {}
        for p in pairs:
            by_parent[p["parent_source_type"]] = by_parent.get(p["parent_source_type"], 0) + 1
        print(f"    pairs by parent source_type: {dict(sorted(by_parent.items()))}")

        rrng = random.Random(f"{SEED}:{split}:replay")
        replay = [{"id": r["id"], "origin": r.get("origin"), "source_type": r.get("source_type"),
                   "messages": r["messages"]}
                  for r in rrng.sample(rows, replay_n[split])]
        for p in pairs:                       # counts were for stats only; keep metadata lean
            del p["prefix_lines"], p["target_lines"]
        mix = pairs + replay
        random.Random(f"{SEED}:{split}:shuffle").shuffle(mix)
        write_jsonl(mixed[split], mix)
        print(f"    -> {mixed[split]}: {len(mix)} rows ({n} pairs + {len(replay)} replay)")


if __name__ == "__main__":
    main()
