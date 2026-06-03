"""14-gram decontamination of holdout candidates against the training set.

Builds the set of 14-token shingles over all training Python (comment-stripped,
whitespace-tokenized) and flags any candidate whose shingles overlap the
training set above a threshold. This is the non-negotiable ordering guard: the
eval holdout must share no near-duplicate task with anything trained on.
"""

import json;
import from dedup { code_tokens }
import from pathlib { Path }
import from typing { Any }


def extract_python(rec: dict) -> str {
    msgs: list = list(rec["messages"]);
    user: dict = dict(msgs[0]);
    content: str = str(user["content"]);
    if "```python" not in content { return ""; }
    after: str = content.split("```python", 1)[1];
    return after.rsplit("```", 1)[0].strip();
}


def shingles(code: str, n: int = 14) -> list[str] {
    toks: list[str] = code_tokens(code);
    out: list[str] = [];
    if len(toks) < n { return out; }
    for i in range(len(toks) - n + 1) {
        out.append(" ".join(toks[i:i + n]));
    }
    return out;
}


def build_training_shingles(paths: list[str]) -> dict {
    s: dict[str, int] = {};
    for path in paths {
        p = Path(path);
        if not p.exists() { continue; }
        for line in p.read_text().split("\n") {
            if line.strip() == "" { continue; }
            rec: dict = dict(json.loads(line));
            py: str = extract_python(rec);
            for sh in shingles(py) { s[sh] = 1; }
        }
    }
    return s;
}


def is_contaminated(py: str, train: dict, threshold: float = 0.5) -> bool {
    sh: list[str] = shingles(py);
    if len(sh) == 0 { return False; }
    hit: int = 0;
    for x in sh { if x in train { hit += 1; } }
    return (hit / len(sh)) >= threshold;
}
