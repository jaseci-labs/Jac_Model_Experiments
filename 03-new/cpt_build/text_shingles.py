"""14-gram word-shingle sets, shared by build_cpt.py's decontam step and
shingle_dedup.py's within-source near-duplicate detector. Containment
(|A & B| / |A|), not symmetric Jaccard, is the meaningful direction when
comparing a small item against a much larger one (a holdout snippet against
a whole doc row, or a short chunk against a longer one) -- callers compute
containment themselves using the sets this returns."""
import re

_WORD_RE = re.compile(r"\S+")


def shingles(text: str, n: int = 14) -> set:
    words = _WORD_RE.findall(text)
    if len(words) < n:
        return set()
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}
