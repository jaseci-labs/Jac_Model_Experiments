"""
ROUGE-L (LCS) similarity, duplicated from ../jacgen/dedup.jac's
`code_tokens`/`rouge_l` for jacgen2's seed_pool.jac.

NOTE on duplication: `import from ..jacgen.dedup { code_tokens, rouge_l }`
type-checks under `jac check` but fails at runtime here (empirically
verified: "attempted relative import with no known parent package") --
identical failure mode Task 3/Task 4 hit and solved by inlining rather than
fighting the import system. Same move here. jacgen/dedup.jac itself is
untouched (frozen, never edited).
"""

def code_tokens(code: str) -> list[str] {
    lines: list[str] = [];
    for line in code.split("\n") {
        stripped: str = line.split("#", 1)[0];
        if stripped.strip() != "" { lines.append(stripped); }
    }
    return " ".join(lines).split();
}


def lcs_len(a: list[str], b: list[str]) -> int {
    n: int = len(a);
    m: int = len(b);
    prev: list[int] = [0] * (m + 1);
    for i in range(1, n + 1) {
        cur: list[int] = [0] * (m + 1);
        for j in range(1, m + 1) {
            val: int = max(prev[j], cur[j - 1]);
            if a[i - 1] == b[j - 1] { val = prev[j - 1] + 1; }
            cur[j] = val;
        }
        prev = cur;
    }
    return prev[m];
}


def rouge_l(code_a: str, code_b: str) -> float {
    a: list[str] = code_tokens(code_a);
    b: list[str] = code_tokens(code_b);
    if len(a) == 0 or len(b) == 0 { return 0.0; }
    return (2.0 * lcs_len(a, b)) / (len(a) + len(b));
}
