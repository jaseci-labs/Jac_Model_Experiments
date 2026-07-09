"""RL driver (synthetic): per-character counts formatted in first-seen order."""

def letter_counts(s: str) -> str {
    __HOLE__
}

with entry {
    for x in ["aabbc", "hello", "abc", "zzz"] {
        print(letter_counts(x));
    }
}
