"""RL driver (synthetic): count the whitespace-separated words in a string."""

def word_count(s: str) -> int {
    # >>>HOLE id="syn_word_count" instruction="Implement word_count(s): return the number of whitespace-separated words in s (split s on whitespace and return the length of the result)."
    return len(s.split());
    # <<<HOLE
}

with entry {
    for x in ["hello world", "one", "a b c d e", "   spaced   out  "] {
        print(word_count(x));
    }
}
