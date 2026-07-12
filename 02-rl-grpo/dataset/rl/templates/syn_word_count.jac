"""RL driver (synthetic): count the whitespace-separated words in a string."""

def word_count(s: str) -> int {
    __HOLE__
}

with entry {
    for x in ["hello world", "one", "a b c d e", "   spaced   out  "] {
        print(word_count(x));
    }
}
