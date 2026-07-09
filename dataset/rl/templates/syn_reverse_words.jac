"""RL driver (synthetic): reverse the order of words in a string."""

def reverse_words(s: str) -> str {
    __HOLE__
}

with entry {
    for x in ["hello world foo", "one two three four", "single", "a b c"] {
        print(reverse_words(x));
    }
}
