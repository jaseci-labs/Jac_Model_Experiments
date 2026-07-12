"""RL driver (synthetic): reverse the order of words in a string."""

def reverse_words(s: str) -> str {
    # >>>HOLE id="syn_reverse_words" instruction="Implement reverse_words(s): split s on whitespace, reverse the order of the words, and join them back with single spaces."
    return " ".join(s.split()[::-1]);
    # <<<HOLE
}

with entry {
    for x in ["hello world foo", "one two three four", "single", "a b c"] {
        print(reverse_words(x));
    }
}
