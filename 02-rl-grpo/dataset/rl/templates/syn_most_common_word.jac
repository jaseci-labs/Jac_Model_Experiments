"""RL driver (synthetic): most frequent word, ties broken by first appearance."""

def most_common_word(s: str) -> str {
    __HOLE__
}

with entry {
    for x in ["a b a c a", "one two two three", "hi", "x y y x x"] {
        print(most_common_word(x));
    }
}
