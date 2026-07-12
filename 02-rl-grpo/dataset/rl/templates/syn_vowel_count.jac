"""RL driver (synthetic): count the vowels (a,e,i,o,u) in a string, case-insensitive."""

def vowel_count(s: str) -> int {
    __HOLE__
}

with entry {
    for x in ["Hello", "AEIOU", "xyz", "Programming"] {
        print(vowel_count(x));
    }
}
