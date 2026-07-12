"""RL driver (synthetic): swap the case of every alphabetic character in a string."""

def swapcase(s: str) -> str {
    __HOLE__
}

with entry {
    for x in ["Hello World", "jAC ml", "ABC123xyz", "!?."] {
        print(swapcase(x));
    }
}
