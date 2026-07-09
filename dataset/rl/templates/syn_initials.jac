"""RL driver (synthetic): build uppercase initials from a full name."""

def initials(name: str) -> str {
    __HOLE__
}

with entry {
    for x in ["john smith", "ada lovelace", "the quick brown fox", "solo"] {
        print(initials(x));
    }
}
