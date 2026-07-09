"""RL driver (synthetic): remove duplicates from a list, preserving first-seen order."""

def dedupe(items: list) -> list {
    __HOLE__
}

with entry {
    for x in [[1, 2, 2, 3, 1], [5, 5, 5], [1, 2, 3], [4, 4, 3, 2, 3]] {
        print(dedupe(x));
    }
}
