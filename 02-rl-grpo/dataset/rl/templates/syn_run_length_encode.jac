"""RL driver (synthetic): run-length encode a string as char+count pairs."""

def run_length_encode(s: str) -> str {
    __HOLE__
}

with entry {
    for x in ["aaab", "abc", "wwwww", "aabbaa"] {
        print(run_length_encode(x));
    }
}
