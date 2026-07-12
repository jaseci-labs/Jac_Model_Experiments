"""RL driver (synthetic): round a positive float to 2 decimals without round()."""

def round2(x: float) -> float {
    # >>>HOLE id="syn_round2" instruction="Implement round2(x): round the non-negative float x to two decimal places WITHOUT the round() builtin — multiply x by 100.0, add 0.5, truncate to int, then divide by 100.0 and return the float."
    return int(x * 100.0 + 0.5) / 100.0;
    # <<<HOLE
}

with entry {
    for x in [3.14159, 2.71828, 1.41421, 0.5, 9.999] {
        print(round2(x));
    }
}
