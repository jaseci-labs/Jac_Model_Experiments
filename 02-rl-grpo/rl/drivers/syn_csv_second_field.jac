"""RL driver (synthetic): extract the second comma-separated field, trimmed."""

def csv_second_field(row: str) -> str {
    # >>>HOLE id="syn_csv_second_field" instruction="Implement csv_second_field(row): split row on commas, take the field at index 1 (the second field), strip surrounding whitespace from it, and return it."
    return row.split(",")[1].strip();
    # <<<HOLE
}

with entry {
    for x in ["a,b,c", "name, age, city", "1,2,3", "x, y"] {
        print(csv_second_field(x));
    }
}
