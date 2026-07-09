"""RL driver (synthetic): extract the second comma-separated field, trimmed."""

def csv_second_field(row: str) -> str {
    __HOLE__
}

with entry {
    for x in ["a,b,c", "name, age, city", "1,2,3", "x, y"] {
        print(csv_second_field(x));
    }
}
