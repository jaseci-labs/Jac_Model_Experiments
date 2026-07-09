"""RL driver (synthetic): convert Celsius to Fahrenheit."""

def celsius_to_fahrenheit(c: float) -> float {
    __HOLE__
}

with entry {
    for c in [0.0, 100.0, 37.0, -40.0, 25.0] {
        print(celsius_to_fahrenheit(c));
    }
}
