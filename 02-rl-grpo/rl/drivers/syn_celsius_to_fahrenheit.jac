"""RL driver (synthetic): convert Celsius to Fahrenheit."""

def celsius_to_fahrenheit(c: float) -> float {
    # >>>HOLE id="syn_celsius_to_fahrenheit" instruction="Implement celsius_to_fahrenheit(c): convert a Celsius temperature to Fahrenheit and return it — multiply c by 9.0, divide by 5.0, and add 32.0."
    return c * 9.0 / 5.0 + 32.0;
    # <<<HOLE
}

with entry {
    for c in [0.0, 100.0, 37.0, -40.0, 25.0] {
        print(celsius_to_fahrenheit(c));
    }
}
