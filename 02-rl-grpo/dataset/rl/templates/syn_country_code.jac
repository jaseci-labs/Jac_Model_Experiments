"""RL driver (synthetic): dict lookup of country name to ISO code, default '??'."""

def country_code(name: str) -> str {
    __HOLE__
}

with entry {
    for x in ["usa", "japan", "france", "brazil"] {
        print(country_code(x));
    }
}
