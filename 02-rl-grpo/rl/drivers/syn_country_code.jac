"""RL driver (synthetic): dict lookup of country name to ISO code, default '??'."""

def country_code(name: str) -> str {
    # >>>HOLE id="syn_country_code" instruction="Implement country_code(name): look name up in the dict {'usa':'US','canada':'CA','japan':'JP','brazil':'BR'} and return the matching code, or '??' if name is not a key."
    m = {"usa": "US", "canada": "CA", "japan": "JP", "brazil": "BR"};
    return m.get(name, "??");
    # <<<HOLE
}

with entry {
    for x in ["usa", "japan", "france", "brazil"] {
        print(country_code(x));
    }
}
