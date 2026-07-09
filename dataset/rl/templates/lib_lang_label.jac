"""RL driver (pure-lib): source_index._lang_label.

Origin: this_is_jac/source_index.jac. Deterministic: maps a file path to its
human language label using the _LANG table. The marked body is the unit the
policy must regenerate; the `with entry` block exercises it and prints a fixed,
deterministic projection (no ids, no timestamps).
"""

import os;

glob _LANG: dict = {
    "jac": "Jac", "py": "Python", "css": "CSS", "toml": "TOML",
    "md": "Markdown", "json": "JSON", "svg": "SVG", "sh": "Shell",
    "ts": "TypeScript", "tsx": "TSX", "js": "JavaScript", "jsx": "JSX",
    "zig": "Zig", "html": "HTML", "txt": "Text", "yml": "YAML", "yaml": "YAML"
};

def _lang_label(rel: str) -> str {
    __HOLE__
}

with entry {
    cases = ["a/b/main.jac", "x.py", "style.css", "deep/nested.TS", "README", "data.unknown", "Makefile.toml"];
    for c in cases {
        print(c + " -> " + _lang_label(c));
    }
}
