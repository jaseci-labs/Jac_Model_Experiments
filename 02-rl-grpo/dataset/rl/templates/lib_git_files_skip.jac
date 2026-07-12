"""RL driver (pure-lib): source_index._git_files (path-exclusion predicate).

Origin: this_is_jac/source_index.jac (the string-only filtering rules inside the
`_git_files` loop, isolated from the git/filesystem calls). Deterministic:
decides whether a repo-relative path should be skipped based on its text alone:
__pycache__ components, dot-prefixed basenames, and binary extensions. The
marked body is the unit the policy must regenerate; the `with entry` block
exercises it on fixed inputs and prints a deterministic projection.
"""

import os;

glob _BINARY_EXTS: set = {
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".webp", ".bmp",
    ".woff", ".woff2", ".ttf", ".otf", ".eot",
    ".wasm", ".so", ".dylib", ".dll", ".a", ".o", ".bin",
    ".zip", ".gz", ".tar", ".pdf", ".mp4", ".mov", ".jbc", ".pyc"
};

def _should_skip(rel: str) -> bool {
    __HOLE__
}

with entry {
    cases = [
        "src/main.jac",
        "a/__pycache__/x.pyc",
        "dir/.hidden",
        "logo.PNG",
        "build/app.wasm",
        "readme.md",
        "nested/.gitignore"
    ];
    for c in cases {
        print(c + " -> " + str(_should_skip(c)));
    }
}
