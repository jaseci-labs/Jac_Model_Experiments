"""This is Jac - the live source-browser backend.

The whole showcase is written in Jac; this module lets the site show its OWN
source. Two `def:pub` endpoints (reachable from the browser as RPCs, exactly
like the guestbook walkers) enumerate the repo's source files and highlight any
one on demand. Highlighting runs server-side through `source_index.py`, which
uses Jac's *own* canonical Pygments lexer - so the `.jac` you read here is
coloured by the same grammar the official docs use. The client just renders the
returned HTML.
"""

import from source_index { list_files, get_file }

"""All viewable source files in the repo (git-tracked + new, build output
excluded), each as {path, name, dir, lang, size}."""
def:pub list_source_files -> list[dict] {
    return list_files();
}

"""One highlighted file: {path, lang, html, lines, raw, truncated, error}. The
`html` is pre-highlighted token spans for direct `unsafe_html` insertion; `raw`
is the plain text for the copy button. Path is sandboxed to the repo."""
def:pub get_source_file(path: str) -> dict {
    return get_file(path);
}
