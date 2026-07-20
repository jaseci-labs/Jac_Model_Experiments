"""This is Jac - source-index + syntax-highlight helper for the "Browse the
source" section.

Enumerates the showcase repo's own source files (via git, so build artifacts and
node_modules are excluded for free) and highlights any one of them with Pygments
- using Jac's *own* canonical lexer (`source_lexer.JacLexer`) for every `.jac`
variant, and Pygments' built-in lexers for Python/CSS/TOML/Markdown/JSON/etc. The
client just drops the returned HTML into a <pre> via `unsafe_html`.
"""

import os;
import subprocess;
import from functools { lru_cache }
import from pygments { highlight }
import from pygments.formatters { HtmlFormatter }
import from pygments.lexers { get_lexer_by_name, get_lexer_for_filename }
import from pygments.lexers.special { TextLexer }
import from pygments.util { ClassNotFound }
import from source_lexer { JacLexer }

glob ROOT: str = os.path.dirname(os.path.abspath(__file__));
glob MAX_BYTES: int = 260000;  # cap a single file so a huge generated file can't wedge the UI

# Files git tracks (or that are untracked-but-not-ignored) we still don't want to
# render as text.
glob _BINARY_EXTS: set = {
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".webp", ".bmp",
    ".woff", ".woff2", ".ttf", ".otf", ".eot",
    ".wasm", ".so", ".dylib", ".dll", ".a", ".o", ".bin",
    ".zip", ".gz", ".tar", ".pdf", ".mp4", ".mov", ".jbc", ".pyc"
};

# Human-facing language label for the file badge.
glob _LANG: dict = {
    "jac": "Jac", "py": "Python", "css": "CSS", "toml": "TOML",
    "md": "Markdown", "json": "JSON", "svg": "SVG", "sh": "Shell",
    "ts": "TypeScript", "tsx": "TSX", "js": "JavaScript", "jsx": "JSX",
    "zig": "Zig", "html": "HTML", "txt": "Text", "yml": "YAML", "yaml": "YAML"
};

"""Pretty language label for a file path's badge."""
def _lang_label(rel: str) -> str {
    if rel.endswith(".jac") {
        return "Jac";
    }
    ext = rel.rsplit(".", 1)[-1].lower() if "." in os.path.basename(rel) else "";
    return _LANG.get(ext, ext.upper() or "Text");
}

"""The canonical Jac lexer for any `*.jac`; Pygments' best guess otherwise."""
def _lexer_for(path: str) -> any {
    if path.endswith(".jac") {
        return JacLexer(stripnl=False);
    }
    try {
        return get_lexer_for_filename(path, stripnl=False);
    } except ClassNotFound {
        if path.endswith(".svg") {
            return get_lexer_by_name("xml", stripnl=False);
        }
        return TextLexer(stripnl=False);
    }
}

"""Repo-tracked + untracked-non-ignored paths, relative to ROOT, sorted.

`--others --exclude-standard` means a freshly-added source file shows up
immediately while node_modules / .jac build output stay excluded."""
@lru_cache(maxsize=1)
def _git_files() -> tuple {
    rels: list = [];
    try {
        out = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=ROOT, capture_output=True, text=True, timeout=15
        );
        rels = out.stdout.splitlines();
    } except Exception {
        rels = [];
    }
    keep: list = [];
    for raw in rels {
        rel = raw.strip();
        if not rel {
            continue;
        }
        if "__pycache__" in rel or os.path.basename(rel).startswith(".") {
            continue;
        }
        ext = os.path.splitext(rel)[1].lower();
        if ext in _BINARY_EXTS {
            continue;
        }
        if not os.path.isfile(os.path.join(ROOT, rel)) {
            continue;
        }
        keep.append(rel);
    }
    return tuple(sorted(keep));
}

"""Flat, sorted list of viewable source files with light metadata."""
def list_files() -> list {
    files: list = [];
    for rel in _git_files() {
        try {
            size = os.path.getsize(os.path.join(ROOT, rel));
        } except OSError {
            continue;
        }
        files.append({
            "path": rel,
            "name": os.path.basename(rel),
            "dir": os.path.dirname(rel),
            "lang": _lang_label(rel),
            "size": size
        });
    }
    return files;
}

"""Empty file payload carrying an error message."""
def _blank(rel: str, err: str) -> dict {
    return {
        "path": rel, "lang": "", "html": "", "lines": 0,
        "raw": "", "truncated": False, "error": err
    };
}

"""Read + highlight one file. Sandboxed to the repo's own file set."""
def get_file(rel: str) -> dict {
    path = rel.lstrip("/").replace("\\", "/");
    if path not in _git_files() {
        return _blank(path, "File not found.");
    }
    full = os.path.realpath(os.path.join(ROOT, path));
    if os.path.commonpath([full, os.path.realpath(ROOT)]) != os.path.realpath(ROOT) {
        return _blank(path, "Invalid path.");
    }
    try {
        data = open(full, encoding="utf-8", errors="replace").read();
    } except OSError {
        return _blank(path, "Could not read file.");
    }
    truncated = len(data) > MAX_BYTES;
    if truncated {
        data = data[:MAX_BYTES] + "\n... (truncated)\n";
    }
    data = data.rstrip("\n");
    # Count from the rendered HTML so the client's line gutter lines up exactly
    # (newlines render as line breaks inside <pre> regardless of span nesting).
    html = highlight(data, _lexer_for(path), HtmlFormatter(nowrap=True)).rstrip("\n");
    return {
        "path": path,
        "lang": _lang_label(path),
        "html": html,
        "lines": html.count("\n") + 1,
        "raw": data,
        "truncated": truncated,
        "error": ""
    };
}
