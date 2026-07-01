import os
_LANG = {"jac": "Jac", "py": "Python", "css": "CSS", "toml": "TOML", "md": "Markdown",
         "json": "JSON", "svg": "SVG", "sh": "Shell", "ts": "TypeScript", "tsx": "TSX",
         "js": "JavaScript", "jsx": "JSX", "zig": "Zig", "html": "HTML", "txt": "Text",
         "yml": "YAML", "yaml": "YAML"}

def lang_label(rel: str) -> str:
    if rel.endswith(".jac"):
        return "Jac"
    ext = rel.rsplit(".", 1)[-1].lower() if "." in os.path.basename(rel) else ""
    return _LANG.get(ext, ext.upper() or "Text")
