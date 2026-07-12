import os

_BINARY_EXTS = {
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".webp", ".bmp",
    ".woff", ".woff2", ".ttf", ".otf", ".eot",
    ".wasm", ".so", ".dylib", ".dll", ".a", ".o", ".bin",
    ".zip", ".gz", ".tar", ".pdf", ".mp4", ".mov", ".jbc", ".pyc",
}

def _should_skip(rel: str) -> bool:
    if "__pycache__" in rel or os.path.basename(rel).startswith("."):
        return True
    ext = os.path.splitext(rel)[1].lower()
    if ext in _BINARY_EXTS:
        return True
    return False
