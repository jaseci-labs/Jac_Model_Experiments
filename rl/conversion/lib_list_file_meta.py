import os
# _lang_label(rel) -> str (uses the _LANG extension map) provided elsewhere.
def _file_meta(rel: str, size: int) -> dict:
    return {
        "path": rel,
        "name": os.path.basename(rel),
        "dir": os.path.dirname(rel),
        "lang": _lang_label(rel),
        "size": size,
    }
