from pygments.util import shebang_matches

def analyse_text(text: str) -> bool:
    return shebang_matches(text, r"pythonw?(3(\.\d)?)?") or "import " in text[:1000]
