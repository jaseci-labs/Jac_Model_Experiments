def signature_phrase(signatures: int) -> str:
    plural = "s" if signatures != 1 else ""
    return f"{signatures} signature{plural}"
