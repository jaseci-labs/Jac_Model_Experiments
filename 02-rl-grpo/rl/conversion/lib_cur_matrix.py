# sh is a Shim (has proj: list, mv: list, mode: int); provided elsewhere.
def _cur(sh) -> list:
    return sh.proj if sh.mode == 0x1701 else sh.mv
