# sh is a Shim (has proj: list, mv: list, mode: int); provided elsewhere.
def _setcur(sh, m: list) -> None:
    if sh.mode == 0x1701:
        sh.proj = m
    else:
        sh.mv = m
