# _m_ident() -> flat 4x4 identity is provided elsewhere.
def m_translate(x: float, y: float, z: float) -> list:
    m = _m_ident()
    m[12] = x
    m[13] = y
    m[14] = z
    return m
