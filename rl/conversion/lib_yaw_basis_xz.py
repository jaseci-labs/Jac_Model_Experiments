DEG2RAD = 0.017453292519943295
# jcos(a)/jsin(a) -> float helpers provided elsewhere.
def yaw_basis(yaw: float) -> list:
    yaw_r = yaw * DEG2RAD
    fwd_x = -jsin(yaw_r)
    fwd_z = -jcos(yaw_r)
    rgt_x = jcos(yaw_r)
    rgt_z = -jsin(yaw_r)
    return [fwd_x, fwd_z, rgt_x, rgt_z]
