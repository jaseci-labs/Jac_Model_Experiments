DEG2RAD = 0.017453292519943295
# Globs yaw, pitch, px, py, pz and helpers jcos()/jsin() provided elsewhere.
# bullets is a list of Bullet (has x, y, z, vx, vy, vz, life).
def fire(bullets: list) -> None:
    yaw_r = yaw * DEG2RAD
    pitch_r = pitch * DEG2RAD
    cp = jcos(pitch_r)
    dx = -jsin(yaw_r) * cp
    dy = jsin(pitch_r)
    dz = -jcos(yaw_r) * cp
    speed = 40.0
    i = 0
    while i < len(bullets):
        b = bullets[i]
        if b.life <= 0.0:
            b.x = px
            b.y = py
            b.z = pz
            b.vx = dx * speed
            b.vy = dy * speed
            b.vz = dz * speed
            b.life = 2.0
            return
        i += 1
