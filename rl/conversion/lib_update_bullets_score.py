# Global `score`, helper respawn(t), and Bullet/Target objs provided elsewhere.
score = 0

def update_bullets(bullets: list, targets: list, step: float) -> None:
    global score
    bi = 0
    while bi < len(bullets):
        b = bullets[bi]
        if b.life > 0.0:
            b.x += b.vx * step
            b.y += b.vy * step
            b.z += b.vz * step
            b.life -= step
            ti = 0
            while ti < len(targets):
                t = targets[ti]
                if t.alive:
                    ddx = b.x - t.x
                    ddy = b.y - t.y
                    ddz = b.z - t.z
                    if ddx * ddx + ddy * ddy + ddz * ddz < 1.4:
                        t.alive = False
                        b.life = 0.0
                        score += 1
                        respawn(t)
                ti += 1
        bi += 1
