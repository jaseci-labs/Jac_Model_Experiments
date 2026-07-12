"""RL driver (pure-lib): main.fire (fire a bullet along the current aim into a dead pool slot).

Origin: this_is_jac/raylib_shooter/web/main.jac (the fire logic, isolated from any
io/time/graph machinery). Deterministic: fixed camera yaw/pitch and player position
globs feed the deterministic Taylor-series trig helpers (jcos/jsin), and all printed
floats are rounded to 6 dp. The marked body is the unit the policy must regenerate; the
with entry block exercises it on fixed inputs and prints a deterministic projection.
"""

glob DEG2RAD: float = 0.017453292519943295,
     yaw: float = 30.0,
     pitch: float = 15.0,
     px: float = 1.0,
     py: float = 2.0,
     pz: float = 3.0;

obj Bullet {
    has x: float,
        y: float,
        z: float,
        vx: float,
        vy: float,
        vz: float,
        life: float;
}

"""cos via range-reduced Taylor series (good to ~1e-4 on the reduced range)."""
def jcos(a: float) -> float {
    twopi = 6.283185307179586;
    pi = 3.141592653589793;
    while a > pi {
        a -= twopi;
    }
    while a < -pi {
        a += twopi;
    }
    x2 = a * a;
    x4 = x2 * x2;
    x6 = x4 * x2;
    x8 = x4 * x4;
    x10 = x8 * x2;
    x12 = x8 * x4;
    return 1.0 - x2 / 2.0 + x4 / 24.0 - x6 / 720.0 + x8 / 40320.0 - x10 / 3628800.0 + x12 / 479001600.0;
}

"""sin(a) = cos(a - pi/2)."""
def jsin(a: float) -> float {
    return jcos(a - 1.5707963267948796);
}

def fire(bullets: list[Bullet]) -> None {
    __HOLE__
}

with entry {
    bullets = [
        Bullet(x=0.0, y=0.0, z=0.0, vx=0.0, vy=0.0, vz=0.0, life=5.0),
        Bullet(x=0.0, y=0.0, z=0.0, vx=0.0, vy=0.0, vz=0.0, life=0.0)
    ];
    fire(bullets);
    b = bullets[1];
    print(
        str(round(b.x, 6)) + " " + str(round(b.y, 6)) + " " + str(round(b.z, 6))
        + " " + str(round(b.vx, 6)) + " " + str(round(b.vy, 6)) + " "
        + str(round(b.vz, 6)) + " " + str(round(b.life, 6))
    );
}
