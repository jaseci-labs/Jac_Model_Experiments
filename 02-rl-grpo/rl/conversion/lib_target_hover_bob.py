# jsin(a) -> float provided elsewhere.
def hover_y(base_y: float, world_t: float, phase: float) -> float:
    return base_y + 0.4 * jsin(world_t * 1.5 + phase)
