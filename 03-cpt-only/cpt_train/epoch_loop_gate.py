"""Stop-loss decision for the CPT-v2 epoch loop (design.md section 4.3, the
floor/target/ceiling numbers the user approved). Pure function -- the actual
CF-check execution and leg orchestration live in Task 11/13, this is just the
decision given a leg number and whether that leg's CF-check passed 16/16."""


def decide_next_action(leg: int, cf_passed: bool, floor: int = 6, ceiling: int = 12) -> str:
    if leg >= ceiling:
        return "halt_keep_this" if cf_passed else "halt_keep_previous"
    if leg <= floor:
        return "continue"
    return "continue" if cf_passed else "halt_keep_previous"
