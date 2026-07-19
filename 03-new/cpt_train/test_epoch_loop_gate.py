from epoch_loop_gate import decide_next_action


def test_below_floor_continues_even_on_regression():
    assert decide_next_action(leg=3, cf_passed=False) == "continue"


def test_at_floor_continues_even_on_regression():
    assert decide_next_action(leg=6, cf_passed=False) == "continue"


def test_past_floor_pass_continues():
    assert decide_next_action(leg=7, cf_passed=True) == "continue"


def test_past_floor_regression_halts_keeping_previous():
    assert decide_next_action(leg=7, cf_passed=False) == "halt_keep_previous"


def test_at_target_pass_continues_toward_ceiling():
    assert decide_next_action(leg=8, cf_passed=True) == "continue"


def test_at_ceiling_pass_halts_keeping_this_leg():
    assert decide_next_action(leg=12, cf_passed=True) == "halt_keep_this"


def test_at_ceiling_regression_halts_keeping_previous():
    assert decide_next_action(leg=12, cf_passed=False) == "halt_keep_previous"


def test_custom_floor_ceiling():
    assert decide_next_action(leg=4, cf_passed=False, floor=4, ceiling=8) == "continue"
    assert decide_next_action(leg=5, cf_passed=False, floor=4, ceiling=8) == "halt_keep_previous"
