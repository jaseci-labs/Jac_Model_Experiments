from track_b_judge_prep import assign_order, build_judge_prompt


def test_assign_order_deterministic_per_question():
    a1 = assign_order("q-walker-1", seed=42)
    a2 = assign_order("q-walker-1", seed=42)
    assert a1 == a2  # same question, same seed -> same order every time


def test_assign_order_varies_across_questions():
    orders = {assign_order(f"q-{i}", seed=42) for i in range(20)}
    assert len(orders) == 2  # both "cpt_v2_first" and "oracle_first" should appear


def test_build_judge_prompt_never_reveals_source():
    prompt = build_judge_prompt(
        question="What is a walker?",
        ground_truth_passage="A walker is a traversal agent.",
        answer_a="Answer text A", answer_b="Answer text B")
    assert "cpt" not in prompt.lower() and "jac-gpt" not in prompt.lower() and "oracle" not in prompt.lower()
    assert "Answer A" in prompt and "Answer B" in prompt


def test_build_judge_prompt_includes_ground_truth():
    prompt = build_judge_prompt("q", "the ground truth passage text", "a1", "a2")
    assert "the ground truth passage text" in prompt
