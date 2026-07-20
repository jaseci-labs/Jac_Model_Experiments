"""Track B: blind pairwise judge prep (design.md section 6.3). Order is
randomized per-question with a fixed seed (reproducible, not true
nondeterminism) so a Sonnet judge can't develop a positional prior. The judge
is never told which answer is CPT-v2's and which is jac-gpt's."""
import random


def assign_order(question_id: str, seed: int = 42) -> str:
    rng = random.Random(f"{seed}:{question_id}")
    return "cpt_v2_first" if rng.random() < 0.5 else "oracle_first"


def build_judge_prompt(question: str, ground_truth_passage: str, answer_a: str, answer_b: str) -> str:
    return f"""You are judging two candidate answers to a Jac programming language question,
against the ground-truth documentation passage below. You do NOT know which system
produced which answer -- judge purely on correctness and completeness against the
passage.

Question: {question}

Ground-truth passage:
{ground_truth_passage}

Answer A:
{answer_a}

Answer B:
{answer_b}

Respond with ONLY a JSON object in this exact shape:
{{"winner": "A"|"B"|"tie", "justification": "one sentence, grounded in the passage"}}"""


def prep_question(q: dict, cpt_v2_answer: str, oracle_answer: str, ground_truth_passage: str, seed: int = 42) -> dict:
    order = assign_order(q["id"], seed)
    if order == "cpt_v2_first":
        answer_a, answer_b = cpt_v2_answer, oracle_answer
    else:
        answer_a, answer_b = oracle_answer, cpt_v2_answer
    return {
        "id": q["id"], "order": order,
        "prompt": build_judge_prompt(q["question"], ground_truth_passage, answer_a, answer_b),
    }
