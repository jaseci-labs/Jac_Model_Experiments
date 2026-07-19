import numpy as np

from track_a_cosine import cosine_similarity, score_answers


def test_cosine_similarity_identical_vectors():
    v = np.array([1.0, 2.0, 3.0])
    assert abs(cosine_similarity(v, v) - 1.0) < 1e-9


def test_cosine_similarity_orthogonal_vectors():
    a, b = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    assert abs(cosine_similarity(a, b)) < 1e-9


def test_score_answers_uses_provided_embedder():
    def fake_embed(texts):
        # embedding = (length, has-word-oracle) so vectors genuinely differ
        return np.array([[len(t), 10.0 if "oracle" in t else 0.0] for t in texts])

    answers = {"oracle": "a longer oracle answer here",
               "base": "short", "cpt_v1": "a medium length answer",
               "cpt_v2": "a longer oracle answer here"}
    scores = score_answers(answers, embed_fn=fake_embed)
    assert set(scores) == {"base", "cpt_v1", "cpt_v2"}  # oracle scored against, never scored
    assert abs(scores["cpt_v2"] - 1.0) < 1e-6  # identical text -> identical vector -> 1.0
    assert scores["base"] < 1.0  # differing vector direction -> below 1.0
