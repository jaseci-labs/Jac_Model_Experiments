import json

from merge_questions_batches import merge_question_batches


def test_merge_combines_and_dedupes_by_id(tmp_path):
    b1 = tmp_path / "b1.json"
    b1.write_text(json.dumps([{"id": "q-walker-1", "question": "What is a walker?",
                                "source_chunk_id": "aaa"}]))
    b2 = tmp_path / "b2.json"
    b2.write_text(json.dumps([{"id": "q-node-1", "question": "What is a node?",
                                "source_chunk_id": "bbb"}]))
    merged = merge_question_batches([b1, b2])
    assert len(merged) == 2
    assert {q["id"] for q in merged} == {"q-walker-1", "q-node-1"}


def test_merge_rejects_invalid_batch(tmp_path):
    b1 = tmp_path / "b1.json"
    b1.write_text(json.dumps([{"id": "q-1", "question": "short"}]))  # missing source_chunk_id, too-short question
    try:
        merge_question_batches([b1])
        assert False, "should have raised"
    except ValueError:
        pass


def test_merge_sample_down_to_target_count(tmp_path):
    items = [{"id": f"q-{i}", "question": f"Question number {i} about Jac?",
              "source_chunk_id": f"c{i}"} for i in range(150)]
    b1 = tmp_path / "b1.json"
    b1.write_text(json.dumps(items))
    merged = merge_question_batches([b1], target_count=100, seed=42)
    assert len(merged) == 100
