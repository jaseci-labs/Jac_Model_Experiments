import json
from pathlib import Path

from merge_curation_batches import merge_batches


def test_merge_combines_disjoint_batches(tmp_path):
    b1 = tmp_path / "batch1.json"
    b1.write_text(json.dumps([{"chunk_id": "a", "verdict": "keep", "reason": "x"}]))
    b2 = tmp_path / "batch2.json"
    b2.write_text(json.dumps([{"chunk_id": "b", "verdict": "drop", "reason": "y"}]))
    merged = merge_batches([b1, b2])
    assert merged == {"a": {"verdict": "keep", "reason": "x"},
                       "b": {"verdict": "drop", "reason": "y"}}


def test_merge_raises_on_duplicate_chunk_id_with_conflicting_verdict(tmp_path):
    b1 = tmp_path / "batch1.json"
    b1.write_text(json.dumps([{"chunk_id": "a", "verdict": "keep", "reason": "x"}]))
    b2 = tmp_path / "batch2.json"
    b2.write_text(json.dumps([{"chunk_id": "a", "verdict": "drop", "reason": "y"}]))
    try:
        merge_batches([b1, b2])
        assert False, "should have raised"
    except ValueError as e:
        assert "a" in str(e)


def test_merge_allows_same_verdict_with_different_reason_text(tmp_path):
    b1 = tmp_path / "batch1.json"
    b1.write_text(json.dumps([{"chunk_id": "a", "verdict": "keep", "reason": "x"}]))
    b2 = tmp_path / "batch2.json"
    b2.write_text(json.dumps([{"chunk_id": "a", "verdict": "keep", "reason": "different wording"}]))
    merged = merge_batches([b1, b2])
    assert merged["a"]["verdict"] == "keep"


def test_merge_allows_same_upweight_and_weight_with_different_reason_text(tmp_path):
    b1 = tmp_path / "batch1.json"
    b1.write_text(json.dumps([{"chunk_id": "a", "verdict": "upweight", "weight": 3.0, "reason": "x"}]))
    b2 = tmp_path / "batch2.json"
    b2.write_text(json.dumps([{"chunk_id": "a", "verdict": "upweight", "weight": 3.0, "reason": "different wording"}]))
    merged = merge_batches([b1, b2])
    assert merged["a"]["verdict"] == "upweight"
    assert merged["a"]["weight"] == 3.0


def test_merge_still_raises_on_same_verdict_different_weight(tmp_path):
    b1 = tmp_path / "batch1.json"
    b1.write_text(json.dumps([{"chunk_id": "a", "verdict": "upweight", "weight": 2.0, "reason": "x"}]))
    b2 = tmp_path / "batch2.json"
    b2.write_text(json.dumps([{"chunk_id": "a", "verdict": "upweight", "weight": 3.0, "reason": "y"}]))
    try:
        merge_batches([b1, b2])
        assert False, "should have raised"
    except ValueError as e:
        assert "a" in str(e)


def test_merge_rejects_invalid_batch(tmp_path):
    b1 = tmp_path / "batch1.json"
    b1.write_text(json.dumps([{"chunk_id": "a", "verdict": "not-a-verdict", "reason": "x"}]))
    try:
        merge_batches([b1])
        assert False, "should have raised"
    except ValueError as e:
        assert "verdict" in str(e)
