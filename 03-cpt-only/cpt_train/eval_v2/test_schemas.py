from schemas import validate_curation_batch


def test_valid_batch_no_errors():
    batch = [{"chunk_id": "abc123def456", "verdict": "keep", "reason": "core content"}]
    assert validate_curation_batch(batch) == []


def test_invalid_verdict_flagged():
    batch = [{"chunk_id": "abc123def456", "verdict": "maybe", "reason": "unsure"}]
    errs = validate_curation_batch(batch)
    assert len(errs) == 1 and "verdict" in errs[0]


def test_missing_reason_flagged():
    batch = [{"chunk_id": "abc123def456", "verdict": "drop"}]
    errs = validate_curation_batch(batch)
    assert len(errs) == 1 and "reason" in errs[0]


def test_missing_chunk_id_flagged():
    batch = [{"verdict": "keep", "reason": "x"}]
    errs = validate_curation_batch(batch)
    assert len(errs) == 1 and "chunk_id" in errs[0]


def test_upweight_without_weight_gets_default_flagged_as_warning_not_error():
    batch = [{"chunk_id": "abc123def456", "verdict": "upweight", "reason": "core concept"}]
    assert validate_curation_batch(batch) == []  # weight is optional, apply_curation defaults it
