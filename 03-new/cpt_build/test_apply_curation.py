from apply_curation import apply_curation


def _row(cid, weight=1):
    return {"text": "x", "meta": {"chunk_id": cid, "upsample_weight": weight}}


def test_keep_passes_through_unchanged():
    rows = [_row("a")]
    out = apply_curation(rows, {"a": {"verdict": "keep", "reason": "fine"}})
    assert out == rows


def test_drop_removes_row():
    rows = [_row("a"), _row("b")]
    out = apply_curation(rows, {"a": {"verdict": "drop", "reason": "boilerplate"}})
    assert [r["meta"]["chunk_id"] for r in out] == ["b"]


def test_upweight_multiplies_weight():
    rows = [_row("a", weight=1)]
    out = apply_curation(rows, {"a": {"verdict": "upweight", "reason": "core concept", "weight": 3.0}})
    assert out[0]["meta"]["upsample_weight"] == 3


def test_missing_chunk_id_defaults_to_keep():
    rows = [_row("a")]
    out = apply_curation(rows, {})
    assert out == rows


def test_upweight_default_multiplier_is_two():
    rows = [_row("a", weight=1)]
    out = apply_curation(rows, {"a": {"verdict": "upweight", "reason": "core"}})
    assert out[0]["meta"]["upsample_weight"] == 2
