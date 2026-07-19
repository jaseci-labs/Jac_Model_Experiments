from shingle_dedup import find_near_duplicates


def _row(cid, text):
    return {"text": text, "meta": {"chunk_id": cid}}


def test_near_duplicate_detected():
    base = " ".join(f"word{i}" for i in range(20))
    rows = [_row("a", base), _row("b", base + " extra tail words here")]
    dups = find_near_duplicates(rows, threshold=0.5)
    ids = {(d["chunk_id_a"], d["chunk_id_b"]) for d in dups}
    assert ("a", "b") in ids or ("b", "a") in ids


def test_unrelated_rows_not_flagged():
    rows = [_row("a", " ".join(f"alpha{i}" for i in range(20))),
            _row("b", " ".join(f"beta{i}" for i in range(20)))]
    assert find_near_duplicates(rows, threshold=0.5) == []


def test_short_rows_skipped_no_crash():
    rows = [_row("a", "too short"), _row("b", "also short")]
    assert find_near_duplicates(rows, threshold=0.5) == []


def test_self_pairs_never_reported():
    base = " ".join(f"word{i}" for i in range(20))
    rows = [_row("a", base)]
    assert find_near_duplicates(rows, threshold=0.5) == []
