from build_cpt import make_chunk_id


def test_chunk_id_deterministic():
    a = make_chunk_id("docs/walkers.md", "Walkers", "A walker is...")
    b = make_chunk_id("docs/walkers.md", "Walkers", "A walker is...")
    assert a == b


def test_chunk_id_changes_with_text():
    a = make_chunk_id("docs/walkers.md", "Walkers", "A walker is...")
    b = make_chunk_id("docs/walkers.md", "Walkers", "A different walker...")
    assert a != b


def test_chunk_id_changes_with_file():
    a = make_chunk_id("docs/walkers.md", "Walkers", "same text")
    b = make_chunk_id("docs/nodes.md", "Walkers", "same text")
    assert a != b


def test_chunk_id_length():
    cid = make_chunk_id("f", "s", "t")
    assert len(cid) == 12
    int(cid, 16)  # hex


def test_chunk_id_same_80char_prefix_different_tail():
    """Regression for the real CPT-v2 build collision: 92/10341 rows shared
    a chunk_id because make_chunk_id only hashed text[:80]. Two rows with
    identical first-80-chars (e.g. a repeated '---' separator run, or
    near-identical short code snippets in the same file+section) but
    different content beyond char 80 must now get DIFFERENT chunk_ids --
    this would have failed under the old text[:80] behavior."""
    prefix = "-" * 80
    a = make_chunk_id("docs/changelog.md", "Changes", prefix + "\nfirst variant of the tail content")
    b = make_chunk_id("docs/changelog.md", "Changes", prefix + "\nsecond, completely different tail")
    assert a != b
