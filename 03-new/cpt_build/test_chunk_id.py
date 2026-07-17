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
