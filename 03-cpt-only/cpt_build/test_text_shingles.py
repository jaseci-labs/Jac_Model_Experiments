from text_shingles import shingles


def test_shingles_basic_count():
    text = " ".join(f"word{i}" for i in range(20))
    s = shingles(text, n=14)
    assert len(s) == 20 - 14 + 1


def test_shingles_short_text_empty():
    assert shingles("only three words", n=14) == set()


def test_shingles_identical_text_full_overlap():
    text = " ".join(f"word{i}" for i in range(20))
    assert shingles(text, n=14) == shingles(text, n=14)


def test_shingles_containment_direction():
    # containment(a subset of b) should be 1.0 even though jaccard isn't
    small = " ".join(f"word{i}" for i in range(14))
    big = small + " " + " ".join(f"extra{i}" for i in range(50))
    s_small, s_big = shingles(small, 14), shingles(big, 14)
    assert len(s_small & s_big) / len(s_small) == 1.0
