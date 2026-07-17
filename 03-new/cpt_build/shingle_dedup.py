"""Within-source near-duplicate candidate detection -- cheap pre-filter before
Fable curation (design.md section 3). Same containment style as build_cpt.py's
decontam() (overlap over one set's size, not symmetric Jaccard), but here
neither chunk is reliably the larger one -- both are same-scale chunks, unlike
decontam()'s rows-vs-holdout-items split -- so we use the smaller of the two
shingle sets as the denominator, which keeps containment symmetric-friendly
for same-scale chunks (see test_near_duplicate_detected)."""
# ponytail: O(n²) pairwise scan, revisit with LSH bucketing if a source exceeds ~20K rows
from text_shingles import shingles


def find_near_duplicates(rows: list, threshold: float = 0.5) -> list:
    shingle_sets = []
    for row in rows:
        s = shingles(row["text"])
        if s:
            shingle_sets.append((row["meta"]["chunk_id"], s))

    out = []
    for i in range(len(shingle_sets)):
        cid_a, s_a = shingle_sets[i]
        for j in range(i + 1, len(shingle_sets)):
            cid_b, s_b = shingle_sets[j]
            overlap = len(s_a & s_b)
            if not overlap:
                continue
            containment = overlap / min(len(s_a), len(s_b))
            if containment >= threshold:
                out.append({"chunk_id_a": cid_a, "chunk_id_b": cid_b,
                             "containment": round(containment, 3)})
    return out
