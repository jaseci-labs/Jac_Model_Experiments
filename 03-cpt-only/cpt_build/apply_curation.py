"""Apply Fable + shingle-dedup curation verdicts (03-new/docs/cpt-2/design.md
section 3) before packing. Deterministic, no LLM calls here -- the judgment
already happened when curation.json was produced (Task 7)."""


def apply_curation(rows: list, curation: dict) -> list:
    out = []
    for row in rows:
        cid = row["meta"].get("chunk_id")
        verdict = curation.get(cid, {}).get("verdict", "keep")
        if verdict == "drop":
            continue
        if verdict == "upweight":
            mult = curation[cid].get("weight", 2.0)
            row = {**row, "meta": {**row["meta"],
                   "upsample_weight": int(row["meta"]["upsample_weight"] * mult)}}
        out.append(row)
    return out
