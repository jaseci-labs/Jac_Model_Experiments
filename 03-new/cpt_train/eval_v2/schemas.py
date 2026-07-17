"""Validates Fable's raw JSON output before it's trusted. Fable is a
subagent call, not a type-checked function -- this is the boundary where its
output either becomes a real artifact or gets rejected and re-prompted."""

VALID_VERDICTS = {"keep", "drop", "upweight"}


def validate_curation_batch(batch: list) -> list:
    errors = []
    for i, item in enumerate(batch):
        if "chunk_id" not in item or not item["chunk_id"]:
            errors.append(f"item {i}: missing chunk_id")
            continue
        cid = item["chunk_id"]
        if item.get("verdict") not in VALID_VERDICTS:
            errors.append(f"chunk {cid}: verdict must be one of {VALID_VERDICTS}, got {item.get('verdict')!r}")
        if not item.get("reason"):
            errors.append(f"chunk {cid}: missing reason")
    return errors


def validate_questions_batch(batch: list) -> list:
    errors = []
    for i, item in enumerate(batch):
        for field in ("id", "question", "source_chunk_id"):
            if not item.get(field):
                errors.append(f"item {i}: missing {field}")
        if item.get("question") and len(item["question"]) < 10:
            errors.append(f"item {i}: question suspiciously short: {item['question']!r}")
    return errors
