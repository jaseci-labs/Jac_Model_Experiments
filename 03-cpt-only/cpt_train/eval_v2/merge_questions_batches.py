"""Merges Fable question-gen batches (design.md section 6.1) into one
questions.json, sampled down to ~100 with a fixed seed for reproducibility."""
import json
import random
from pathlib import Path

from schemas import validate_questions_batch


def merge_question_batches(batch_files: list, target_count: int = 100, seed: int = 42) -> list:
    all_questions = []
    seen_ids = set()
    for path in batch_files:
        batch = json.loads(Path(path).read_text())
        errors = validate_questions_batch(batch)
        if errors:
            raise ValueError(f"{path}: {'; '.join(errors)}")
        for item in batch:
            if item["id"] in seen_ids:
                continue
            seen_ids.add(item["id"])
            all_questions.append(item)

    if len(all_questions) <= target_count:
        return all_questions
    rng = random.Random(seed)
    return rng.sample(all_questions, target_count)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("batch_files", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--target-count", type=int, default=100)
    args = ap.parse_args()
    merged = merge_question_batches(args.batch_files, args.target_count)
    args.out.write_text(json.dumps(merged, indent=2))
    print(f"{len(merged)} questions -> {args.out}")


if __name__ == "__main__":
    main()
