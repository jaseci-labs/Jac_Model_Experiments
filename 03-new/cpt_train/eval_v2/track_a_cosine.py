"""Track A: convergence scoring (design.md section 6.2). Local
sentence-transformer embeddings, no API cost. Answers "did CPT-v2 move
closer to jac-gpt's grounded answers than base/CPT-v1 did" -- capped at 1.0
= tying jac-gpt exactly, cannot show a win (that's Track B, Task 18).

Beyond the scores, main() persists every generated answer (base/cpt_v1/
cpt_v2 + oracle) to track_a_answers.json -- Track B's judge reuses the
cpt_v2 and oracle answers verbatim instead of regenerating them, and the
answers file doubles as the audit trail for both tracks. The oracle loop is
resume-safe: answers are written incrementally, so a crash mid-run (oracle
down, network) resumes where it left off instead of redoing 3x100
generations."""
import json
from pathlib import Path

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def score_answers(answers: dict, embed_fn) -> dict:
    """answers: {"oracle": str, "base": str, "cpt_v1": str, "cpt_v2": str}.
    embed_fn: callable(list[str]) -> np.ndarray of shape (n, dim) -- injected
    so tests don't need to load the real embedding model."""
    keys = list(answers.keys())
    vecs = embed_fn([answers[k] for k in keys])
    oracle_idx = keys.index("oracle")
    return {k: cosine_similarity(vecs[i], vecs[oracle_idx])
            for i, k in enumerate(keys) if k != "oracle"}


_EMBED_MODEL = None


def real_embed_fn(texts: list) -> np.ndarray:
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _EMBED_MODEL = SentenceTransformer("all-mpnet-base-v2")
    return _EMBED_MODEL.encode(texts)


MODELS = {
    "base": "models/qwen-q4",
    "cpt_v1": "models/qwen-cpt-v1-fused-q4",
    "cpt_v2": "models/qwen-cpt-v2-fused-q4",
}


def main():
    import argparse
    from jac_gpt_client import ask_jac_gpt
    from mlx_lm import generate, load

    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="03-new/cpt_train/eval_v2/questions.json")
    ap.add_argument("--answers", default="03-new/results/cpt-v2/track_a_answers.json",
                    help="incremental store of all generated + oracle answers (Track B reuses it)")
    ap.add_argument("--out", default="03-new/results/cpt-v2/track_a.json")
    args = ap.parse_args()

    questions = json.loads(Path(args.questions).read_text())
    answers_path = Path(args.answers)
    answers_path.parent.mkdir(parents=True, exist_ok=True)
    answers = json.loads(answers_path.read_text()) if answers_path.exists() else {}

    def save_answers():
        answers_path.write_text(json.dumps(answers, indent=2, ensure_ascii=False))

    # Phase 1: local model generations, one model in memory at a time.
    for name, path in MODELS.items():
        todo = [q for q in questions if name not in answers.get(q["id"], {})]
        if not todo:
            print(f"{name}: all {len(questions)} answers already present, skipping load")
            continue
        print(f"{name}: generating {len(todo)} answers ({path})")
        model, tok = load(path)
        for q in todo:
            msgs = [{"role": "user", "content": q["question"]}]
            ptoks = tok.apply_chat_template(msgs, add_generation_prompt=True)
            text = generate(model, tok, ptoks, max_tokens=300, verbose=False)
            answers.setdefault(q["id"], {})[name] = text
        del model, tok
        save_answers()

    # Phase 2: oracle answers (resume-safe -- saved after every question).
    for q in questions:
        if "oracle" in answers.get(q["id"], {}):
            continue
        answers.setdefault(q["id"], {})["oracle"] = ask_jac_gpt(q["question"])
        save_answers()
        print(f"oracle: {q['id']} done")

    # Phase 3: score.
    results = {}
    for q in questions:
        results[q["id"]] = score_answers(answers[q["id"]], real_embed_fn)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    for name in MODELS:
        mean = sum(r[name] for r in results.values()) / len(results)
        print(f"{name}: mean cosine-to-oracle = {mean:.3f}")


if __name__ == "__main__":
    main()
