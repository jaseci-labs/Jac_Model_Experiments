#!/usr/bin/env python3
"""Checkpoint 1 semantic MCQ: base qwen-q4 vs fused qwen-cpt-v1 on 20
Jac/OSP concept questions (questions.py). No compiler involved -- pure
concept-recognition test, isolated from syntax. Per design.md's eval design.
"""
import gc
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from questions import QUESTIONS

from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

ROOT = Path(__file__).resolve().parents[3]

LETTER_RE = re.compile(r"\b([ABCD])\b")


def build_prompt(q):
    opts = "\n".join(f"{k}) {v}" for k, v in q["options"].items())
    return (f"{q['q']}\n\n{opts}\n\n"
            f"Respond with only the single letter (A, B, C, or D) of the correct answer.")


def extract_letter(text):
    # last match, not first -- a model that reasons before answering despite
    # the instruction still usually settles on its final letter at the end.
    matches = LETTER_RE.findall(text.strip())
    return matches[-1] if matches else None


def run_model(model_id, path):
    print(f"\n{'='*70}\nLOADING {model_id} ({path})\n{'='*70}", flush=True)
    model, tok = load(str(ROOT / path))
    sampler = make_sampler(temp=0.0)
    results = []
    for q in QUESTIONS:
        msgs = [{"role": "user", "content": build_prompt(q)}]
        ptoks = tok.apply_chat_template(msgs, add_generation_prompt=True)
        text = generate(model, tok, ptoks, max_tokens=20, sampler=sampler, verbose=False)
        letter = extract_letter(text)
        correct = letter == q["correct"]
        results.append({"id": q["id"], "output": text, "letter": letter,
                        "expected": q["correct"], "correct": correct})
        print(f"  [{q['id']:22s}] answered={letter} expected={q['correct']} "
              f"{'OK' if correct else 'WRONG'}", flush=True)
    del model, tok
    gc.collect()
    return results


def main():
    base = run_model("qwen-q4", "models/qwen-q4")
    cpt = run_model("qwen-cpt-v1", "models/qwen-cpt-v1-fused-q4")

    out = {"base": base, "cpt": cpt}
    outp = ROOT / "03-new" / "results" / "cpt-v1" / "mcq_check.json"
    outp.write_text(json.dumps(out, indent=2))

    n = len(QUESTIONS)
    base_correct = sum(r["correct"] for r in base)
    cpt_correct = sum(r["correct"] for r in cpt)
    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    print(f"base qwen-q4:  {base_correct}/{n} ({100*base_correct/n:.0f}%)")
    print(f"cpt-v1:        {cpt_correct}/{n} ({100*cpt_correct/n:.0f}%)")
    print(f"delta:         {cpt_correct - base_correct:+d} questions")
    for b, c in zip(base, cpt):
        if b["correct"] != c["correct"]:
            print(f"  DIFF {b['id']}: base={b['correct']} cpt={c['correct']}")
    print(f"\nwrote {outp}")


if __name__ == "__main__":
    sys.exit(main())
