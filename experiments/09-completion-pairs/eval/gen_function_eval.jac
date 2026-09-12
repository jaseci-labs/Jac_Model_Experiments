"""Generate samples for jac-data-gen's function eval v1 (evals/function/v1).

Reads the PUBLIC problems file only (prompt + id; hidden tests never enter this
process) and writes one greedy sample per problem in the shape jac-data-gen's
own grader (scripts/eval/eval_jac.py) expects:
  completion tasks  -> {"completion": raw}   the grader prepends the task prefix
  translation tasks -> {"output": raw}       the grader extracts one ```jac block or takes raw text
Raw model text is written unmodified: following the prompt's format rules
(e.g. "no Markdown fences" for completion) is part of the task.
Same chat template + mlx_lm.batch_generate mechanism as eval_functional.jac.
Resumable: ids already in JAC_FN_SAMPLES_OUT are skipped.

    JAC_EVAL_MODEL=models/qwen-q4 [JAC_EVAL_ADAPTER=path] \
      JAC_FN_PROBLEMS=../jac-data-gen/evals/function/v1/public/test.jsonl \
      JAC_FN_SAMPLES_OUT=<dir>/samples.jsonl [JAC_EVAL_LIMIT=8] \
      [JAC_EVAL_BATCH_SIZE=32] [JAC_EVAL_MAX_TOKENS=1024] \
      jac run experiments/08-nitin-new2-ds/eval/gen_function_eval.jac
"""

import json;
import os;
import from pathlib { Path }


def load_model(mp: str, adapter: str) -> tuple {
    import mlx_lm;
    if adapter != "" { return mlx_lm.load(mp, adapter_path=adapter); }  # jac:ignore[E1032]
    return mlx_lm.load(mp);  # jac:ignore[E1032]
}


with entry {
    import mlx_lm;

    problems = Path(os.environ["JAC_FN_PROBLEMS"]);
    out = Path(os.environ["JAC_FN_SAMPLES_OUT"]);
    mp: str = os.environ.get("JAC_EVAL_MODEL", "");
    adapter: str = os.environ.get("JAC_EVAL_ADAPTER", "");
    limit: int = int(os.environ.get("JAC_EVAL_LIMIT", "0"));
    batch_size: int = int(os.environ.get("JAC_EVAL_BATCH_SIZE", "32"));
    max_tokens: int = int(os.environ.get("JAC_EVAL_MAX_TOKENS", "1024"));

    rows: list[dict] = [];
    for line in problems.read_text().split("\n") {
        if line.strip() == "" { continue; }
        if limit > 0 and len(rows) >= limit { break; }
        rows.append(dict(json.loads(line)));
    }

    done: set = set();
    if out.exists() {
        for line in out.read_text().split("\n") {
            if line.strip() != "" { done.add(str(json.loads(line)["problem_id"])); }
        }
    }
    todo: list[dict] = [r for r in rows if str(r["id"]) not in done];
    print("function eval gen: " + str(len(rows)) + " problems, " + str(len(done)) + " already done, " + str(len(todo)) + " to generate");
    if len(todo) == 0 { exit(0); }

    (model, tok) = load_model(mp, adapter);
    out.parent.mkdir(parents=True, exist_ok=True);

    for batch_start in range(0, len(todo), batch_size) {
        batch: list[dict] = todo[batch_start:min(batch_start + batch_size, len(todo))];
        encoded_prompts: list[list[int]] = [];
        for rec in batch {
            msgs: list = [{"role": "user", "content": str(rec["prompt"])}];
            encoded_prompts.append(list(tok.apply_chat_template(msgs, add_generation_prompt=True)));
        }
        resp = mlx_lm.batch_generate(model, tok, encoded_prompts, max_tokens=max_tokens, verbose=False);  # jac:ignore[E1032]
        texts: list[str] = list(resp.texts);  # jac:ignore[E1032]
        with open(str(out), "a") as f {
            for i in range(len(batch)) {
                rec: dict = batch[i];
                key: str = "completion" if rec["task"] == "completion" else "output";
                f.write(json.dumps({"problem_id": rec["id"], "sample_id": 0, "task": rec["task"], key: texts[i]}) + "\n");
            }
        }
        print("  " + str(min(batch_start + batch_size, len(todo))) + "/" + str(len(todo)));
    }
}
