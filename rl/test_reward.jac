"""Self-checking test for the Jac GRPO reward (run: `jac run rl/test_reward.jac`).

Written as a `with entry` harness rather than `jac test` blocks (the latter is
flaky for standalone files in jaclang 0.16.1). Exits non-zero on any failure so
it works as a CI gate. Requires a built task set (`jac run rl/build_tasks.jac`).
"""

import os;
import re;
import json;
import sys;
import from pathlib { Path }
import from reward_logic { jac_behavioral }

def ref_body(task_id: str) -> str {
    # The original body the driver author wrote, between the HOLE sentinels.
    for p in Path("rl/drivers").glob("*.jac") {
        text = p.read_text();
        if ('id="' + task_id + '"') not in text {
            continue;
        }
        lines = text.split("\n");
        start = -1; end = -1;
        for (i, ln) in enumerate(lines) {
            if ">>>HOLE" in ln and ('id="' + task_id + '"') in ln { start = i; }
            elif "<<<HOLE" in ln and start >= 0 and end < 0 { end = i; }
        }
        if start >= 0 and end > start {
            return "\n".join(lines[start + 1:end]);
        }
    }
    return "";
}

with entry {
    tasks_path = "dataset/rl/tasks.jsonl";
    if not Path(tasks_path).exists() {
        print("MISSING " + tasks_path + " — run: jac run rl/build_tasks.jac");
        sys.exit(1);
    }
    tasks = [json.loads(l) for l in Path(tasks_path).read_text().split("\n") if l.strip()];
    fails: list[str] = [];

    # 1. each task's own reference body scores ~1.0
    for t in tasks {
        meta = json.loads(t["answer"]);
        tid = str(meta["id"]);
        body = ref_body(tid);
        if not body {
            fails.append("no ref body for " + tid);
            continue;
        }
        comp = "```jac\n" + body + "\n```";
        s = jac_behavioral(prompts=["p"], completions=[comp], answer=[t["answer"]])[0];
        if s < 0.85 {
            fails.append("perfect " + tid + " scored " + str(s) + " (want >=0.85)");
        }
    }

    # 2. noncompiling garbage scores low (only the small dense sim floor, no
    #    compile/run/output credit). Not 0 by design — the floor gives gradient.
    g = jac_behavioral(prompts=["p"], completions=["```jac\n@@@ not jac !!!\n```"], answer=[tasks[0]["answer"]])[0];
    if g >= 0.3 {
        fails.append("garbage scored " + str(g) + " (want < 0.3)");
    }

    # 3. compiles+runs but wrong output -> partial credit, below pass
    lib = [x for x in tasks if json.loads(x["answer"])["id"] == "lib_lang_label"];
    if lib {
        w = jac_behavioral(prompts=["p"], completions=['```jac\nreturn "WRONG";\n```'], answer=[lib[0]["answer"]])[0];
        if not (0.0 < w and w < 0.9) {
            fails.append("wrong-output scored " + str(w) + " (want 0<s<0.9)");
        }
    }

    # 4. batch length matches input
    n = len(tasks);
    batch = jac_behavioral(
        prompts=["p"] * n,
        completions=["```jac\nreturn 1;\n```"] * n,
        answer=[t["answer"] for t in tasks]
    );
    if len(batch) != n {
        fails.append("batch len " + str(len(batch)) + " != " + str(n));
    }

    if fails {
        print("FAIL (" + str(len(fails)) + "):");
        for f in fails {
            print("  - " + f);
        }
        sys.exit(1);
    }
    print("OK — reward tests passed (" + str(n) + " tasks)");
}
