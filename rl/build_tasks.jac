"""Build GRPO tasks from authored RL drivers.

Each `rl/drivers/*.jac` is a complete, deterministic `jac run`-able file with one
unit's body wrapped in sentinels:

    # >>>HOLE id="..." instruction="..."
    <body statements>
    # <<<HOLE

For every driver this emits one GRPO record to `dataset/rl/tasks.jsonl`
(`{prompt, answer}`, the schema `mlx_lm_lora`'s GRPODataset reads) plus a sidecar
`dataset/rl/templates/<id>.jac` holding the file with the body replaced by the
literal `__HOLE__`. The reward function (rl/reward.py) splices a completion into
that sentinel and runs it. `expected_output` is captured by running the driver
as-authored (the real body is present), so it is the ground-truth stdout.
"""

import os;
import re;
import json;
import glob;
import subprocess;
import from pathlib { Path }

def run_jac(jac_code: str, timeout: int = 30) -> tuple[int, str, str] {
    # Mirror of sft_dpo/jacgen/writer.jac:run_jac — the behavioral gate.
    import tempfile;
    with tempfile.TemporaryDirectory() as d {
        f = Path(str(d)) / "snippet.jac";
        f.write_text(jac_code);
        try {
            # cwd=d so jac's `.jac/` persistence is fresh per run (a persistent
            # `root` graph would otherwise accumulate state across drivers).
            p = subprocess.run(
                ["jac", "run", str(f)], capture_output=True, text=True, timeout=timeout, cwd=str(d)
            );
            return (p.returncode, p.stdout, p.stderr);
        } except (subprocess.SubprocessError, OSError) as e {
            return (124, "", str(e));
        }
    }
}

def parse_hole(text: str) -> dict {
    # Returns {ok, id, instruction, indent, start, end} where start/end are line
    # indices of the >>>HOLE and <<<HOLE marker lines (inclusive).
    lines = text.split("\n");
    start = -1;
    end = -1;
    for (i, ln) in enumerate(lines) {
        if ">>>HOLE" in ln { start = i; }
        elif "<<<HOLE" in ln { end = i; }
    }
    if start < 0 or end < 0 or end <= start {
        return {"ok": False};
    }
    m = re.search(r'>>>HOLE\s+id="([^"]+)"\s+instruction="(.*)"', lines[start]);
    if not m {
        return {"ok": False};
    }
    indent = re.match(r"\s*", lines[start]).group(0);
    return {
        "ok": True,
        "id": str(m.group(1)),
        "instruction": str(m.group(2)),
        "indent": indent,
        "start": start,
        "end": end,
        "lines": lines
    };
}

def replace_region(lines: list, start: int, end: int, repl_lines: list) -> str {
    out = lines[:start] + repl_lines + lines[end + 1:];
    return "\n".join(out);
}

with entry {
    drivers = sorted(glob.glob("rl/drivers/*.jac"));
    os.makedirs("dataset/rl/templates", exist_ok=True);
    tasks_path = "dataset/rl/tasks.jsonl";
    open(tasks_path, "w").close();   # truncate — full rebuild each run

    built = 0;
    failed = 0;
    for path in drivers {
        text = Path(path).read_text();
        info = parse_hole(text);
        if not info["ok"] {
            print("SKIP (no valid HOLE): " + path);
            failed += 1;
            continue;
        }
        # ground-truth output: run the driver exactly as authored
        (code, out, err) = run_jac(text);
        if code != 0 or not out.strip() {
            print("FAIL (driver did not run clean): " + path + "  exit=" + str(code));
            print("   " + err.strip()[:200]);
            failed += 1;
            continue;
        }

        lines = info["lines"];
        indent = info["indent"];
        # template: body region -> a single __HOLE__ sentinel line
        template = replace_region(lines, info["start"], info["end"], [indent + "__HOLE__"]);
        # prompt view: body region -> a TODO placeholder
        prompt_view = replace_region(
            lines, info["start"], info["end"],
            [indent + "# TODO: write the body for this unit"]
        );
        prompt = (
            info["instruction"]
            + "\n\nComplete the Jac file below: replace the `# TODO` line with the"
            + " correct Jac code for that unit. Return only the replacement code,"
            + " in a ```jac code block.\n\n```jac\n" + prompt_view + "\n```"
        );

        Path("dataset/rl/templates/" + info["id"] + ".jac").write_text(template);
        # reference body (between the sentinels) -> sidecar, used by the reward's
        # dense similarity-shaping term (NOT shown to the model; reward-side only).
        ref_body = "\n".join(lines[info["start"] + 1:info["end"]]);
        os.makedirs("dataset/rl/refbodies", exist_ok=True);
        Path("dataset/rl/refbodies/" + info["id"] + ".txt").write_text(ref_body);
        rec = {
            "prompt": prompt,
            "answer": json.dumps({
                "id": info["id"],
                "expected_output": out,
                "timeout": 20
            })
        };
        with open(tasks_path, "a") as fh {
            fh.write(json.dumps(rec) + "\n");
        }
        built += 1;
        print("OK  " + info["id"] + "  (" + str(len(out.strip().split(chr(10)))) + " output lines)");
    }
    print("=== built " + str(built) + " tasks, " + str(failed) + " failed -> " + tasks_path + " ===");
}
