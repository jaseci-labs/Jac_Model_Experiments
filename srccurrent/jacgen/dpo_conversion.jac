"""Generate conversion DPO pairs (Recipe 3: de-Python-ification).

A DPO pair needs an idiomatic `chosen`, which only the hand/mined-crafted core
in sft.jsonl provides. For each such example this pairs the idiomatic Jac
(chosen) against a `jac py2jac` transpile of the SAME Python (rejected) -- the
literal Python-shaped Jac the finetune must learn to avoid. The mechanical
transpile IS the adversarial negative, so this scales automatically to the size
of the idiomatic core (not to the 1500-example volume tier, which has no
idiomatic partner).

Run from the repo root:
    jac run srccurrent/jacgen/dpo_conversion.jac
"""

import json;
import subprocess;
import tempfile;
import from corpus { transpile, sanitize_transpile }
import from writer { append_jsonl }
import from pathlib { Path }
import from typing { Any }


def compiles(jac_code: str) -> bool {
    with tempfile.TemporaryDirectory() as d {
        fp = Path(d) / "r.jac";
        fp.write_text(jac_code);
        try {
            proc = subprocess.run(["jac", "check", str(fp)], capture_output=True, text=True, timeout=30);
        } except (subprocess.SubprocessError, OSError) {
            return False;
        }
        return proc.returncode == 0;
    }
}

glob SFT_PATH: str = "dataset/conversion/sft.jsonl";
glob DPO_PATH: str = "dataset/conversion/dpo.jsonl";


def extract_python(user: str) -> str {
    after: str = user.split("```python", 1)[1];
    return after.rsplit("```", 1)[0].strip();
}


def extract_idiomatic(assistant: str) -> str {
    # the idiomatic Jac, minus the demo `with entry` block, so chosen and
    # rejected are compared at the same scope (the translation only)
    block: str = assistant.split("```jac", 1)[1].rsplit("```", 1)[0];
    return block.split("with entry", 1)[0].strip();
}


with entry {
    Path(DPO_PATH).parent.mkdir(parents=True, exist_ok=True);
    open(DPO_PATH, "w").close();

    p = Path(SFT_PATH);
    made: int = 0;
    skipped: int = 0;
    for line in p.read_text().split("\n") {
        if line.strip() == "" { continue; }
        rec: dict[str, Any] = json.loads(line);
        msgs: list[dict] = rec["messages"];
        user: dict[str, str] = msgs[0];
        assistant: dict[str, str] = msgs[1];
        prompt: str = user["content"];

        py: str = extract_python(prompt);
        chosen_jac: str = extract_idiomatic(assistant["content"]);
        rejected_jac: str = sanitize_transpile(transpile(py));
        if rejected_jac.strip() == "" or chosen_jac == "" { skipped += 1; continue; }
        if not compiles(rejected_jac) { skipped += 1; continue; }

        meta: dict[str, str] = rec["meta"];
        append_jsonl(DPO_PATH, {
            "prompt": prompt,
            "chosen": "```jac\n" + chosen_jac + "\n```",
            "rejected": "```jac\n" + rejected_jac.strip() + "\n```",
            "meta": {
                "category": "conversion",
                "recipe": "R3",
                "generator": "claude-code-mcp+py2jac",
                "rejected_style": "python_shaped_jac_transpile",
                "difficulty": meta["difficulty"],
                "source": meta["source"]
            }
        });
        made += 1;
    }
    print("wrote " + str(made) + " DPO pairs -> " + DPO_PATH + "  (skipped " + str(skipped) + ")");
}
