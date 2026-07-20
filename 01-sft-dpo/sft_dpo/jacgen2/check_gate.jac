"""
Self-check for gate.jac: behavioral gate pass/fail, multi-file project gate,
v2 record schema completeness, and category-aware payload extraction.

Run with JAC_RUN_TAG set (make_example_v2 calls require_run_tag()):
    JAC_RUN_TAG=fresh jac run check_gate.jac
"""

import json;
import from gate { run_jac_project, behavioral_gate, make_example_v2, extract_payload }
import from common { example_key }

with entry {
    # behavioral_gate: pass case
    result = behavioral_gate("with entry { print(1+1); }", "2");
    assert result[0] == True;

    # behavioral_gate: fail case (wrong expected output)
    result = behavioral_gate("with entry { print(1+1); }", "3");
    assert result[0] == False;

    # run_jac_project: multiple files written into one tempdir, main_file runs
    files = {
        "main.jac": "with entry { print(1+1); }",
        "helper.jac": "def unused_helper() -> int { return 42; }"
    };
    (code, out, err) = run_jac_project(files, "main.jac");
    assert code == 0;
    assert out.strip() == "2";

    # make_example_v2: full record, all spec.md sec6 fields present
    seed_id = "seed-abc123";
    task_type = "core_language_basics";
    variant_idx = 0;
    messages = [
        {"role": "user", "content": "write hello world in jac"},
        {"role": "assistant", "content": "```jac\nwith entry { print(\"hello\"); }\n```"}
    ];
    rec = make_example_v2(
        "code_gen",
        task_type,
        "simple",
        messages,
        "behavioral",
        True,
        True,
        seed_id,
        1,
        variant_idx,
        "opus-api",
        "claude-opus-4-1-20250805"
    );

    required_fields = [
        "id", "category", "task_type", "complexity", "compiler_pass",
        "test_pass", "manually_reviewed", "generator", "generator_model_id",
        "gate_class", "variant_idx", "generation_date", "source_prompt_version",
        "context_bundle_version", "validator_version", "dataset_version",
        "run_tag", "seed_id"
    ];
    for field in required_fields {
        assert field in rec;
    }

    assert rec["id"] == example_key(seed_id, task_type, variant_idx);
    assert rec["manually_reviewed"] == False;
    assert rec["run_tag"] == "fresh";

    # round-trips through json without error
    roundtripped = json.loads(json.dumps(rec));
    assert roundtripped["id"] == rec["id"];

    # extra_meta merges in
    rec2 = make_example_v2(
        "code_gen", task_type, "simple", messages, "behavioral", True, True,
        seed_id, 1, variant_idx, "opus-api", "claude-opus-4-1-20250805",
        extra_meta={"custom_field": "custom_value"}
    );
    assert rec2["custom_field"] == "custom_value";

    # extract_payload: jac-fenced assistant turn -> fence markers stripped
    fenced_rec = {
        "messages": [
            {"role": "user", "content": "write hello world"},
            {"role": "assistant", "content": "```jac\nwith entry { print(\"hi\"); }\n```"}
        ]
    };
    payload = extract_payload(fenced_rec, "code_gen");
    assert "```" not in payload;
    assert "with entry" in payload;

    # extract_payload: prose (no fence) -> raw content returned
    prose_rec = {
        "messages": [
            {"role": "user", "content": "explain walkers"},
            {"role": "assistant", "content": "A walker traverses the graph, node by node."}
        ]
    };
    prose_payload = extract_payload(prose_rec, "explanation");
    assert prose_payload == "A walker traverses the graph, node by node.";

    # extract_payload: trajectory -> all assistant turns concatenated
    trajectory_rec = {
        "messages": [
            {"role": "user", "content": "fix this bug"},
            {"role": "assistant", "content": "first attempt output"},
            {"role": "user", "content": "still broken"},
            {"role": "assistant", "content": "second attempt output"}
        ]
    };
    traj_payload = extract_payload(trajectory_rec, "trajectory");
    assert "first attempt output" in traj_payload;
    assert "second attempt output" in traj_payload;

    print("all gate.jac checks passed");
}
