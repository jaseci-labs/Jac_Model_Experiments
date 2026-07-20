"""
Self-check for common.jac: run-tag guard, path helpers, example-key
derivation, and existing_keys resumability lookup.

Run with JAC_RUN_TAG=fresh set:
    JAC_RUN_TAG=fresh jac run check_common.jac
"""

import json;
import from pathlib { Path }
import from common { require_run_tag, data_root, out_path, example_key, existing_keys }

with entry {
    tag = require_run_tag();
    assert tag == "fresh";

    assert data_root("fresh") == "model-experiments/04-cpt-sft/dataset/fresh";

    assert out_path("fresh", "clean_dataset", "code_gen") == "model-experiments/04-cpt-sft/dataset/fresh/clean_dataset/code_gen";

    assert example_key("abc123", "core_language_basics", 0) == "abc123__core_language_basics__0";

    scratch_path = "/tmp/jacgen2_test_existing_keys.jsonl";
    with open(scratch_path, 'w') as f {
        f.write(json.dumps({"meta": {"id": "key-one"}}) + "\n");
        f.write(json.dumps({"meta": {"id": "key-two"}}) + "\n");
    }
    keys = existing_keys(scratch_path);
    assert "key-one" in keys;
    assert "key-two" in keys;

    assert existing_keys("/tmp/jacgen2_does_not_exist.jsonl") == {};

    print("all common.jac checks passed");
}
