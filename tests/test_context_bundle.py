from data_generation.context_bundle import build_default_context_bundle, load_context_bundle, write_context_bundle


def test_build_default_context_bundle_includes_task_sources_and_jac_guidance():
    bundle = build_default_context_bundle(
        docs_context="Jac Coding Agent strategy",
        dataset_foundation="Dataset Foundation policy",
        jac_guidance={
            "jac://guide/pitfalls": "Jac is NOT Python; semicolons are required.",
            "jac://guide/patterns": "walker Explorer { }",
        },
    )

    assert "jac-context-v1" in bundle
    assert "Jac Coding Agent strategy" in bundle
    assert "Dataset Foundation policy" in bundle
    assert "code_gen" in bundle
    assert "debug" in bundle
    assert "conversion" in bundle
    assert "Jac is NOT Python" in bundle
    assert "walker Explorer" in bundle


def test_write_and_load_context_bundle_round_trip(tmp_path):
    path = write_context_bundle(
        tmp_path,
        content="context bundle contents",
        version="jac-context-v9",
    )

    assert path == tmp_path / "dataset/context/jac-context-v9.md"
    assert load_context_bundle(tmp_path, version="jac-context-v9") == "context bundle contents"
