import json

from data_generation.artifacts import artifact_path, ensure_dataset_tree, write_json, write_jsonl


def test_ensure_dataset_tree_creates_task4_paths(tmp_path):
    ensure_dataset_tree(tmp_path)

    assert (tmp_path / "dataset/raw_output/code_gen").is_dir()
    assert (tmp_path / "dataset/clean_dataset/debug").is_dir()
    assert (tmp_path / "dataset/rejected/conversion").is_dir()
    assert (tmp_path / "dataset/review/explanation").is_dir()
    assert (tmp_path / "dataset/logs/generation").is_dir()
    assert (tmp_path / "dataset/logs/compiler").is_dir()
    assert (tmp_path / "dataset/logs/prompt_revisions").is_dir()
    assert (tmp_path / "dataset/logs/scale_decisions").is_dir()
    assert (tmp_path / "dataset/context").is_dir()


def test_artifact_path_keeps_storage_areas_separate(tmp_path):
    batch_id = "20260508-code_gen-001"

    raw_path = artifact_path(tmp_path, "raw_output", "code_gen", batch_id)
    clean_path = artifact_path(tmp_path, "clean_dataset", "code_gen", batch_id, suffix=".jsonl")
    scale_path = artifact_path(tmp_path, "logs", "scale_decisions", batch_id)

    assert raw_path == tmp_path / "dataset/raw_output/code_gen/20260508-code_gen-001.json"
    assert clean_path == tmp_path / "dataset/clean_dataset/code_gen/20260508-code_gen-001.jsonl"
    assert scale_path == tmp_path / "dataset/logs/scale_decisions/20260508-code_gen-001.json"
    assert len({raw_path.parent, clean_path.parent, scale_path.parent}) == 3


def test_write_json_and_jsonl_are_readable(tmp_path):
    json_path = tmp_path / "nested" / "record.json"
    jsonl_path = tmp_path / "records.jsonl"

    write_json(json_path, {"batch_id": "20260508-code_gen-001"})
    write_jsonl(jsonl_path, [{"id": "one"}, {"id": "two"}])

    assert json.loads(json_path.read_text()) == {"batch_id": "20260508-code_gen-001"}
    assert [json.loads(line) for line in jsonl_path.read_text().splitlines()] == [
        {"id": "one"},
        {"id": "two"},
    ]
