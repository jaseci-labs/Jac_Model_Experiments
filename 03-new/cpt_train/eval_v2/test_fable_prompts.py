from fable_prompts import build_curation_prompt


def test_prompt_includes_every_chunk_id():
    batch = [{"chunk_id": "aaa111", "text": "Walkers traverse the graph.", "meta": {"file": "walkers.md"}},
             {"chunk_id": "bbb222", "text": "License: MIT", "meta": {"file": "LICENSE.md"}}]
    prompt = build_curation_prompt(batch, near_dup_candidates=[])
    assert "aaa111" in prompt and "bbb222" in prompt


def test_prompt_includes_near_dup_candidates():
    batch = [{"chunk_id": "aaa111", "text": "x", "meta": {"file": "a.md"}}]
    dups = [{"chunk_id_a": "aaa111", "chunk_id_b": "ccc333", "containment": 0.9}]
    prompt = build_curation_prompt(batch, near_dup_candidates=dups)
    assert "ccc333" in prompt and "0.9" in prompt


def test_prompt_requests_json_output():
    prompt = build_curation_prompt([{"chunk_id": "a", "text": "x", "meta": {}}], [])
    assert "json" in prompt.lower()
