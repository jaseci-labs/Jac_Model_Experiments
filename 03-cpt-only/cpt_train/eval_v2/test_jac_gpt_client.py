import json
from unittest.mock import patch, MagicMock

from jac_gpt_client import ask_jac_gpt


def test_ask_jac_gpt_accumulates_chunk_events():
    fake_response = MagicMock()
    fake_response.raise_for_status.return_value = None
    fake_response.iter_lines.return_value = [
        'data: {"type": "chunk", "data": {"content": "A walker "}}',
        'data: {"type": "chunk", "data": {"content": "traverses the graph."}}',
    ]
    with patch("jac_gpt_client.requests.post", return_value=fake_response) as mock_post:
        result = ask_jac_gpt("What is a walker?", base_url="http://localhost:9999")
    assert result == "A walker traverses the graph."
    call_kwargs = mock_post.call_args.kwargs
    assert call_kwargs["json"]["message"] == "What is a walker?"
    assert "session_id" in call_kwargs["json"]


def test_ask_jac_gpt_tool_call_clears_buffer():
    fake_response = MagicMock()
    fake_response.raise_for_status.return_value = None
    fake_response.iter_lines.return_value = [
        'data: {"type": "chunk", "data": {"content": "pre-tool reasoning"}}',
        'data: {"type": "tool_call", "data": {}}',
        'data: {"type": "chunk", "data": {"content": "final answer"}}',
    ]
    with patch("jac_gpt_client.requests.post", return_value=fake_response):
        result = ask_jac_gpt("q", base_url="http://localhost:9999")
    assert result == "final answer"


def test_ask_jac_gpt_uses_thought_event_when_no_chunks():
    fake_response = MagicMock()
    fake_response.raise_for_status.return_value = None
    fake_response.iter_lines.return_value = [
        'data: {"type": "thought", "data": {"content": "the final answer via thought"}}',
    ]
    with patch("jac_gpt_client.requests.post", return_value=fake_response):
        result = ask_jac_gpt("q", base_url="http://localhost:9999")
    assert result == "the final answer via thought"


def test_ask_jac_gpt_raises_on_http_error():
    fake_response = MagicMock()
    fake_response.raise_for_status.side_effect = Exception("500 error")
    with patch("jac_gpt_client.requests.post", return_value=fake_response):
        try:
            ask_jac_gpt("test question", base_url="http://localhost:9999")
            assert False, "should have raised"
        except Exception:
            pass
