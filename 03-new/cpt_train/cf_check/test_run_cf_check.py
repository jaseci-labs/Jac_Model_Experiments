import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent))
import run_cf_check
from run_cf_check import run_model

# One trivial task so grade()'s subprocess grading stays fast -- we're not
# testing grading correctness here, only the load()/generate() call shape.
FAKE_TASKS = [{
    "id": "t0",
    "prompt": "write a function add(a, b) that returns a + b",
    "entry_point": "add",
    "tests": [((1, 2), 3)],
}]

FAKE_GENERATED_TEXT = "```python\ndef add(a, b):\n    return a + b\n```"


def _mock_model_and_tokenizer():
    fake_model = MagicMock()
    fake_tok = MagicMock()
    fake_tok.apply_chat_template.return_value = [1, 2, 3]
    return fake_model, fake_tok


def test_run_model_without_adapter_omits_adapter_path_kwarg():
    fake_model, fake_tok = _mock_model_and_tokenizer()
    with patch("run_cf_check.TASKS", FAKE_TASKS), \
         patch("run_cf_check.load", return_value=(fake_model, fake_tok)) as mock_load, \
         patch("run_cf_check.generate", return_value=FAKE_GENERATED_TEXT):
        run_model("qwen-q4", "models/qwen-q4")

    mock_load.assert_called_once()
    assert mock_load.call_args.args == (str(run_cf_check.ROOT / "models/qwen-q4"),)
    assert mock_load.call_args.kwargs == {}
    assert "adapter_path" not in mock_load.call_args.kwargs


def test_run_model_with_adapter_includes_adapter_path_kwarg():
    fake_model, fake_tok = _mock_model_and_tokenizer()
    adapter_checkpoint = "03-new/adapters/cpt-v2/0000570_adapters.safetensors"
    with patch("run_cf_check.TASKS", FAKE_TASKS), \
         patch("run_cf_check.load", return_value=(fake_model, fake_tok)) as mock_load, \
         patch("run_cf_check.generate", return_value=FAKE_GENERATED_TEXT):
        run_model("cpt-v2-leg", "models/qwen-q4", adapter_path=adapter_checkpoint)

    mock_load.assert_called_once()
    assert mock_load.call_args.args == (str(run_cf_check.ROOT / "models/qwen-q4"),)
    assert mock_load.call_args.kwargs == {
        "adapter_path": str(run_cf_check.ROOT / adapter_checkpoint),
    }
