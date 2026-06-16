"""Thin shim — the reward LOGIC lives in rl/reward_logic.jac (all Jac).

mlx_lm_lora loads the reward via `importlib.spec_from_file_location`, which
requires a `.py` entrypoint, so a pure-`.jac` reward can't be passed to
`--reward-functions-file` directly. This shim imports the Jac module, whose
`with entry` block registers `jac_behavioral` into mlx_lm_lora's reward
registry. Nothing here is reward logic — only the bridge.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jaclang  # noqa: F401  -> activates the .jac import hook
import reward_logic  # noqa: F401,E402  -> its `with entry` registers jac_behavioral

# Re-export so `from reward import jac_behavioral` also works (tests / smoke).
jac_behavioral = reward_logic.jac_behavioral
