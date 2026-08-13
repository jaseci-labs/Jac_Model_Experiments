"""Drop-in replacement for `python -m mlx_lm_lora.train` that fixes DPO's
chat-template bug, then delegates to mlx_lm_lora's own CLI entry point.

THE BUG
-------
`mlx_lm_lora.trainer.datasets.DPODataset.__init__` builds each preference
sequence as:

    chosen_messages = base_messages + [{"role": "assistant", "content": ...}]
    tokenizer.apply_chat_template(chosen_messages, add_generation_prompt=True)

`add_generation_prompt=True` is for an UNFINISHED conversation -- it appends the
opening tag of a new assistant turn so the model knows to start generating. But
the messages list here already ends with a COMPLETE assistant turn, so every
chosen AND rejected sequence gets a spurious `<|im_start|>assistant\\n` stitched
on AFTER the real `<|im_end|>`. Verified against this project's own tokenizer
(models/qwen-q4):

    add_generation_prompt=True   ...HELLO<|im_end|>\\n<|im_start|>assistant\\n
    add_generation_prompt=False  ...HELLO<|im_end|>\\n

Because the corruption is byte-identical on both sides of every pair, it does
NOT bias which side DPO prefers -- pairwise accuracy can still read 1.0, which
is exactly what the collapsed runs showed. What it DOES do is train the model
that a correctly-terminated answer is followed by re-opening a new turn, i.e.
it corrupts precisely the stop-cleanly behaviour that a compiles-or-doesn't
functional eval is most sensitive to. SFT is unaffected: it goes through
`mlx_lm.lora`, a different package that never calls this code path -- consistent
with SFT working while every DPO variant collapsed to ~1-2% by iter 20.

WHY A DRIVER AND NOT A PATCH
----------------------------
Same call this repo already made for CPT-v2's optimizer-state persistence
(`03-cpt-only/cpt_train/run_cpt_leg.py`; rationale in
`03-cpt-only/docs/cpt-2/design.md` section 4.2): our own script composing the
installed package's public API, never an edit to `.venv/`. It survives
`pip install --upgrade` and venv rebuilds by construction, and leaves every
other recipe in this repo on the stock code path.

The patch is applied to the module ATTRIBUTE
`mlx_lm_lora.trainer.datasets.DPODataset`. Verified safe: nothing in the package
imports `DPODataset` by name (only `create_dataset`, inside that same module,
references it -- as a module global resolved at call time), so rebinding the
attribute before `main()` runs is sufficient and total.

Guarded, not assumed: `_assert_upstream_still_buggy()` fails loudly if a future
version already passes `add_generation_prompt=False` (patch now redundant --
delete this driver) or if the class/signature moved (patch may no longer apply).
Follows run_cpt_leg.py's "fail at import time, never silently" property.

USAGE
-----
    python dpo_fixed_train.py --model ... --train --data ...   # same flags as
                                                               # `-m mlx_lm_lora.train`
    python dpo_fixed_train.py --verify-tokenization --model MODEL --data DIR
"""

import inspect
import sys
from typing import Dict, List

import mlx_lm_lora.trainer.datasets as _ds
from transformers import PreTrainedTokenizer


class FixedDPODataset:
    """`DPODataset` with `add_generation_prompt=False`.

    Byte-for-byte identical to upstream except for that flag -- the message
    assembly below is copied verbatim from the installed
    `mlx_lm_lora.trainer.datasets.DPODataset.__init__` (Goekdeniz-Guelmez,
    Apache-2.0) so that a future upstream change to the assembly shows up as a
    diff here rather than being silently overridden.
    """

    def __init__(
        self,
        data: List[Dict[str, str]],
        tokenizer: PreTrainedTokenizer,
        prompt_key: str = "prompt",
        chosen_key: str = "chosen",
        rejected_key: str = "rejected",
        system_key: str = "system",
    ):
        self._chosen_data = []
        self._rejected_data = []

        for d in data:
            messages = (
                [{"role": "system", "content": d[system_key]}]
                if system_key and system_key in d
                else []
            )
            messages.append({"role": "user", "content": d[prompt_key]})

            base_messages = messages.copy()
            chosen_messages = base_messages + [
                {"role": "assistant", "content": d[chosen_key]}
            ]
            rejected_messages = base_messages + [
                {"role": "assistant", "content": d[rejected_key]}
            ]

            # THE FIX: the assistant turn is already present and terminated, so
            # do NOT append a generation prompt for a turn that never comes.
            self._chosen_data.append(
                tokenizer.apply_chat_template(
                    chosen_messages, add_generation_prompt=False
                )
            )
            self._rejected_data.append(
                tokenizer.apply_chat_template(
                    rejected_messages, add_generation_prompt=False
                )
            )

    def __getitem__(self, idx: int):
        return {"chosen": self._chosen_data[idx], "rejected": self._rejected_data[idx]}

    def __len__(self):
        return len(self._chosen_data)

    def process(self, d):
        return d


def _assert_upstream_still_buggy() -> None:
    """Fail loudly if the installed package no longer matches what we patched."""
    orig = getattr(_ds, "DPODataset", None)
    if orig is None:
        raise RuntimeError(
            "mlx_lm_lora.trainer.datasets.DPODataset is gone -- this driver's "
            "patch target moved. Re-read the installed source before trusting "
            "any DPO run."
        )
    src = "".join(inspect.getsource(orig.__init__).split())  # strip all whitespace
    if "add_generation_prompt=True" not in src:
        raise RuntimeError(
            "Installed DPODataset no longer passes add_generation_prompt=True. "
            "Upstream may have fixed this -- verify, then DELETE this driver "
            "and go back to `python -m mlx_lm_lora.train`."
        )
    ours = set(inspect.signature(FixedDPODataset.__init__).parameters)
    theirs = set(inspect.signature(orig.__init__).parameters)
    if ours != theirs:
        raise RuntimeError(
            f"DPODataset.__init__ signature changed: upstream={sorted(theirs)} "
            f"ours={sorted(ours)}. Update FixedDPODataset before running."
        )


def apply_patch() -> None:
    _assert_upstream_still_buggy()
    _ds.DPODataset = FixedDPODataset


def _verify_tokenization(model_path: str, data_dir: str, n: int = 2) -> int:
    """Decode real training rows through both datasets and diff the tails.

    Uses `mlx_lm.utils.load_tokenizer` -- the SAME loader `mlx_lm_lora.train`
    uses -- not a bare `transformers.AutoTokenizer`. This matters: under
    transformers 5.x a raw AutoTokenizer's `apply_chat_template` returns a
    `BatchEncoding` dict (so `len()` is 2, the key count), while mlx_lm's
    `TokenizerWrapper` normalizes to a flat `list[int]`, which is what the
    trainer's batcher requires. Verifying through the wrong object would both
    misreport token counts and invent a crash the real pipeline never hits.
    """
    import json
    from pathlib import Path

    from mlx_lm.utils import load_tokenizer

    tok = load_tokenizer(Path(model_path))
    rows = []
    with open(Path(data_dir) / "train.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
            if len(rows) >= n:
                break

    stock_cls = _ds.DPODataset
    stock = stock_cls(data=rows, tokenizer=tok)
    fixed = FixedDPODataset(data=rows, tokenizer=tok)

    bad = 0
    for i in range(len(rows)):
        for side in ("chosen", "rejected"):
            s_txt = tok.decode(stock[i][side])
            f_txt = tok.decode(fixed[i][side])
            print(f"--- row {i} / {side}")
            print(f"    stock tail: {s_txt[-70:]!r}")
            print(f"    fixed tail: {f_txt[-70:]!r}")
            print(f"    stock tokens={len(stock[i][side])} fixed tokens={len(fixed[i][side])}")
            if not s_txt.endswith("<|im_start|>assistant\n"):
                print("    !! stock did NOT show the expected spurious tag")
                bad += 1
            if f_txt.endswith("<|im_start|>assistant\n"):
                print("    !! FIXED STILL HAS THE SPURIOUS TAG")
                bad += 1
            if not f_txt.endswith("<|im_end|>\n"):
                print("    !! fixed does not end on a clean <|im_end|>")
                bad += 1
            # the fixed sequence must be a strict prefix of the stock one
            if not s_txt.startswith(f_txt):
                print("    !! fixed is not a prefix of stock -- more than the tag changed")
                bad += 1
    print()
    print("VERIFY:", "PASS -- only the trailing generation prompt differs" if bad == 0
          else f"FAIL -- {bad} problem(s)")
    return 1 if bad else 0


def main() -> None:
    if "--verify-tokenization" in sys.argv:
        args = sys.argv[1:]
        model = args[args.index("--model") + 1]
        data = args[args.index("--data") + 1]
        _assert_upstream_still_buggy()
        sys.exit(_verify_tokenization(model, data))

    apply_patch()
    print(
        "[dpo_fixed_train] DPODataset patched: add_generation_prompt=False "
        "(no spurious trailing assistant tag)",
        flush=True,
    )
    from mlx_lm_lora.train import main as _upstream_main

    _upstream_main()  # parses sys.argv exactly as `python -m mlx_lm_lora.train` does


if __name__ == "__main__":
    main()
