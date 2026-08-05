# Dataset structure — 04-cpt-sft

Frozen, byte-identical across both arms (`dataset/fresh/releases/`). Both arms
train on MD5-verified identical copies; both eval on the same 1428-row
holdout (855 code-graded).

## SFT — 7 categories (9608 rows)

| category | n | example prompt |
|---|---|---|
| explanation | 2552 | "What are the two operators Jac adds on top of Python's standard comprehensions for working with collections of graph nodes and objects...?" |
| code_gen | 2301 | "I'm building an AI Day Planner web app in Jac and I need a login/signup form component. It should keep its own username, password, and error state..." |
| conversion | 1943 | "Convert this Python to idiomatic Jac: ```python from collections import deque def main(): # Binary tree as dict-of-children adjacency..." |
| documentation | 1369 | "Document this Jac code: ```jac \"\"\"Chess engine — all method implementations.\"\"\"..." |
| trajectory | 1225 | "I'm writing a terminal chess engine in Jac. I already have my declarations in `chess.jac`..." (multi-turn build-up) |
| debug | 209 | "This Jac code has a bug. Symptom: Clicking Sign Out does nothing -- the logout handler is never invoked..." |
| migration | 9 | "Migrate this deprecated Jac code to current syntax: ```jac node Item { has price: int; }..." |

Under the hood, `category` further splits into 28 finer `task_type`s — e.g.
`python_to_jac_function` 1920, `concept_recall` 1213, `docstring_authoring`
1078 — down to single-digit tails like `type_mismatch` n=1.

`migration` n=9 stands out — `docs/datagen/spec.md` flags this category's
original target as infeasible (inventory of distinct deprecated patterns ran
dry); the buffer category absorbed the shortfall.

## DPO — 5 axes (1652 pairs, chosen/rejected format)

| axis | n | example prompt |
|---|---|---|
| graph_native | 646 | "Write graph-native Jac code for this task -- use node/edge/walker traversal, not a dict-of-lists adjacency representation..." |
| idiomatic | 578 | "Write idiomatic, graph-native Jac code for this task -- use node/edge/walker OSP primitives..." |
| correctness | 388 | "Write correct Jac code that avoids this bug: Clicking Sign Out does nothing -- the logout handler is never invoked..." |
| auth_security | 38 | "Write secure Jac code with correct auth gating that avoids this bug: get_tasks is registered as a public endpoint..." |
| typing | 2 | "Write well-typed Jac code that avoids this bug: Type error: enum Tag is declared with base type `int` but its members hold str values..." |

`typing` n=2 — near-nonexistent, likely noise-level not a real axis at this
size.

Sources: `dataset/fresh/releases/sft_train.jsonl` (n=9608),
`dataset/fresh/releases/dpo_train.jsonl` (n=826, canonical DPO set —
`dpo_train_free.jsonl`/`dpo_train_llm.jsonl` are alt releases, combined
n=1652 above).
