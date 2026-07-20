"""
Self-check for llm.jac's wrapper logic: retry counting, budget enforcement,
and call logging.

There is no ANTHROPIC_API_KEY in this environment, so every real LLM call
attempt fails with an auth error -- that failure is the EXPECTED path here.
This check proves the wrapper handles it correctly (all attempts counted and
logged, a clean ok:False dict returned, never a raised exception out of a
single call) rather than proving any call succeeds.

Run with:
    JAC_RUN_TAG=fresh JAC_CALL_BUDGET=5 jac run check_llm_wrap.jac
"""

import json;
import from pathlib { Path }
import from common { data_root }
import from llm { call_with_retry, get_calls_made }

with entry {
    tag: str = "fresh";
    log_path: str = data_root(tag) + "/logs/generation/calls_" + tag + ".jsonl";

    # 1. A real call attempt with no API key: exhausts all 3 retries and
    #    returns a graceful ok:False dict (never raises).
    result: dict = call_with_retry("gen_text_opus", "say pong", "opus");
    assert result["ok"] == False;
    assert result["err"] != "";
    assert result["text"] == "";

    # 2. All 3 attempts were made and counted.
    assert get_calls_made() == 3;

    # 3. All 3 attempts were logged as failures for the right fn/model.
    with open(log_path, 'r') as f {
        lines: list = f.readlines();
    }
    assert len(lines) >= 3;
    last_three: list = lines[-3:];
    for line in last_three {
        rec: dict = json.loads(line.strip());
        assert rec["ok"] == False;
        assert rec["fn"] == "gen_text_opus";
        assert rec["model"] == "opus";
    }

    # 4. A bogus model name raises instead of returning ok:False.
    raised_bad_model: bool = False;
    try {
        call_with_retry("gen_text_opus", "hi", "not_a_real_model");
    } except Exception {
        raised_bad_model = True;
    }
    assert raised_bad_model;

    # 5. One more full call_with_retry (3 more failed attempts) pushes
    #    CALLS_MADE from 3 to 6, past the JAC_CALL_BUDGET=5 ceiling.
    result2: dict = call_with_retry("gen_text_fable", "say pong", "fable");
    assert result2["ok"] == False;
    assert get_calls_made() == 6;

    # 6. Budget is now exhausted (6 >= 5) -- the NEXT call must hard-stop by
    #    raising before making any further attempt, not just warn.
    raised_budget: bool = False;
    try {
        call_with_retry("gen_text_opus", "say pong", "opus");
    } except Exception {
        raised_budget = True;
    }
    assert raised_budget;
    assert get_calls_made() == 6;

    print("all llm.jac wrapper checks passed");
}
