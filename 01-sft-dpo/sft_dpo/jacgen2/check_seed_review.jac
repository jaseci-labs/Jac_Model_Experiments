"""
Self-check for gen_seed_review.jac's response-parsing logic.

There is no ANTHROPIC_API_KEY in this environment, so a real Fable verdict
can't be produced -- this proves the PARSER handles all three shapes a real
response (or a failed call) can take: a clean ACCEPT, a clean REJECT, and
garbage/unparseable text. review_candidate's call-failure branch is proven
separately by the real end-to-end run (seed_scrape.jac then
gen_seed_review.jac against one small cloned repo, JAC_CALL_BUDGET set
small) -- see task-6-brief.md for the exact invocation and observed output.

NOTE on duplicating parse_review_response instead of importing it:
gen_seed_review.jac has a real, file-writing `with entry` block (reads
tier4_candidates.jsonl, rewrites seed_pool.jsonl) that -- empirically
verified -- runs as a side effect of merely IMPORTING the module, same as
running it directly. Importing it here just to reach one pure function
would silently re-run that whole merge/dedup/write pass on every check
invocation. Copying this ~10-line pure parser avoids that hazard entirely;
same "duplicate a small pure helper rather than fight an import
complication" move dedup2.jac's header documents for this codebase.

Run:
    jac run check_seed_review.jac
"""


def parse_review_response(text: str) -> dict {
    stripped: str = text.strip();
    if stripped.startswith("ACCEPT:") {
        return {"accepted": True, "reason": stripped[len("ACCEPT:"):].strip()};
    }
    if stripped.startswith("REJECT:") {
        return {"accepted": False, "reason": stripped[len("REJECT:"):].strip()};
    }
    return {"accepted": False, "reason": "review_unparseable: " + stripped[:100]};
}

with entry {
    accept: dict = parse_review_response("ACCEPT: looks fine, idiomatic OSP");
    assert accept["accepted"] == True;
    assert accept["reason"] == "looks fine, idiomatic OSP";

    reject: dict = parse_review_response("REJECT: reinvents stdlib dict.get");
    assert reject["accepted"] == False;
    assert reject["reason"] == "reinvents stdlib dict.get";

    garbage: dict = parse_review_response("uh, I guess this is okay??");
    assert garbage["accepted"] == False;
    assert garbage["reason"].startswith("review_unparseable:");

    empty: dict = parse_review_response("");
    assert empty["accepted"] == False;
    assert empty["reason"].startswith("review_unparseable:");

    # leading/trailing whitespace around a valid verdict still parses
    padded: dict = parse_review_response("  \n ACCEPT: fine \n ");
    assert padded["accepted"] == True;
    assert padded["reason"] == "fine";

    print("all gen_seed_review.jac parser checks passed");
}
