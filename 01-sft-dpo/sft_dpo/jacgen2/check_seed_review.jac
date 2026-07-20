"""
Self-check for gen_seed_review.jac's response-parsing logic AND its
merge/dedup-into-the-pool logic.

There is no ANTHROPIC_API_KEY in this environment, so a real Fable verdict
can't be produced -- this proves the PARSER handles all three shapes a real
response (or a failed call) can take: a clean ACCEPT, a clean REJECT, and
garbage/unparseable text. review_candidate's call-failure branch is proven
separately by the real end-to-end run (seed_scrape.jac then
gen_seed_review.jac against one small cloned repo, JAC_CALL_BUDGET set
small) -- see task-6-brief.md for the exact invocation and observed output.

It also proves the merge/dedup path (`dedup_pool`) against SYNTHETIC data,
since the real run never accepted any tier-4 candidates (0 accepts, no API
key), so that path was otherwise never exercised against non-empty input.

NOTE on importing gen_seed_review directly (this used to duplicate
parse_review_response instead): gen_seed_review.jac's top-level pipeline
block is now `with entry:__main__ { ... }`, not a plain `with entry { ... }`
-- Jac's main-guard, which (empirically verified) only runs on direct
`jac run gen_seed_review.jac` invocation, not on import. Importing
gen_seed_review here for its `parse_review_response` and `dedup_pool`
functions is now safe: it does NOT re-trigger the scrape/review/merge
pipeline or touch seed_pool.jsonl.

Run:
    jac run check_seed_review.jac
"""

import from gen_seed_review { parse_review_response, dedup_pool }


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

    # -- merge/dedup path (dedup_pool), synthetic data --
    #
    # dedup_pool is a pure in-memory function (list[dict] -> list[dict]) --
    # it does no file I/O itself (that only happens in gen_seed_review's
    # with entry:__main__ block, which we never trigger by importing). So
    # this test needs no temp file at all to stay safe: there is zero disk
    # I/O in the code path under test, meaning the real
    # seed_pool.jsonl (1373 records) can't be touched no matter what.
    #
    # bucket_key = str(len(code) // 200) + "|" + code[:20] -- so for the
    # near-dup/exact-dup candidates to land in the same bucket as the
    # existing "anchor" record they must duplicate, their code must share
    # its first 20 characters. Below, ANCHOR_CODE / EXACT_DUP_CODE /
    # NEAR_DUP_CODE all start with "alpha beta gamma del" (20 chars) and
    # are all well under 200 chars, so all three land in bucket "0|alpha
    # beta gamma del". NEAR_DUP_CODE only swaps the LAST token, so its
    # rouge_l against ANCHOR_CODE is 19-of-20 tokens matching in order
    # (LCS=19) => (2*19)/(20+20) = 0.95 >= NEAR_DUP_THRESHOLD (0.85).
    # UNIQUE_CODE shares no vocabulary with anything else in the pool, so
    # it lands in its own bucket and can't collide with anything.

    ANCHOR_CODE: str = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon";
    EXACT_DUP_CODE: str = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon";
    NEAR_DUP_CODE: str = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau CHANGED";
    UNIQUE_CODE: str = "zulu yankee xray whiskey victor uniform tango sierra romeo quebec papa oscar november mike lima kilo juliet india hotel golf";

    synthetic_pool: list[dict] = [
        {"seed_id": "existing_1_anchor", "jac_code": ANCHOR_CODE},
        {"seed_id": "existing_2", "jac_code": "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen"},
        {"seed_id": "existing_3", "jac_code": "cat dog bird fish frog snake lizard turtle rabbit hamster gerbil mouse rat squirrel chipmunk"},
        {"seed_id": "existing_4", "jac_code": "red orange yellow green blue indigo violet black white gray brown pink teal maroon navy"},
        {"seed_id": "existing_5", "jac_code": "monday tuesday wednesday thursday friday saturday sunday january february march april may june july"}
    ];

    synthetic_new_accepted: list[dict] = [
        {"seed_id": "new_unique", "jac_code": UNIQUE_CODE},
        {"seed_id": "new_exact_dup", "jac_code": EXACT_DUP_CODE},
        {"seed_id": "new_near_dup", "jac_code": NEAR_DUP_CODE}
    ];

    merged: list[dict] = synthetic_pool + synthetic_new_accepted;
    deduped: list[dict] = dedup_pool(merged);

    deduped_ids: dict[str, bool] = {};
    for rec in deduped {
        deduped_ids[str(rec["seed_id"])] = True;
    }

    assert len(synthetic_pool) == 5;
    assert len(deduped) == 6, "expected 5 existing + 1 surviving new record, got " + str(len(deduped));
    assert "new_unique" in deduped_ids, "unique new record should survive dedup";
    assert "new_exact_dup" not in deduped_ids, "exact-duplicate new record should be dropped by dedup";
    assert "new_near_dup" not in deduped_ids, "near-duplicate (rouge_l >= 0.85) new record should be dropped by dedup";

    new_survivor_count: int = 0;
    for rec in synthetic_new_accepted {
        if str(rec["seed_id"]) in deduped_ids {
            new_survivor_count += 1;
        }
    }
    assert new_survivor_count == 1, "expected exactly 1 of the 3 synthetic new records to survive dedup, got " + str(new_survivor_count);

    # all 5 pre-existing records must still be present untouched
    for rec in synthetic_pool {
        assert str(rec["seed_id"]) in deduped_ids, "pre-existing pool record dropped unexpectedly: " + str(rec["seed_id"]);
    }

    print("gen_seed_review.jac merge/dedup check passed: 1 unique survived, exact-dup dropped, near-dup dropped");
}
