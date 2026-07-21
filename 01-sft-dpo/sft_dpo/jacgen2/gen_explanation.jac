"""
gen_explanation.jac -- pilot generator for the `explanation` category
(spec.md 04-cpt-sft/docs/datagen/spec.md section 3), built on the
prompt-batch handoff protocol in batch.jac. Same two-phase shape as
gen_code_gen.jac / gen_debug.jac -- read gen_code_gen.jac's docstring for
the general pattern, not repeated here.

`explanation` is docs-grounded quiz Q&A: every question must be answerable
from a specific doc_chunks.jsonl chunk (schema: {chunk_id, source_page,
heading, text}, produced by Task 5's chunking pass), and the source
chunk(s) are stored alongside the answer for traceability. Unlike code_gen
and debug, there is NO compiler gate here -- this is prose-only, so
generation quality (Fable, per spec.md section 4.1) plus a lexical
groundedness check are the only defenses against hallucination.

This is a SCOPED-DOWN v1: one quiz item per eligible doc_chunk (not up to 3
per spec.md section 3's "one minimum, up to 3 for multi-concept chunks"),
capped at JAC_PILOT_LIMIT total pending items. All 5 of spec.md section 3's
quiz formats ARE implemented structurally (cheap to add, unlike gen_debug's
domain x bug matrix) and assigned round-robin (index % 5) across the
filtered chunk pool, with a straightforward if/elif applicability check and
fallback to concept_recall when a chunk doesn't fit the round-robin pick:

    concept_recall          -- always applicable (the fallback format).
    pitfall_identification  -- only if "pitfall" appears in the chunk's
                                source_page or heading; else falls back.
    pattern_application     -- always applicable (a judgment question
                                grounded in the chunk, no special gate).
    compare_contrast        -- only if an adjacent chunk (index-1 or
                                index+1 in the filtered pool) shares the
                                same source_page (cheap proxy for "plausibly
                                discusses a related/contrastable concept");
                                else falls back. When applicable, the prompt
                                and pending item carry BOTH chunks.
    doc_example_reading     -- only if the chunk's text contains a
                                code-like fragment ("```" or "def "); else
                                falls back.

Chunks with `text` shorter than ~50 chars are skipped entirely (too thin to
quiz on) before round-robin assignment / cap counting.

Two phases, selected by env var JAC_BATCH_PHASE:

  prepare -- for each eligible doc_chunk (round-robin format, applicability
    fallback as above), write a pending batch item asking Fable to write one
    QUESTION:/ANSWER: pair grounded ONLY in the given excerpt(s). Writes
    pending_batch.jsonl and stops -- no LLM call happens in this phase.
    model="fable" per spec.md section 4.1 (explanation is Fable-assigned;
    no compiler gate exists here so generation quality is the only defense
    against hallucination).

  collect -- reads responses_batch.jsonl, joins each response back to its
    pending-item context by batch_id, parses the QUESTION:/ANSWER: format
    (tolerant parser, reject-with-reason on parse failure), then runs a
    LEXICAL groundedness gate in place of a compiler gate (spec.md section
    3's note: "the groundedness check's known false modes ... are logged as
    an overlap score in metadata, with the accept threshold calibrated on
    the pilot rather than hard-coded"): a stopword-filtered word-overlap
    score between the ANSWER and the source chunk text(s), accepted if
    >= JAC_GROUND_MIN (env var, default "0.25"). The real overlap score is
    recorded in extra_meta for EVERY response, accepted or rejected, so a
    human can recalibrate the threshold later. Persists accepted examples
    via gate.make_example_v2 + common.append_jsonl.

Run (from repo root):
    JAC_RUN_TAG=fresh JAC_BATCH_PHASE=prepare JAC_PILOT_LIMIT=25 \
        /Volumes/ExtremePro/JaseciLabs/jac_model_studio/.venv/bin/jac \
        run model-experiments/01-sft-dpo/sft_dpo/jacgen2/gen_explanation.jac
    JAC_RUN_TAG=fresh JAC_BATCH_PHASE=collect \
        /Volumes/ExtremePro/JaseciLabs/jac_model_studio/.venv/bin/jac \
        run model-experiments/01-sft-dpo/sft_dpo/jacgen2/gen_explanation.jac
"""

import os;
import json;
import from pathlib { Path }
import from common { require_run_tag, data_root, append_jsonl, existing_keys }
import from gate { make_example_v2 }
import from batch { make_batch_id, write_pending_batch, read_response_batch, pending_path, responses_path }

glob CATEGORY: str = "explanation";
glob DOC_CHUNK_POOL_PATH: str = "model-experiments/04-cpt-sft/dataset/shared/doc_chunks.jsonl";
glob GENERATOR_MODEL_ID: str = "claude-fable-5";
glob MIN_CHUNK_TEXT_LEN: int = 50;

glob FORMATS: list[str] = [
    "concept_recall",
    "pitfall_identification",
    "pattern_application",
    "compare_contrast",
    "doc_example_reading"
];

"""One extra guidance sentence per format, folded into the shared prompt
template -- lifted from spec.md section 3's format table."""
glob GUIDANCE: dict[str, str] = {
    "concept_recall": "Ask a direct factual recall question about something explicitly stated in the excerpt.",
    "pitfall_identification": "Ask why the described pattern or approach fails, or what gotcha it documents, and have the answer cite the fix.",
    "pattern_application": "Describe a short scenario grounded in the excerpt's concepts and ask which pattern or approach applies -- the answer should justify the choice using the excerpt.",
    "compare_contrast": "Ask a question that forces holding the two excerpts' concepts apart -- e.g. when does each apply, or how do they differ.",
    "doc_example_reading": "Ask about the code example in the excerpt -- e.g. what it outputs, or what would break if a specific part were removed or changed."
};

"""Small hardcoded stopword list -- lexical groundedness only needs to
filter the most common function words, not a real NLP-grade list."""
glob STOPWORDS: list[str] = [
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "for",
    "is", "are", "was", "were", "be", "been", "being", "this", "that",
    "these", "those", "it", "its", "as", "with", "by", "from", "at",
    "which", "what", "who", "whom", "how", "when", "where", "why", "not",
    "no", "do", "does", "did", "can", "could", "will", "would", "should",
    "may", "might", "must", "shall", "you", "your", "we", "our", "they",
    "their", "he", "she", "his", "her", "i", "my", "if", "then", "than",
    "so", "such", "also", "into", "about", "up", "down", "out", "over",
    "under", "again", "further", "once", "here", "there", "all", "any",
    "both", "each", "few", "more", "most", "other", "some", "only", "own",
    "same", "just", "because"
];

glob PUNCT_CHARS: str = ".,;:!?\"'()[]{}`*_-";


def load_doc_chunks() -> list[dict] {
    chunks: list[dict] = [];
    path: Path = Path(DOC_CHUNK_POOL_PATH);
    if not path.exists() {
        raise RuntimeError("doc chunk pool not found at " + DOC_CHUNK_POOL_PATH + " -- run Task 5's chunking pass first.");
    }
    with open(DOC_CHUNK_POOL_PATH, "r") as f {
        lines: list[str] = f.readlines();
    }
    for line in lines {
        s: str = line.strip();
        if s {
            rec = json.loads(s);
            chunks.append(dict(rec));
        }
    }
    return chunks;
}

"""Chunks with text shorter than MIN_CHUNK_TEXT_LEN are too thin to quiz on.
Order is preserved from the pool file (which is already grouped by
source_page), so index-adjacency in the RETURNED list is a reasonable
same-page proxy for compare_contrast pairing below."""
def eligible_chunks(chunks: list[dict]) -> list[dict] {
    out: list[dict] = [];
    for chunk in chunks {
        if len(str(chunk["text"])) >= MIN_CHUNK_TEXT_LEN {
            out.append(chunk);
        }
    }
    return out;
}

"""compare_contrast pairing heuristic: an adjacent chunk (next, else
previous) in the filtered pool that shares the same source_page. Returns
None if neither neighbor matches, in which case the caller falls back to
concept_recall."""
def find_partner(chunks: list[dict], idx: int) -> dict | None {
    source_page: str = str(chunks[idx]["source_page"]);
    if idx + 1 < len(chunks) {
        candidate: dict = dict(chunks[idx + 1]);
        if str(candidate["source_page"]) == source_page {
            return candidate;
        }
    }
    if idx - 1 >= 0 {
        candidate: dict = dict(chunks[idx - 1]);
        if str(candidate["source_page"]) == source_page {
            return candidate;
        }
    }
    return None;
}

"""Round-robins FORMATS by pool index, then falls back to concept_recall
per-chunk when the round-robin pick isn't applicable (see this file's top
docstring for the per-format rule). Returns (format_name, partner) where
partner is only non-None for an applicable compare_contrast pick. Returns a
real tuple (not a dict) so the dict | None type of partner survives the
round trip -- reading a value back out of a dict widens it to `any`, which
Jac's type checker then rejects when passed into a dict | None parameter."""
def choose_format(chunks: list[dict], idx: int) -> tuple[str, dict | None] {
    chunk: dict = dict(chunks[idx]);
    format_name: str = FORMATS[idx % len(FORMATS)];
    partner: dict | None = None;

    if format_name == "pitfall_identification" {
        source_page: str = str(chunk["source_page"]).lower();
        heading: str = str(chunk["heading"]).lower();
        if "pitfall" not in source_page and "pitfall" not in heading {
            format_name = "concept_recall";
        }
    } elif format_name == "compare_contrast" {
        partner = find_partner(chunks, idx);
        if partner is None {
            format_name = "concept_recall";
        }
    } elif format_name == "doc_example_reading" {
        text: str = str(chunk["text"]);
        if "```" not in text and "def " not in text {
            format_name = "concept_recall";
        }
    }

    return (format_name, partner);
}

def build_prompt(format_name: str, chunk: dict, partner: dict | None) -> str {
    guidance: str = GUIDANCE[format_name];
    heading: str = str(chunk["heading"]);
    text: str = str(chunk["text"]);

    if format_name == "compare_contrast" and partner is not None {
        heading2: str = str(partner["heading"]);
        text2: str = str(partner["text"]);
        return (
            "You are writing a quiz question for a Jac programming language training "
            + "dataset. The question MUST be answerable using ONLY the information in "
            + "the two doc excerpts below -- do not require outside knowledge. Write ONE "
            + "question and its correct answer, format type: compare_contrast. " + guidance
            + "\n\nReply in EXACTLY this format:\n"
            + "QUESTION: <the question text>\n"
            + "ANSWER: <the answer text, grounded in the excerpts>\n\n"
            + "Doc excerpt 1 (from \"" + heading + "\"):\n" + text
            + "\n\nDoc excerpt 2 (from \"" + heading2 + "\"):\n" + text2 + "\n"
        );
    }

    return (
        "You are writing a quiz question for a Jac programming language training "
        + "dataset. The question MUST be answerable using ONLY the information in the "
        + "doc excerpt below -- do not require outside knowledge. Write ONE question and "
        + "its correct answer, format type: " + format_name + ". " + guidance
        + "\n\nReply in EXACTLY this format:\n"
        + "QUESTION: <the question text>\n"
        + "ANSWER: <the answer text, grounded in the excerpt>\n\n"
        + "Doc excerpt (from \"" + heading + "\"):\n" + text + "\n"
    );
}


def run_prepare() {
    tag: str = require_run_tag();
    limit: int = int(os.environ.get("JAC_PILOT_LIMIT", "25"));

    all_chunks: list[dict] = load_doc_chunks();
    filtered: list[dict] = eligible_chunks(all_chunks);
    skipped_thin: int = len(all_chunks) - len(filtered);

    items: list[dict] = [];
    capped: int = 0;
    format_counts: dict[str, int] = {};

    for idx in range(len(filtered)) {
        if len(items) >= limit {
            capped += 1;
            continue;
        }
        chunk: dict = dict(filtered[idx]);
        (format_name, partner) = choose_format(filtered, idx);

        prompt: str = build_prompt(format_name, chunk, partner);
        chunk_id: str = str(chunk["chunk_id"]);
        batch_id: str = make_batch_id(CATEGORY, format_name, chunk_id, 0);

        chunk_id2: str = str(partner["chunk_id"]) if partner is not None else "";
        source_page2: str = str(partner["source_page"]) if partner is not None else "";
        heading2: str = str(partner["heading"]) if partner is not None else "";
        doc_chunk_text2: str = str(partner["text"]) if partner is not None else "";

        items.append({
            "batch_id": batch_id,
            "model": "fable",
            "prompt": prompt,
            "category": CATEGORY,
            "task_type": format_name,
            "variant_idx": 0,
            "seed_id": chunk_id,
            "seed_tier": 2,
            "chunk_id": chunk_id,
            "source_page": chunk["source_page"],
            "heading": chunk["heading"],
            "doc_chunk_text": chunk["text"],
            "chunk_id2": chunk_id2,
            "source_page2": source_page2,
            "heading2": heading2,
            "doc_chunk_text2": doc_chunk_text2
        });
        format_counts[format_name] = format_counts.get(format_name, 0) + 1;
    }

    out_path: str = pending_path(tag, CATEGORY);
    write_result: dict = write_pending_batch(out_path, items);

    print("");
    print("=== gen_explanation prepare summary ===");
    print("doc_chunks in pool: " + str(len(all_chunks)));
    print("skipped (text < " + str(MIN_CHUNK_TEXT_LEN) + " chars): " + str(skipped_thin));
    print("eligible chunks: " + str(len(filtered)));
    print("candidate items this run: " + str(len(items)) + " (JAC_PILOT_LIMIT=" + str(limit) + ")");
    print("skipped due to cap: " + str(capped));
    print("format distribution (post-fallback): " + str(format_counts));
    print("newly written to pending batch: " + str(write_result["written"]));
    print("skipped as already-present in pending batch: " + str(write_result["skipped"]));
    print("wrote " + out_path);
}

"""Tolerant parser for the QUESTION:/ANSWER: response format. Returns {}
(falsy) rather than raising on anything that doesn't contain both markers
in order -- callers must check `if not parsed` and reject/log rather than
crash."""
def parse_explanation_response(text: str) -> dict {
    if "QUESTION:" not in text or "ANSWER:" not in text {
        return {};
    }
    parts: list[str] = text.split("ANSWER:", 1);
    before: str = parts[0];
    after: str = parts[1];
    if "QUESTION:" not in before {
        return {};
    }
    question: str = before.split("QUESTION:", 1)[1].strip();
    answer: str = after.strip();
    if not question or not answer {
        return {};
    }
    return {"question": question, "answer": answer};
}

"""Lowercase, strip surrounding punctuation, drop stopwords/empties, dedupe
-- deliberately simple (no stemming, no library) since this is a
pilot-calibrated lexical heuristic, not a real NLP scorer (spec.md section
3's note on this gate's known false modes)."""
def significant_words(text: str) -> list[str] {
    words: list[str] = [];
    for raw in text.lower().split() {
        w: str = raw.strip(PUNCT_CHARS);
        if w and w not in STOPWORDS and w not in words {
            words.append(w);
        }
    }
    return words;
}

"""Fraction of the ANSWER's significant words that also appear in the
source chunk text(s). Returns 0.0 for an answer with no significant words
at all (rather than dividing by zero)."""
def overlap_score(answer: str, chunk_text: str) -> float {
    answer_words: list[str] = significant_words(answer);
    chunk_words: list[str] = significant_words(chunk_text);
    if not answer_words {
        return 0.0;
    }
    matched: int = 0;
    for w in answer_words {
        if w in chunk_words {
            matched += 1;
        }
    }
    return matched / len(answer_words);
}

def score_stats(scores: list[float]) -> str {
    if not scores {
        return "n/a (no scored items)";
    }
    lo: float = min(scores);
    hi: float = max(scores);
    avg: float = sum(scores) / len(scores);
    return "min=" + ("%.3f" % lo) + " max=" + ("%.3f" % hi) + " avg=" + ("%.3f" % avg) + " (n=" + str(len(scores)) + ")";
}


def run_collect() {
    tag: str = require_run_tag();
    p_path: str = pending_path(tag, CATEGORY);
    r_path: str = responses_path(tag, CATEGORY);

    if not Path(p_path).exists() {
        print("ERROR: pending batch not found at " + p_path + " -- run prepare mode first.");
        return;
    }

    responses: dict = read_response_batch(r_path);
    if not responses {
        print("no responses found yet at " + r_path + ".");
        print("this is normal between prepare and the orchestrator's generation pass --");
        print("nothing to collect. Re-run collect mode once " + r_path + " exists.");
        return;
    }

    pending_by_id: dict = {};
    with open(p_path, "r") as f {
        lines: list[str] = f.readlines();
    }
    for line in lines {
        s: str = line.strip();
        if s {
            rec = json.loads(s);
            pending_by_id[rec["batch_id"]] = rec;
        }
    }

    clean_path: str = data_root(tag) + "/clean_dataset/explanation/examples.jsonl";
    rejected_path: str = data_root(tag) + "/rejected/explanation/rejected.jsonl";
    already_persisted: dict = existing_keys(clean_path);
    ground_min: float = float(os.environ.get("JAC_GROUND_MIN", "0.25"));

    processed: int = 0;
    accepted: int = 0;
    already_persisted_skipped: int = 0;
    reason_counts: dict[str, int] = {};
    accepted_scores: list[float] = [];
    rejected_scores: list[float] = [];

    for batch_id in responses {
        processed += 1;
        resp: dict = dict(responses[batch_id]);
        ctx = pending_by_id.get(batch_id, None);

        if ctx is None {
            reason: str = "no matching pending context for batch_id";
            reason_counts[reason] = reason_counts.get(reason, 0) + 1;
            append_jsonl(rejected_path, {"batch_id": batch_id, "category": CATEGORY, "reason": reason, "overlap_score": None});
            continue;
        }

        task_type: str = str(ctx["task_type"]);

        if not bool(resp["ok"]) {
            reason: str = "generation failed (ok:false)";
            reason_counts[reason] = reason_counts.get(reason, 0) + 1;
            append_jsonl(rejected_path, {
                "batch_id": batch_id, "category": CATEGORY, "seed_id": ctx["chunk_id"],
                "task_type": task_type, "reason": reason, "overlap_score": None
            });
            continue;
        }

        parsed: dict = parse_explanation_response(str(resp["text"]));
        if not parsed {
            reason: str = "failed to parse QUESTION:/ANSWER: format";
            reason_counts[reason] = reason_counts.get(reason, 0) + 1;
            append_jsonl(rejected_path, {
                "batch_id": batch_id, "category": CATEGORY, "seed_id": ctx["chunk_id"],
                "task_type": task_type, "reason": reason, "overlap_score": None
            });
            continue;
        }

        question: str = str(parsed["question"]);
        answer: str = str(parsed["answer"]);

        chunk_text: str = str(ctx["doc_chunk_text"]);
        chunk_text2: str = str(ctx.get("doc_chunk_text2", ""));
        if task_type == "compare_contrast" and chunk_text2 {
            chunk_text = chunk_text + "\n" + chunk_text2;
        }

        score: float = overlap_score(answer, chunk_text);

        if score < ground_min {
            reason: str = "groundedness gate failed: overlap " + ("%.3f" % score) + " < JAC_GROUND_MIN " + str(ground_min);
            reason_counts[reason] = reason_counts.get(reason, 0) + 1;
            rejected_scores.append(score);
            append_jsonl(rejected_path, {
                "batch_id": batch_id, "category": CATEGORY, "seed_id": ctx["chunk_id"],
                "task_type": task_type, "reason": reason, "overlap_score": score
            });
            continue;
        }

        accepted_scores.append(score);

        messages: list[dict] = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer}
        ];

        rec: dict = make_example_v2(
            category=CATEGORY,
            task_type=task_type,
            complexity="simple",
            messages=messages,
            gate_class="prose_lexical",
            compiler_pass=True,
            test_pass=None,
            seed_id=str(ctx["chunk_id"]),
            seed_tier=int(ctx["seed_tier"]),
            variant_idx=int(ctx["variant_idx"]),
            generator="fable-api",
            generator_model_id=GENERATOR_MODEL_ID,
            extra_meta={
                "overlap_score": score,
                "source_page": ctx["source_page"],
                "heading": ctx["heading"],
                "chunk_id2": ctx.get("chunk_id2", ""),
                "source_page2": ctx.get("source_page2", ""),
                "heading2": ctx.get("heading2", ""),
                "compiler_pass_note": "explanation is prose-only, no compiler gate exists -- "
                    + "compiler_pass=True is vacuous here, gate_class=prose_lexical (the "
                    + "overlap_score above) is the real accept/reject signal",
                "generator_model_id_note": "placeholder -- no models.json-style pinning "
                    + "exists yet for the Agent-dispatch (Opus/Fable subagent) path"
            }
        );
        if rec["id"] in already_persisted {
            already_persisted_skipped += 1;
            continue;
        }

        append_jsonl(clean_path, rec);
        already_persisted[rec["id"]] = True;
        accepted += 1;
    }

    rejected_count: int = processed - accepted - already_persisted_skipped;
    print("");
    print("=== gen_explanation collect summary ===");
    print("responses processed: " + str(processed));
    print("accepted (newly written): " + str(accepted));
    print("skipped as already-persisted: " + str(already_persisted_skipped));
    print("rejected: " + str(rejected_count));
    for reason in reason_counts {
        print("  - " + reason + ": " + str(reason_counts[reason]));
    }
    print("JAC_GROUND_MIN=" + str(ground_min));
    print("overlap score distribution, accepted: " + score_stats(accepted_scores));
    print("overlap score distribution, rejected (groundedness only): " + score_stats(rejected_scores));
    print("wrote accepted examples to " + clean_path);
}


with entry:__main__ {
    phase: str | None = os.environ.get("JAC_BATCH_PHASE");
    if phase == "prepare" {
        run_prepare();
    } elif phase == "collect" {
        run_collect();
    } else {
        raise ValueError(
            "JAC_BATCH_PHASE must be set to 'prepare' or 'collect', got: " + str(phase)
        );
    }
}
