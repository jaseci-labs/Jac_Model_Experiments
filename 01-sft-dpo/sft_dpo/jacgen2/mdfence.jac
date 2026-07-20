"""
Fence-aware markdown extraction for the jacgen2 seed pool (tier 2 — jac-mcp
docs). Pure functions, no side effects, no imports beyond stdlib-free string
scanning.

extract_jac_fences: pulls every ```jac fenced code block out of a markdown
doc, tagged with the nearest preceding heading.

chunk_doc: splits a markdown doc into heading-bounded prose chunks capped at
a rough word-count token proxy, for gen_explanation's doc_chunks.jsonl. This
is a rough heuristic chunker (word-count via str.split(), not a real
tokenizer) — deliberately simple, not a precision tool. It does not skip
lines inside fenced code blocks when heading-scanning (a "#" inside a code
comment could be mistaken for a heading) — acceptable for a rough chunker
per the ponytail simplicity bar; extract_jac_fences (used for the actual
code seeds) does track fence state properly since precision matters there.
"""

def extract_jac_fences(md_text: str) -> list[dict] {
    lines: list[str] = md_text.split("\n");
    fences: list[dict] = [];
    heading: str = "";
    in_fence: bool = False;
    capturing: bool = False;
    buf: list[str] = [];
    for line in lines {
        stripped: str = line.strip();
        if not in_fence and stripped.startswith("#") {
            heading = stripped.lstrip("#").strip();
            continue;
        }
        if stripped.startswith("```") {
            if not in_fence {
                in_fence = True;
                capturing = (stripped == "```jac");
                buf = [];
            } else {
                in_fence = False;
                if capturing {
                    fences.append({"content": "\n".join(buf), "heading": heading});
                }
                capturing = False;
                buf = [];
            }
            continue;
        }
        if in_fence and capturing {
            buf.append(line);
        }
    }
    return fences;
}


def split_paragraphs(text: str) -> list[str] {
    parts: list[str] = [];
    cur: list[str] = [];
    for line in text.split("\n") {
        if line.strip() == "" {
            if cur {
                parts.append("\n".join(cur));
                cur = [];
            }
        } else {
            cur.append(line);
        }
    }
    if cur {
        parts.append("\n".join(cur));
    }
    return parts;
}


def pack_paragraphs(paragraphs: list[str], max_tokens: int) -> list[str] {
    chunks: list[str] = [];
    cur_parts: list[str] = [];
    cur_tokens: int = 0;
    for p in paragraphs {
        ptoks: int = len(p.split());
        if cur_parts and (cur_tokens + ptoks) > max_tokens {
            chunks.append("\n\n".join(cur_parts));
            cur_parts = [];
            cur_tokens = 0;
        }
        cur_parts.append(p);
        cur_tokens += ptoks;
    }
    if cur_parts {
        chunks.append("\n\n".join(cur_parts));
    }
    return chunks;
}


def chunk_doc(md_text: str, max_tokens: int = 500) -> list[dict] {
    lines: list[str] = md_text.split("\n");
    sections: list[dict] = [];
    heading: str = "";
    buf: list[str] = [];
    for line in lines {
        if line.strip().startswith("#") {
            if buf {
                sections.append({"heading": heading, "text": "\n".join(buf)});
            }
            heading = line.strip().lstrip("#").strip();
            buf = [];
        } else {
            buf.append(line);
        }
    }
    if buf {
        sections.append({"heading": heading, "text": "\n".join(buf)});
    }

    chunks: list[dict] = [];
    for sec in sections {
        text: str = str(sec["text"]);
        if len(text.split()) <= max_tokens {
            if text.strip() != "" {
                chunks.append({"heading": sec["heading"], "text": text.strip()});
            }
        } else {
            paragraphs: list[str] = split_paragraphs(text);
            packed: list[str] = pack_paragraphs(paragraphs, max_tokens);
            for p in packed {
                if p.strip() != "" {
                    chunks.append({"heading": sec["heading"], "text": p.strip()});
                }
            }
        }
    }
    return chunks;
}
