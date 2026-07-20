"""A framed Jac code panel with window chrome, a copy button, and a small
hand-rolled tokenizer (keywords / strings / comments / numbers). No
syntax-highlight npm dependency - the highlighter is a character scan written
in Jac, so the panel ships with the bundle and uses only jac2js-safe string ops
(slicing, membership, len)."""

import from "lucide-react" { Copy, Check }

glob _LETTERS: str = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_";
glob _DIGITS: str = "0123456789";
glob _NL: str = chr(10);  # jac2js emits "\n" as a literal backslash-n in cl;
                          # use a real newline char for line-end detection.
glob _KEYWORDS: list[str] = [
    "import", "from", "include", "def", "obj", "node", "edge", "walker", "enum",
    "has", "can", "with", "entry", "exit", "visit", "report", "spawn", "return",
    "if", "else", "elif", "for", "while", "in", "not", "and", "or", "is", "by",
    "glob", "cl", "na", "sv", "lambda", "async", "await", "yield", "True",
    "False", "None", "self", "here", "root", "disengage", "del", "grant", "as",
    "global", "let", "abs"
];

"""Classify a token kind into a Tailwind color class."""
def _tok_class(kind: str) -> str {
    if kind == "kw" {
        return "text-primary";
    }
    if kind == "st" {
        return "text-emerald-400";
    }
    if kind == "cm" {
        return "text-muted-foreground/70 italic";
    }
    if kind == "nu" {
        return "text-amber-300";
    }
    if kind == "ty" {
        return "text-sky-300";
    }
    return "text-foreground/85";
}

"""Scan `src` into a flat list of (text, kind) tokens. Strings and comments are
captured whole (they may span newlines for triple-quoted strings); `<pre>`
preserves the newlines, so rendering each token as a span reproduces the
source exactly."""
def _tokenize(src: str) -> list[dict] {
    out: list[dict] = [];
    i = 0;
    n = len(src);
    while i < n {
        c = src[i];
        # line comment
        if c == "#" {
            j = i;
            while j < n and src[j] != _NL {
                j += 1;
            }
            out.append({"t": src[i:j], "k": "cm"});
            i = j;
            continue;
        }
        # string literal (triple or single). NOTE: `j` is assigned here at the
        # branch's top scope *before* the if/else - jac2js block-scopes a `let`
        # at the first assignment, so a first-assignment nested inside if/else
        # would leave `j` undefined for the `src[i:j]` use below (a TDZ crash).
        if c == '"' or c == "'" {
            triple = i + 2 < n and src[i + 1] == c and src[i + 2] == c;
            j = i + 3 if triple else i + 1;
            if triple {
                while j < n and not (
                    src[j] == c
                    and j + 2 < n
                    and src[j + 1] == c
                    and src[j + 2] == c
                ) {
                    j += 1;
                }
                j = min(n, j + 3);
            } else {
                while j < n and src[j] != c {
                    if src[j] == "\\" {
                        j += 1;
                    }
                    j += 1;
                }
                j = min(n, j + 1);
            }
            out.append({"t": src[i:j], "k": "st"});
            i = j;
            continue;
        }
        # identifier / keyword / type
        if c in _LETTERS {
            j = i;
            while j < n and (src[j] in _LETTERS or src[j] in _DIGITS) {
                j += 1;
            }
            word = src[i:j];
            kind = "id";
            if word in _KEYWORDS {
                kind = "kw";
            } elif word[0] in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" {
                kind = "ty";
            }
            out.append({"t": word, "k": kind});
            i = j;
            continue;
        }
        # number
        if c in _DIGITS {
            j = i;
            while j < n and (src[j] in _DIGITS or src[j] in ".xXabcdefABCDEF_") {
                j += 1;
            }
            out.append({"t": src[i:j], "k": "nu"});
            i = j;
            continue;
        }
        # everything else (punctuation / whitespace), coalesced
        j = i;
        while j < n
            and src[j] != "#"
            and src[j] != '"'
            and src[j] != "'"
            and src[j] not in _LETTERS
            and src[j] not in _DIGITS {
            j += 1;
        }
        if j == i {
            j = i + 1;
        }
        out.append({"t": src[i:j], "k": "pn"});
        i = j;
    }
    return out;
}

def:pub CodeBlock(code: str, filename: str = "main.jac", className: str = "", dense: bool = False) -> JsxElement {
    has copied: bool = False;

    tokens = _tokenize(code);

    do_copy = lambda {
        navigator.clipboard.writeText(code);
        copied = True;
    };

    return
        <div
            className={"group/code overflow-hidden rounded-xl border border-border bg-[#0c0e14] shadow-xl shadow-black/30 " + className}
        >
            <div
                className="flex items-center gap-2 border-b border-border/70 bg-white/[0.02] px-4 py-2.5"
            >
                <span className="size-3 rounded-full bg-[#ff5f57]"/>
                <span className="size-3 rounded-full bg-[#febc2e]"/>
                <span className="size-3 rounded-full bg-[#28c840]"/>
                <span
                    className="ml-2 font-mono text-xs text-muted-foreground"
                >
                    {filename}
                </span>
                <button
                    className="ml-auto flex items-center gap-1.5 rounded-md px-2 py-1 text-xs text-muted-foreground transition hover:bg-white/5 hover:text-foreground"
                    onClick={do_copy}
                    title="Copy code"
                >
                    {if copied {
                        <Check size={13} className="text-emerald-400"/>
                    } else {
                        <Copy size={13}/>
                    } }
                    {"Copied" if copied else "Copy"}
                </button>
            </div>
            <pre
                className={"overflow-x-auto p-4 text-[13px] " + ("leading-snug" if dense else "leading-relaxed")}
            >
                <code className="font-mono">
                    {[
                        <span className={_tok_class(tok["k"])} key={idx}>
                            {tok["t"]}
                        </span>
                        for (idx, tok) in enumerate(tokens)
                    ]}
                </code>
            </pre>
        </div>;
}
