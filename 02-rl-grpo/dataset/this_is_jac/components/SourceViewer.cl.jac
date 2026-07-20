"""Source browser - the closing section. The whole showcase is written in Jac,
so this lets visitors read its OWN source. The file list and per-file syntax
highlighting both come from the `source_viewer.jac` server endpoints
(`source_index.py` colours each file with Jac's canonical Pygments lexer - the
same grammar the docs use). This component is the presentation layer: a collapsible
directory tree on the left (expanded by default), a highlighted code pane with a line gutter on the right.
Server state (`files` / `current` / `loading`) is owned by the frontend shell
and passed in; `onOpen(path)` asks it to load a file."""

import from "lucide-react" { Search, Copy, Check, FileCode2, Folder, FolderOpen, Code2, ChevronDown, ChevronRight }
import from .SectionShell { SectionShell }

"""Header label for the repo-root file group (files with no directory)."""
glob _ROOT_LABEL: str = "this_is_jac";

"""Tailwind text colour for a language, used on badges and active file icons."""
def _lang_color(lang: str) -> str {
    if lang == "Jac" {
        return "text-primary";
    }
    if lang == "Python" {
        return "text-sky-300";
    }
    if lang == "CSS" {
        return "text-violet-300";
    }
    if lang == "TOML" or lang == "JSON" {
        return "text-amber-300";
    }
    if lang == "Markdown" {
        return "text-emerald-300";
    }
    return "text-sky-300";
}

def:pub SourceViewer(
    files: list[dict],
    current: dict | None,
    loading: bool,
    onOpen: Callable[[str], None],
) -> JsxElement {
    has query: str = "",
        active: str = "",
        copied: bool = False,
        collapsed: dict = {};

    # Keep the highlighted item in sync once a file finishes loading.
    can with [current] entry {
        if current {
            active = current["path"];
        }
    }

    # Filter, then build a real directory TREE: bucket files by their dir,
    # collect every ancestor directory, and flatten to indented rows honoring
    # the collapsed set. Everything is expanded by default; a non-empty filter
    # force-expands so matches are never hidden.
    q = query.strip().lower();
    filtered = [
        f
        for f in files
        if q == "" or q in f["path"].lower()
    ];
    buckets: dict = {};
    dirset: dict = {};
    for f in filtered {
        d = f["dir"];
        if d not in buckets {
            buckets[d] = [];
        }
        buckets[d].append(f);
        while d != "" {
            dirset[d] = True;
            d = d.rsplit("/", 1)[0] if "/" in d else "";
        }
    }
    coll = {} if q != "" else collapsed;
    rows: list[dict] = [];
    rows.append({"kind": "dir", "path": "", "name": _ROOT_LABEL, "depth": 0});
    if not coll.get("", False) {
        for f in buckets.get("", []) {
            rows.append({"kind": "file", "file": f, "depth": 1});
        }
        for d in sorted(dirset.keys()) {
            # Skip rows under any collapsed ancestor.
            hidden = False;
            a = d.rsplit("/", 1)[0] if "/" in d else "";
            while a != "" {
                if coll.get(a, False) {
                    hidden = True;
                }
                a = a.rsplit("/", 1)[0] if "/" in a else "";
            }
            if hidden {
                continue;
            }
            depth = d.count("/") + 1;
            rows.append({
                "kind": "dir",
                "path": d,
                "name": d.rsplit("/", 1)[-1],
                "depth": depth
            });
            if not coll.get(d, False) {
                for f in buckets.get(d, []) {
                    rows.append({"kind": "file", "file": f, "depth": depth + 1});
                }
            }
        }
    }

    toggleDir = lambda d: str {
        nc = {**collapsed};
        nc[d] = not collapsed.get(d, False);
        collapsed = nc;
    };

    curPath = current["path"] if current else "";
    curLang = current["lang"] if current else "";
    curHtml = current["html"] if current else "";
    curErr = current["error"] if current else "";
    curLines = current["lines"] if current else 0;
    curRaw = current["raw"] if current else "";

    do_copy = lambda {
        navigator.clipboard.writeText(curRaw);
        copied = True;
    };

    return
        <SectionShell
            id="source"
            eyebrow="Open source"
            title="This Demo is One Typechecked Jac Program. Let Me Prove It."
            subtitle="Every pixel you have scrolled is Jac - client, server, and native, in one project. Browse the actual files below, I'd recommend checking out analytics.jac, main.jac, social_graph.jac, etc."
        >
            <div className="mx-auto max-w-6xl">
                <div
                    className="dark grid overflow-hidden rounded-2xl border border-border bg-card text-foreground shadow-xl shadow-black/30 lg:grid-cols-[290px_1fr]"
                >
                    <div
                        className="flex max-h-[600px] min-h-[440px] flex-col border-b border-border lg:border-b-0 lg:border-r"
                    >
                        <div
                            className="flex items-center gap-2 border-b border-border/70 px-3 py-2.5"
                        >
                            <Search
                                size={14}
                                className="shrink-0 text-muted-foreground"
                            />
                            <input
                                value={query}
                                onChange={lambda e: ChangeEvent { query = e.target.value; }}
                                placeholder="Filter files..."
                                className="w-full bg-transparent text-sm text-foreground placeholder:text-muted-foreground focus:outline-none"
                            />
                            <span
                                className="shrink-0 font-mono text-[10px] text-muted-foreground"
                            >
                                {len(filtered)}
                            </span>
                        </div>
                        <div className="flex-1 overflow-y-auto py-1">
                            {[
                                (
                                    <button
                                        key={"d:" + r["path"]}
                                        onClick={lambda { toggleDir(r["path"]); }}
                                        className="flex w-full items-center gap-1.5 px-3 py-1 text-left text-[10px] font-semibold uppercase tracking-wider text-muted-foreground/70 transition hover:bg-white/5 hover:text-foreground"
                                    >
                                        {[
                                            <span key={i} className="w-3 shrink-0"/>
                                            for i in range(r["depth"])
                                        ]}
                                        {if coll.get(r["path"], False) {
                                            <ChevronRight size={11} className="shrink-0"/>
                                        } else {
                                            <ChevronDown size={11} className="shrink-0"/>
                                        }}
                                        {if coll.get(r["path"], False) {
                                            <Folder size={11} className="shrink-0"/>
                                        } else {
                                            <FolderOpen size={11} className="shrink-0"/>
                                        }}
                                        <span className="truncate">{r["name"]}</span>
                                    </button>
                                    if r["kind"] == "dir"
                                    else <button
                                        key={r["file"]["path"]}
                                        onClick={lambda {
                                            active = r["file"]["path"];
                                            onOpen(r["file"]["path"]);
                                        }}
                                        className={"flex w-full items-center gap-2 px-3 py-1 text-left text-[13px] transition " + (
                                            "bg-primary/15 text-foreground"
                                                if r["file"]["path"] == active
                                                else "text-muted-foreground hover:bg-white/5 hover:text-foreground"
                                        )}
                                    >
                                        {[
                                            <span key={i} className="w-3 shrink-0"/>
                                            for i in range(r["depth"])
                                        ]}
                                        <FileCode2
                                            size={13}
                                            className={"shrink-0 " + (
                                                _lang_color(r["file"]["lang"])
                                                    if r["file"]["path"] == active
                                                    else "text-muted-foreground/50"
                                            )}
                                        />
                                        <span className="truncate">
                                            {r["file"]["name"]}
                                        </span>
                                    </button>
                                )
                                for r in rows
                            ]}
                        </div>
                    </div>
                    <div className="flex max-h-[600px] min-w-0 flex-col bg-[#0c0e14]">
                        <div
                            className="flex items-center gap-2 border-b border-border/70 px-4 py-2.5"
                        >
                            <span
                                className="size-3 shrink-0 rounded-full bg-[#ff5f57]"
                            />
                            <span
                                className="size-3 shrink-0 rounded-full bg-[#febc2e]"
                            />
                            <span
                                className="size-3 shrink-0 rounded-full bg-[#28c840]"
                            />
                            <span
                                className="ml-2 truncate font-mono text-xs text-muted-foreground"
                            >
                                {curPath or "—"}
                            </span>
                            {if curLang {
                                <span
                                    className={"ml-1 shrink-0 rounded-md border border-border/60 px-1.5 py-0.5 font-mono text-[10px] " + _lang_color(
                                        curLang
                                    )}
                                >
                                    {curLang}
                                </span>
                            }}
                            <button
                                className="ml-auto flex shrink-0 items-center gap-1.5 rounded-md px-2 py-1 text-xs text-muted-foreground transition hover:bg-white/5 hover:text-foreground"
                                onClick={do_copy}
                                title="Copy file"
                            >
                                {if copied {
                                    <Check size={13} className="text-emerald-400"/>
                                } else {
                                    <Copy size={13}/>
                                } }
                                {"Copied" if copied else "Copy"}
                            </button>
                        </div>
                        <div className="relative flex-1 overflow-y-auto">
                            {if curErr {
                                <div
                                    className="p-8 text-center text-sm text-muted-foreground"
                                >
                                    {curErr}
                                </div>
                            } elif curHtml {
                                <div className="flex min-h-full">
                                    <div
                                        className="select-none border-r border-border/40 px-3 py-3 text-right font-mono text-[12.5px] leading-[1.55] text-muted-foreground/40"
                                    >
                                        {[
                                            <div key={i}>{i}</div>
                                            for i in range(1, curLines + 1)
                                        ]}
                                    </div>
                                    <pre
                                        className="tij-code flex-1 overflow-x-auto px-4 py-3 font-mono text-[12.5px] leading-[1.55]"
                                    >
                                        {unsafe_html(curHtml)}
                                    </pre>
                                </div>
                            } else {
                                <div
                                    className="flex h-full items-center justify-center gap-2 p-8 text-sm text-muted-foreground"
                                >
                                    <Code2 size={16}/>
                                    Select a file to view its source
                                </div>
                            }  }
                            {if loading {
                                <div
                                    className="absolute inset-0 flex items-center justify-center bg-[#0c0e14]/60 backdrop-blur-[1px]"
                                >
                                    <div
                                        className="size-5 animate-spin rounded-full border-2 border-primary border-t-transparent"
                                    />
                                </div>
                            }}
                        </div>
                    </div>
                </div>
            </div>
        </SectionShell>;
}
