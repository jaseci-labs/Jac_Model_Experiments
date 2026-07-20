"""Cohesion section: the closing argument. We rebuilt this exact site WITHOUT
Jac (React + FastAPI + Postgres + a C/WASM module). The line count barely moved -
but five seams reappeared, each a boundary between systems you keep in sync by
hand. Jac collapses every one of them into a single language. This section makes
that visible: per seam, the conventional stack's joint vs Jac's one-thing."""

import from "lucide-react" { Database, Check, X, GitMerge, Cpu, Server, Globe, Box }
import from .SectionShell { SectionShell }

"""A bottom-line collapse chip: `<from> -> <dest>`."""
def:pub CollapseChip(from_: str, dest: str) -> JsxElement {
    return
        <span
            className="inline-flex items-center gap-1.5 rounded-full border border-primary/30 bg-primary/10 px-3 py-1 font-mono text-xs text-primary"
        >
            <span
                className="text-muted-foreground line-through decoration-destructive/60"
            >
                {from_}
            </span>
            <GitMerge size={13}/>
            <span className="font-semibold">{dest}</span>
        </span>;
}

"""The conventional twin's processes - the N programs that have to talk to each
other over the wire (the not-jac build ran exactly these five)."""
glob _PROGRAMS: list[dict] = [
         {"label": "Browser SPA", "icon": <Globe size={15}/>},
         {"label": "API server", "icon": <Server size={15}/>},
         {"label": "Analytics svc", "icon": <Cpu size={15}/>},
         {"label": "Postgres", "icon": <Database size={15}/>},
         {"label": "Redis", "icon": <Box size={15}/>}
     ];

"""One small dashed box in the 'N programs' cluster."""
def:pub ProgBox(label: str, icon: any) -> JsxElement {
    return
        <div
            className="flex items-center gap-1.5 rounded-lg border border-dashed border-border/70 bg-background/40 px-2.5 py-1.5 text-xs text-muted-foreground"
        >
            {icon}{label}
        </div>;
}

def:pub SeamsSection -> JsxElement {
    return
        <SectionShell
            id="seams"
            eyebrow="Cohesion"
            title="Jac is the Antidote to Polyglot Complexity."
            subtitle="We rebuilt this exact site without Jac - React, FastAPI, Postgres, a C/WASM module, etc. But six seams appeared - boundaries between systems you must keep in sync by hand, where things drift and break. Jac collapses each into one language, one program, with one next generation compiler and runtime."
        >
            <div className="mx-auto flex max-w-4xl flex-col gap-4">
                <div
                    className="mt-4 rounded-2xl border border-primary/40 bg-primary/[0.05] p-6 shadow-lg shadow-primary/10 md:p-8"
                >
                    <div className="mb-6 text-center">
                        <div
                            className="mb-2 font-mono text-[11px] uppercase tracking-widest text-primary"
                        >
                            the bottom line
                        </div>
                        <h3
                            className="font-display text-2xl font-bold text-foreground md:text-3xl"
                        >
                            One program - not{" "}
                            <span className="tij-gradient-text">
                                5 programs talking to each other
                            </span>
                            .
                        </h3>
                    </div>
                    <div
                        className="grid items-stretch gap-4 md:grid-cols-[1fr_auto_1fr]"
                    >
                        <div
                            className="rounded-xl border border-destructive/25 bg-destructive/[0.05] p-4"
                        >
                            <div
                                className="mb-3 flex items-center gap-1.5 font-mono text-[10px] font-semibold uppercase tracking-wider text-destructive/80"
                            >
                                <X size={13}/>
                                without jac - 5 programs
                            </div>
                            <div className="flex flex-wrap gap-2">
                                {[
                                    <ProgBox
                                        key={p["label"]}
                                        label={p["label"]}
                                        icon={p["icon"]}
                                    /> for p in _PROGRAMS
                                ]}
                            </div>
                            <p
                                className="mt-3 text-xs leading-relaxed text-muted-foreground"
                            >
                                Five processes, talking over HTTP, SQL and RESP. Every arrow between them is a thing that can fail, drift, or fall out of sync.
                            </p>
                        </div>
                        <div className="flex items-center justify-center">
                            <span
                                className="font-mono text-xs uppercase tracking-widest text-muted-foreground"
                            >
                                vs
                            </span>
                        </div>
                        <div
                            className="flex flex-col rounded-xl border border-primary/30 bg-primary/[0.05] p-4"
                        >
                            <div
                                className="mb-3 flex items-center gap-1.5 font-mono text-[10px] font-semibold uppercase tracking-wider text-primary"
                            >
                                <Check size={13}/>
                                with jac - 1 program
                            </div>
                            <div
                                className="flex items-center gap-3 rounded-lg border border-primary/40 bg-primary/10 p-4"
                            >
                                <span
                                    className="flex size-10 shrink-0 items-center justify-center rounded-xl bg-primary font-mono text-base font-bold text-primary-foreground"
                                >
                                    J
                                </span>
                                <div>
                                    <div
                                        className="font-mono text-sm font-bold text-foreground"
                                    >
                                        your_app.jac
                                    </div>
                                    <div className="text-xs text-muted-foreground">
                                        cl + sv + na - one process
                                    </div>
                                </div>
                            </div>
                            <p
                                className="mt-3 text-xs leading-relaxed text-muted-foreground"
                            >
                                Client, server, and native in one program. There is no wire between them to break.
                            </p>
                        </div>
                    </div>
                </div>
                <div
                    className="mt-4 flex flex-wrap items-center justify-center gap-2.5"
                >
                    <CollapseChip from_="5 languages" dest="1"/>
                    <CollapseChip from_="5 programs" dest="1"/>
                    <CollapseChip from_="hand-synced types" dest="0"/>
                    <CollapseChip from_="migrations" dest="0"/>
                    <CollapseChip from_="proxy + CORS glue" dest="0"/>
                    <CollapseChip from_="native toolchain" dest="0"/>
                </div>
            </div>
        </SectionShell>;
}
