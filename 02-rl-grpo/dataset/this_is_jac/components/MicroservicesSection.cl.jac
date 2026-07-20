"""Microservices section (LIVE): the SAME `sv import` that erases the
client/server line also erases the SERVICE line. `guestbook.jac` serves its own
walkers to the browser AND, via one relative `sv import from .analytics`, calls
the analytics microservice - which the runtime spawns as its OWN process on
first call. The button runs it for real; the two distinct pids prove the
processes are separate, and the analytics result comes back as a typed object."""

import from "lucide-react" { Cpu, Server, ArrowRight, Boxes, Play, Radio }
import from .ui.button { Button }
import from .CodeBlock { CodeBlock }
import from .SectionShell { SectionShell }

glob _CONSUMER: str = """# guestbook.jac - serves its walkers to the browser AND
# consumes a microservice. One relative sv import is the whole integration.
sv import from .analytics { analyze, narrate }

def:pub guestbook_stats() -> dict {
    rep = analyze(count_signatures());      # -> SEPARATE service
    return {
        "gateway_pid": os.getpid(),         # this server
        "analytics_pid": rep.pid,           # the analytics microservce
        "score": rep.score                  # typed, end to end
    };
}

# The SAME import also streams: the sv->sv call returns a LIVE
# generator (one SSE frame per yield) - re-yield it to the browser.
def:pub guestbook_story() -> Generator {
    for chunk in narrate(count_signatures()) {
        yield chunk;                        # frame in, frame out
    }
}""";

glob _PROVIDER: str = """# analytics.jac - the provider service.
import os;
obj AnalyticsReport { has pid: int, score: float; }

def:pub analyze(signatures: int) -> AnalyticsReport {
    # runs in analytics' OWN service process - pid proves it
    return AnalyticsReport(pid=os.getpid(), score=float(signatures) * 1.5);
}

# A generator def:pub IS a streaming endpoint: every yield leaves
# this process as one server-sent event.
def:pub narrate(signatures: int) -> Generator {
    for word in commentary(signatures).split(" ") {
        yield word + " ";                   # one SSE frame per word
    }
}""";

"""One labelled process box in the topology diagram."""
def:pub ProcBox(label: str, sub: str, pid: str, icon: any, accent: bool) -> JsxElement {
    return
        <div
            className={"flex flex-1 flex-col items-center gap-1 rounded-xl border p-4 text-center " + (
                "border-primary/40 bg-primary/10"
                    if accent
                    else "border-border bg-card/60"
            )}
        >
            <div
                className="flex items-center gap-2 text-sm font-semibold text-foreground"
            >
                {icon}{label}
            </div>
            <div className="font-mono text-xs text-muted-foreground">
                pid{" "}
                <span className="text-primary">{pid}</span>
            </div>
            <div className="font-mono text-[0.65rem] text-muted-foreground/70">
                {sub}
            </div>
        </div>;
}

def:pub MicroservicesSection(
    stats: dict | None,
    running: bool,
    story: str,
    storyOn: bool,
    onRun: Callable[[], None],
) -> JsxElement {
    gpid = str(stats["gateway_pid"]) if stats else "—";
    apid = str(stats["analytics_pid"]) if stats else "—";

    return
        <SectionShell
            id="microservices"
            eyebrow="Microservices"
            title="One import is a Microservice boundary"
            subtitle="One relative `sv import from .analytics`, specifies a microservice. Done. Typechecked acrross. The runtime spawns analytics as its own service and routes the call - The compiler generate HTTP APIs, serializers, service URL, etc. This is real, Click the button. Watch the same import STREAM, and the analytics commentary arrives word by word over SSE."
        >
            <div className="mx-auto max-w-5xl">
                <div className="grid items-stretch gap-4 lg:grid-cols-2">
                    <CodeBlock filename="guestbook.jac" code={_CONSUMER} dense={True}/>
                    <CodeBlock filename="analytics.jac" code={_PROVIDER} dense={True}/>
                </div>
                <div className="mt-4 rounded-xl border border-border bg-card/60 p-6">
                    <div className="mb-4 flex items-center justify-between">
                        <span
                            className="flex items-center gap-2 text-sm font-medium text-foreground"
                        >
                            <Boxes size={15} className="text-primary"/>
                            Live cross-process call
                        </span>
                        <Button
                            size="sm"
                            className="gap-1.5 rounded-full"
                            disabled={running}
                            onClick={lambda { onRun(); }}
                        >
                            <Play size={13}/>
                            {"Calling..." if running else "Run guestbook_stats"}
                        </Button>
                    </div>
                    <div className="flex items-center gap-2">
                        <ProcBox
                            label="guestbook"
                            sub="the showcase server"
                            pid={gpid}
                            icon={<Server size={15} className="text-primary"/>}
                            accent={False}
                        />
                        <div
                            className="flex shrink-0 flex-col items-center text-muted-foreground"
                        >
                            <ArrowRight size={20}/>
                            <span className="font-mono text-[0.65rem]">sv import</span>
                        </div>
                        <ProcBox
                            label="analytics"
                            sub="auto-spawned microservice"
                            pid={apid}
                            icon={<Cpu size={15} className="text-primary"/>}
                            accent={True}
                        />
                    </div>
                    {if stats {
                        <div
                            className="mt-4 rounded-lg border border-border bg-background/50 p-3 text-sm"
                        >
                            <div className="text-muted-foreground">
                                analytics computed a reach score of{" "}
                                <span className="font-mono font-bold text-primary">
                                    {stats["score"]}
                                </span>{" "}
                                over{" "}
                                <span className="font-mono text-foreground">
                                    {stats["signatures"]}
                                </span>{" "}
                                guestbook signatures.
                            </div>
                            <div className="mt-1 text-xs text-muted-foreground">
                                {"Two different PIDs - analytics ran in a separate process the runtime spawned from one sv import."
                                    if stats["gateway_pid"] != stats["analytics_pid"]
                                    else "Same PID (in-process)."}
                            </div>
                        </div>
                    } else {
                        <div
                            className="mt-4 rounded-lg border border-border bg-background/50 p-3 text-center text-sm text-muted-foreground"
                        >
                            Run it - the guestbook server forwards your live signature count to the analytics microservice and reports both process ids.
                        </div>
                    } }
                    {if storyOn or story != "" {
                        <div
                            className="mt-4 rounded-lg border border-primary/30 bg-primary/[0.05] p-4"
                        >
                            <div
                                className="mb-2 flex items-center gap-1.5 font-mono text-[10px] font-semibold uppercase tracking-wider text-primary"
                            >
                                <Radio
                                    size={13}
                                    className={"animate-pulse" if storyOn else ""}
                                />
                                streaming live from the analytics process - one SSE frame per word
                            </div>
                            <p className="text-sm leading-relaxed text-foreground/85">
                                {story}
                                {if storyOn {
                                    <span className="animate-pulse text-primary">
                                        ▍
                                    </span>
                                }}
                            </p>
                        </div>
                    }}
                </div>
            </div>
        </SectionShell>;
}
