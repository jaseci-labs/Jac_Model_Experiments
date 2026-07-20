"""A small framed panel that embeds the runtime's Jac Graph Visualizer
(`/graph`). It renders the live node/edge topology the server holds - including
the Day and Visitor nodes the guestbook (just above) writes, each signature
grouped under the day it was signed. Sits below the live guestbook in the
fullstack section."""

import from "lucide-react" { Waypoints, ArrowUpRight }

def:pub GraphView(refreshKey: int = 0) -> JsxElement {
    return
        <div className="overflow-hidden rounded-xl border border-border bg-card shadow-lg shadow-black/20">
            <div
                className="flex items-center gap-1.5 border-b border-border/70 bg-white/[0.02] px-3 py-2 font-mono text-xs text-muted-foreground"
            >
                <Waypoints size={13} className="text-primary"/>
                /graph - live Jac graph visualizer
                <a
                    href="/graph"
                    target="_blank"
                    rel="noreferrer"
                    className="ml-auto flex items-center gap-1 rounded-md px-2 py-0.5 transition hover:bg-white/5 hover:text-foreground"
                >
                    open
                    <ArrowUpRight size={13}/>
                </a>
            </div>
            <iframe
                src={"/graph?public=1&v=" + str(refreshKey)}
                title="Jac Graph Visualizer"
                className="h-[440px] w-full border-0 bg-background"
            />
            <p
                className="border-t border-border/70 px-3 py-2 text-[0.7rem] leading-snug text-muted-foreground"
            >
                This is the <span className="text-foreground">public graph</span> -
                the Day and Visitor nodes the guestbook above writes, each signature
                grouped under the day it was signed. Logged into littleX? Hit
                "open" to see your own user graph in a full tab.
            </p>
        </div>;
}
