"""A framed panel that embeds the auto-generated FastAPI Swagger UI
(`/docs`). Every endpoint this server exposes - littleX's social walkers and the
showcase's - is documented and callable here. Sits beside the single-file tweet
example, showing that same code surfaced as a live REST API."""

import from "lucide-react" { FileText, ArrowUpRight }

def:pub DocsView -> JsxElement {
    return
        <div
            className="flex h-full flex-col overflow-hidden rounded-xl border border-border bg-card shadow-lg shadow-black/20"
        >
            <div
                className="flex items-center gap-1.5 border-b border-border/70 bg-white/[0.02] px-3 py-2 font-mono text-xs text-muted-foreground"
            >
                <FileText size={13} className="text-primary"/>
                /docs - FastAPI Swagger UI
                <a
                    href="/docs"
                    target="_blank"
                    rel="noreferrer"
                    className="ml-auto flex items-center gap-1 rounded-md px-2 py-0.5 transition hover:bg-white/5 hover:text-foreground"
                >
                    open
                    <ArrowUpRight size={13}/>
                </a>
            </div>
            <iframe
                src="/docs"
                title="API docs (Swagger UI)"
                className="w-full flex-1 min-h-[420px] border-0 bg-white"
            />
        </div>;
}
