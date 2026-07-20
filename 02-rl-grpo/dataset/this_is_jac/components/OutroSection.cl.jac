"""Closing call-to-action: install line, primary links, and a footer."""

import from "lucide-react" { Terminal, BookOpen, Github, ArrowUpRight }
import from .ui.button { Button }

def:pub OutroSection -> JsxElement {
    return
        <section
            className="relative flex min-h-screen flex-col justify-center overflow-hidden px-6 py-32"
        >
            <div
                className="tij-glow absolute inset-x-0 bottom-0 -z-10 h-[400px] rotate-180"
            />
            <div className="mx-auto flex max-w-3xl flex-col items-center text-center">
                <h2
                    className="font-display text-4xl font-bold tracking-tight text-foreground md:text-5xl"
                >
                    Build anything in{" "}
                    <span className="tij-gradient-text">one language</span>
                    .
                </h2>
                <p className="mt-4 max-w-xl text-lg text-muted-foreground">
                    The graph, the API, the UI, and the native core - no glue, no context-switching, no second runtime.
                </p>
                <div
                    className="mt-8 flex items-center gap-3 rounded-full border border-border bg-card/60 px-5 py-3 font-mono text-base text-foreground backdrop-blur"
                >
                    <Terminal size={18} className="text-primary"/>
                    pip install jaclang
                </div>
                <div className="mt-8 flex flex-wrap items-center justify-center gap-3">
                    <a href="https://www.jac-lang.org" target="_blank" rel="noreferrer">
                        <Button size="lg" className="rounded-full px-6 font-semibold">
                            <BookOpen size={18}/>
                            Read the docs
                            <ArrowUpRight size={16}/>
                        </Button>
                    </a>
                    <a
                        href="https://github.com/jaseci-labs/jaseci"
                        target="_blank"
                        rel="noreferrer"
                    >
                        <Button
                            size="lg"
                            variant="outline"
                            className="rounded-full px-6 font-semibold"
                        >
                            <Github size={18}/>
                            Star on GitHub
                        </Button>
                    </a>
                </div>
                <div
                    className="mt-20 flex flex-col items-center gap-1 text-sm text-muted-foreground"
                >
                    <div
                        className="flex items-center gap-2 font-display font-bold text-foreground"
                    >
                        <span
                            className="flex size-6 items-center justify-center rounded-md bg-primary font-mono text-xs text-primary-foreground"
                        >
                            J
                        </span>
                        this is jac
                    </div>
                    <span>
                        Built end-to-end in Jac. Every demo on this page is real.
                    </span>
                </div>
            </div>
        </section>;
}
