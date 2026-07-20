"""Landing hero: the pitch, an install line, and a one-file teaser that hints at
the cl / sv / na trio the rest of the page unpacks."""

import from "lucide-react" { ArrowDown, Terminal }
import from .ui.button { Button }
import from .CodeBlock { CodeBlock }

def:pub HeroSection -> JsxElement {
    return
        <section
            id="top"
            className="relative flex min-h-screen flex-col items-center justify-center overflow-hidden px-6 pt-16"
        >
            <div className="tij-grid absolute inset-0 -z-10"/>
            <div className="tij-glow absolute inset-x-0 top-0 -z-10 h-[600px]"/>
            <div
                className="tij-fade-up flex w-full max-w-5xl flex-col items-center text-center"
            >
                <div
                    className="mb-6 inline-flex items-center gap-2 rounded-full border border-border bg-card/60 px-4 py-1.5 text-sm text-muted-foreground backdrop-blur"
                >
                    <span className="size-2 rounded-full bg-primary tij-pulse"/>
                    This is a technology demonstration.
                </div>
                <h1
                    className="font-display text-6xl font-bold leading-[0.95] tracking-tight text-foreground md:text-8xl"
                >
                    This is{" "}
                    <span className="tij-gradient-text">Jac</span>
                    .
                </h1>
                <p
                    className="mt-6 max-w-2xl text-balance text-xl text-muted-foreground"
                >
                    "This language shouldn't be possible."
                </p>
                <div className="mt-8 flex flex-wrap items-center justify-center gap-3">
                    <a href="#fullstack">
                        <Button size="lg" className="rounded-full px-7 font-semibold">
                            Keep going
                            <ArrowDown size={18}/>
                        </Button>
                    </a>
                </div>
            </div>
            <a
                href="#fullstack"
                className="tij-bounce absolute bottom-8 text-muted-foreground"
                aria-label="Scroll down"
            >
                <ArrowDown size={24}/>
            </a>
        </section>;
}
