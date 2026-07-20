"""Sticky top navigation: brand mark, in-page section anchors, a theme toggle,
and a GitHub link. Anchors rely on CSS smooth-scroll + `scroll-mt` on each
SectionShell."""

import from "lucide-react" { Sun, Moon, Github }
import from .ui.button { Button }

glob _LINKS: list[dict] = [
    {"href": "#interop", "label": "Interop"},
    {"href": "#fullstack", "label": "Fullstack"},
    {"href": "#ai", "label": "AI-native"},
    {"href": "#native", "label": "Native"},
    {"href": "#wasm", "label": "WebAssembly"},
    {"href": "#littlex", "label": "littleX"},
    {"href": "#source", "label": "Source"},
    {"href": "#seams", "label": "Cohesion"}
];

def:pub Nav(darkMode: bool, onToggleTheme: Callable[[], None]) -> JsxElement {
    return
        <header
            className="fixed inset-x-0 top-0 z-50 border-b border-border/60 bg-background/70 backdrop-blur-xl"
        >
            <nav
                className="mx-auto flex h-16 w-full max-w-6xl items-center gap-6 px-6"
            >
                <a href="#top" className="flex items-center gap-2 font-display font-bold">
                    <span
                        className="flex size-7 items-center justify-center rounded-lg bg-primary font-mono text-sm text-primary-foreground"
                    >
                        J
                    </span>
                    <span className="text-foreground">
                        this is <span className="tij-gradient-text">jac</span>
                    </span>
                </a>
                <div className="ml-4 hidden items-center gap-1 md:flex">
                    {[
                        <a
                            href={link["href"]}
                            key={link["href"]}
                            className="rounded-md px-3 py-1.5 text-sm text-muted-foreground transition hover:bg-muted hover:text-foreground"
                        >
                            {link["label"]}
                        </a>
                        for link in _LINKS
                    ]}
                </div>
                <div className="ml-auto flex items-center gap-1">
                    <Button
                        variant="ghost"
                        size="icon"
                        className="rounded-full text-muted-foreground"
                        title="Toggle theme"
                        onClick={lambda { onToggleTheme(); }}
                    >
                        {if darkMode {
                            <Sun size={18}/>
                        } else {
                            <Moon size={18}/>
                        } }
                    </Button>
                    <a
                        href="https://github.com/jaseci-labs/jaseci"
                        target="_blank"
                        rel="noreferrer"
                    >
                        <Button
                            variant="ghost"
                            size="icon"
                            className="rounded-full text-muted-foreground"
                            title="GitHub"
                        >
                            <Github size={18}/>
                        </Button>
                    </a>
                </div>
            </nav>
        </header>;
}
