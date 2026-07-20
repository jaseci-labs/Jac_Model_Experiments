"""Interop section: one `.jac` module imports straight from the Python and npm
ecosystems (packages AND local files) and links any C-ABI shared library by
name. An impressive funnel diagram - three ecosystem sources flowing into a
single Jac module."""

import from "lucide-react" {
    Boxes,
    Package,
    Cpu,
    Terminal,
    Binary,
    Server,
    Network,
    Box,
    Library,
    Layers,
    Globe,
    Monitor,
    Smartphone,
    Puzzle,
    TabletSmartphone
}
import from .SectionShell { SectionShell }

glob _PY_CODE: str = """import numpy as np;
import from sklearn.cluster { KMeans }
import from .models { User }""";

glob _NPM_CODE: str = """import from "react" { useState }
import from "d3-scale" { scaleLinear }
import from .Chart { Chart }""";

glob _C_CODE: str = """import from raylib {
    def InitWindow(w: i32, h: i32, t: str);
}
import from "libm.so.6" {
    def sqrt(x: f64) -> f64;
}""";

glob _JAC_CODE: str = """import from .guestbook { GuestEntry }
import from .components.Nav { Nav }
sv import from .analytics { analyze }""";

"""Every shippable thing the SAME module compiles to - one box per project kind
from the docs' "What You Can Build" page, each tagged with the verb that builds
it. The accent groups them: backend/CLI (sky), native (orange), packages (rose),
apps (violet)."""
glob _BUILDS: list[dict] = [
         {
             "name": "CLI tool",
             "cmd": "jac run",
             "accent": "bg-sky-400/15 text-sky-400",
             "icon": <Terminal size={15}/>
         },
         {
             "name": "Native binary",
             "cmd": "jac nacompile",
             "accent": "bg-primary/15 text-primary",
             "icon": <Binary size={15}/>
         },
         {
             "name": "API service",
             "cmd": "jac start --no-client",
             "accent": "bg-sky-400/15 text-sky-400",
             "icon": <Server size={15}/>
         },
         {
             "name": "Microservices",
             "cmd": "sv import × N",
             "accent": "bg-sky-400/15 text-sky-400",
             "icon": <Network size={15}/>
         },
         {
             "name": "Python package",
             "cmd": "jac bundle",
             "accent": "bg-rose-400/15 text-rose-400",
             "icon": <Box size={15}/>
         },
         {
             "name": "npm package",
             "cmd": "jac bundle --target npm",
             "accent": "bg-rose-400/15 text-rose-400",
             "icon": <Package size={15}/>
         },
         {
             "name": "Shared library",
             "cmd": "jac nacompile --shared",
             "accent": "bg-primary/15 text-primary",
             "icon": <Library size={15}/>
         },
         {
             "name": "Full-stack app",
             "cmd": "jac start",
             "accent": "bg-violet-400/15 text-violet-400",
             "icon": <Layers size={15}/>
         },
         {
             "name": "In-browser (wasm)",
             "cmd": "na → wasm",
             "accent": "bg-violet-400/15 text-violet-400",
             "icon": <Globe size={15}/>
         },
         {
             "name": "Desktop app",
             "cmd": "jac build --client desktop",
             "accent": "bg-violet-400/15 text-violet-400",
             "icon": <Monitor size={15}/>
         },
         {
             "name": "Mobile app (webview iOS/Android)",
             "cmd": "jac build --client mobile",
             "accent": "bg-violet-400/15 text-violet-400",
             "icon": <Smartphone size={15}/>
         }
     ];

"""The two project kinds the docs list as not-yet-wired (the roadmap rows)."""
glob _SOON: list[dict] = [
         {"name": "Full-stack package", "icon": <Puzzle size={15}/>},
         {"name": "Mobile (React Native)", "icon": <TabletSmartphone size={15}/>}
     ];

"""One ecosystem source card: an accent badge, a name + scope line, and a small
snippet of the real Jac import syntax for that ecosystem."""
def:pub EcoCard(name: str, sub: str, code: str, accent: str, icon: any) -> JsxElement {
    return
        <div className="flex flex-col rounded-xl border border-border bg-card/60 p-5">
            <div className="flex items-center gap-3">
                <span
                    className={"flex size-9 shrink-0 items-center justify-center rounded-lg " + accent}
                >
                    {icon}
                </span>
                <div className="min-w-0">
                    <div
                        className="font-display font-bold leading-tight text-foreground"
                    >
                        {name}
                    </div>
                    <div className="text-xs text-muted-foreground">{sub}</div>
                </div>
            </div>
            <pre
                className="mt-4 overflow-x-auto rounded-lg border border-border bg-[#0c0e14] p-3 font-mono text-[12px] leading-relaxed text-foreground/85"
            >
                {code}
            </pre>
        </div>;
}

"""One build-target box: an accent icon, the project-kind name, and the verb that
builds it. These are the outputs the single module fans out into."""
def:pub BuildCard(name: str, cmd: str, accent: str, icon: any) -> JsxElement {
    return
        <div
            className="group flex items-center gap-2.5 rounded-lg border border-border bg-card/60 px-3 py-2.5 transition hover:border-primary/40 hover:bg-card"
        >
            <span
                className={"flex size-7 shrink-0 items-center justify-center rounded-md " + accent}
            >
                {icon}
            </span>
            <div className="min-w-0">
                <div
                    className="truncate text-[13px] font-semibold leading-tight text-foreground"
                >
                    {name}
                </div>
                <div className="truncate font-mono text-[10px] text-muted-foreground">
                    {cmd}
                </div>
            </div>
        </div>;
}

"""A roadmap project kind: same footprint as a build box but dashed and muted,
with a `soon` tag - the docs' not-yet-wired rows."""
def:pub SoonChip(name: str, icon: any) -> JsxElement {
    return
        <div
            className="flex items-center gap-2.5 rounded-lg border border-dashed border-border/70 px-3 py-2.5 text-muted-foreground"
        >
            <span
                className="flex size-7 shrink-0 items-center justify-center rounded-md bg-muted/40"
            >
                {icon}
            </span>
            <div className="text-[13px] font-medium leading-tight">{name}</div>
            <span
                className="ml-1 rounded-full border border-border/60 px-1.5 py-0.5 font-mono text-[9px] uppercase tracking-wide"
            >
                soon
            </span>
        </div>;
}

def:pub InteropSection -> JsxElement {
    return
        <SectionShell
            id="interop"
            eyebrow="Interop"
            title="Import anything, link anything, build anything."
            subtitle="One .jac module imports straight from Python, npm, and other Jac modules - and links any C-ABI library by name. No generated bindings, no FFI boilerplate, no glue layer."
        >
            <div className="mx-auto max-w-5xl">
                <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
                    <EcoCard
                        name="Python"
                        sub="PyPI packages + local .py files"
                        code={_PY_CODE}
                        accent="bg-sky-400/15 text-sky-400"
                        icon={<Boxes size={18}/>}
                    />
                    <EcoCard
                        name="npm / web"
                        sub="npmjs packages + js · ts · jsx · tsx"
                        code={_NPM_CODE}
                        accent="bg-rose-400/15 text-rose-400"
                        icon={<Package size={18}/>}
                    />
                    <EcoCard
                        name="C-ABI FFI"
                        sub="any .so / .dylib / .dll, by logical name"
                        code={_C_CODE}
                        accent="bg-violet-400/15 text-violet-400"
                        icon={<Cpu size={18}/>}
                    />
                    <EcoCard
                        name="Jac"
                        sub="your .jac modules - any target, even services"
                        code={_JAC_CODE}
                        accent="bg-primary/15 text-primary"
                        icon={<span className="font-mono text-sm font-bold">J</span>}
                    />
                </div>
                <svg
                    viewBox="0 0 600 64"
                    preserveAspectRatio="none"
                    className="mx-auto h-12 w-full max-w-3xl"
                    aria-hidden="true"
                >
                    <path
                        className="tij-flow"
                        d="M75,2 C75,42 300,22 300,62"
                        fill="none"
                        stroke="var(--primary)"
                        strokeOpacity={0.55}
                        strokeWidth={2}
                    />
                    <path
                        className="tij-flow"
                        d="M225,2 C225,42 300,30 300,62"
                        fill="none"
                        stroke="var(--primary)"
                        strokeOpacity={0.55}
                        strokeWidth={2}
                    />
                    <path
                        className="tij-flow"
                        d="M375,2 C375,42 300,30 300,62"
                        fill="none"
                        stroke="var(--primary)"
                        strokeOpacity={0.55}
                        strokeWidth={2}
                    />
                    <path
                        className="tij-flow"
                        d="M525,2 C525,42 300,22 300,62"
                        fill="none"
                        stroke="var(--primary)"
                        strokeOpacity={0.55}
                        strokeWidth={2}
                    />
                </svg>
                <div
                    className="mx-auto flex max-w-sm items-center gap-3 rounded-2xl border border-primary/40 bg-primary/10 px-5 py-4 shadow-lg shadow-primary/10"
                >
                    <span
                        className="flex size-10 shrink-0 items-center justify-center rounded-xl bg-primary font-mono text-base font-bold text-primary-foreground"
                    >
                        J
                    </span>
                    <div>
                        <div className="font-mono text-sm font-bold text-foreground">
                            your_module.jac
                        </div>
                        <div className="text-xs text-muted-foreground">
                            one file - type-checked across every boundary
                        </div>
                    </div>
                </div>
                <p
                    className="mx-auto mt-6 max-w-md text-center font-mono text-[11px] uppercase tracking-widest text-muted-foreground/70"
                >
                    ...and compiles to anything you ship
                </p>
                <svg
                    viewBox="0 0 600 56"
                    preserveAspectRatio="none"
                    className="mx-auto h-12 w-full"
                    aria-hidden="true"
                >
                    <path
                        className="tij-flow"
                        d="M300,2 C300,38 75,18 75,54"
                        fill="none"
                        stroke="var(--primary)"
                        strokeOpacity={0.5}
                        strokeWidth={2}
                    />
                    <path
                        className="tij-flow"
                        d="M300,2 C300,38 225,18 225,54"
                        fill="none"
                        stroke="var(--primary)"
                        strokeOpacity={0.5}
                        strokeWidth={2}
                    />
                    <path
                        className="tij-flow"
                        d="M300,2 C300,38 375,18 375,54"
                        fill="none"
                        stroke="var(--primary)"
                        strokeOpacity={0.5}
                        strokeWidth={2}
                    />
                    <path
                        className="tij-flow"
                        d="M300,2 C300,38 525,18 525,54"
                        fill="none"
                        stroke="var(--primary)"
                        strokeOpacity={0.5}
                        strokeWidth={2}
                    />
                </svg>
                <div className="grid grid-cols-2 gap-2.5 sm:grid-cols-3 lg:grid-cols-4">
                    {[
                        <BuildCard
                            key={b["name"]}
                            name={b["name"]}
                            cmd={b["cmd"]}
                            accent={b["accent"]}
                            icon={b["icon"]}
                        /> for b in _BUILDS
                    ]}
                </div>
                <div className="mt-5 flex flex-col items-center gap-2.5">
                    <span
                        className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground/55"
                    >
                        on the roadmap
                    </span>
                    <div className="flex flex-wrap justify-center gap-2.5">
                        {[
                            <SoonChip
                                key={s["name"]}
                                name={s["name"]}
                                icon={s["icon"]}
                            /> for s in _SOON
                        ]}
                    </div>
                </div>
            </div>
        </SectionShell>;
}
