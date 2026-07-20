"""Native-compilation section. `jac nacompile` turns Jac into a standalone
binary with a pure-Python linker (no cc/ld), linking a precompiled C library by
its logical name. The bars compare the Jac-native shooter against a faithful Zig
twin; numbers + screenshots come from `capture.jac` (fetched at runtime), with a
representative fallback baked in so the page is complete before any capture run."""

import from "lucide-react" { Cpu, Zap, Camera }
import from .ui.card { Card, CardContent }
import from .ui.badge { Badge }
import from .CodeBlock { CodeBlock }
import from .StatBar { StatBar }
import from .SectionShell { SectionShell }

glob _CODE: str = """# Bind a precompiled C library by its LOGICAL name - no
# headers, no `cc`, no per-OS path. The native backend picks
# libraylib.so / .dylib / raylib.dll from the target triple.
import from raylib {
    def InitWindow(width: i32, height: i32, title: str) -> None;
    def rlVertex3f(x: f32, y: f32, z: f32) -> None;
    def rlColor4ub(r: u8, g: u8, b: u8, a: u8) -> None;
}

# `jac nacompile shooter.na.jac` emits a standalone ELF/Mach-O.
# The linker is PURE PYTHON - there is no cc/ld invocation.""";

def:pub NativeSection -> JsxElement {
    has jacAvg: float = 461.7,
        jacMax: float = 719.9,
        jacFrames: int = 3694,
        zigAvg: float = 470.5,
        zigMax: float = 719.1,
        zigFrames: int = 3764,
        capturedAt: str = "",
        jacShotOk: bool = True,
        zigShotOk: bool = True;

    can with entry {
        loadBench();
    }

    async def loadBench -> None {
        try {
            resp = await fetch("/static/captures/benchmark.json");
            if resp.ok {
                data = await resp.json();
                if data["jac"] and data["jac"]["avg"] {
                    jacAvg = data["jac"]["avg"];
                    jacMax = data["jac"]["max"];
                    jacFrames = data["jac"]["frames"];
                }
                if data["zig"] and data["zig"]["avg"] {
                    zigAvg = data["zig"]["avg"];
                    zigMax = data["zig"]["max"];
                    zigFrames = data["zig"]["frames"];
                }
                capturedAt = data["captured_at"] if data["captured_at"] else "";
            }
        } except Exception as e {
            # No capture artifact yet - keep the baked-in representative numbers.
            capturedAt = "";
        }
    }

    scale = max(jacAvg, zigAvg) * 1.08;

    return
        <SectionShell
            id="native"
            eyebrow="Native"
            title="Compiles to machine code, no C toolchain"
            subtitle="A first-person shooter written in Jac, linked against precompiled raylib and built to a standalone binary - benchmarked head-to-head with a hand-written Zig twin doing identical work."
        >
            <div className="grid items-start gap-8 lg:grid-cols-2">
                <CodeBlock code={_CODE} filename="shooter.na.jac"/>

                <Card className="border-border bg-card/60">
                    <CardContent className="flex flex-col gap-5 p-6">
                        <div className="flex items-center justify-between">
                            <span className="flex items-center gap-2 text-sm font-medium text-foreground">
                                <Zap size={15} className="text-primary"/>
                                raylib cube shooter - uncapped FPS
                            </span>
                            <Badge variant="secondary" className="font-normal">
                                {capturedAt if capturedAt else "representative"}
                            </Badge>
                        </div>

                        <StatBar
                            label="Jac (nacompile)"
                            value={jacAvg}
                            max_value={scale}
                            display={str(round(jacAvg, 1)) + " avg fps"}
                            barClass="bg-primary"
                            sub={"peak " + str(round(jacMax, 1)) + " · " + str(jacFrames) + " frames"}
                        />
                        <StatBar
                            label="Zig (baseline)"
                            value={zigAvg}
                            max_value={scale}
                            display={str(round(zigAvg, 1)) + " avg fps"}
                            barClass="bg-sky-400"
                            sub={"peak " + str(round(zigMax, 1)) + " · " + str(zigFrames) + " frames"}
                        />

                        <div className="flex items-center gap-2 rounded-lg border border-border bg-background/50 p-3 text-sm text-muted-foreground">
                            <Cpu size={15} className="shrink-0 text-primary"/>
                            Same precompiled raylib, identical per-frame work - Jac
                            tracks the conventional toolchain within the margin.
                        </div>
                    </CardContent>
                </Card>
            </div>

            <div className="mt-8 grid gap-4 md:grid-cols-2">
                <div className="overflow-hidden rounded-xl border border-border bg-[#0b0e16]">
                    <div className="flex items-center gap-2 border-b border-border/60 px-4 py-2 text-xs text-muted-foreground">
                        <Camera size={13}/>
                        Jac build (jac nacompile) - captured by the engine itself
                    </div>
                    {if jacShotOk {
                        <img
                            src="/static/captures/shooter_jac.png"
                            alt="Jac native raylib cube shooter"
                            className="aspect-[16/10] w-full object-cover"
                            onError={lambda { jacShotOk = False; }}
                        />
                    } else {
                        <div className="flex aspect-[16/10] w-full flex-col items-center justify-center gap-2 text-muted-foreground">
                            <Camera size={24} className="opacity-40"/>
                            <span className="text-xs">run capture.jac to add a screenshot</span>
                        </div>
                    } }
                </div>
                <div className="overflow-hidden rounded-xl border border-border bg-[#0b0e16]">
                    <div className="flex items-center gap-2 border-b border-border/60 px-4 py-2 text-xs text-muted-foreground">
                        <Camera size={13}/>
                        Zig twin (zig build-exe) - captured by the engine itself
                    </div>
                    {if zigShotOk {
                        <img
                            src="/static/captures/shooter_zig.png"
                            alt="Zig twin raylib cube shooter"
                            className="aspect-[16/10] w-full object-cover"
                            onError={lambda { zigShotOk = False; }}
                        />
                    } else {
                        <div className="flex aspect-[16/10] w-full flex-col items-center justify-center gap-2 text-muted-foreground">
                            <Camera size={24} className="opacity-40"/>
                            <span className="text-xs">run capture.jac to add a screenshot</span>
                        </div>
                    } }
                </div>
            </div>
        </SectionShell>;
}
