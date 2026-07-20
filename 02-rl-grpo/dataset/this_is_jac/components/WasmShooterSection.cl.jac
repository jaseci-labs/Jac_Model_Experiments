"""The native section's twin: the SAME `na {}` rlgl shooter from `main.jac`,
compiled to `/static/main.wasm` by the client build and run live in the browser
on a WebGL canvas via the shim in `raylib_shim.cl.jac`. Click to load the wasm
module and play - no plugin, no install."""

import from "lucide-react" { Play, Gamepad2 }
import from .ui.button { Button }
import from ..raylib_shim { run_game }
import from .SectionShell { SectionShell }

def:pub WasmShooterSection -> JsxElement {
    has launched: bool = False;

    launch = lambda {
        launched = True;
        canvas: any = document.getElementById("tij-glcanvas");
        run_game(
            canvas,
            lambda exports: any , fps: int {
                score_el: any = document.getElementById("tij-hud-score");
                fps_el: any = document.getElementById("tij-hud-fps");
                if score_el {
                    score_el.textContent = exports.get_score();
                }
                if fps_el {
                    fps_el.textContent = fps;
                }
            }
        );
    };

    return
        <SectionShell
            id="wasm"
            eyebrow="WebAssembly"
            title="Oh, you can play it here, because... WASM!"
            subtitle="This tech demo integrates and typechecks across native jac to WebAssembly and the raylib externs via wasm imports, satisfied by a WebGL shim. Click to load the module and play."
        >
            <div className="mx-auto max-w-4xl">
                <div
                    className="relative overflow-hidden rounded-2xl border border-border bg-[#0b0e16] shadow-2xl shadow-black/40"
                >
                    <div
                        className="relative mx-auto"
                        style={{"aspectRatio": "960 / 600", "maxWidth": "960px"}}
                    >
                        <canvas
                            id="tij-glcanvas"
                            width={960}
                            height={600}
                            className="h-full w-full"
                            style={{"display": "block", "cursor": "crosshair"}}
                        />
                        <div
                            className="pointer-events-none absolute left-3 top-3 font-mono text-sm text-[#cdd6e6]"
                            style={{"textShadow": "0 1px 2px #000"}}
                        >
                            score{" "}
                            <b id="tij-hud-score" className="text-primary">0</b>
                            {" · "}
                            <b id="tij-hud-fps" className="text-primary">0</b>{" "}
                            fps
                        </div>
                        {if not launched {
                            <div
                                className="absolute inset-0 flex flex-col items-center justify-center gap-4 bg-[#0b0e16]/90 backdrop-blur-sm"
                            >
                                <div
                                    className="flex size-16 items-center justify-center rounded-2xl bg-primary/15 text-primary"
                                >
                                    <Gamepad2 size={30}/>
                                </div>
                                <div className="text-center">
                                    <div
                                        className="font-display text-lg font-bold text-foreground"
                                    >
                                        Jac Cube Shooter
                                    </div>
                                    <div className="text-sm text-muted-foreground">
                                        WebAssembly + WebGL, compiled from Jac
                                    </div>
                                </div>
                                <Button
                                    className="rounded-full px-6 font-semibold"
                                    onClick={launch}
                                >
                                    <Play size={16}/>
                                    Launch demo
                                </Button>
                            </div>
                        }}
                    </div>
                </div>
                <p className="mt-4 text-center text-sm text-muted-foreground">
                    Click the canvas to capture the mouse
                    {" · "}
                    <b className="text-foreground">WASD</b>{" "}
                    move
                    {" · "}
                    mouse / arrows aim
                    {" · "}
                    <b className="text-foreground">Space</b>{" "}
                    fire
                    {" · "}
                    <b className="text-foreground">Tab</b>{" "}
                    release
                </p>
            </div>
        </SectionShell>;
}
