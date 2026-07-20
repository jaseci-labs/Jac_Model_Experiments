# This is Jac

A scroll-driven showcase site that demonstrates Jac's range in one codebase -
**fullstack, object-spatial, AI-native, and native-compiled** - and a Jac
automation script that benchmarks and screenshots the native demos for the page
to embed.

Everything here is built in Jac. The live demos really run.

![sections](assets/jac-logo.svg)

## What it shows

Each section is its own component under [`components/`](components/), composed by
[`frontend.cl.jac`](frontend.cl.jac):

| Section | Capability | Live? |
| ------- | ---------- | ----- |
| **Hero** | the pitch + a one-file `cl` / `na` / walker teaser | static |
| **Interop** | one `.jac` imports Python (PyPI + files), npm (js/ts/jsx + npmjs) and links any C-ABI library | static - a funnel diagram of the three ecosystems flowing into one module |
| **Fullstack** | one language across the wire | **live** - a guestbook backed by `sign_guestbook` / `get_guestbook` walkers, persisted to a real graph |
| **Object-Spatial** | the program is a graph; walkers walk it | **live** - `explore_graph` traverses the **real** graph (root -> every guestbook signature) and the section draws the actual path; no demo data |
| **AI-native** | a function body delegated to an LLM with `by llm()` | static (code + illustrated call) |
| **Native** | `jac nacompile` to machine code, no C toolchain | **data** - Jac-vs-Zig FPS bars + screenshots from `capture.jac` |
| **WebAssembly** | the *same* `na {}` shooter, compiled to wasm | **live** - playable in-browser via a WebGL shim |
| **littleX** | a whole social app as one `<LittleX/>` component | **live** - the full littleX app (auth/feed/follows/channels/profiles) embedded in an app-window frame; its walkers run in *this* server, persisting to the **same graph**. Below the embed, two smaller side-by-side panels surface server views as components: **`/graph`** (the live Jac Graph Visualizer of the graph littleX writes to) and **`/docs`** (the FastAPI Swagger UI). Also standalone at **`/littlex`** |
| **Outro** | install + links | static |

The same `na {}` cube-shooter game lives in [`main.jac`](main.jac): the client
build compiles it to `/static/main.wasm`, and the WebAssembly section renders it
live through [`raylib_shim.cl.jac`](raylib_shim.cl.jac) - byte-for-byte the rlgl
pipeline from the bundled [`raylib_shooter/`](raylib_shooter) example.

## Run it

```bash
jac install        # first time: installs python + npm deps
jac start          # build cl bundle + na->wasm, serve on http://localhost:8000
jac start --dev    # same, with hot reload
```

Open <http://localhost:8000> and scroll. Sign the guestbook, spawn the walker,
and launch the in-browser shooter - all of it hits real Jac.

`jac build` produces the same artifacts under `.jac/client/dist/` without serving.

## Layout

```
main.jac              cl{} app delegate + na{} rlgl shooter (-> main.wasm)
frontend.cl.jac       app shell: shared state, section order, sv handler decls
frontend.impl.jac     handler bodies (root spawn ... walkers)
guestbook.jac         object-spatial backend: Day/Visitor nodes, guestbook/explore walkers
raylib_shim.cl.jac    WebGL/DOM shim for the wasm shooter
capture.jac           the automation script (below)
components/           Nav, SectionShell, CodeBlock, StatBar + one file per section
components/ui/         shadcn primitives (button, card, badge, ...)
assets/captures/      benchmark.json + shooter screenshots (refreshed by capture.jac)
raylib_shooter/       the bundled native demo capture.jac benchmarks + screenshots
littlex/              the bundled littleX app (frontend + social_graph.jac + components);
                      its `app` is embedded by the littleX section, its walkers share
                      this server's graph
```

## The capture script (`capture.jac`)

A Jac program that drives the bundled `raylib_shooter/` example and writes
artifacts the **Native** section embeds. It locates `raylib_shooter` by walking
up from its own directory, so it works bundled here or beside the upstream
example. Each step is independently guarded - a failure logs and the run
continues.

```bash
jac run capture.jac                 # benchmark + screenshots
jac run capture.jac --skip-shots    # benchmark only
jac run capture.jac --skip-bench    # screenshots only
```

1. **Benchmark** - runs `raylib_shooter/bench.jac` (a pure-Jac orchestrator;
   first run downloads precompiled raylib + the Zig toolchain + the Zig twin's
   source and builds both binaries), reads the Jac-vs-Zig avg/max FPS, and writes
   `assets/captures/benchmark.json`. The Native section fetches it at runtime; a
   representative fallback is committed so the page is complete before any
   capture run.
2. **Screenshot** - the Jac shooter screenshots *itself*: `capture.jac` writes a
   `.screenshot` sentinel, the binary renders a few frames and calls raylib's own
   `TakeScreenshot` (reading its GL framebuffer directly), then exits, and the PNG
   is moved to `assets/captures/shooter_jac.png`. No external window grab. If the
   frame comes back blank - e.g. **WSLg/llvmpipe**, where the GL buffer can't be
   read back by *any* tool - it's detected and skipped, keeping the committed
   image; on a normal display it captures a real native frame.
3. **Manifest** - `manifest.json` records what each step did, with timestamps.

`benchmark.json` and `shooter_jac.png` are committed as representative artifacts
(re-run `capture.jac` to refresh them); only `manifest.json` is git-ignored.

## Requirements

- `jac` with the `jac-client` plugin (this repo's `.venv`, or `pip install jaclang jac-client`).
- For `capture.jac`: network access on first run + a GL-capable display (the
  benchmark builds + runs both binaries; the Zig toolchain and Zig source are
  auto-downloaded by `bench.jac`). The screenshot is written by the engine itself
  - no ImageMagick needed to capture; `identify`, if present, is used only as a
  best-effort blank-frame guard.
