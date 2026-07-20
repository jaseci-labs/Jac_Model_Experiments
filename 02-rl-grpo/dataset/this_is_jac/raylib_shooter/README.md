# raylib cube shooter (bundled)

A trimmed copy of the Jac native `raylib_shooter` example, bundled so the
**This is Jac** site's Native section can benchmark it. The whole demo is written
in Jac - there is **no shell script and no committed Zig source** in this repo.

| File | Role |
| ---- | ---- |
| `shooter.na.jac` | the game: raylib FFI bindings + the scalar `rlgl` render loop, compiled to a native binary by `jac nacompile` (pure-Python linker, no `cc`/`ld`). Has two dormant paths gated by sentinel files: a benchmark (`.bench_seconds`) and a self-screenshot (`.screenshot` → raylib `TakeScreenshot`) |
| `bench.jac` | pure-Jac benchmark orchestrator (replaces the original `demo.sh`) |
| `web/` | the same game as an `na{}`→WebAssembly + `cl{}` WebGL build (`jac start`) |

## Benchmark it

```bash
jac run bench.jac                 # build both, run each ~8s, print the FPS table
jac run bench.jac --json          # ... and emit a `BENCH_JSON {...}` line for tooling
jac run bench.jac --seconds 5     # shorter run
```

`bench.jac`:

1. detects the platform and downloads the matching **precompiled raylib** release
   into a git-ignored `.build/` cache, staging the shared library beside the binary;
2. builds the Jac shooter with `jac nacompile shooter.na.jac` → `./shooter`;
3. for the conventional-toolchain baseline, **downloads the Zig twin's source on
   demand** from the jaseci repo and the **Zig toolchain** into `.build/`, then
   builds `./shooter_zig`. This leg is best-effort - if it fails, the Jac number
   is still reported and Zig shows `n/a`;
4. runs each binary in a dormant benchmark mode (the duration is handed over via a
   sibling `.bench_seconds` file; each binary writes its `avg max frames` back to
   `.bench_result` using raylib's own file I/O) and prints the comparison.

The Jac and Zig builds link the *same* precompiled raylib and do identical
per-frame work, so they render pixel-for-pixel and track each other closely.

## Requirements

- `jac` (for `nacompile` and to run `bench.jac`).
- network access on first run (the raylib release + Zig toolchain + the Zig
  source are fetched into `.build/`).
- A GL-capable display to actually run/benchmark the binaries.

Nothing here is system-installed: the toolchain and library land in `.build/`,
which is git-ignored.
