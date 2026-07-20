"""This is Jac - capture/automation script (written in Jac).

Drives the sibling `raylib_shooter` example and writes artifacts into
`assets/captures/` for the site to embed:

  1. Benchmark   - runs `raylib_shooter/demo.sh --bench`, parses the Jac-vs-Zig
                   avg/max FPS table, and writes `benchmark.json` (the
                   NativeSection fetches it at runtime). RELIABLE.
  2. Screenshots - best-effort: launches the freshly-built native binaries,
                   grabs their window with ImageMagick `import`, and saves
                   `shooter_jac.png` / `shooter_zig.png`. Needs an X display;
                   on headless / pure-Wayland boxes it records "skipped" and
                   keeps going (the live WASM section still shows the game).
  3. Manifest    - `manifest.json` records what each step did + timestamps.

Each step is independently guarded: a failure logs and the run continues.

    jac run capture.jac                 # benchmark + screenshots
    jac run capture.jac --skip-bench    # screenshots only
    jac run capture.jac --skip-shots    # benchmark only
"""

import os;
import sys;
import json;
import time;
import shutil;
import subprocess;
import datetime;

glob HERE: str = os.path.dirname(os.path.abspath(__file__));

"""Locate the sibling `raylib_shooter` example by walking up from HERE, so this
script works wherever `this_is_jac` lives (jac/examples/, _planning/, or a
standalone checkout next to raylib_shooter). Returns "" if not found - the
benchmark/screenshot steps then skip cleanly."""
def _find_shooter_dir -> str {
    d = HERE;
    for _i in range(10) {
        sibling = os.path.join(d, "raylib_shooter");
        if os.path.exists(os.path.join(sibling, "bench.jac")) {
            return sibling;
        }
        in_repo = os.path.join(d, "jac", "examples", "raylib_shooter");
        if os.path.exists(os.path.join(in_repo, "bench.jac")) {
            return in_repo;
        }
        parent = os.path.dirname(d);
        if parent == d {
            break;
        }
        d = parent;
    }
    return "";
}

glob SHOOTER_DIR: str = _find_shooter_dir();
glob CAPTURES: str = os.path.join(HERE, "assets", "captures");
glob BENCH_TIMEOUT: int = 900;


def _today -> str {
    return datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d");
}

def _iso -> str {
    return datetime.datetime.now(datetime.UTC).replace(tzinfo=None).isoformat() + "Z";
}

def _platform -> str {
    try {
        u = os.uname();
        return f"{u.sysname} {u.machine}";
    } except Exception as e {
        return "unknown";
    }
}

"""Resolve the `jac` CLI (PATH, then the active venv's bin) to run bench.jac."""
def _jac_exe -> str {
    found = shutil.which("jac");
    if found {
        return found;
    }
    cand = os.path.join(sys.prefix, "bin", "jac");
    if os.path.exists(cand) {
        return cand;
    }
    return "jac";
}

"""Extract the `BENCH_JSON {...}` payload emitted by `bench.jac --json`
(returns the decoded dict, or None if absent)."""
def _parse_bench_json(text: str) -> any {
    marker = "BENCH_JSON ";
    for line in text.splitlines() {
        s = line.strip();
        if s.startswith(marker) {
            try {
                return json.loads(s[len(marker):]);
            } except Exception as e {
                return None;
            }
        }
    }
    return None;
}


obj CaptureRunner {
    has results: list[dict] = [];

    def record(name: str, status: str, detail: str) -> None {
        self.results.append({"step": name, "status": status, "detail": detail});
        mark = "ok  " if status == "ok" else ("skip" if status == "skip" else "FAIL");
        print(f"  [{mark}] {name}: {detail}");
    }

    # ── 1. Benchmark ─────────────────────────────────────────────────────────
    def run_benchmark -> None {
        if not SHOOTER_DIR {
            self.record("benchmark", "fail", "raylib_shooter example not found (walked up from script dir)");
            return;
        }
        bench = os.path.join(SHOOTER_DIR, "bench.jac");
        if not os.path.exists(bench) {
            self.record("benchmark", "fail", f"bench.jac not found at {bench}");
            return;
        }
        print(">> benchmarking via bench.jac (downloads raylib/zig + builds both, ~8s each)...");
        try {
            proc = subprocess.run(
                [_jac_exe(), "run", bench, "--json"],
                cwd=SHOOTER_DIR,
                capture_output=True,
                text=True,
                timeout=BENCH_TIMEOUT
            );
        } except subprocess.TimeoutExpired as e {
            self.record("benchmark", "fail", f"timed out after {BENCH_TIMEOUT}s");
            return;
        } except Exception as e {
            self.record("benchmark", "fail", f"{e}");
            return;
        }
        out = proc.stdout + "\n" + proc.stderr;
        parsed = _parse_bench_json(out);
        if parsed is None {
            tail = "\n".join(out.strip().splitlines()[-8:]);
            self.record("benchmark", "fail", "no BENCH_JSON from bench.jac (no display/GPU?)");
            print(f"     --- last output ---\n{tail}");
            return;
        }
        data = {
            "captured_at": _today(),
            "platform": _platform(),
            "note": "Measured by capture.jac via bench.jac",
            "jac": parsed["jac"],
            "zig": parsed["zig"]
        };
        with open(os.path.join(CAPTURES, "benchmark.json"), "w") as f {
            json.dump(data, f, indent=2);
        }
        ja = parsed["jac"]["avg"];
        za = parsed["zig"]["avg"];
        self.record("benchmark", "ok", f"Jac {ja} / Zig {za} avg fps -> benchmark.json");
    }

    # ── 2. Screenshot (via the shooter's own raylib TakeScreenshot) ───────────
    """A frame with almost no distinct colors means the GL framebuffer couldn't
    be read back. The real shooter frame is deliberately flat-shaded - black sky,
    grid lines, solid cubes, FPS text - so it only has ~6 distinct colors; a
    failed readback has 1-2. Best-effort: with no `identify` available we accept."""
    def _looks_blank(path: str) -> bool {
        try {
            r = subprocess.run(
                ["identify", "-format", "%k", path],
                capture_output=True, text=True, timeout=15
            );
            return int(r.stdout.strip()) < 3;
        } except Exception as e {
            return False;
        }
    }

    """Drive the shooter's dormant self-screenshot path: write the warmup frame
    into `.screenshot`, run the (already-built) binary - which renders, calls
    raylib's `TakeScreenshot`, and exits - then move the PNG into assets/captures/.
    The engine writes the PNG from its own GL framebuffer, so this needs no
    external window grab and works on any display server that can actually read
    that buffer back."""
    def capture_self(outname: str, bin_name: str) -> None {
        if not SHOOTER_DIR {
            self.record(outname, "skip", "raylib_shooter not found");
            return;
        }
        bin_path = os.path.join(SHOOTER_DIR, bin_name);
        if not os.path.exists(bin_path) {
            self.record(outname, "skip", f"{bin_name} not built (run without --skip-bench first)");
            return;
        }
        shot_file = os.path.join(SHOOTER_DIR, ".screenshot");
        bench_file = os.path.join(SHOOTER_DIR, ".bench_seconds");
        produced = os.path.join(SHOOTER_DIR, "shooter_shot.png");
        if os.path.exists(bench_file) {
            os.remove(bench_file);
        }
        if os.path.exists(produced) {
            os.remove(produced);
        }
        # Two attempts: the platform's native GL first, then Mesa software
        # rendering (llvmpipe) - on WSLg the D3D12-backed driver renders fine
        # but glReadPixels comes back blank, while llvmpipe reads back exactly
        # what it drew. Same real game either way.
        attempts: list[dict[str, str]] = [{}, {"LIBGL_ALWAYS_SOFTWARE": "1"}];
        got = False;
        for extra_env in attempts {
            with open(shot_file, "w") as f {
                f.write("60");
            }
            if os.path.exists(produced) {
                os.remove(produced);
            }
            run_env = dict(os.environ);
            run_env.update(extra_env);
            try {
                subprocess.run(
                    [bin_path], cwd=SHOOTER_DIR,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    timeout=60, env=run_env
                );
            } except Exception as e {
                self.record(outname, "fail", f"{e}");
            }
            if os.path.exists(shot_file) {
                os.remove(shot_file);
            }
            if os.path.exists(produced) and not self._looks_blank(produced) {
                got = True;
                break;
            }
        }
        if not got {
            if os.path.exists(produced) {
                os.remove(produced);
            }
            self.record(outname, "skip", "no readable frame (no display/GL) - kept existing image");
            return;
        }
        dst = os.path.join(CAPTURES, outname);
        shutil.move(produced, dst);
        kb = os.path.getsize(dst) // 1024;
        self.record(outname, "ok", f"{outname} ({kb} KB) via in-engine TakeScreenshot");
    }

    def capture_screenshots -> None {
        print(">> capturing native shooter screenshots via the engines' TakeScreenshot...");
        self.capture_self("shooter_jac.png", "shooter");
        self.capture_self("shooter_zig.png", "shooter_zig");
    }

    # ── 3. Manifest + summary ──────────────────────────────────────────────────
    def write_manifest -> None {
        manifest = {
            "generated_at": _iso(),
            "platform": _platform(),
            "steps": self.results
        };
        with open(os.path.join(CAPTURES, "manifest.json"), "w") as f {
            json.dump(manifest, f, indent=2);
        }
    }

    def summary -> int {
        oks = len([r for r in self.results if r["status"] == "ok"]);
        print("\n" + "=" * 56);
        print(f"  capture complete - {oks}/{len(self.results)} step(s) ok");
        print("  artifacts in: assets/captures/");
        print("=" * 56);
        return 0 if oks > 0 else 1;
    }
}


with entry {
    skip_bench = "--skip-bench" in sys.argv;
    skip_shots = "--skip-shots" in sys.argv;

    print("=" * 56);
    print("  this-is-jac :: capture");
    print(f"  shooter dir: {SHOOTER_DIR}");
    print("=" * 56);

    os.makedirs(CAPTURES, exist_ok=True);
    runner = CaptureRunner();
    if not skip_bench {
        runner.run_benchmark();
    }
    if not skip_shots {
        runner.capture_screenshots();
    }
    runner.write_manifest();
    sys.exit(runner.summary());
}
