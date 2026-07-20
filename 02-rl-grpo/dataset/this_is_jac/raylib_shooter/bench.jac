"""raylib cube shooter benchmark - pure-Jac orchestration (replaces demo.sh).

Builds the Jac-native shooter (`jac nacompile shooter.na.jac`) and, on demand, a
Zig twin - downloading the Zig *source* from the jaseci repo plus the Zig
toolchain and precompiled raylib into a git-ignored `.build/` cache - then runs
each for a fixed number of seconds and reports avg/max FPS. No shell scripts and
no committed Zig source live in this repo.

    jac run bench.jac                 # benchmark both, print a table
    jac run bench.jac --json          # ... and emit a `BENCH_JSON {...}` line
    jac run bench.jac --seconds 5     # shorter run (default 8)

The Zig leg is best-effort: if its source/toolchain download or build fails, the
Jac number is still reported and Zig shows `n/a`.
"""

import os;
import sys;
import json;
import shutil;
import tarfile;
import platform;
import subprocess;
import from urllib.request { urlretrieve }

glob HERE: str = os.path.dirname(os.path.abspath(__file__));
glob BUILD_DIR: str = os.path.join(HERE, ".build");
glob RAYLIB_VERSION: str = "6.0";
glob ZIG_VERSION: str = "0.13.0";
glob ZIG_SOURCE_URL: str = "https://raw.githubusercontent.com/jaseci-labs/jaseci/main/jac/examples/raylib_shooter/shooter.zig";

obj BenchResult {
    has avg: float = 0.0,
        max: float = 0.0,
        frames: int = 0;
}


def _log(msg: str) -> None {
    print(">> " + msg);
}

"""Resolve the `jac` CLI that should drive `nacompile` (prefer PATH, then the
active venv's bin)."""
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

"""Download `url` to `dest` (cached: skip if present)."""
def _download(url: str, dest: str) -> None {
    if os.path.exists(dest) {
        _log("using cached " + os.path.basename(dest));
        return;
    }
    _log("fetching " + url);
    tmp = dest + ".part";
    urlretrieve(url, tmp);
    os.replace(tmp, dest);
}

"""Map this platform to its raylib release asset + library naming."""
def _platform_info -> dict[str, str] {
    osname = platform.system();
    arch = platform.machine();
    if osname == "Linux" {
        if arch in ["x86_64", "amd64"] {
            asset = "raylib-" + RAYLIB_VERSION + "_linux_amd64.tar.gz";
        } elif arch in ["aarch64", "arm64"] {
            asset = "raylib-" + RAYLIB_VERSION + "_linux_arm64.tar.gz";
        } elif arch in ["i386", "i686"] {
            asset = "raylib-" + RAYLIB_VERSION + "_linux_i386.tar.gz";
        } else {
            raise RuntimeError("Unsupported Linux architecture: " + arch);
        }
        return {"os": "Linux", "arch": arch, "asset": asset, "rpath": "$ORIGIN"};
    }
    if osname == "Darwin" {
        asset = "raylib-" + RAYLIB_VERSION + "_macos.tar.gz";
        return {"os": "Darwin", "arch": arch, "asset": asset, "rpath": "@loader_path"};
    }
    raise RuntimeError("Unsupported OS: " + osname);
}

"""True for a staged raylib shared-library filename on this OS."""
def _is_lib(fn: str, osname: str) -> bool {
    if osname == "Darwin" {
        return fn.startswith("libraylib") and fn.endswith(".dylib");
    }
    return fn.startswith("libraylib.so");
}

"""Download + extract the precompiled raylib release and stage its shared
library set (preserving the SONAME version symlinks) beside the binaries."""
def ensure_raylib(info: dict[str, str]) -> None {
    osname = info["os"];
    asset = info["asset"];
    os.makedirs(BUILD_DIR, exist_ok=True);
    url = "https://github.com/raysan5/raylib/releases/download/" + RAYLIB_VERSION + "/" + asset;
    tarball = os.path.join(BUILD_DIR, asset);
    _download(url, tarball);

    extract_dir = os.path.join(BUILD_DIR, "extracted");
    if os.path.exists(extract_dir) {
        shutil.rmtree(extract_dir);
    }
    os.makedirs(extract_dir);
    with tarfile.open(tarball, "r:gz") as t {
        t.extractall(extract_dir);
    }

    # Find the real (non-symlink) library to locate its source directory.
    real = "";
    for (walk_root, dirs, files) in os.walk(extract_dir) {
        for fn in files {
            fn_s = str(fn);
            full = os.path.join(str(walk_root), fn_s);
            if _is_lib(fn_s, osname) and not os.path.islink(full) {
                real = full;
                break;
            }
        }
        if real {
            break;
        }
    }
    if not real {
        raise RuntimeError("could not locate libraylib in the release tree");
    }
    lib_src_dir = os.path.dirname(real);

    # Stage every libraylib.* entry, recreating symlinks (cp -P equivalent) so
    # whatever name a binary records (plain name for Jac, SONAME for Zig) resolves.
    for fn in os.listdir(lib_src_dir) {
        if not _is_lib(fn, osname) {
            continue;
        }
        src = os.path.join(lib_src_dir, fn);
        dst = os.path.join(HERE, fn);
        if os.path.lexists(dst) {
            os.remove(dst);
        }
        if os.path.islink(src) {
            os.symlink(os.readlink(src), dst);
        } else {
            shutil.copy2(src, dst);
        }
    }
    _log("staged raylib shared library set -> ./");
}

"""Compile shooter.na.jac -> ./shooter with `jac nacompile` (no cc/ld)."""
def build_jac -> str {
    _log("compiling: jac nacompile shooter.na.jac");
    r = subprocess.run([_jac_exe(), "nacompile", "shooter.na.jac"], cwd=HERE);
    if r.returncode != 0 {
        raise RuntimeError("jac nacompile failed (exit " + str(r.returncode) + ")");
    }
    return os.path.join(HERE, "shooter");
}

"""Locate a Zig toolchain, downloading the single-archive distribution into
`.build/` if `zig` is not already on PATH. Returns the zig executable path."""
def ensure_zig -> str {
    found = shutil.which("zig");
    if found {
        return found;
    }
    arch = platform.machine();
    if arch in ["x86_64", "amd64"] {
        zarch = "x86_64";
    } elif arch in ["aarch64", "arm64"] {
        zarch = "aarch64";
    } else {
        raise RuntimeError("no prebuilt Zig auto-download for architecture: " + arch);
    }
    zos = "linux" if platform.system() == "Linux" else "macos";
    asset = "zig-" + zos + "-" + zarch + "-" + ZIG_VERSION + ".tar.xz";
    url = "https://ziglang.org/download/" + ZIG_VERSION + "/" + asset;
    tarball = os.path.join(BUILD_DIR, asset);
    zig_root = os.path.join(BUILD_DIR, "zig-" + zos + "-" + zarch + "-" + ZIG_VERSION);
    zig_exe = os.path.join(zig_root, "zig");
    if not os.path.exists(zig_exe) {
        _download(url, tarball);
        _log("unpacking " + asset + " -> .build/");
        with tarfile.open(tarball, "r:xz") as t {
            t.extractall(BUILD_DIR);
        }
    }
    if not os.path.exists(zig_exe) {
        raise RuntimeError("zig toolchain not found after unpack");
    }
    return zig_exe;
}

"""Download the Zig twin's source on demand from the jaseci repo, then build
./shooter_zig against the staged sibling library. The source lives in `.build/`
(never committed) and the linker bakes an $ORIGIN/@loader_path runpath."""
def build_zig(info: dict[str, str]) -> str {
    src = os.path.join(BUILD_DIR, "shooter.zig");
    os.makedirs(BUILD_DIR, exist_ok=True);
    _download(ZIG_SOURCE_URL, src);
    zig = ensure_zig();
    _log("compiling: zig build-exe (downloaded shooter.zig) -> ./shooter_zig");
    r = subprocess.run(
        [
            zig, "build-exe", src,
            "-O", "ReleaseFast",
            "-lc",
            "-L" + HERE, "-lraylib",
            "-rpath", info["rpath"],
            "-femit-bin=shooter_zig"
        ],
        cwd=HERE
    );
    leftover = os.path.join(HERE, "shooter_zig.o");
    if os.path.exists(leftover) {
        os.remove(leftover);
    }
    if r.returncode != 0 {
        raise RuntimeError("zig build failed (exit " + str(r.returncode) + ")");
    }
    return os.path.join(HERE, "shooter_zig");
}

"""Run one binary in benchmark mode for `seconds` and read back its result.

Hands the duration over via a sibling `.bench_seconds` file (the binary reads it
with raylib's own file helpers) and reads the `BENCH_RESULT avg max frames
seconds` line the binary writes to `.bench_result`. Returns the parsed dict, or
None if the binary produced no result (e.g. no display/GPU)."""
def run_bench(binary: str, seconds: int) -> BenchResult | None {
    seconds_file = os.path.join(HERE, ".bench_seconds");
    result_file = os.path.join(HERE, ".bench_result");
    if os.path.exists(result_file) {
        os.remove(result_file);
    }
    with open(seconds_file, "w") as f {
        f.write(str(seconds));
    }
    try {
        subprocess.run(
            [binary], cwd=HERE,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=seconds + 20
        );
    } except Exception as e {
        # timeout / nonzero / no display - fall through to the result check
        _log("run note: " + str(e));
    }
    if os.path.exists(seconds_file) {
        os.remove(seconds_file);
    }
    if not os.path.exists(result_file) {
        return None;
    }
    line = open(result_file).read().strip();
    os.remove(result_file);
    parts = line.split();
    if len(parts) >= 4 and parts[0] == "BENCH_RESULT" {
        try {
            return BenchResult(
                avg=float(parts[1]), max=float(parts[2]), frames=int(parts[3])
            );
        } except ValueError as e {
            return None;
        }
    }
    return None;
}

"""One-decimal string without the `round` builtin (its 2-arg overload trips
`jac check`)."""
def _one_dp(x: float) -> str {
    return str(int(x * 10.0 + 0.5) / 10.0);
}

def _fmt_row(label: str, d: BenchResult | None) -> str {
    if d is None {
        return "  " + label.ljust(6) + "n/a".rjust(14) + "n/a".rjust(14) + "n/a".rjust(12);
    }
    return "  " + label.ljust(6) + _one_dp(d.avg).rjust(14) + _one_dp(d.max).rjust(14) + str(d.frames).rjust(12);
}


with entry {
    args = sys.argv;
    as_json = "--json" in args;
    seconds = 8;
    if "--seconds" in args {
        i = args.index("--seconds");
        if i + 1 < len(args) {
            seconds = int(args[i + 1]);
        }
    }

    info = _platform_info();
    _log("platform : " + info["os"] + " / " + info["arch"]);
    ensure_raylib(info);

    jac_result: BenchResult | None = None;
    zig_result: BenchResult | None = None;

    try {
        jac_bin = build_jac();
        jac_result = run_bench(jac_bin, seconds);
    } except Exception as e {
        _log("jac build/bench failed: " + str(e));
    }

    try {
        zig_bin = build_zig(info);
        zig_result = run_bench(zig_bin, seconds);
    } except Exception as e {
        _log("zig leg skipped: " + str(e));
    }

    print("");
    print("====== raylib cube shooter benchmark (" + str(seconds) + "s each, uncapped) ======");
    print("  " + "build".ljust(6) + "avg FPS".rjust(14) + "max FPS".rjust(14) + "frames".rjust(12));
    print(_fmt_row("Jac", jac_result));
    print(_fmt_row("Zig", zig_result));
    print("=" * 63);

    if as_json {
        jac_d = {"avg": jac_result.avg, "max": jac_result.max, "frames": jac_result.frames}
            if jac_result
            else {"avg": 0.0, "max": 0.0, "frames": 0};
        zig_d = {"avg": zig_result.avg, "max": zig_result.max, "frames": zig_result.frames}
            if zig_result
            else {"avg": 0.0, "max": 0.0, "frames": 0};
        print("BENCH_JSON " + json.dumps({"jac": jac_d, "zig": zig_d}));
    }

    if jac_result is None {
        sys.exit(1);
    }
}
