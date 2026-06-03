"""Zero-dependency ASCII training dashboard (no matplotlib needed).

Parses the mlx_lm.lora training log (results/<name>-train.log) for train/val
loss, learning rate, tokens/sec, and the per-checkpoint learning-curve metrics
(results/<name>-metrics.jsonl from eval_probe), and renders terminal charts.
run_probe.sh re-runs this on a timer for a live view.

    jac run srccurrent/jacgen/dashboard.jac
Override paths via env JAC_TRAIN_LOG / JAC_METRICS.
"""

import json;
import re;
import os;
import from pathlib { Path }
import from typing { Any }


def _num(line: str, pattern: str) -> float {
    m = re.search(pattern, line);
    if m is None { return -1.0; }
    return float(m.group(1));
}


def parse_train(path: str) -> dict {
    train: list = [];
    val: list = [];
    lr: list = [];
    tps: list = [];
    p = Path(path);
    if not p.exists() { return {"train": train, "val": val, "lr": lr, "tps": tps}; }
    for line in p.read_text().split("\n") {
        m = re.search("Iter (\\d+)", line);
        if m is None { continue; }
        it: int = int(m.group(1));
        tl: float = _num(line, "Train loss ([0-9.]+)");
        if tl >= 0.0 { train.append([it, tl]); }
        vl: float = _num(line, "Val loss ([0-9.]+)");
        if vl >= 0.0 { val.append([it, vl]); }
        lv: float = _num(line, "Learning Rate ([0-9.eE+-]+)");
        if lv >= 0.0 { lr.append([it, lv]); }
        tv: float = _num(line, "Tokens/sec ([0-9.]+)");
        if tv >= 0.0 { tps.append([it, tv]); }
    }
    return {"train": train, "val": val, "lr": lr, "tps": tps};
}


def bar(value: float, lo: float, hi: float, width: int) -> str {
    if hi <= lo { return "#"; }
    n: int = int((value - lo) / (hi - lo) * width);
    out: str = "";
    for _ in range(n) { out += "#"; }
    return out;
}


def render(label: str, points: list) {
    if len(points) == 0 {
        print("--- " + label + ": (no data yet) ---");
        return;
    }
    lo: float = points[0][1];
    hi: float = points[0][1];
    for pt in points {
        if pt[1] < lo { lo = pt[1]; }
        if pt[1] > hi { hi = pt[1]; }
    }
    print("--- " + label + "  [min " + str(lo) + ", max " + str(hi) + "] ---");
    start: int = 0;
    if len(points) > 12 { start = len(points) - 12; }
    for i in range(start, len(points)) {
        pt = points[i];
        print("  " + str(pt[0]) + "\t" + str(pt[1]) + "\t" + bar(pt[1], lo, hi, 28));
    }
}


with entry {
    train_log: str = os.environ.get("JAC_TRAIN_LOG", "results/train.log");
    metrics: str = os.environ.get("JAC_METRICS", "results/metrics.jsonl");

    t: dict = parse_train(train_log);
    print("==================== TRAINING DASHBOARD ====================");
    render("train loss", t["train"]);
    render("val loss", t["val"]);
    render("learning rate", t["lr"]);
    render("tokens/sec", t["tps"]);

    lc: list = [];
    mp = Path(metrics);
    if mp.exists() {
        for line in mp.read_text().split("\n") {
            if line.strip() == "" { continue; }
            r: dict[str, Any] = json.loads(line);
            lc.append([r["step"], r["test_pass_pct"]]);
        }
    }
    print("");
    render("HOLDOUT test-pass % per checkpoint (LEARNING CURVE)", lc);
    print("============================================================");
}
