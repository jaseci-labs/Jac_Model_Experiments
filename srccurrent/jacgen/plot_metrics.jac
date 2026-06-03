"""Render training graphs as PNGs (matplotlib).

SCAFFOLD — needs `pip install matplotlib` in your jac Python env. Parses the
mlx_lm.lora training log + the per-checkpoint learning-curve metrics and writes
results/*.png: train loss, val loss, learning rate, tokens/sec, and the holdout
test-pass learning curve. The zero-dep ASCII view is dashboard.jac.

    JAC_TRAIN_LOG=results/x-train.log JAC_METRICS=results/x-metrics.jsonl \
        jac run srccurrent/jacgen/plot_metrics.jac
"""

import json;
import re;
import os;
import matplotlib;
import from matplotlib { pyplot }
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


def plot_one(points: list, title: str, ylabel: str, outpath: str) {
    if len(points) == 0 { return; }
    xs: list = [p[0] for p in points];
    ys: list = [p[1] for p in points];
    pyplot.figure();
    pyplot.plot(xs, ys, marker="o");
    pyplot.title(title);
    pyplot.xlabel("iteration");
    pyplot.ylabel(ylabel);
    pyplot.grid(True);
    pyplot.savefig(outpath);
    pyplot.close();
    print("  wrote " + outpath);
}


with entry {
    matplotlib.use("Agg");
    train_log: str = os.environ.get("JAC_TRAIN_LOG", "results/train.log");
    metrics: str = os.environ.get("JAC_METRICS", "results/metrics.jsonl");

    t: dict = parse_train(train_log);
    plot_one(t["train"], "Train Loss", "loss", "results/train_loss.png");
    plot_one(t["val"], "Validation Loss", "loss", "results/val_loss.png");
    plot_one(t["lr"], "Learning Rate", "lr", "results/learning_rate.png");
    plot_one(t["tps"], "Throughput", "tokens/sec", "results/tokens_per_sec.png");

    lc: list = [];
    mp = Path(metrics);
    if mp.exists() {
        for line in mp.read_text().split("\n") {
            if line.strip() == "" { continue; }
            r: dict[str, Any] = json.loads(line);
            lc.append([r["step"], r["test_pass_pct"]]);
        }
    }
    plot_one(lc, "Holdout test-pass % per checkpoint (learning curve)", "test-pass %",
             "results/learning_curve.png");
    print("graphs in results/");
}
