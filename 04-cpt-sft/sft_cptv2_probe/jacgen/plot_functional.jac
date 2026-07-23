"""Per-category functional pass-rate learning curve, from eval_functional.jac's
metrics_functional.jsonl. Extends plot_metrics.jac's style -- separate file
because it groups by (category, gate_class) instead of a single scalar series.

    JAC_FUNC_METRICS=path JAC_PLOT_DIR=path jac run .../plot_functional.jac
"""

import json;
import os;
import matplotlib;
import from matplotlib { pyplot }
import from pathlib { Path }

with entry {
    matplotlib.use("Agg");
    metrics: str = os.environ.get("JAC_FUNC_METRICS", "results/sft/metrics_functional.jsonl");
    d: str = os.environ.get("JAC_PLOT_DIR", "results/sft/images");
    Path(d).mkdir(parents=True, exist_ok=True);

    p = Path(metrics);
    series: dict = {};   # category -> list[(step, pct)]
    if p.exists() {
        for line in p.read_text().split("\n") {
            if not line.strip().startswith("{") { continue; }
            r: dict = dict(json.loads(line));
            cat: str = str(r["category"]);
            if cat not in series { series[cat] = []; }
            series[cat].append((r["step"], r["runs_pct"]));
        }
    }

    pyplot.figure(figsize=(9, 6));  # jac:ignore[E1051]
    for (cat, pts) in series.items() {
        xs: list = [x[0] for x in pts];
        ys: list = [x[1] for x in pts];
        pyplot.plot(xs, ys, marker="o", label=cat);  # jac:ignore[E1051,E1052]
    }
    pyplot.title("Functional pass rate (runs) by category over training");  # jac:ignore[E1051]
    pyplot.xlabel("iteration");  # jac:ignore[E1051]
    pyplot.ylabel("runs %");  # jac:ignore[E1051]
    pyplot.ylim(0, 100);  # jac:ignore[E1051]
    pyplot.legend();  # jac:ignore[E1051]
    pyplot.grid(True);  # jac:ignore[E1051]
    pyplot.savefig(d + "/functional_pass_rate.png");  # jac:ignore[E1051]
    pyplot.close();  # jac:ignore[E1051]
    print("wrote " + d + "/functional_pass_rate.png");
}
