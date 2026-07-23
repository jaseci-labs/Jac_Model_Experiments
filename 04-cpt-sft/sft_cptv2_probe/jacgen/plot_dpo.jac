"""DPO-stage graphs, parsing mlx_lm_lora's own train-log format (confirmed
DIFFERENT from mlx_lm.lora's -- lowercase `loss`/`lr`/`it/s`/`tok/s`, plus
native `acc`/`margin`/`chosen_r`/`rejected_r` fields plot_metrics.jac has no
regex for). acc/margin ARE the preference-accuracy metric this probe needs --
the trainer computes them from its own frozen reference-model comparison, no
separate scoring script required.

    JAC_DPO_LOG=path JAC_PLOT_DIR=path jac run .../plot_dpo.jac
"""

import re;
import os;
import matplotlib;
import from matplotlib { pyplot }
import from pathlib { Path }

def _num(line: str, pattern: str) -> float {
    m = re.search(pattern, line);
    if m is None { return -1.0; }
    return float(str(m.group(1)));
}

def parse(path: str) -> dict {
    loss: list = []; acc: list = []; margin: list = [];
    chosen_r: list = []; rejected_r: list = [];
    val_loss: list = []; val_acc: list = []; val_margin: list = [];
    p = Path(path);
    if not p.exists() { return {"loss": loss, "acc": acc, "margin": margin,
                                  "chosen_r": chosen_r, "rejected_r": rejected_r,
                                  "val_loss": val_loss, "val_acc": val_acc, "val_margin": val_margin}; }
    for line in p.read_text().split("\n") {
        m = re.search("Iter (\\d+)", line);
        if m is None { continue; }
        it: int = int(str(m.group(1)));
        is_val: bool = "Val loss" in line;
        if is_val {
            vl: float = _num(line, "Val loss ([0-9.]+)");
            if vl >= 0.0 { val_loss.append([it, vl]); }
            va: float = _num(line, "Val accuracy ([0-9.]+)");
            if va >= 0.0 { val_acc.append([it, va]); }
            vm: float = _num(line, "Val margin ([0-9.-]+)");
            if vm > -999.0 { val_margin.append([it, vm]); }
        } else {
            tl: float = _num(line, "loss ([0-9.]+)");
            if tl >= 0.0 { loss.append([it, tl]); }
            ac: float = _num(line, "acc ([0-9.]+)");
            if ac >= 0.0 { acc.append([it, ac]); }
            mg: float = _num(line, "margin ([0-9.-]+)");
            if mg > -999.0 { margin.append([it, mg]); }
            cr: float = _num(line, "chosen_r ([0-9.-]+)");
            if cr > -999.0 { chosen_r.append([it, cr]); }
            rr: float = _num(line, "rejected_r ([0-9.-]+)");
            if rr > -999.0 { rejected_r.append([it, rr]); }
        }
    }
    return {"loss": loss, "acc": acc, "margin": margin, "chosen_r": chosen_r,
            "rejected_r": rejected_r, "val_loss": val_loss, "val_acc": val_acc, "val_margin": val_margin};
}

def plot_pair(train_pts: list, val_pts: list, title: str, ylabel: str, outpath: str) {
    if len(train_pts) == 0 and len(val_pts) == 0 { return; }
    pyplot.figure();  # jac:ignore[E1051]
    if len(train_pts) > 0 {
        pyplot.plot([x[0] for x in train_pts], [x[1] for x in train_pts], marker="o", label="train");  # jac:ignore[E1051,E1052]
    }
    if len(val_pts) > 0 {
        pyplot.plot([x[0] for x in val_pts], [x[1] for x in val_pts], marker="s", label="val");  # jac:ignore[E1051,E1052]
    }
    pyplot.title(title);  # jac:ignore[E1051]
    pyplot.xlabel("iteration");  # jac:ignore[E1051]
    pyplot.ylabel(ylabel);  # jac:ignore[E1051]
    pyplot.legend();  # jac:ignore[E1051]
    pyplot.grid(True);  # jac:ignore[E1051]
    pyplot.savefig(outpath);  # jac:ignore[E1051]
    pyplot.close();  # jac:ignore[E1051]
    print("wrote " + outpath);
}

with entry {
    matplotlib.use("Agg");
    log_path: str = os.environ.get("JAC_DPO_LOG", "results/dpo/train.log");
    d: str = os.environ.get("JAC_PLOT_DIR", "results/dpo/images");
    Path(d).mkdir(parents=True, exist_ok=True);
    r: dict = parse(log_path);

    plot_pair(list(r["loss"]), list(r["val_loss"]), "DPO Loss", "loss", d + "/dpo_loss.png");
    plot_pair(list(r["acc"]), list(r["val_acc"]), "DPO Preference Accuracy", "accuracy", d + "/dpo_accuracy.png");
    plot_pair(list(r["margin"]), list(r["val_margin"]), "DPO Reward Margin", "margin", d + "/dpo_margin.png");

    pyplot.figure();  # jac:ignore[E1051]
    if len(r["chosen_r"]) > 0 {
        pyplot.plot([x[0] for x in r["chosen_r"]], [x[1] for x in r["chosen_r"]], marker="o", label="chosen reward");  # jac:ignore[E1051,E1052]
    }
    if len(r["rejected_r"]) > 0 {
        pyplot.plot([x[0] for x in r["rejected_r"]], [x[1] for x in r["rejected_r"]], marker="s", label="rejected reward");  # jac:ignore[E1051,E1052]
    }
    pyplot.title("DPO Chosen vs Rejected Reward");  # jac:ignore[E1051]
    pyplot.xlabel("iteration");  # jac:ignore[E1051]
    pyplot.ylabel("reward");  # jac:ignore[E1051]
    pyplot.legend();  # jac:ignore[E1051]
    pyplot.grid(True);  # jac:ignore[E1051]
    pyplot.savefig(d + "/dpo_rewards.png");  # jac:ignore[E1051]
    pyplot.close();  # jac:ignore[E1051]
    print("wrote " + d + "/dpo_rewards.png");
}
