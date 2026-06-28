"""Record ALL ladder results into docs/rl/02-results.md (the living results doc).

Runs show_ladder.jac (the single source of truth for the pivot table) and wraps
its output with a header, the method/reward summary, and provenance. Re-run after
each ladder stage to refresh the doc.

  jac run rl/record_results.jac
"""

import os;
import datetime;
import subprocess;
import from pathlib { Path }

glob ROOT: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)));
glob JAC: str = os.path.join(ROOT, ".venv", "bin", "jac") if os.path.exists(os.path.join(ROOT, ".venv", "bin", "jac")) else "jac";

with entry {
    p = subprocess.run([JAC, "run", "rl/show_ladder.jac"], capture_output=True, text=True, cwd=ROOT);
    table = p.stdout.strip();
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M");

    doc = (
        "# RL Ladder — Results\n\n"
        + "*Auto-recorded " + ts + " by `rl/record_results.jac` from `results/rl_ladder.jsonl`. "
        + "Re-run after each ladder stage. Live version also renders in the Studio RL section.*\n\n"
        + "**Method:** SFT-first ladder (train-N = 1,3,5,10,20,all) × 2 models × conditions "
        + "{base, SFT(gold), SFT+GRPO, raw-GRPO control}. Holdout is file-disjoint + decontaminated; "
        + "headline = exact-stdout pass@1 with Wilson 95% CI + sampled pass@k.\n\n"
        + "**Reward (GRPO):** tiered monotone — exact=1.0 > runs-wrong≤0.80 > compiles≤0.35 > none≤0.15, "
        + "with a dense body-sim term in every tier (breaks the σ=0 trap).\n\n"
        + "## Ladder cells\n\n"
        + table + "\n\n"
        + "See [strat.md](strat.md) for hypotheses, [references.md](references.md) for the "
        + "literature, and [RL_WEEKEND_RESULTS.md](RL_WEEKEND_RESULTS.md) for the prior GRPO archive.\n"
    );
    Path(os.path.join(ROOT, "docs", "rl", "02-results.md")).write_text(doc);
    print("wrote docs/rl/02-results.md (" + str(len(table)) + " chars of table)");
}
