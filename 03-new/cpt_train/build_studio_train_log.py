"""Rewrites run_epoch_loop.py's raw epoch_loop.log into a train.log Studio's
metrics.parse_train_log() can chart correctly. Each leg is a fresh mlx_lm
subprocess (Task 13's isolation-per-leg design) so its own 'Iter N' counter
restarts at 1 every leg -- a plain concatenation would report last_iter=544
(one leg's worth) instead of the true cumulative 6528, and every leg's
points would overlap on the same x-range. Reads each leg header's
--done-steps value as that leg's offset and rewrites 'Iter N' -> 'Iter
(N+offset)' for every line until the next leg header."""
import re
import sys
from pathlib import Path

LEG_HEADER = re.compile(r"^=== leg \d+/\d+: .*--done-steps (\d+)")
ITER_LINE = re.compile(r"Iter (\d+)")


def rewrite(text: str) -> str:
    out_lines = []
    offset = 0
    for line in text.split("\n"):
        m = LEG_HEADER.match(line)
        if m:
            offset = int(m.group(1))
            out_lines.append(line)
            continue
        if offset and "Iter " in line:
            line = ITER_LINE.sub(lambda mm: f"Iter {int(mm.group(1)) + offset}", line)
        out_lines.append(line)
    return "\n".join(out_lines)


def main():
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "03-new/results/cpt-v2/epoch_loop.log")
    dst = Path(sys.argv[2] if len(sys.argv) > 2 else "03-new/results/cpt-v2/train.log")
    dst.write_text(rewrite(src.read_text()))
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
