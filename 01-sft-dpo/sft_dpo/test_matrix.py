"""Self-check for the bake-off parsers (ponytail: asserts, no framework).

Imports make_comparison for its functions only — plotting is guarded under
main(), so importing has no side effects.
"""
import make_comparison as mc


def test_pass_pct(tmp):
    p = tmp / "finetuned.txt"
    p.write_text("=== conversion probe (mlx) on 150 holdout tasks ===\n"
                 "runs (compiles+executes): 96%  (145/150)\n"
                 "cross-compiled test pass: 93%  (140/150)\n")
    assert mc.pass_pct(str(p)) == 93
    assert mc.pass_pct(str(tmp / "missing.txt")) is None


def test_avg_sim(tmp):
    p = tmp / "idiom-metrics.jsonl"
    p.write_text('{"total":150,"avg_sim":0.968}\n{"total":150,"avg_sim":0.338}\n')
    assert mc.avg_sim(str(p)) == 0.338          # last row wins
    assert mc.avg_sim(str(tmp / "missing.jsonl")) is None


if __name__ == "__main__":
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as d:
        test_pass_pct(pathlib.Path(d))
        test_avg_sim(pathlib.Path(d))
    print("matrix parser self-check: PASS")
