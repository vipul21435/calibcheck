import json
from pathlib import Path

import numpy as np
import pytest

from calibcheck.cli import main


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return str(path)


def confidence_rows(seed, n=600, sharpen=2.5):
    rng = np.random.default_rng(seed)
    score = rng.normal(1.0, 1.0, size=n)
    correct = rng.random(n) < 1 / (1 + np.exp(-score))
    conf = np.round(1 / (1 + np.exp(-sharpen * score)), 3)
    return [
        {"id": f"q{i}", "topic": f"t{i % 12}", "confidence": float(c), "correct": bool(y)}
        for i, (c, y) in enumerate(zip(conf, correct, strict=True))
    ]


def vector_rows(seed, n=500, scale=2.0):
    rng = np.random.default_rng(seed)
    logits = rng.normal(0.0, 1.5, size=(n, 4))
    p = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    labels = [int(rng.choice(4, p=row)) for row in p]
    return [
        {"id": f"q{i}", "topic": f"t{i % 10}", "logits": (logits[i] * scale).tolist(), "label": y}
        for i, y in enumerate(labels)
    ]


@pytest.fixture
def overconfident(tmp_path):
    return write_jsonl(tmp_path / "model_a.jsonl", confidence_rows(0))


def run_json(capsys, args):
    """Run a command and parse its JSON report (asked for unless a format is given)."""
    if "--format" not in args:
        args = [*args, "--format", "json"]
    assert main(args) == 0
    return json.loads(capsys.readouterr().out)


def test_metrics_with_bootstrap_intervals(overconfident, capsys):
    args = ["metrics", overconfident, "--bootstrap", "200", "--seed", "3", "--format", "json"]
    report = run_json(capsys, args)
    intervals = report["intervals"]
    assert intervals["n_resamples"] == 200 and intervals["level"] == 0.95
    assert intervals["n_groups"] is None
    assert set(intervals["bounds"]) == set(report["metrics"])
    for key in ("accuracy", "brier", "ece_debiased", "auroc", "aurc", "log_loss"):
        lo, hi = intervals["bounds"][key]
        assert lo < report["metrics"][key] < hi, key
    assert run_json(capsys, args)["intervals"] == intervals
    other = run_json(capsys, args[:5] + ["4"] + args[6:])
    assert other["intervals"]["bounds"] != intervals["bounds"]


def test_metrics_markdown_shows_intervals_and_groups(overconfident, capsys):
    assert main(["metrics", overconfident, "--bootstrap", "50", "--group", "topic"]) == 0
    out = capsys.readouterr().out
    assert "95% intervals: percentile bootstrap over 12 groups, 50 resamples." in out
    assert "| metric | value | 95% interval |" in out
    assert "| ECE, debiased | 0." in out
    assert "| accuracy at 80% coverage | 0." in out
    assert main(["metrics", overconfident, "--bootstrap", "50", "--coverage", "0.5"]) == 0
    out = capsys.readouterr().out
    assert "percentile bootstrap over items, 50 resamples." in out
    assert "| accuracy at 50% coverage | 0." in out


def test_metrics_from_logit_vectors(tmp_path, capsys):
    path = write_jsonl(tmp_path / "v.jsonl", vector_rows(1))
    report = run_json(capsys, ["metrics", path, "--logits", "logits", "--bootstrap", "20"])
    assert report == run_json(capsys, ["metrics", path, "--logits", "logits", "--bootstrap", "20"])
    assert main(["metrics", path, "--logits", "logits", "--bootstrap", "0"]) == 0
    out = capsys.readouterr().out
    assert "Confidence is the largest probability of each vector (top-label)." in out
    assert "| classwise ECE, plug-in | 0." in out
    assert "| log loss of the full vector | " in out


def test_metrics_reads_stated_confidence_and_reports_digit_preference(tmp_path, capsys):
    rows = [
        {"confidence": text, "correct": ok}
        for text, ok in [
            ("90%", True),
            ("90%", False),
            ("95%", True),
            ("sure", True),
            ("80", False),
        ]
    ]
    path = write_jsonl(tmp_path / "stated.jsonl", rows)
    words = tmp_path / "words.json"
    words.write_text(json.dumps({"sure": 0.99}), encoding="utf-8")
    args = ["metrics", path, "--words", str(words), "--bootstrap", "0"]
    report = run_json(capsys, args)
    assert report["stated"]["distinct_values"] == 4
    assert report["stated"]["top_values"][0] == {"confidence": 0.9, "count": 2, "share": 0.4}
    assert main(args) == 0
    out = capsys.readouterr().out
    assert "## Stated confidence" in out
    assert "4 distinct values; 60.0% of answers are multiples of 10 % and 80.0%" in out
    assert "| 90.00% | 2 | 40.0% |" in out
    words.write_text("[1, 2]", encoding="utf-8")
    assert main(args) == 2
    assert "expected a JSON object" in capsys.readouterr().err
    assert main(["metrics", path, "--bootstrap", "0"]) == 2  # not numeric without --stated


def test_recalibrate_reports_before_and_after(overconfident, capsys):
    report = run_json(capsys, ["recalibrate", overconfident, "--in-sample", "--seed", "1"])
    rows = {(row["method"], row["fit"]): row for row in report["rows"]}
    assert report["folds"] == 5 and not report["grouped"]
    before = rows[("none", "-")]
    assert before["ece_debiased"] > 0.1
    for method in ("temperature", "platt", "isotonic", "histogram"):
        assert rows[(method, "5 folds")]["ece_debiased"] < before["ece_debiased"] - 0.02, method
        assert rows[(method, "5 folds")]["log_loss"] < before["log_loss"]
    for method in ("temperature", "platt"):
        assert rows[(method, "5 folds")]["ece_debiased"] < 0.06, method
    same = rows[("isotonic", "same items (optimistic)")]
    assert same["ece"] < 1e-9 < rows[("isotonic", "5 folds")]["ece"]

    assert main(["recalibrate", overconfident, "--method", "platt", "--group", "topic"]) == 0
    out = capsys.readouterr().out
    assert "fitted on 4 of 5 folds and scored on the held-out fold (groups kept together)" in out
    assert "| none | - | " in out and "| platt | 5 folds | " in out
    assert "isotonic" not in out


def test_recalibrate_on_vectors_and_with_a_separate_fitting_file(tmp_path, capsys):
    fit = write_jsonl(tmp_path / "fit.jsonl", vector_rows(2))
    score = write_jsonl(tmp_path / "score.jsonl", vector_rows(3))
    base = ["recalibrate", score, "--logits", "logits", "--method", "temperature"]
    report = run_json(capsys, base)
    vector_row = report["rows"][-1]
    assert vector_row["method"] == "temperature on vectors"
    assert vector_row["temperatures"] == pytest.approx([2.0] * 5, rel=0.25)
    assert vector_row["ece_debiased"] < report["rows"][0]["ece_debiased"]

    report = run_json(capsys, [*base, "--fit-on", fit])
    assert report["folds"] is None
    assert [row["fit"] for row in report["rows"][1:]] == ["fitted on the other file"] * 2
    assert len(report["rows"][-1]["temperatures"]) == 1
    assert main([*base, "--fit-on", fit]) == 0
    out = capsys.readouterr().out
    assert "each map is fitted on the other file" in out
    assert "Temperatures fitted on vectors: " in out
    assert "| Brier | top-label log loss | AUROC |" in out
    assert "The log loss of the whole vector is " in out
    assert "Log loss of the whole vector after scaling: " in out


def test_recalibration_report_separates_top_label_and_vector_log_loss(tmp_path, capsys):
    path = write_jsonl(tmp_path / "v.jsonl", vector_rows(4))
    args = ["recalibrate", path, "--logits", "logits", "--method", "temperature"]
    report = run_json(capsys, args)
    before = report["vector_log_loss_before"]
    after = report["rows"][-1]["vector_log_loss"]
    assert after < before  # the logits were twice too large, so scaling helps the vector loss
    assert before > report["rows"][0]["log_loss"]  # four-way loss, not the binary one
    plain = write_jsonl(tmp_path / "c.jsonl", confidence_rows(0))
    report = run_json(capsys, ["recalibrate", plain, "--method", "platt"])
    assert report["vector_log_loss_before"] is None
    assert main(["recalibrate", plain, "--method", "platt"]) == 0
    out = capsys.readouterr().out
    assert "| top-label log loss |" in out and "whole vector" not in out


def test_compare_matches_items_by_id(tmp_path, capsys, overconfident):
    rows = confidence_rows(0)
    shuffled = [rows[i] for i in np.random.default_rng(5).permutation(len(rows))]
    same = write_jsonl(tmp_path / "same.jsonl", shuffled)
    args = ["compare", overconfident, same, "--id", "id", "--bootstrap", "30"]
    report = run_json(capsys, args)
    assert report["matched_by"] == "id" and report["n_items"] == 600
    for row in report["rows"]:
        assert row["difference"] == 0.0 and row["interval"] == [0.0, 0.0], row["metric"]

    better = write_jsonl(tmp_path / "model_b.jsonl", confidence_rows(0, sharpen=1.0))
    report = run_json(
        capsys, ["compare", overconfident, better, "--id", "id", "--bootstrap", "200"]
    )
    by_metric = {row["metric"]: row for row in report["rows"]}
    assert by_metric["accuracy"]["difference"] == 0.0
    lo, hi = by_metric["ece_debiased"]["interval"]
    assert 0 < lo < by_metric["ece_debiased"]["difference"] < hi
    assert main(["compare", overconfident, better, "--bootstrap", "20", "--group", "topic"]) == 0
    out = capsys.readouterr().out
    assert "600 items matched by position. Differences are A - B; 95% intervals: paired" in out
    assert "bootstrap over 12 groups, 20 resamples." in out
    assert "| ECE, debiased | 0." in out


def test_compare_rejects_files_that_do_not_line_up(tmp_path, capsys, overconfident):
    short = write_jsonl(tmp_path / "short.jsonl", confidence_rows(0)[:10])
    assert main(["compare", overconfident, short]) == 2
    assert "differ in length" in capsys.readouterr().err
    assert main(["compare", overconfident, short, "--id", "id"]) == 2
    assert "do not hold the same ids" in capsys.readouterr().err


def test_diagram_writes_a_png_and_prints_the_test(tmp_path, capsys, overconfident):
    out_path = tmp_path / "diagram.png"
    args = ["diagram", overconfident, "-o", str(out_path), "--resamples", "100", "--title", "T"]
    assert main(args) == 0
    assert out_path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    out = capsys.readouterr().out
    assert out.startswith(f"wrote {out_path}\n")
    assert "consistency test: plug-in ECE 0." in out
    assert "p = 0.0099 (100 resamples)" in out
    assert main([*args, "--binning", "mass", "--bins", "8"]) == 0


def test_report_holds_the_risk_coverage_curve(overconfident, capsys):
    report = run_json(capsys, ["metrics", overconfident, "--bootstrap", "0"])
    curve = report["risk_coverage"]
    assert [point["coverage"] for point in curve] == [q / 10 for q in range(1, 11)]
    assert curve[-1]["accuracy"] == pytest.approx(report["metrics"]["accuracy"])
    assert curve[7]["accuracy"] == pytest.approx(report["metrics"]["accuracy_at_coverage"])
    assert curve[0]["accuracy"] > curve[-1]["accuracy"]
    assert main(["metrics", overconfident, "--bootstrap", "0"]) == 0
    out = capsys.readouterr().out
    assert "| coverage | 10% | 20% | 30% | 40% | 50% | 60% | 70% | 80% | 90% | 100% |" in out


def test_compare_runs_on_the_two_bundled_examples(capsys):
    examples = Path(__file__).resolve().parents[1] / "examples"
    args = ["compare", str(examples / "example.jsonl"), str(examples / "example_b.jsonl")]
    report = run_json(capsys, [*args, "--id", "id", "--bootstrap", "100"])
    by_metric = {row["metric"]: row for row in report["rows"]}
    assert report["n_items"] == 400 and report["matched_by"] == "id"
    assert by_metric["ece"]["a"] == pytest.approx(0.1210, abs=5e-5)
    assert by_metric["l2_ce_debiased"]["difference"] > 0


def test_recalibrate_warns_when_the_temperature_ends_on_a_bound(tmp_path, capsys):
    rows = [{"confidence": 0.6 + 0.003 * i, "correct": True} for i in range(100)]
    path = write_jsonl(tmp_path / "all_right.jsonl", rows)
    assert main(["recalibrate", path, "--method", "temperature"]) == 0
    captured = capsys.readouterr()
    assert (
        "calibcheck: warning: temperature search stopped at the edge of its range" in captured.err
    )
    assert "| temperature | 5 folds | " in captured.out


def test_compare_includes_vector_metrics_when_both_files_hold_vectors(tmp_path, capsys):
    a = write_jsonl(tmp_path / "a.jsonl", vector_rows(6))
    b = write_jsonl(tmp_path / "b.jsonl", vector_rows(6, scale=1.0))
    args = ["compare", a, b, "--logits", "logits", "--id", "id", "--bootstrap", "40"]
    report = run_json(capsys, args)
    by_metric = {row["metric"]: row for row in report["rows"]}
    assert report["vectors"]
    for key in ("classwise_ece", "classwise_ece_debiased", "nll", "brier_multiclass"):
        assert key in by_metric
    lo, hi = by_metric["nll"]["interval"]
    assert 0 < lo <= by_metric["nll"]["difference"] <= hi  # logits of A are twice too large
    assert by_metric["accuracy"]["difference"] == 0.0
    same = run_json(capsys, ["compare", a, a, "--logits", "logits", "--bootstrap", "20"])
    assert all(row["difference"] == 0.0 for row in same["rows"])
    assert main(args) == 0
    out = capsys.readouterr().out
    assert "| classwise ECE, plug-in | 0." in out
    assert "| log loss of the full vector | " in out


def test_compare_of_plain_confidences_has_no_vector_rows(tmp_path, capsys, overconfident):
    report = run_json(capsys, ["compare", overconfident, overconfident, "--bootstrap", "10"])
    assert not report["vectors"]
    assert "nll" not in {row["metric"] for row in report["rows"]}
