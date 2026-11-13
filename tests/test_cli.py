import json
import subprocess
import sys
from pathlib import Path

import pytest

from calibcheck.cli import main
from calibcheck.report import build_report, render_json, render_markdown

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "example.jsonl"
ROWS = [(0.9, 1), (0.9, 1), (0.9, 1), (0.9, 0), (0.6, 1), (0.6, 0)]


@pytest.fixture
def small_file(tmp_path):
    path = tmp_path / "small.jsonl"
    lines = [json.dumps({"confidence": c, "correct": y}) + "\n" for c, y in ROWS]
    path.write_text("".join(lines), encoding="utf-8")
    return path


def test_metrics_as_json(small_file, capsys):
    assert main(["metrics", str(small_file), "--bins", "10", "--format", "json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["source"] == str(small_file)
    assert report["n_items"] == 6
    assert report["binning"] == {
        "strategy": "width",
        "requested_bins": 10,
        "bins": 10,
        "empty_bins": 8,
        "singleton_bins": 0,
    }
    metrics = report["metrics"]
    assert metrics["ece"] == pytest.approx(0.4 / 3)
    assert metrics["mce"] == pytest.approx(0.15)
    assert metrics["accuracy"] == pytest.approx(4 / 6)
    assert metrics["mean_confidence"] == pytest.approx(0.8)
    assert metrics["sq_ce_debiased"] < 0 and metrics["l2_ce_debiased"] == 0.0
    assert len(report["bins"]) == 10
    assert report["bins"][0] == {
        "lower": 0.0,
        "upper": 0.1,
        "count": 0,
        "confidence": None,
        "accuracy": None,
        "gap": None,
    }
    assert report["bins"][9]["count"] == 4
    assert report["bins"][9]["accuracy"] == pytest.approx(0.75)


def test_metrics_as_markdown(small_file, capsys):
    assert main(["metrics", str(small_file), "--bins", "10"]) == 0
    out = capsys.readouterr().out
    assert out.startswith(f"# Calibration report: {small_file}\n")
    assert "6 items, 10 equal-width bins (8 empty)." in out
    assert "| ECE, plug-in | 0.1333 |" in out
    assert "| MCE | 0.1500 |" in out
    assert "| [0.000, 0.100) | 0 | - | - | - |" in out
    assert "| [0.900, 1.000] | 4 | 0.9000 | 0.7500 | -0.1500 |" in out


def test_equal_mass_binning_merges_tied_values(small_file, capsys):
    assert main(["metrics", str(small_file), "--bins", "10", "--binning", "mass"]) == 0
    out = capsys.readouterr().out
    assert "6 items, 2 equal-mass bins." in out
    assert "| [0.000, 0.900) | 2 | 0.6000 | 0.5000 | -0.1000 |" in out
    assert "| [0.900, 1.000] | 4 | 0.9000 | 0.7500 | -0.1500 |" in out


def test_output_file_and_csv_field_names(tmp_path, capsys):
    data = tmp_path / "answers.csv"
    data.write_text("p,ok\n" + "".join(f"{c},{y}\n" for c, y in ROWS), encoding="utf-8")
    out_path = tmp_path / "report.json"
    code = main(
        ["metrics", str(data), "--confidence", "p", "--correct", "ok", "--bins", "10"]
        + ["--format", "json", "--output", str(out_path)]
    )
    assert code == 0
    assert capsys.readouterr().out == ""
    assert json.loads(out_path.read_text(encoding="utf-8"))["metrics"]["ece"] == pytest.approx(
        0.4 / 3
    )


def test_single_item_bins_are_mentioned(tmp_path, capsys):
    path = tmp_path / "few.jsonl"
    path.write_text(
        '{"confidence": 0.15, "correct": 1}\n{"confidence": 0.85, "correct": 1}\n'
        '{"confidence": 0.85, "correct": 0}\n',
        encoding="utf-8",
    )
    assert main(["metrics", str(path), "--bins", "10"]) == 0
    assert "1 bin(s) hold a single item" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["metrics", "absent.jsonl"], "No such file"),
        (["metrics", "{file}", "--bins", "0"], "n_bins must be at least 1"),
        (["metrics", "{file}", "--confidence", "prob"], "missing field 'prob'"),
    ],
)
def test_errors_exit_with_code_2(small_file, capsys, args, message):
    args = [a.replace("{file}", str(small_file)) for a in args]
    assert main(args) == 2
    err = capsys.readouterr().err
    assert err.startswith("calibcheck: error: ")
    assert message in err


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as info:
        main(["--version"])
    assert info.value.code == 0
    assert capsys.readouterr().out.startswith("calibcheck ")


def test_report_without_a_source_and_json_round_trip():
    conf, correct = zip(*ROWS, strict=True)
    report = build_report(conf, correct, n_bins=10)
    assert render_markdown(report).startswith("# Calibration report\n")
    assert json.loads(render_json(report)) == report


def test_tiny_negative_values_are_not_printed_as_minus_zero():
    conf, correct = zip(*ROWS, strict=True)
    report = build_report(conf, correct, n_bins=10)
    report["metrics"]["within_bin"] = -2.8e-17
    text = render_markdown(report)
    assert "| within-bin term | 0.0000 |" in text
    assert "-0.0000" not in text


def test_module_entry_point_runs_on_the_bundled_example():
    done = subprocess.run(
        [sys.executable, "-m", "calibcheck", "metrics", str(EXAMPLE)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    assert "400 items, 15 equal-width bins (9 empty)." in done.stdout
    assert "| ECE, plug-in | 0.1210 |" in done.stdout
