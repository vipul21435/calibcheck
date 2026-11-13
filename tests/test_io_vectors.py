import json
import math
import warnings

import numpy as np
import pytest

from calibcheck.io import parse_label, parse_vector, read_items


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_read_probability_vectors(tmp_path):
    path = write_jsonl(
        tmp_path / "v.jsonl",
        [
            {"id": "a", "probs": [0.7, 0.2, 0.1], "label": 0, "topic": "x"},
            {"id": "b", "probs": [0.1, 0.3, 0.6], "label": 1, "topic": "y"},
        ],
    )
    items = read_items(path, probs="probs", group="topic", id_field="id")
    assert items.confidence.tolist() == [0.7, 0.6]
    assert items.correct.tolist() == [1.0, 0.0]
    assert items.label.tolist() == [0, 1]
    assert items.probs.shape == (2, 3) and not items.ragged
    assert items.logits == pytest.approx(np.log(items.probs))
    assert items.group == ["x", "y"] and items.ids == ["a", "b"]


def test_read_logits_of_different_lengths(tmp_path):
    path = write_jsonl(
        tmp_path / "v.jsonl",
        [
            {"scores": [0.0, math.log(3.0)], "answer": 1},
            {"scores": [2.0, 0.0, 0.0, 0.0], "answer": 3},
        ],
    )
    items = read_items(path, logits="scores", label="answer")
    assert items.ragged
    assert items.probs[0].tolist() == pytest.approx([0.25, 0.75, 0.0, 0.0])
    assert items.logits[0, 2] == -math.inf
    assert items.confidence[0] == pytest.approx(0.75)
    assert items.correct.tolist() == [1.0, 0.0]


def test_read_vectors_from_csv_cells(tmp_path):
    path = tmp_path / "v.csv"
    path.write_text(
        'id,probs,label\n1,"[0.25, 0.75]",1\n2,0.9;0.1,1\n3,0.5 0.5,0\n', encoding="utf-8"
    )
    items = read_items(path, probs="probs")
    assert items.confidence.tolist() == [0.75, 0.9, 0.5]
    assert items.correct.tolist() == [1.0, 0.0, 1.0]


def test_read_stated_confidences(tmp_path):
    path = write_jsonl(
        tmp_path / "s.jsonl",
        [
            {"confidence": "85%", "correct": True},
            {"confidence": "high", "correct": False},
            {"confidence": 40, "correct": False},
        ],
    )
    assert read_items(path, stated=True).confidence.tolist() == [0.85, 0.7, 0.4]
    custom = read_items(path, stated=True, words={"high": 0.99})
    assert custom.confidence.tolist() == [0.85, 0.99, 0.4]
    with pytest.raises(ValueError, match=r"s\.jsonl:1: confidence must be a number"):
        read_items(path)


@pytest.mark.parametrize(
    ("row", "message"),
    [
        ({"probs": [0.5, 0.6], "label": 0}, "sum to 1"),
        ({"probs": [1.5, -0.5], "label": 0}, "non-negative"),
        ({"probs": [0.5, 0.5], "label": 2}, r"class index in \[0, 2\)"),
        ({"probs": [0.5, 0.5], "label": "x"}, "class index"),
        ({"probs": [0.5, 0.5], "label": 0.5}, "class index"),
        ({"probs": [1.0], "label": 0}, "at least two numbers"),
        ({"probs": "[0.5, ", "label": 0}, "cannot read a vector"),
        ({"probs": ["a", "b"], "label": 0}, "cannot read a vector"),
        ({"probs": [0.5, 0.5]}, "missing field 'label'"),
    ],
)
def test_bad_vector_rows_name_the_line(tmp_path, row, message):
    path = write_jsonl(tmp_path / "v.jsonl", [{"probs": [0.5, 0.5], "label": 0}, row])
    with pytest.raises(ValueError, match=r"v\.jsonl:2: .*" + message):
        read_items(path, probs="probs")


def test_ids_must_be_unique_and_fields_exclusive(tmp_path):
    path = write_jsonl(
        tmp_path / "d.jsonl",
        [
            {"id": "a", "confidence": 0.5, "correct": 1},
            {"id": "a", "confidence": 0.6, "correct": 0},
        ],
    )
    with pytest.raises(ValueError, match="not unique"):
        read_items(path, id_field="id")
    with pytest.raises(ValueError, match="not both"):
        read_items(path, probs="p", logits="z")


def test_parse_helpers():
    assert parse_vector("[1, 2.5]") == [1.0, 2.5]
    assert parse_vector("0.1|0.9") == [0.1, 0.9]
    assert parse_vector((0.2, 0.8)) == [0.2, 0.8]
    assert parse_label("3", 4) == 3
    assert parse_label(2.0, 4) == 2
    for bad in (True, None, -1, 4, "1.5"):
        with pytest.raises(ValueError):
            parse_label(bad, 4)
    for bad in ([True, False], [float("nan"), 1.0], 0.5):
        with pytest.raises(ValueError):
            parse_vector(bad)


def test_numbers_too_large_for_a_float_are_ordinary_errors(tmp_path):
    huge = "1" + "0" * 400
    for line, kwargs in [
        ('{"confidence": HUGE, "correct": 1}', {}),
        ('{"confidence": HUGE, "correct": 1}', {"stated": True}),
        ('{"probs": [HUGE, 0.5], "label": 0}', {"probs": "probs"}),
        ('{"probs": [0.5, 0.5], "label": HUGE}', {"probs": "probs"}),
    ]:
        path = tmp_path / "huge.jsonl"
        path.write_text(line.replace("HUGE", huge) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match=r"huge\.jsonl:1: "):
            read_items(path, **kwargs)


def test_broken_csv_is_reported_with_its_line(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_bytes(b'confidence,correct\n0.5,1\n"0.7,1\n' + b"x" * 200_000 + b"\n")
    with pytest.raises(ValueError, match=r"bad\.csv:\d+: "):
        read_items(path)


def test_infinite_logits_are_rejected_with_the_line(tmp_path):
    path = tmp_path / "inf.jsonl"
    path.write_text(
        '{"z": [0.25, 0.75], "label": 0}\n{"z": [1e309, 0.0], "label": 0}\n', encoding="utf-8"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no numpy warning before the error
        with pytest.raises(ValueError, match=r"inf\.jsonl:2: a vector must hold finite numbers"):
            read_items(path, logits="z")
        with pytest.raises(ValueError, match=r"inf\.jsonl:2: a vector must hold finite numbers"):
            read_items(path, probs="z")
    for bad in ([float("inf"), 0.0], [0.0, float("-inf")], "[1e309, 1]", "1e999;0"):
        with pytest.raises(ValueError, match="finite"):
            parse_vector(bad)
