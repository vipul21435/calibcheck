import json

import pytest

from calibcheck.io import parse_confidence, parse_correct, read_items


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_read_jsonl(tmp_path):
    path = write_jsonl(
        tmp_path / "items.jsonl",
        [
            {"id": "a", "confidence": 0.9, "correct": True},
            {"id": "b", "confidence": "0.25", "correct": 0},
            {"id": "c", "confidence": 1, "correct": "yes"},
        ],
    )
    items = read_items(path)
    assert len(items) == 3
    assert items.confidence.tolist() == [0.9, 0.25, 1.0]
    assert items.correct.tolist() == [1.0, 0.0, 1.0]
    assert items.group is None


def test_read_jsonl_skips_blank_lines_and_accepts_a_byte_order_mark(tmp_path):
    path = tmp_path / "items.ndjson"
    body = '\n{"confidence": 0.5, "correct": false}\n\n{"confidence": 0.7, "correct": true}\n'
    path.write_bytes(b"\xef\xbb\xbf" + body.encode("utf-8"))
    items = read_items(path)
    assert items.confidence.tolist() == [0.5, 0.7]
    assert items.correct.tolist() == [0.0, 1.0]


def test_read_csv_with_custom_fields_and_group(tmp_path):
    path = tmp_path / "items.csv"
    path.write_text(
        "question,p,right,topic\nq1,0.8,true,maths\nq2,0.6,False,history\nq3,0.95,1,maths\n",
        encoding="utf-8",
    )
    items = read_items(path, confidence="p", correct="right", group="topic")
    assert items.confidence.tolist() == [0.8, 0.6, 0.95]
    assert items.correct.tolist() == [1.0, 0.0, 1.0]
    assert items.group == ["maths", "history", "maths"]


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ('{"confidence": 1.5, "correct": true}', r"items\.jsonl:2: confidence must lie in \[0"),
        ('{"confidence": "high", "correct": true}', r"items\.jsonl:2: confidence must be a number"),
        ('{"confidence": true, "correct": true}', r"items\.jsonl:2: confidence must be a number"),
        ('{"confidence": NaN, "correct": true}', r"items\.jsonl:2: confidence must lie in"),
        ('{"confidence": 0.5, "correct": "maybe"}', r"items\.jsonl:2: correct must be a boolean"),
        ('{"confidence": 0.5, "correct": 2}', r"items\.jsonl:2: correct must be a boolean"),
        ('{"confidence": 0.5}', r"items\.jsonl:2: missing field 'correct'"),
        ('{"confidence": null, "correct": 1}', r"items\.jsonl:2: missing field 'confidence'"),
        ('{"confidence": 0.5, "correct": 1', r"items\.jsonl:2: invalid JSON"),
        ("[0.5, 1]", r"items\.jsonl:2: expected a JSON object"),
    ],
)
def test_bad_jsonl_lines_are_reported_with_their_line_number(tmp_path, line, message):
    path = tmp_path / "items.jsonl"
    path.write_text('{"confidence": 0.5, "correct": true}\n' + line + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        read_items(path)


def test_csv_errors_carry_the_line_number(tmp_path):
    path = tmp_path / "items.csv"
    path.write_text("confidence,correct\n0.5,1\n0.7,\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"items\.csv:3: missing field 'correct'"):
        read_items(path)
    path.write_text("confidence,right\n0.5,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"items\.csv:2: missing field 'correct'"):
        read_items(path)


def test_empty_file_and_unknown_extension(tmp_path):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no items"):
        read_items(empty)
    other = tmp_path / "items.txt"
    other.write_text("0.5 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported file type"):
        read_items(other)
    with pytest.raises(FileNotFoundError):
        read_items(tmp_path / "absent.jsonl")


@pytest.mark.parametrize(
    ("value", "expected"),
    [(True, 1.0), (False, 0.0), (1, 1.0), (0, 0.0), (1.0, 1.0), ("TRUE", 1.0), (" no ", 0.0)]
    + [("correct", 1.0), ("incorrect", 0.0), ("1.0", 1.0), ("0", 0.0)],
)
def test_parse_correct_accepts_common_spellings(value, expected):
    assert parse_correct(value) == expected


@pytest.mark.parametrize("value", [None, 0.5, 2, "maybe", "", [1]])
def test_parse_correct_rejects_everything_else(value):
    with pytest.raises(ValueError):
        parse_correct(value)


def test_parse_confidence():
    assert parse_confidence(0) == 0.0
    assert parse_confidence(" 0.85 ") == 0.85
    assert parse_confidence(1) == 1.0
    for bad in (None, True, "85%", -0.01, 1.01, float("inf"), [0.5]):
        with pytest.raises(ValueError):
            parse_confidence(bad)
