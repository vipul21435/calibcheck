import pytest

from calibcheck.stated import DEFAULT_WORDS, digit_preference, parse_stated


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("85%", 0.85),
        ("85 %", 0.85),
        (" 12.5% ", 0.125),
        ("100%", 1.0),
        ("0%", 0.0),
        ("0.85", 0.85),
        (".85", 0.85),
        ("85", 0.85),
        (85, 0.85),
        (0.85, 0.85),
        ("1", 1.0),
        (1, 1.0),
        (0, 0.0),
        ("High", 0.7),
        (" very  high. ", 0.9),
        ('"medium"', 0.5),
    ],
)
def test_parse_stated_reads_numbers_percentages_and_words(value, expected):
    assert parse_stated(value) == pytest.approx(expected)


def test_scale_decides_what_a_bare_number_means():
    assert parse_stated("1", scale="percent") == pytest.approx(0.01)
    assert parse_stated(50, scale="percent") == 0.5
    assert parse_stated("0.5", scale="unit") == 0.5
    assert parse_stated("50%", scale="unit") == 0.5  # a percent sign is never ambiguous
    with pytest.raises(ValueError, match="outside the unit scale"):
        parse_stated("50", scale="unit")
    with pytest.raises(ValueError, match="unknown scale"):
        parse_stated("50", scale="odds")


def test_custom_word_scale_replaces_the_default():
    words = {"Almost certain": 0.95, "about even": 0.5}
    assert parse_stated("almost  CERTAIN", words=words) == 0.95
    with pytest.raises(ValueError, match="cannot read"):
        parse_stated("high", words=words)
    with pytest.raises(ValueError, match="outside"):
        parse_stated("sure", words={"sure": 1.5})
    assert set(DEFAULT_WORDS) == {"very low", "low", "medium", "high", "very high"}


@pytest.mark.parametrize("value", ["abc", "", "150", 150, "150%", -1, "-5%", None, True, [0.5]])
def test_parse_stated_rejects_what_it_cannot_read(value):
    with pytest.raises(ValueError):
        parse_stated(value)
    assert True


def test_nan_is_rejected():
    with pytest.raises(ValueError):
        parse_stated(float("nan"))


def test_digit_preference_counts_round_values():
    stated = [0.9] * 5 + [0.8] * 3 + [0.95] * 2 + [0.87, 0.5]
    out = digit_preference(stated, top=3)
    assert out["n_items"] == 12
    assert out["distinct_values"] == 5
    assert out["top_values"] == [
        {"confidence": 0.9, "count": 5, "share": 5 / 12},
        {"confidence": 0.8, "count": 3, "share": 3 / 12},
        {"confidence": 0.95, "count": 2, "share": 2 / 12},
    ]
    assert out["share_multiple_of_10"] == pytest.approx(9 / 12)
    assert out["share_multiple_of_5"] == pytest.approx(11 / 12)
    with pytest.raises(ValueError, match="no items"):
        digit_preference([])


def test_digit_preference_ignores_floating_point_noise():
    out = digit_preference([0.1 + 0.2, 0.3, 0.30000000000000004])
    assert out["distinct_values"] == 1
    assert out["share_multiple_of_10"] == 1.0


@pytest.mark.parametrize("value", ["1.5", 1.5, "5", 9.99, 2])
def test_bare_values_between_one_and_ten_are_ambiguous_under_auto(value):
    with pytest.raises(ValueError, match="ambiguous"):
        parse_stated(value)
    assert parse_stated(value, scale="percent") == pytest.approx(float(value) / 100)
    with pytest.raises(ValueError, match="outside the unit scale"):
        parse_stated(value, scale="unit")


def test_the_edges_of_the_ambiguous_range_are_read_as_before():
    assert parse_stated(1) == 1.0
    assert parse_stated("10") == pytest.approx(0.10)
    assert parse_stated("5%") == pytest.approx(0.05)
    assert parse_stated("1.5 %") == pytest.approx(0.015)
