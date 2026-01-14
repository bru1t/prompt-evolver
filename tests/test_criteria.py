"""Tests for evaluation criteria."""

from prompt_evolver.criteria import (
    ContainsKeywordsCriterion,
    ExactMatchCriterion,
    JsonValidCriterion,
    LengthRangeCriterion,
    RegexCriterion,
    criteria_from_config,
)


def test_exact_match() -> None:
    """Verify exact match criterion passes for identical strings.

    Returns:
        None
    """
    criterion = ExactMatchCriterion()
    result = criterion.evaluate("ok", "ok")
    assert result.passed is True
    assert result.score == 1.0


def test_contains_keywords() -> None:
    """Verify keyword criterion passes for partial matches.

    Returns:
        None
    """
    criterion = ContainsKeywordsCriterion(keywords=("alpha", "beta"), match_all=False)
    result = criterion.evaluate("alpha only", "ignored")
    assert result.passed is True
    assert result.score == 0.5


def test_regex() -> None:
    """Verify regex criterion passes when pattern matches.

    Returns:
        None
    """
    criterion = RegexCriterion(pattern=r"\d+")
    result = criterion.evaluate("value=42", "ignored")
    assert result.passed is True


def test_length_range() -> None:
    """Verify length range criterion passes within bounds.

    Returns:
        None
    """
    criterion = LengthRangeCriterion(min_length=2, max_length=5)
    result = criterion.evaluate("test", "ignored")
    assert result.passed is True


def test_json_valid() -> None:
    """Verify JSON validity criterion passes for valid JSON.

    Returns:
        None
    """
    criterion = JsonValidCriterion()
    result = criterion.evaluate('{"key": "value"}', "ignored")
    assert result.passed is True


def test_criteria_from_config() -> None:
    """Verify criteria factory builds criteria from config.

    Returns:
        None
    """
    config = [
        {"type": "exact_match", "name": "match", "weight": 1.5},
        {"type": "length_range", "min_length": 1, "max_length": 3},
    ]
    criteria = criteria_from_config(config)
    assert len(criteria) == 2
    assert criteria[0].name == "match"
