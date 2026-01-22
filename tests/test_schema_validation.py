"""Tests for JSON schema validation of evaluator responses."""

import json

import pytest

from prompt_evolver.pipeline import EVALUATION_SCHEMA, _parse_evaluation_response


def test_valid_evaluation_response():
    """Test parsing a valid evaluation response."""
    response = json.dumps(
        {"pass": True, "score": 0.9, "issues": [], "suggestions": ["Good work"]}
    )

    feedback = _parse_evaluation_response(response)

    assert feedback.passed is True
    assert feedback.score == 0.9
    assert feedback.issues == []
    assert feedback.suggestions == ["Good work"]


def test_valid_failure_response():
    """Test parsing a valid failure response."""
    response = json.dumps(
        {
            "pass": False,
            "score": 0.4,
            "issues": ["Missing field", "Wrong format"],
            "suggestions": ["Add field X", "Use format Y"],
        }
    )

    feedback = _parse_evaluation_response(response)

    assert feedback.passed is False
    assert feedback.score == 0.4
    assert len(feedback.issues) == 2
    assert "Missing field" in feedback.issues
    assert len(feedback.suggestions) == 2


def test_score_out_of_range_high():
    """Test that scores above 1.0 trigger schema validation error."""
    response = json.dumps({"pass": True, "score": 1.5, "issues": [], "suggestions": []})

    feedback = _parse_evaluation_response(response)

    # Schema validation should fail for out-of-range scores
    assert feedback.passed is False
    assert any("schema_validation_error" in issue for issue in feedback.issues)


def test_score_out_of_range_low():
    """Test that scores below 0.0 are clamped."""
    response = json.dumps({"pass": False, "score": -0.5, "issues": ["Bad"], "suggestions": []})

    feedback = _parse_evaluation_response(response)

    assert feedback.score == 0.0  # Should be clamped


def test_missing_issues_field():
    """Test that missing 'issues' field triggers schema validation error."""
    response = json.dumps({"pass": True, "score": 1.0, "suggestions": []})

    feedback = _parse_evaluation_response(response)

    # Should return a failure feedback with schema validation error
    assert feedback.passed is False
    assert any("schema_validation_error" in issue for issue in feedback.issues)


def test_missing_pass_field():
    """Test that missing 'pass' field triggers schema validation error."""
    response = json.dumps({"score": 0.8, "issues": [], "suggestions": []})

    feedback = _parse_evaluation_response(response)

    assert feedback.passed is False
    assert any("schema_validation_error" in issue for issue in feedback.issues)


def test_wrong_type_for_pass():
    """Test that wrong type for 'pass' field triggers schema validation error."""
    response = json.dumps({"pass": "yes", "score": 0.8, "issues": [], "suggestions": []})

    feedback = _parse_evaluation_response(response)

    assert feedback.passed is False
    assert any("schema_validation_error" in issue for issue in feedback.issues)


def test_wrong_type_for_score():
    """Test that wrong type for 'score' field triggers schema validation error."""
    response = json.dumps({"pass": True, "score": "high", "issues": [], "suggestions": []})

    feedback = _parse_evaluation_response(response)

    assert feedback.passed is False
    assert any("schema_validation_error" in issue for issue in feedback.issues)


def test_issues_not_array():
    """Test that non-array 'issues' field triggers schema validation error."""
    response = json.dumps({"pass": False, "score": 0.5, "issues": "some issue", "suggestions": []})

    feedback = _parse_evaluation_response(response)

    assert feedback.passed is False
    assert any("schema_validation_error" in issue for issue in feedback.issues)


def test_suggestions_not_array():
    """Test that non-array 'suggestions' field triggers schema validation error."""
    response = json.dumps(
        {"pass": False, "score": 0.5, "issues": [], "suggestions": "some suggestion"}
    )

    feedback = _parse_evaluation_response(response)

    assert feedback.passed is False
    assert any("schema_validation_error" in issue for issue in feedback.issues)


def test_json_in_markdown_code_block():
    """Test parsing JSON wrapped in markdown code block."""
    response = """Here's the evaluation:
```json
{
  "pass": true,
  "score": 0.85,
  "issues": [],
  "suggestions": ["Consider adding more detail"]
}
```
"""

    feedback = _parse_evaluation_response(response)

    assert feedback.passed is True
    assert feedback.score == 0.85
    assert feedback.suggestions == ["Consider adding more detail"]


def test_json_with_extra_fields():
    """Test that extra fields are allowed (additionalProperties: true)."""
    response = json.dumps(
        {
            "pass": True,
            "score": 0.95,
            "issues": [],
            "suggestions": [],
            "extra_field": "extra_value",
            "confidence": 0.99,
        }
    )

    feedback = _parse_evaluation_response(response)

    assert feedback.passed is True
    assert feedback.score == 0.95
    # Extra fields should be in raw dict
    assert feedback.raw.get("extra_field") == "extra_value"
    assert feedback.raw.get("confidence") == 0.99


def test_empty_issues_when_failed():
    """Test that empty issues list for failed evaluation gets 'unspecified_issues'."""
    response = json.dumps({"pass": False, "score": 0.3, "issues": [], "suggestions": []})

    feedback = _parse_evaluation_response(response)

    assert feedback.passed is False
    assert "unspecified_issues" in feedback.issues


def test_whitespace_in_issues_and_suggestions():
    """Test that whitespace-only strings are filtered out."""
    response = json.dumps(
        {
            "pass": False,
            "score": 0.5,
            "issues": ["Real issue", "  ", "Another issue", ""],
            "suggestions": ["", "Real suggestion", "   "],
        }
    )

    feedback = _parse_evaluation_response(response)

    assert len(feedback.issues) == 2
    assert "Real issue" in feedback.issues
    assert "Another issue" in feedback.issues
    assert len(feedback.suggestions) == 1
    assert "Real suggestion" in feedback.suggestions


def test_evaluation_schema_structure():
    """Test that EVALUATION_SCHEMA has required structure."""
    assert EVALUATION_SCHEMA["type"] == "object"
    assert "properties" in EVALUATION_SCHEMA
    assert "pass" in EVALUATION_SCHEMA["properties"]
    assert "score" in EVALUATION_SCHEMA["properties"]
    assert "issues" in EVALUATION_SCHEMA["properties"]
    assert "suggestions" in EVALUATION_SCHEMA["properties"]
    assert EVALUATION_SCHEMA["required"] == ["pass", "score", "issues", "suggestions"]


def test_score_bounds_in_schema():
    """Test that schema defines correct score bounds."""
    score_schema = EVALUATION_SCHEMA["properties"]["score"]
    assert score_schema["type"] == "number"
    assert score_schema["minimum"] == 0.0
    assert score_schema["maximum"] == 1.0


def test_invalid_json_returns_failure():
    """Test that completely invalid JSON returns failure feedback."""
    response = "This is not JSON at all"

    feedback = _parse_evaluation_response(response)

    assert feedback.passed is False
    assert "invalid_json" in feedback.issues


def test_partial_json_extraction():
    """Test extraction of JSON from text with surrounding content."""
    response = """
    The model output looks good overall.

    {"pass": true, "score": 0.88, "issues": [], "suggestions": ["Minor improvement needed"]}

    That's my evaluation.
    """

    feedback = _parse_evaluation_response(response)

    assert feedback.passed is True
    assert feedback.score == 0.88
