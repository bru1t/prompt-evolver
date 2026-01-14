"""
Define evaluation criteria utilities.

This module provides simple scoring criteria and a factory for building
criteria from configuration dictionaries.

Public API:
- Criterion
- ExactMatchCriterion
- ContainsKeywordsCriterion
- RegexCriterion
- LengthRangeCriterion
- JsonValidCriterion
- criteria_from_config(...)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Iterable, Protocol

from .models import CriterionResult


class Criterion(Protocol):
    """Defines the interface for a scoring criterion."""

    name: str
    weight: float

    def evaluate(self, output: str, expected_output: str) -> CriterionResult:
        """Evaluate output against expected output.

        Args:
            output: Model output to evaluate.
            expected_output: Target output to compare against.

        Returns:
            CriterionResult: Structured criterion result.
        """


@dataclass(frozen=True)
class ExactMatchCriterion:
    """Checks whether the output exactly matches the expected output."""

    name: str = "exact_match"
    weight: float = 1.0

    def evaluate(self, output: str, expected_output: str) -> CriterionResult:
        """Score 1.0 when output matches expected output exactly.

        Args:
            output: Model output to evaluate.
            expected_output: Target output to compare against.

        Returns:
            CriterionResult: Structured criterion result.
        """

        passed = output == expected_output
        return CriterionResult(
            name=self.name,
            passed=passed,
            score=1.0 if passed else 0.0,
        )


@dataclass(frozen=True)
class ContainsKeywordsCriterion:
    """Checks whether output contains required keywords."""

    keywords: tuple[str, ...]
    match_all: bool = True
    name: str = "contains_keywords"
    weight: float = 1.0

    def evaluate(self, output: str, expected_output: str) -> CriterionResult:
        """Score based on keyword coverage in the output.

        Args:
            output: Model output to evaluate.
            expected_output: Target output (unused).

        Returns:
            CriterionResult: Structured criterion result.
        """

        lower_output = output.lower()
        hits = [kw for kw in self.keywords if kw.lower() in lower_output]
        total = max(len(self.keywords), 1)
        passed = len(hits) == total if self.match_all else len(hits) > 0
        score = len(hits) / total
        detail = f"matched={len(hits)}/{total}"
        return CriterionResult(
            name=self.name,
            passed=passed,
            score=score,
            detail=detail,
        )


@dataclass(frozen=True)
class RegexCriterion:
    """Checks whether output matches a regex pattern."""

    pattern: str
    name: str = "regex_match"
    weight: float = 1.0
    flags: int = 0

    def evaluate(self, output: str, expected_output: str) -> CriterionResult:
        """Score 1.0 when the pattern matches the output.

        Args:
            output: Model output to evaluate.
            expected_output: Target output (unused).

        Returns:
            CriterionResult: Structured criterion result.
        """

        compiled = re.compile(self.pattern, self.flags)
        matched = compiled.search(output) is not None
        return CriterionResult(
            name=self.name,
            passed=matched,
            score=1.0 if matched else 0.0,
        )


@dataclass(frozen=True)
class LengthRangeCriterion:
    """Checks whether output length is within a min/max range."""

    min_length: int | None = None
    max_length: int | None = None
    name: str = "length_range"
    weight: float = 1.0

    def evaluate(self, output: str, expected_output: str) -> CriterionResult:
        """Score 1.0 when output length is within bounds.

        Args:
            output: Model output to evaluate.
            expected_output: Target output (unused).

        Returns:
            CriterionResult: Structured criterion result.
        """

        length = len(output)
        min_ok = self.min_length is None or length >= self.min_length
        max_ok = self.max_length is None or length <= self.max_length
        passed = min_ok and max_ok
        return CriterionResult(
            name=self.name,
            passed=passed,
            score=1.0 if passed else 0.0,
            detail=f"length={length}",
        )


@dataclass(frozen=True)
class JsonValidCriterion:
    """Checks whether output is valid JSON."""

    name: str = "json_valid"
    weight: float = 1.0

    def evaluate(self, output: str, expected_output: str) -> CriterionResult:
        """Score 1.0 when output parses as JSON.

        Args:
            output: Model output to evaluate.
            expected_output: Target output (unused).

        Returns:
            CriterionResult: Structured criterion result.
        """

        try:
            json.loads(output)
        except json.JSONDecodeError as exc:
            return CriterionResult(
                name=self.name,
                passed=False,
                score=0.0,
                detail=str(exc),
            )
        return CriterionResult(name=self.name, passed=True, score=1.0)


def criteria_from_config(entries: Iterable[dict]) -> list[Criterion]:
    """Build a list of Criterion objects from config dictionaries.

    Args:
        entries: Iterable of criterion config mappings.

    Returns:
        list[Criterion]: Initialized criterion instances.

    Raises:
        ValueError: If an unsupported criterion type is provided.
    """

    criteria: list[Criterion] = []
    for entry in entries:
        kind = entry.get("type", "").lower()
        name_override = entry.get("name")
        weight = float(entry.get("weight", 1.0))
        if kind == "exact_match":
            criteria.append(
                ExactMatchCriterion(name=name_override or "exact_match", weight=weight)
            )
        elif kind == "contains_keywords":
            keywords = tuple(entry.get("keywords", []))
            match_all = bool(entry.get("match_all", True))
            criteria.append(
                ContainsKeywordsCriterion(
                    keywords=keywords,
                    match_all=match_all,
                    name=name_override or "contains_keywords",
                    weight=weight,
                )
            )
        elif kind == "regex":
            pattern = entry.get("pattern", "")
            flags = int(entry.get("flags", 0))
            criteria.append(
                RegexCriterion(
                    pattern=pattern,
                    flags=flags,
                    name=name_override or "regex_match",
                    weight=weight,
                )
            )
        elif kind == "length_range":
            min_length = entry.get("min_length")
            max_length = entry.get("max_length")
            criteria.append(
                LengthRangeCriterion(
                    min_length=int(min_length) if min_length is not None else None,
                    max_length=int(max_length) if max_length is not None else None,
                    name=name_override or "length_range",
                    weight=weight,
                )
            )
        elif kind == "json_valid":
            criteria.append(
                JsonValidCriterion(name=name_override or "json_valid", weight=weight)
            )
        else:
            raise ValueError(f"Unsupported criterion type: {kind}")
    return criteria
