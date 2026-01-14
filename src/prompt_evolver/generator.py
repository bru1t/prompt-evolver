"""
Generate candidate prompt rewrites.

This module defines a heuristic generator used for lightweight prompt
variants and includes an interface for custom generators.

Public API:
- PromptGenerator
- HeuristicPromptGenerator
"""

from __future__ import annotations

import re
from typing import Protocol

from .models import PromptRecord

_WHITESPACE_RE = re.compile(r"\s+")
_FILLER_WORD_RE = re.compile(r"\b(?:please|kindly|just|actually|basically)\b", re.IGNORECASE)


class PromptGenerator(Protocol):
    """Defines the interface for prompt candidate generation."""

    def generate(self, record: PromptRecord, max_generations: int) -> list[str]:
        """Generate prompt candidates based on the record.

        Args:
            record: Source prompt record.
            max_generations: Maximum number of candidates to return.

        Returns:
            list[str]: Prompt candidates in preference order.
        """


class HeuristicPromptGenerator:
    """Generates a small set of heuristic prompt rewrites."""

    def generate(self, record: PromptRecord, max_generations: int) -> list[str]:
        """Return unique prompt variants constrained by max_generations.

        Args:
            record: Source prompt record.
            max_generations: Maximum number of candidates to return.

        Returns:
            list[str]: Unique prompt candidates.
        """

        original = record.prompt.strip()
        candidates = [original]
        collapsed = _WHITESPACE_RE.sub(" ", original).strip()
        candidates.append(collapsed)
        stripped_fillers = _FILLER_WORD_RE.sub("", collapsed)
        stripped_fillers = _WHITESPACE_RE.sub(" ", stripped_fillers).strip()
        candidates.append(stripped_fillers)
        if record.expected_output:
            concise = f"{stripped_fillers}\n\nRespond concisely."
            candidates.append(concise)
        unique_candidates: list[str] = []
        seen = set()
        for candidate in candidates:
            if candidate and candidate not in seen:
                seen.add(candidate)
                unique_candidates.append(candidate)
            if len(unique_candidates) >= max_generations:
                break
        return unique_candidates
