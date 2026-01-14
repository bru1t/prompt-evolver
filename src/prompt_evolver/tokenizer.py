"""
Provide token counting utilities for prompts and outputs.

This module offers a lightweight, deterministic token counter used for
sanity checks, leakage heuristics, and reporting token deltas.

Public API:
- TokenCounter
- SimpleTokenCounter

Notes:
- Token counts are heuristic, not model-specific.
"""

from __future__ import annotations

import re
from typing import Protocol

_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


class TokenCounter(Protocol):
    """Defines the interface for token counting."""

    def count(self, text: str) -> int:
        """Return the estimated token count for the given text.

        Args:
            text: Input text to tokenize.

        Returns:
            int: Estimated token count.
        """


class SimpleTokenCounter:
    """Estimates tokens using a lightweight regex heuristic."""

    def count(self, text: str) -> int:
        """Return the number of regex tokens found in the text.

        Args:
            text: Input text to tokenize.

        Returns:
            int: Estimated token count.
        """

        if not text:
            return 0
        return len(_TOKEN_PATTERN.findall(text))
