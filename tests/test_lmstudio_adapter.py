"""Tests for LM Studio adapter behavior."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

from prompt_evolver.llm import LMStudioClient


class _DummyResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        """Initialize with a payload to return as JSON.

        Args:
            payload: Response payload to return.
        """

        self._payload = payload

    def read(self) -> bytes:
        """Serialize payload to JSON bytes.

        Returns:
            bytes: JSON-encoded payload.
        """

        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self) -> "_DummyResponse":
        """Return self for context manager usage.

        Returns:
            _DummyResponse: Context manager instance.
        """

        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """Exit the context manager.

        Args:
            exc_type: Exception type, if any.
            exc: Exception instance, if any.
            tb: Traceback, if any.

        Returns:
            None
        """

        return None


def test_lmstudio_client_parses_response() -> None:
    """Verify LM Studio client parses chat completions output.

    Returns:
        None
    """
    payload = {"choices": [{"message": {"content": "hello"}}]}

    def _fake_urlopen(request, timeout=0):  # type: ignore[no-untyped-def]
        return _DummyResponse(payload)

    with patch("urllib.request.urlopen", _fake_urlopen):
        client = LMStudioClient(model="mistralai/ministral-3-3b")
        response = client.generate("Say hello")
        assert response == "hello"
