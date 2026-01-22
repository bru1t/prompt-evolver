"""
Define LLM client interfaces and adapters.

This module provides client implementations for local and OpenAI-compatible
backends, including LM Studio. It supports prompt execution, improvement,
and evaluation workflows.

Public API:
- LLMClient
- EchoLLMClient
- OpenAICompatibleClient
- LMStudioClient
- create_llm_client(...)

Notes:
- Network calls are performed via urllib.
"""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Any, Protocol

from .config import LLMConfig
from .retry import retry_with_backoff


# [ ] FIX: MODEL_RUN_CONTEXT
# Problem: LLM metadata is ad hoc; define a structured run context and persist it.
class LLMClient(Protocol):
    """Interface for producing model outputs for candidate prompts."""

    def generate(
        self,
        prompt: str,
        *,
        expected_output: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Return the model output for the provided prompt.

        Args:
            prompt: Prompt content sent to the model.
            expected_output: Optional expected output for stub clients.
            metadata: Optional metadata (temperature, etc.).

        Returns:
            str: Model response content.
        """


class EchoLLMClient:
    """Returns the expected output as a placeholder model response."""

    def generate(
        self,
        prompt: str,
        *,
        expected_output: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Return expected output when available, otherwise an empty string.

        Args:
            prompt: Prompt content sent to the model.
            expected_output: Expected output to echo.
            metadata: Optional metadata (unused).

        Returns:
            str: Model response content.
        """

        return expected_output or ""


class OpenAICompatibleClient:
    """OpenAI-compatible REST client for chat completions."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key_env: str | None = None,
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
        base_delay_seconds: float = 1.0,
        max_delay_seconds: float = 30.0,
    ) -> None:
        """Initialize the OpenAI-compatible client.

        Args:
            base_url: Base URL for the OpenAI-compatible endpoint.
            model: Model identifier.
            api_key_env: Environment variable holding the API key.
            timeout_seconds: Request timeout in seconds.
            max_retries: Maximum number of retry attempts for transient errors.
            base_delay_seconds: Initial delay in seconds before first retry.
            max_delay_seconds: Maximum delay in seconds between retries.
        """

        self._endpoint = _normalize_chat_endpoint(base_url)
        self._model = model
        self._api_key_env = api_key_env
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries
        self._base_delay_seconds = base_delay_seconds
        self._max_delay_seconds = max_delay_seconds

    def generate(
        self,
        prompt: str,
        *,
        expected_output: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Return the assistant response for the given prompt.

        Args:
            prompt: Prompt content sent to the model.
            expected_output: Optional expected output (unused).
            metadata: Optional metadata (temperature, etc.).

        Returns:
            str: Model response content.

        Raises:
            RuntimeError: If the response is missing content.
            urllib.error.URLError: If the HTTP request fails.
        """

        return self._generate_with_retry(prompt, metadata)

    def _generate_with_retry(
        self,
        prompt: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Internal method that performs the API call with retry logic.

        Args:
            prompt: Prompt content sent to the model.
            metadata: Optional metadata (temperature, etc.).

        Returns:
            str: Model response content.

        Raises:
            RuntimeError: If the response is missing content.
            urllib.error.URLError: If the HTTP request fails after retries.
        """

        @retry_with_backoff(
            max_retries=self._max_retries,
            base_delay=self._base_delay_seconds,
            max_delay=self._max_delay_seconds,
        )
        def _make_request() -> str:
            payload = {
                "model": self._model,
                "messages": [{"role": "user", "content": prompt}],
            }
            if metadata and metadata.get("temperature") is not None:
                payload["temperature"] = float(metadata["temperature"])
            headers = {"Content-Type": "application/json"}
            api_key = os.getenv(self._api_key_env) if self._api_key_env else None
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            request = urllib.request.Request(
                self._endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=self._timeout_seconds) as response:
                body = response.read()
            data = json.loads(body)
            choices = data.get("choices", [])
            if not choices:
                raise RuntimeError("LLM response contained no choices.")
            message = choices[0].get("message", {})
            content = message.get("content")
            if content is None:
                content = choices[0].get("text")
            if content is None:
                raise RuntimeError("LLM response missing content.")
            return str(content)

        return _make_request()


class LMStudioClient(OpenAICompatibleClient):
    """Preconfigured OpenAI-compatible client for LM Studio."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str = "http://127.0.0.1:1234",
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
        base_delay_seconds: float = 1.0,
        max_delay_seconds: float = 30.0,
    ) -> None:
        """Initialize LM Studio client pointing at the local server.

        Args:
            model: Model identifier in LM Studio.
            base_url: Base URL for the LM Studio server.
            timeout_seconds: Request timeout in seconds.
            max_retries: Maximum number of retry attempts for transient errors.
            base_delay_seconds: Initial delay in seconds before first retry.
            max_delay_seconds: Maximum delay in seconds between retries.
        """

        super().__init__(
            base_url=base_url,
            model=model,
            api_key_env=None,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            base_delay_seconds=base_delay_seconds,
            max_delay_seconds=max_delay_seconds,
        )


def _normalize_chat_endpoint(base_url: str) -> str:
    """Normalize base URL into a chat completions endpoint.

    Args:
        base_url: Base URL or full endpoint.

    Returns:
        str: Chat completions endpoint URL.
    """

    trimmed = base_url.rstrip("/")
    if trimmed.endswith("/chat/completions"):
        return trimmed
    if trimmed.endswith("/v1"):
        return f"{trimmed}/chat/completions"
    return f"{trimmed}/v1/chat/completions"


def create_llm_client(config: LLMConfig) -> LLMClient:
    """Create an LLM client based on configuration.

    Args:
        config: LLM configuration object.

    Returns:
        LLMClient: Initialized client instance.

    Raises:
        ValueError: If the configuration mode is unsupported or missing model.
    """

    mode = (config.mode or "local").lower()
    if mode in {"echo", "local"}:
        return EchoLLMClient()
    if mode in {"openai_compatible", "lmstudio"}:
        if not config.model:
            raise ValueError("LLM model must be set for OpenAI-compatible mode.")
        base_url = config.api_url or "http://127.0.0.1:1234"
        if mode == "lmstudio":
            return LMStudioClient(
                model=config.model,
                base_url=base_url,
                timeout_seconds=config.timeout_seconds,
                max_retries=config.max_retries,
                base_delay_seconds=config.base_delay_seconds,
                max_delay_seconds=config.max_delay_seconds,
            )
        return OpenAICompatibleClient(
            base_url=base_url,
            model=config.model,
            api_key_env=config.api_key_env,
            timeout_seconds=config.timeout_seconds,
            max_retries=config.max_retries,
            base_delay_seconds=config.base_delay_seconds,
            max_delay_seconds=config.max_delay_seconds,
        )
    raise ValueError(f"Unsupported LLM mode: {config.mode}")
