"""
Retry logic with exponential backoff for LLM API calls.

This module provides a retry decorator that handles transient HTTP errors
with exponential backoff strategy.

Public API:
- retry_with_backoff(...)
- RetryConfig

Notes:
- Handles HTTP 429 (rate limit), 500, 503 errors
- Configurable max retries and backoff delays
- Logs retry attempts with backoff duration
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, TypeVar
from urllib.error import HTTPError, URLError

F = TypeVar("F", bound=Callable[..., Any])


@dataclass(frozen=True)
class RetryConfig:
    """Configuration for retry behavior."""

    max_retries: int = 3
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 30.0
    exponential_base: float = 2.0


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exponential_base: float = 2.0,
) -> Callable[[F], F]:
    """Decorator that retries a function with exponential backoff on transient errors.

    Args:
        max_retries: Maximum number of retry attempts (default: 3).
        base_delay: Initial delay in seconds before first retry (default: 1.0).
        max_delay: Maximum delay in seconds between retries (default: 30.0).
        exponential_base: Base for exponential backoff calculation (default: 2.0).

    Returns:
        Decorator function that wraps the target function with retry logic.

    Example:
        @retry_with_backoff(max_retries=3, base_delay=1.0)
        def call_api():
            # API call that might fail
            pass
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            logger = logging.getLogger("prompt_evolver.retry")
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except (HTTPError, URLError) as exc:
                    last_exception = exc

                    # Check if error is retryable
                    if isinstance(exc, HTTPError):
                        # Retry on rate limit (429), server errors (500, 503)
                        if exc.code not in {429, 500, 503}:
                            func_name = getattr(func, "__name__", "unknown")
                            logger.debug(
                                f"Non-retryable HTTP error {exc.code} in {func_name}, "
                                f"raising immediately"
                            )
                            raise

                    # If this was the last attempt, raise the exception
                    if attempt >= max_retries:
                        func_name = getattr(func, "__name__", "unknown")
                        logger.warning(
                            f"Max retries ({max_retries}) exceeded for {func_name}, "
                            f"raising exception"
                        )
                        raise

                    # Calculate backoff delay
                    delay = min(base_delay * (exponential_base**attempt), max_delay)

                    # Log retry attempt
                    error_msg = str(exc)
                    if isinstance(exc, HTTPError):
                        error_msg = f"HTTP {exc.code}: {exc.reason}"

                    func_name = getattr(func, "__name__", "unknown")
                    logger.info(
                        f"Retry attempt {attempt + 1}/{max_retries} for {func_name} "
                        f"after error: {error_msg}. Waiting {delay:.2f}s before retry."
                    )

                    time.sleep(delay)

            # This should never be reached, but just in case
            if last_exception:
                raise last_exception
            return None

        return wrapper  # type: ignore

    return decorator


def create_retry_decorator(config: RetryConfig) -> Callable[[F], F]:
    """Create a retry decorator from a RetryConfig object.

    Args:
        config: Retry configuration object.

    Returns:
        Configured retry decorator.

    Example:
        retry_config = RetryConfig(max_retries=5, base_delay_seconds=2.0)
        retry_decorator = create_retry_decorator(retry_config)

        @retry_decorator
        def my_function():
            pass
    """

    return retry_with_backoff(
        max_retries=config.max_retries,
        base_delay=config.base_delay_seconds,
        max_delay=config.max_delay_seconds,
        exponential_base=config.exponential_base,
    )
