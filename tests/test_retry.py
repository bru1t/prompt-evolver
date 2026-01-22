"""Tests for retry logic with exponential backoff."""

import time
from unittest.mock import Mock
from urllib.error import HTTPError, URLError

import pytest

from prompt_evolver.retry import RetryConfig, create_retry_decorator, retry_with_backoff


def test_retry_on_http_429():
    """Test that HTTP 429 errors trigger retries."""
    mock_func = Mock()
    # Fail twice with 429, then succeed
    mock_func.side_effect = [
        HTTPError("http://test", 429, "Too Many Requests", {}, None),
        HTTPError("http://test", 429, "Too Many Requests", {}, None),
        "success",
    ]

    decorated = retry_with_backoff(max_retries=3, base_delay=0.01, max_delay=0.1)(mock_func)
    result = decorated()

    assert result == "success"
    assert mock_func.call_count == 3


def test_retry_on_http_500():
    """Test that HTTP 500 errors trigger retries."""
    mock_func = Mock()
    mock_func.side_effect = [
        HTTPError("http://test", 500, "Internal Server Error", {}, None),
        "success",
    ]

    decorated = retry_with_backoff(max_retries=2, base_delay=0.01)(mock_func)
    result = decorated()

    assert result == "success"
    assert mock_func.call_count == 2


def test_retry_on_http_503():
    """Test that HTTP 503 errors trigger retries."""
    mock_func = Mock()
    mock_func.side_effect = [
        HTTPError("http://test", 503, "Service Unavailable", {}, None),
        "success",
    ]

    decorated = retry_with_backoff(max_retries=2, base_delay=0.01)(mock_func)
    result = decorated()

    assert result == "success"
    assert mock_func.call_count == 2


def test_no_retry_on_http_400():
    """Test that HTTP 400 errors do not trigger retries."""
    mock_func = Mock()
    mock_func.side_effect = HTTPError("http://test", 400, "Bad Request", {}, None)

    decorated = retry_with_backoff(max_retries=3, base_delay=0.01)(mock_func)

    with pytest.raises(HTTPError) as exc_info:
        decorated()

    assert exc_info.value.code == 400
    assert mock_func.call_count == 1  # No retries


def test_no_retry_on_http_401():
    """Test that HTTP 401 errors do not trigger retries."""
    mock_func = Mock()
    mock_func.side_effect = HTTPError("http://test", 401, "Unauthorized", {}, None)

    decorated = retry_with_backoff(max_retries=3, base_delay=0.01)(mock_func)

    with pytest.raises(HTTPError) as exc_info:
        decorated()

    assert exc_info.value.code == 401
    assert mock_func.call_count == 1  # No retries


def test_max_retries_exceeded():
    """Test that retries stop after max_retries."""
    mock_func = Mock()
    mock_func.side_effect = HTTPError("http://test", 429, "Too Many Requests", {}, None)

    decorated = retry_with_backoff(max_retries=2, base_delay=0.01)(mock_func)

    with pytest.raises(HTTPError):
        decorated()

    assert mock_func.call_count == 3  # Initial + 2 retries


def test_exponential_backoff_delay():
    """Test that delays increase exponentially."""
    mock_func = Mock()
    mock_func.side_effect = [
        HTTPError("http://test", 429, "Too Many Requests", {}, None),
        HTTPError("http://test", 429, "Too Many Requests", {}, None),
        "success",
    ]

    start_time = time.time()
    decorated = retry_with_backoff(max_retries=3, base_delay=0.1, exponential_base=2.0)(mock_func)
    result = decorated()
    elapsed = time.time() - start_time

    assert result == "success"
    # First retry: 0.1s, second retry: 0.2s = ~0.3s total
    assert elapsed >= 0.3


def test_max_delay_cap():
    """Test that delays are capped at max_delay."""
    mock_func = Mock()
    mock_func.side_effect = [
        HTTPError("http://test", 429, "Too Many Requests", {}, None),
        "success",
    ]

    start_time = time.time()
    decorated = retry_with_backoff(
        max_retries=2, base_delay=10.0, max_delay=0.1, exponential_base=2.0
    )(mock_func)
    result = decorated()
    elapsed = time.time() - start_time

    assert result == "success"
    # Delay should be capped at 0.1s, not 10.0s
    assert elapsed < 1.0


def test_retry_on_url_error():
    """Test that URLError triggers retries."""
    mock_func = Mock()
    mock_func.side_effect = [
        URLError("Connection refused"),
        "success",
    ]

    decorated = retry_with_backoff(max_retries=2, base_delay=0.01)(mock_func)
    result = decorated()

    assert result == "success"
    assert mock_func.call_count == 2


def test_success_on_first_attempt():
    """Test that successful calls work without retries."""
    mock_func = Mock(return_value="success")

    decorated = retry_with_backoff(max_retries=3, base_delay=0.01)(mock_func)
    result = decorated()

    assert result == "success"
    assert mock_func.call_count == 1


def test_create_retry_decorator_from_config():
    """Test creating retry decorator from RetryConfig."""
    config = RetryConfig(max_retries=2, base_delay_seconds=0.01, max_delay_seconds=0.1)
    decorator = create_retry_decorator(config)

    mock_func = Mock()
    mock_func.side_effect = [
        HTTPError("http://test", 429, "Too Many Requests", {}, None),
        "success",
    ]

    decorated = decorator(mock_func)
    result = decorated()

    assert result == "success"
    assert mock_func.call_count == 2


def test_retry_preserves_function_metadata():
    """Test that decorator preserves function name and docstring."""

    @retry_with_backoff(max_retries=1, base_delay=0.01)
    def test_function():
        """Test docstring."""
        return "result"

    assert test_function.__name__ == "test_function"
    assert test_function.__doc__ == "Test docstring."
