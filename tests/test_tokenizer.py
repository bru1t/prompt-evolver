"""Tests for token counting."""

from prompt_evolver.tokenizer import SimpleTokenCounter


def test_simple_token_counter_counts_tokens() -> None:
    """Verify token counts for a simple sentence.

    Returns:
        None
    """

    counter = SimpleTokenCounter()
    assert counter.count("Hello, world!") == 4


def test_simple_token_counter_empty() -> None:
    """Verify token count for empty input.

    Returns:
        None
    """

    counter = SimpleTokenCounter()
    assert counter.count("") == 0
