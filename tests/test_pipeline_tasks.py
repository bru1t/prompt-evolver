"""Tests for task-based pipeline behavior."""

from __future__ import annotations

import csv
from pathlib import Path

from prompt_evolver.config import EvolverConfig
from prompt_evolver.io import build_tasks_frame, to_task_records
from prompt_evolver.pipeline import run_pipeline


class _FakeExecutionLLM:
    def generate(self, prompt: str, *, expected_output=None, metadata=None) -> str:  # type: ignore[override]
        """Return deterministic output for execution prompts.

        Args:
            prompt: Prompt content.
            expected_output: Expected output for pass-through behavior.
            metadata: Optional metadata.

        Returns:
            str: Simulated model output.
        """

        if "IMPROVED" in prompt:
            return expected_output or ""
        return "bad output"


class _FakeImprovementLLM:
    def generate(self, prompt: str, *, expected_output=None, metadata=None) -> str:  # type: ignore[override]
        """Return a fixed improved prompt.

        Args:
            prompt: Prompt content.
            expected_output: Unused expected output.
            metadata: Optional metadata.

        Returns:
            str: Improved prompt text.
        """

        return "IMPROVED PROMPT"


class _FakeEvaluationLLM:
    def generate(self, prompt: str, *, expected_output=None, metadata=None) -> str:  # type: ignore[override]
        """Return JSON evaluation based on prompt content.

        Args:
            prompt: Prompt content.
            expected_output: Unused expected output.
            metadata: Optional metadata.

        Returns:
            str: JSON evaluation string.
        """

        if "bad output" in prompt:
            return '{"pass": false, "score": 0.1, "issues": ["wrong"], "suggestions": ["fix"]}'
        return '{"pass": true, "score": 0.9, "issues": [], "suggestions": []}'


class _FakeLongPromptLLM:
    def generate(self, prompt: str, *, expected_output=None, metadata=None) -> str:  # type: ignore[override]
        """Return an overly long prompt for sanity check testing.

        Args:
            prompt: Prompt content.
            expected_output: Unused expected output.
            metadata: Optional metadata.

        Returns:
            str: Long prompt text.
        """

        return "Long prompt. " * 200


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    """Write rows to a CSV file.

    Args:
        path: CSV path to write.
        fieldnames: CSV column names.
        rows: Row mappings to write.

    Returns:
        None
    """
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_build_tasks_frame_merges_inputs(tmp_path: Path) -> None:
    """Verify prompts/texts/tasks merge into task records.

    Args:
        tmp_path: Temporary directory provided by pytest.

    Returns:
        None
    """
    prompts_path = tmp_path / "prompts.csv"
    texts_path = tmp_path / "texts.csv"
    tasks_path = tmp_path / "tasks.csv"
    _write_csv(
        prompts_path,
        ["id", "prompt", "tokens"],
        [{"id": "p1", "prompt": "Say hello", "tokens": "2"}],
    )
    _write_csv(
        texts_path,
        ["id", "text", "tokens"],
        [{"id": "t1", "text": "Hello text", "tokens": "2"}],
    )
    _write_csv(
        tasks_path,
        ["id", "id_text", "id_prompt", "task_type", "expected_output"],
        [
            {
                "id": "task_1",
                "id_text": "t1",
                "id_prompt": "p1",
                "task_type": "Writing",
                "expected_output": "Hello",
            }
        ],
    )
    frame = build_tasks_frame(prompts_path, texts_path, tasks_path)
    records = to_task_records(frame)
    assert records[0].task_id == "task_1"
    assert records[0].prompt == "Say hello"


def test_run_pipeline_improves_prompt(tmp_path: Path) -> None:
    """Verify pipeline improves prompt when evaluation fails.

    Args:
        tmp_path: Temporary directory provided by pytest.

    Returns:
        None
    """
    prompts_path = tmp_path / "prompts.csv"
    texts_path = tmp_path / "texts.csv"
    tasks_path = tmp_path / "tasks.csv"
    output_path = tmp_path / "results.csv"
    _write_csv(
        prompts_path,
        ["id", "prompt", "tokens"],
        [{"id": "p1", "prompt": "Original prompt", "tokens": ""}],
    )
    _write_csv(
        texts_path,
        ["id", "text", "tokens"],
        [{"id": "t1", "text": "Sample text", "tokens": ""}],
    )
    _write_csv(
        tasks_path,
        ["id", "id_text", "id_prompt", "task_type", "expected_output"],
        [
            {
                "id": "task_1",
                "id_text": "t1",
                "id_prompt": "p1",
                "task_type": "Writing",
                "expected_output": "IMPROVED RESULT",
            }
        ],
    )
    config = EvolverConfig(max_generations=1)
    frame = run_pipeline(
        prompts_path,
        texts_path,
        tasks_path,
        output_path,
        config=config,
        execution_prompt_template="Task: {task_type}\nPrompt: {prompt}\nText: {text}",
        improvement_prompt_template="Improve: {prompt}\nIssues: {issues}\nSuggestions: {suggestions}",
        evaluation_prompt_template="Task output:\n{output}\nExpected:\n{expected_output}\n",
        execution_llm=_FakeExecutionLLM(),
        improvement_llm=_FakeImprovementLLM(),
        evaluation_llm=_FakeEvaluationLLM(),
    )
    assert output_path.exists()
    assert frame.loc[0, "prompt_improved"] == "IMPROVED PROMPT"


def test_sanity_check_rejects_long_prompt(tmp_path: Path) -> None:
    """Verify sanity check rejects excessively long prompts.

    Args:
        tmp_path: Temporary directory provided by pytest.

    Returns:
        None
    """

    prompts_path = tmp_path / "prompts.csv"
    texts_path = tmp_path / "texts.csv"
    tasks_path = tmp_path / "tasks.csv"
    output_path = tmp_path / "results.csv"
    _write_csv(
        prompts_path,
        ["id", "prompt", "tokens"],
        [{"id": "p1", "prompt": "Short prompt", "tokens": ""}],
    )
    _write_csv(
        texts_path,
        ["id", "text", "tokens"],
        [{"id": "t1", "text": "Sample text", "tokens": ""}],
    )
    _write_csv(
        tasks_path,
        ["id", "id_text", "id_prompt", "task_type", "expected_output"],
        [
            {
                "id": "task_1",
                "id_text": "t1",
                "id_prompt": "p1",
                "task_type": "Writing",
                "expected_output": "IMPROVED RESULT",
            }
        ],
    )
    config = EvolverConfig(max_generations=1, max_prompt_tokens=10)
    frame = run_pipeline(
        prompts_path,
        texts_path,
        tasks_path,
        output_path,
        config=config,
        execution_prompt_template="Task: {task_type}\nPrompt: {prompt}\nText: {text}",
        improvement_prompt_template="Improve: {prompt}\nIssues: {issues}\nSuggestions: {suggestions}",
        evaluation_prompt_template="Task output:\n{output}\nExpected:\n{expected_output}\n",
        execution_llm=_FakeExecutionLLM(),
        improvement_llm=_FakeLongPromptLLM(),
        evaluation_llm=_FakeEvaluationLLM(),
    )
    assert frame.loc[0, "prompt_improved"] == "Short prompt"
