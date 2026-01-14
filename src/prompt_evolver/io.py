"""
Read and write CSV files for the prompt evolver.

This module loads prompts, texts, and tasks into pandas dataframes,
merges them into task records, and writes results back to CSV.

Public API:
- read_prompts_df(...)
- read_texts_df(...)
- read_tasks_df(...)
- build_tasks_frame(...)
- to_task_records(...)
- results_to_frame(...)
- write_results_csv(...)

Notes:
- Side effects: reads and writes CSV files on disk.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import json

from .models import EvaluationFeedback, TaskRecord, TaskResult


def read_prompts_df(path: str | Path) -> pd.DataFrame:
    """Read prompts CSV into a dataframe.

    Args:
        path: Path to the prompts CSV file.

    Returns:
        pd.DataFrame: Dataframe with prompt rows.

    Raises:
        ValueError: If required columns are missing.
        FileNotFoundError: If the file does not exist.
    """

# [ ] FIX: ID_DTYPE_NORMALIZATION
# Problem: IDs can be read as numeric/float; enforce string dtype and strip whitespace.
    df = pd.read_csv(path)
    _require_columns(df, {"id", "prompt"}, "prompts")
    return df


def read_texts_df(path: str | Path) -> pd.DataFrame:
    """Read texts CSV into a dataframe.

    Args:
        path: Path to the texts CSV file.

    Returns:
        pd.DataFrame: Dataframe with text rows.

    Raises:
        ValueError: If required columns are missing.
        FileNotFoundError: If the file does not exist.
    """

    df = pd.read_csv(path)
    _require_columns(df, {"id", "text"}, "texts")
    return df


def read_tasks_df(path: str | Path) -> pd.DataFrame:
    """Read tasks CSV into a dataframe.

    Args:
        path: Path to the tasks CSV file.

    Returns:
        pd.DataFrame: Dataframe with task rows.

    Raises:
        ValueError: If required columns are missing.
        FileNotFoundError: If the file does not exist.
    """

    df = pd.read_csv(path)
    _require_columns(
        df,
        {"id", "id_text", "id_prompt", "task_type", "expected_output"},
        "tasks",
    )
# [ ] FIX: TASK_TYPE_VALIDATION
# Problem: task_type is not validated against an allow-list; reject unknown values.
    return df


def build_tasks_frame(
    prompts_path: str | Path,
    texts_path: str | Path,
    tasks_path: str | Path,
) -> pd.DataFrame:
    """Merge prompts, texts, and tasks into a single dataframe.

    Args:
        prompts_path: Path to prompts CSV.
        texts_path: Path to texts CSV.
        tasks_path: Path to tasks CSV.

    Returns:
        pd.DataFrame: Merged dataframe with task context.

    Raises:
        ValueError: If referenced prompt/text IDs are missing.
    """

# [ ] FIX: CSV_VALIDATION_NORMALIZATION
# Problem: Input CSVs lack token/text validation and normalization (whitespace, NaN, token fields).
    prompts_df = read_prompts_df(prompts_path).rename(
        columns={"id": "id_prompt", "tokens": "prompt_tokens"}
    )
    texts_df = read_texts_df(texts_path).rename(
        columns={"id": "id_text", "tokens": "text_tokens"}
    )
    tasks_df = read_tasks_df(tasks_path).rename(columns={"id": "id_task"})
    merged = tasks_df.merge(prompts_df, on="id_prompt", how="left", validate="many_to_one")
    if merged["prompt"].isna().any():
        missing = merged.loc[merged["prompt"].isna(), "id_prompt"].unique()
        raise ValueError(f"Missing prompts for ids: {', '.join(map(str, missing))}")
    merged = merged.merge(texts_df, on="id_text", how="left", validate="many_to_one")
    if merged["text"].isna().any():
        missing = merged.loc[merged["text"].isna(), "id_text"].unique()
        raise ValueError(f"Missing texts for ids: {', '.join(map(str, missing))}")
    return merged


def to_task_records(frame: pd.DataFrame) -> list[TaskRecord]:
    """Convert a merged dataframe into TaskRecord entries.

    Args:
        frame: Merged dataframe from prompts/texts/tasks.

    Returns:
        list[TaskRecord]: Task records for pipeline execution.
    """

    records: list[TaskRecord] = []
    required = {
        "id_task",
        "id_text",
        "id_prompt",
        "task_type",
        "prompt",
        "text",
        "expected_output",
    }
    for _, row in frame.iterrows():
        metadata = {
            key: row[key]
            for key in frame.columns
            if key not in required | {"prompt_tokens", "text_tokens", "format_requirements"}
        }
        records.append(
            TaskRecord(
                task_id=str(row["id_task"]),
                prompt_id=str(row["id_prompt"]),
                text_id=str(row["id_text"]),
                task_type=str(row["task_type"]),
                format_requirements=_to_optional_str(row.get("format_requirements")),
                prompt=str(row["prompt"]),
                text=str(row["text"]),
                expected_output=str(row["expected_output"]),
                prompt_tokens=_to_int(row.get("prompt_tokens")),
                text_tokens=_to_int(row.get("text_tokens")),
                metadata=metadata,
            )
        )
    return records


def results_to_frame(results: list[TaskResult]) -> pd.DataFrame:
    """Convert TaskResult entries into a dataframe for output.

    Args:
        results: List of task results.

    Returns:
        pd.DataFrame: Output dataframe for CSV export.
    """

    rows = []
    for result in results:
        rows.append(
            {
                "id_task": result.record.task_id,
                "id_text": result.record.text_id,
                "id_prompt": result.record.prompt_id,
                "prompt_original": result.record.prompt,
                "prompt_improved": result.prompt_improved,
                "tokens_original": result.tokens_original,
                "tokens_improved": result.tokens_improved,
                "tokens_delta": result.tokens_delta,
                "iterations_used": result.iterations_used,
                "output_original": result.output_original,
                "output_improved": result.output_improved,
                "output_tokens_original": result.output_tokens_original,
                "output_tokens_improved": result.output_tokens_improved,
                "evaluation_original": _dump_evaluation(result.evaluation_original),
                "evaluation_improved": _dump_evaluation(result.evaluation_improved),
                "model_task": result.model_task or "",
                "model_improve": result.model_improve or "",
                "model_eval": result.model_eval or "",
                "leakage_flag": "yes" if result.leakage_flag else "no",
                "sanity_check_details": result.sanity_check_details or "",
                "failure_reason": result.failure_reason or "",
            }
        )
    return pd.DataFrame(rows)


def write_results_csv(path: str | Path, results: list[TaskResult]) -> None:
    """Write results to a CSV file.

    Args:
        path: Output CSV path.
        results: List of task results.
    """

    frame = results_to_frame(results)
    frame.to_csv(path, index=False)


def _require_columns(df: pd.DataFrame, required: set[str], label: str) -> None:
    """Raise if required columns are missing.

    Args:
        df: Dataframe to validate.
        required: Required column names.
        label: Label for error messages.

    Raises:
        ValueError: If required columns are missing.
    """

    missing = required - set(df.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"{label} CSV missing required columns: {missing_list}")


def _to_int(value: object) -> int:
    """Normalize a dataframe cell to int, defaulting missing values to 0.

    Args:
        value: Cell value to normalize.

    Returns:
        int: Parsed integer value or 0.
    """

    if value is None or (isinstance(value, float) and pd.isna(value)):
        return 0
    text = str(value).strip()
    if not text:
        return 0
    return int(float(text))


def _to_optional_str(value: object) -> str | None:
    """Normalize a dataframe cell to a stripped string or None.

    Args:
        value: Cell value to normalize.

    Returns:
        str | None: Normalized string or None.
    """

    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    return text or None


def _dump_evaluation(feedback: EvaluationFeedback) -> str:
    """Serialize evaluation feedback to a JSON string.

    Args:
        feedback: Evaluation feedback to serialize.

    Returns:
        str: JSON string representation.
    """

    payload = {
        "pass": feedback.passed,
        "score": feedback.score,
        "issues": feedback.issues,
        "suggestions": feedback.suggestions,
    }
    return json.dumps(payload, ensure_ascii=True)
