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
        ValueError: If required columns are missing or invalid.
        FileNotFoundError: If the file does not exist.
    """

    # Force IDs to be read as strings to prevent float conversion (e.g. "1.0" -> "1")
    df = pd.read_csv(path, dtype={"id": str})
    _require_columns(df, {"id", "prompt"}, "prompts")

    # Normalize ID dtype: strip whitespace and reject empty IDs
    df["id"] = df["id"].str.strip()
    if df["id"].isna().any() or (df["id"] == "").any():
        invalid_rows = df[df["id"].isna() | (df["id"] == "")].index.tolist()
        raise ValueError(
            f"CSV validation failed: prompts.csv contains empty or NaN IDs.\n\n"
            f"Invalid rows (0-indexed): {invalid_rows}\n\n"
            f"Please ensure all rows have non-empty ID values."
        )

    # Validate ID uniqueness
    if df["id"].duplicated().any():
        duplicates = df[df["id"].duplicated(keep=False)]["id"].tolist()
        raise ValueError(
            f"CSV validation failed: prompts.csv contains duplicate IDs.\n\n"
            f"Duplicate IDs: {', '.join(set(duplicates))}\n\n"
            f"Please ensure all prompt IDs are unique."
        )

    # Normalize prompt field: strip whitespace and reject empty prompts
    df["prompt"] = df["prompt"].str.strip()
    if df["prompt"].isna().any() or (df["prompt"] == "").any():
        invalid_rows = df[df["prompt"].isna() | (df["prompt"] == "")].index.tolist()
        raise ValueError(
            f"CSV validation failed: prompts.csv contains empty or NaN prompts.\n\n"
            f"Invalid rows (0-indexed): {invalid_rows}\n\n"
            f"Please ensure all rows have non-empty prompt values."
        )

    return df


def read_texts_df(path: str | Path) -> pd.DataFrame:
    """Read texts CSV into a dataframe.

    Args:
        path: Path to the texts CSV file.

    Returns:
        pd.DataFrame: Dataframe with text rows.

    Raises:
        ValueError: If required columns are missing or invalid.
        FileNotFoundError: If the file does not exist.
    """

    # Force IDs to be read as strings
    df = pd.read_csv(path, dtype={"id": str})
    _require_columns(df, {"id", "text"}, "texts")

    # Normalize ID dtype: strip whitespace and reject empty IDs
    df["id"] = df["id"].str.strip()
    if df["id"].isna().any() or (df["id"] == "").any():
        invalid_rows = df[df["id"].isna() | (df["id"] == "")].index.tolist()
        raise ValueError(
            f"CSV validation failed: texts.csv contains empty or NaN IDs.\n\n"
            f"Invalid rows (0-indexed): {invalid_rows}\n\n"
            f"Please ensure all rows have non-empty ID values."
        )

    # Validate ID uniqueness
    if df["id"].duplicated().any():
        duplicates = df[df["id"].duplicated(keep=False)]["id"].tolist()
        raise ValueError(
            f"CSV validation failed: texts.csv contains duplicate IDs.\n\n"
            f"Duplicate IDs: {', '.join(set(duplicates))}\n\n"
            f"Please ensure all text IDs are unique."
        )

    # Normalize text field: strip whitespace and reject empty text
    df["text"] = df["text"].str.strip()
    if df["text"].isna().any() or (df["text"] == "").any():
        invalid_rows = df[df["text"].isna() | (df["text"] == "")].index.tolist()
        raise ValueError(
            f"CSV validation failed: texts.csv contains empty or NaN text values.\n\n"
            f"Invalid rows (0-indexed): {invalid_rows}\n\n"
            f"Please ensure all rows have non-empty text values."
        )

    return df


def read_tasks_df(path: str | Path) -> pd.DataFrame:
    """Read tasks CSV into a dataframe.

    Args:
        path: Path to the tasks CSV file.

    Returns:
        pd.DataFrame: Dataframe with task rows.

    Raises:
        ValueError: If required columns are missing or invalid.
        FileNotFoundError: If the file does not exist.
    """

    # Force all ID columns to be read as strings
    df = pd.read_csv(path, dtype={"id": str, "id_text": str, "id_prompt": str})
    _require_columns(
        df,
        {"id", "id_text", "id_prompt", "task_type", "expected_output"},
        "tasks",
    )

    # Normalize task ID: strip whitespace and reject empty IDs
    df["id"] = df["id"].str.strip()
    if df["id"].isna().any() or (df["id"] == "").any():
        invalid_rows = df[df["id"].isna() | (df["id"] == "")].index.tolist()
        raise ValueError(
            f"CSV validation failed: tasks.csv contains empty or NaN task IDs.\n\n"
            f"Invalid rows (0-indexed): {invalid_rows}\n\n"
            f"Please ensure all rows have non-empty ID values."
        )

    # Validate task ID uniqueness
    if df["id"].duplicated().any():
        duplicates = df[df["id"].duplicated(keep=False)]["id"].tolist()
        raise ValueError(
            f"CSV validation failed: tasks.csv contains duplicate task IDs.\n\n"
            f"Duplicate IDs: {', '.join(set(duplicates))}\n\n"
            f"Please ensure all task IDs are unique."
        )

    # Normalize foreign key IDs: strip whitespace
    df["id_text"] = df["id_text"].str.strip()
    df["id_prompt"] = df["id_prompt"].str.strip()

    # Validate foreign key IDs are not empty
    if df["id_text"].isna().any() or (df["id_text"] == "").any():
        invalid_rows = df[df["id_text"].isna() | (df["id_text"] == "")].index.tolist()
        raise ValueError(
            f"CSV validation failed: tasks.csv contains empty or NaN id_text values.\n\n"
            f"Invalid rows (0-indexed): {invalid_rows}\n\n"
            f"Please ensure all rows have valid text ID references."
        )

    if df["id_prompt"].isna().any() or (df["id_prompt"] == "").any():
        invalid_rows = df[df["id_prompt"].isna() | (df["id_prompt"] == "")].index.tolist()
        raise ValueError(
            f"CSV validation failed: tasks.csv contains empty or NaN id_prompt values.\n\n"
            f"Invalid rows (0-indexed): {invalid_rows}\n\n"
            f"Please ensure all rows have valid prompt ID references."
        )

    # Normalize task_type: strip whitespace
    df["task_type"] = df["task_type"].str.strip()

    # Validate task_type is not empty
    if df["task_type"].isna().any() or (df["task_type"] == "").any():
        invalid_rows = df[df["task_type"].isna() | (df["task_type"] == "")].index.tolist()
        raise ValueError(
            f"CSV validation failed: tasks.csv contains empty or NaN task_type values.\n\n"
            f"Invalid rows (0-indexed): {invalid_rows}\n\n"
            f"Please ensure all rows have non-empty task_type values."
        )

    # Validate task_type against allow-list (case-insensitive)
    # Common task types found in prompt engineering workflows
    ALLOWED_TASK_TYPES = {
        # General categories
        "classification",
        "generation",
        "summarization",
        "extraction",
        "translation",
        "question_answering",
        "writing",
        # Specific task types (from example data and common use cases)
        "editing",
        "comparison",
        "evaluation",
        "research",
        "ops",
        "planning",
        "redaction",
        "consistency",
        "sql",
        "math",
        "product",
        "analysis",
        "formatting",
        "validation",
        "transformation",
    }

    # Convert to lowercase for case-insensitive comparison
    df_types_lower = df["task_type"].str.lower()
    invalid_types = set(df_types_lower.unique()) - ALLOWED_TASK_TYPES
    if invalid_types:
        # Get original case versions for better error message
        invalid_original = df[df_types_lower.isin(invalid_types)]["task_type"].unique()
        invalid_list = ", ".join(sorted(invalid_original))
        allowed_list = ", ".join(sorted(ALLOWED_TASK_TYPES))
        raise ValueError(
            f"CSV validation failed: tasks.csv contains invalid task_type values.\n\n"
            f"Invalid task_type values: {invalid_list}\n"
            f"Allowed task_type values: {allowed_list}\n\n"
            f"Please update task_type values to use one of the allowed types.\n"
            f"If you need to use a custom task_type, it must be added to the validation list."
        )

    # Normalize expected_output: strip whitespace and reject empty values
    df["expected_output"] = df["expected_output"].str.strip()
    if df["expected_output"].isna().any() or (df["expected_output"] == "").any():
        invalid_rows = df[df["expected_output"].isna() | (df["expected_output"] == "")].index.tolist()
        raise ValueError(
            f"CSV validation failed: tasks.csv contains empty or NaN expected_output values.\n\n"
            f"Invalid rows (0-indexed): {invalid_rows}\n\n"
            f"Please ensure all rows have non-empty expected output values."
        )

    return df


def build_tasks_frame(
    prompts_path: str | Path,
    texts_path: str | Path,
    tasks_path: str | Path,
) -> pd.DataFrame:
    """Merge prompts, texts, and tasks into a single dataframe.

    All input validation and normalization is performed in the individual
    read functions (read_prompts_df, read_texts_df, read_tasks_df).

    Args:
        prompts_path: Path to prompts CSV.
        texts_path: Path to texts CSV.
        tasks_path: Path to tasks CSV.

    Returns:
        pd.DataFrame: Merged dataframe with task context.

    Raises:
        ValueError: If referenced prompt/text IDs are missing.
    """

    # Read and validate individual CSVs
    # (ID normalization, uniqueness, and required field validation happen here)
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
        missing_ids = ", ".join(map(str, missing))
        available_prompts = ", ".join(map(str, prompts_df["id_prompt"].unique()))
        error_msg = (
            f"Foreign key validation failed: tasks.csv references missing prompt IDs.\n\n"
            f"Missing prompt IDs: {missing_ids}\n"
            f"Available prompt IDs: {available_prompts}\n\n"
            f"Please ensure all id_prompt values in tasks.csv exist in prompts.csv.\n"
            f"Check for typos or add the missing prompts to prompts.csv."
        )
        raise ValueError(error_msg)

    merged = merged.merge(texts_df, on="id_text", how="left", validate="many_to_one")
    if merged["text"].isna().any():
        missing = merged.loc[merged["text"].isna(), "id_text"].unique()
        missing_ids = ", ".join(map(str, missing))
        available_texts = ", ".join(map(str, texts_df["id_text"].unique()))
        error_msg = (
            f"Foreign key validation failed: tasks.csv references missing text IDs.\n\n"
            f"Missing text IDs: {missing_ids}\n"
            f"Available text IDs: {available_texts}\n\n"
            f"Please ensure all id_text values in tasks.csv exist in texts.csv.\n"
            f"Check for typos or add the missing texts to texts.csv."
        )
        raise ValueError(error_msg)
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
        available_cols = ", ".join(sorted(df.columns))
        error_msg = (
            f"CSV validation failed: {label}.csv is missing required columns.\n\n"
            f"Missing columns: {missing_list}\n"
            f"Available columns: {available_cols}\n"
            f"Required columns: {', '.join(sorted(required))}\n\n"
            f"Please ensure your CSV file includes all required columns.\n"
            f"See docs/data-model.md for CSV format examples."
        )
        raise ValueError(error_msg)


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
