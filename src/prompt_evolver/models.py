"""
Define data models for prompt evolution.

This module contains dataclasses that describe prompts, tasks, evaluation
feedback, and pipeline results.

Public API:
- PromptRecord
- TaskRecord
- EvaluationFeedback
- OutputScore
- TaskResult
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PromptRecord:
    """Represents a single prompt example loaded from the legacy input CSV."""

    prompt: str
    expected_output: str
    input_text: str | None = None
    prompt_tokens: int | None = None
    record_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskRecord:
    """Represents a task assembled from prompt, text, and task CSVs."""

    task_id: str
    prompt_id: str
    text_id: str
    task_type: str
    format_requirements: str | None
    prompt: str
    text: str
    expected_output: str
    prompt_tokens: int
    text_tokens: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OutputScore:
    """Captures scoring details for a model output."""

    total_score: float
    similarity: float
    length_score: float


@dataclass(frozen=True)
class EvaluationFeedback:
    """Structured feedback returned by the evaluation step."""

    passed: bool
    score: float
    issues: list[str]
    suggestions: list[str]
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CriterionResult:
    """Holds the result of a single evaluation criterion."""

    name: str
    passed: bool
    score: float
    detail: str | None = None


@dataclass(frozen=True)
class EvaluationResult:
    """Aggregated evaluation results across criteria."""

    total_score: float
    passed_count: int
    results: list[CriterionResult]


@dataclass(frozen=True)
class EvolutionResult:
    """Captures the best prompt found for a legacy record and its scores."""

    record: PromptRecord
    prompt_improved: str
    tokens_original: int
    tokens_improved: int
    token_delta: int
    score_original: float
    score_improved: float
    criteria_original: list[CriterionResult]
    criteria_improved: list[CriterionResult]
    iterations_used: int


@dataclass(frozen=True)
class TaskResult:
    """Captures the best prompt found for a task and its scores."""

    record: TaskRecord
    prompt_improved: str
    tokens_original: int
    tokens_improved: int
    tokens_delta: int
    score_original: OutputScore
    score_improved: OutputScore
    output_original: str
    output_improved: str
    output_tokens_original: int
    output_tokens_improved: int
    evaluation_original: EvaluationFeedback
    evaluation_improved: EvaluationFeedback
    model_task: str | None
    model_improve: str | None
    model_eval: str | None
    leakage_flag: bool
    sanity_check_details: str | None
    failure_reason: str | None
    iterations_used: int
