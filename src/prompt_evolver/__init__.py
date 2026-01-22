"""
Expose the public package API.

This module re-exports the key configuration types, IO helpers, and
pipeline entry points for external use.

Public API:
- EvolverConfig
- LLMConfig
- load_config(...)
- run_pipeline(...)
- RetryConfig
- retry_with_backoff(...)
"""

from .config import EvolverConfig, LLMConfig, load_config
from .generator import HeuristicPromptGenerator, PromptGenerator
from .io import (
    build_tasks_frame,
    read_prompts_df,
    read_tasks_df,
    read_texts_df,
    results_to_frame,
    to_task_records,
    write_results_csv,
)
from .llm import EchoLLMClient, LLMClient, LMStudioClient, OpenAICompatibleClient, create_llm_client
from .models import EvaluationFeedback, OutputScore, PromptRecord, TaskRecord, TaskResult
from .pipeline import EVALUATION_SCHEMA, run_pipeline
from .retry import RetryConfig, create_retry_decorator, retry_with_backoff
from .tokenizer import SimpleTokenCounter, TokenCounter

__all__ = [
    "EchoLLMClient",
    "EVALUATION_SCHEMA",
    "EvaluationFeedback",
    "EvolverConfig",
    "HeuristicPromptGenerator",
    "LLMClient",
    "LLMConfig",
    "LMStudioClient",
    "OpenAICompatibleClient",
    "OutputScore",
    "PromptGenerator",
    "PromptRecord",
    "RetryConfig",
    "SimpleTokenCounter",
    "TaskRecord",
    "TaskResult",
    "TokenCounter",
    "build_tasks_frame",
    "create_llm_client",
    "create_retry_decorator",
    "load_config",
    "read_prompts_df",
    "read_tasks_df",
    "read_texts_df",
    "results_to_frame",
    "retry_with_backoff",
    "run_pipeline",
    "to_task_records",
    "write_results_csv",
]
