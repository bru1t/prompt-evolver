"""
Load and validate pipeline configuration.

This module defines configuration dataclasses and YAML loading helpers for
LLM backends, leakage checks, and iteration limits.

Public API:
- EvolverConfig
- LLMConfig
- load_config(...)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LLMConfig:
    """LLM connection configuration."""

    mode: str = "local"
    model: str | None = None
    api_url: str | None = None
    api_key_env: str | None = None
    timeout_seconds: float = 30.0


@dataclass(frozen=True)
class EvolverConfig:
    """Top-level configuration for the evolution pipeline."""

    max_generations: int = 5
    similarity_weight: float = 0.8
    length_weight: float = 0.2
    llm_execution: LLMConfig = field(default_factory=LLMConfig)
    llm_improvement: LLMConfig = field(default_factory=LLMConfig)
    llm_evaluation: LLMConfig = field(default_factory=LLMConfig)
    max_no_improve: int = 2
    leakage_similarity_threshold: float = 0.45
    leakage_ngram_size: int = 3
    leakage_ngram_overlap_threshold: float = 0.1
    min_prompt_tokens: int = 2
    max_prompt_tokens: int = 200
    max_prompt_increase_ratio: float = 2.0
    max_prompt_increase_tokens: int = 40
    min_improvement_attempts: int = 2


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file using PyYAML when available.

    Args:
        path: Path to the YAML file.

    Returns:
        dict[str, Any]: Parsed YAML contents.

    Raises:
        ModuleNotFoundError: If PyYAML is not installed.
        FileNotFoundError: If the YAML file does not exist.
        OSError: If the file cannot be read.
    """

    try:
        import yaml  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyYAML is required to load config.yaml. "
            "Install it with `pip install pyyaml`."
        ) from exc
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def load_config(path: str | Path) -> EvolverConfig:
    """Load an EvolverConfig from a YAML file.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        EvolverConfig: Parsed configuration object.

    Raises:
        ModuleNotFoundError: If PyYAML is not installed.
        FileNotFoundError: If the YAML file does not exist.
        OSError: If the file cannot be read.
        ValueError: If numeric config values cannot be parsed.
    """

    config_path = Path(path)
    payload = _load_yaml(config_path)
    llm_payload = payload.get("llm", {})
    execution_payload = payload.get("llm_execution", llm_payload)
    improvement_payload = payload.get("llm_improvement", execution_payload)
    evaluation_payload = payload.get("llm_evaluation", execution_payload)
    return EvolverConfig(
        max_generations=int(payload.get("max_generations", 5)),
        similarity_weight=float(payload.get("similarity_weight", 0.8)),
        length_weight=float(payload.get("length_weight", 0.2)),
        llm_execution=LLMConfig(
            mode=execution_payload.get("mode", "local"),
            model=execution_payload.get("model"),
            api_url=execution_payload.get("api_url"),
            api_key_env=execution_payload.get("api_key_env"),
            timeout_seconds=float(execution_payload.get("timeout_seconds", 30.0)),
        ),
        llm_improvement=LLMConfig(
            mode=improvement_payload.get("mode", "local"),
            model=improvement_payload.get("model"),
            api_url=improvement_payload.get("api_url"),
            api_key_env=improvement_payload.get("api_key_env"),
            timeout_seconds=float(improvement_payload.get("timeout_seconds", 30.0)),
        ),
        llm_evaluation=LLMConfig(
            mode=evaluation_payload.get("mode", "local"),
            model=evaluation_payload.get("model"),
            api_url=evaluation_payload.get("api_url"),
            api_key_env=evaluation_payload.get("api_key_env"),
            timeout_seconds=float(evaluation_payload.get("timeout_seconds", 30.0)),
        ),
        max_no_improve=int(payload.get("max_no_improve", 2)),
        leakage_similarity_threshold=float(payload.get("leakage_similarity_threshold", 0.45)),
        leakage_ngram_size=int(payload.get("leakage_ngram_size", 3)),
        leakage_ngram_overlap_threshold=float(
            payload.get("leakage_ngram_overlap_threshold", 0.1)
        ),
        min_prompt_tokens=int(payload.get("min_prompt_tokens", 2)),
        max_prompt_tokens=int(payload.get("max_prompt_tokens", 200)),
        max_prompt_increase_ratio=float(payload.get("max_prompt_increase_ratio", 2.0)),
        max_prompt_increase_tokens=int(payload.get("max_prompt_increase_tokens", 40)),
        min_improvement_attempts=int(payload.get("min_improvement_attempts", 2)),
    )
