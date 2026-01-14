"""
Provide the CLI entry point for running the pipeline.

This module parses command-line arguments, loads configuration, and
invokes the pipeline with the selected input and prompt templates.

Public API:
- main()

Notes:
- Side effects: reads CSV inputs and writes results CSV.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import EvolverConfig, load_config
from .pipeline import run_pipeline


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the evolver.

    Returns:
        argparse.Namespace: Parsed CLI arguments.
    """

    parser = argparse.ArgumentParser(description="Run the Prompt Evolver pipeline.")
    parser.add_argument("--prompts", required=True, help="Path to prompts CSV file.")
    parser.add_argument("--texts", required=True, help="Path to texts CSV file.")
    parser.add_argument("--tasks", required=True, help="Path to tasks CSV file.")
    parser.add_argument("--output", required=True, help="Path for output CSV file.")
    parser.add_argument(
        "--config",
        help="Optional path to config.yaml. Defaults to built-in settings.",
    )
    parser.add_argument("--execution-prompt", required=True, help="Path to execution prompt template.")
    parser.add_argument("--improvement-prompt", required=True, help="Path to improvement prompt template.")
    parser.add_argument("--evaluation-prompt", required=True, help="Path to evaluation prompt template.")
    parser.add_argument("--max-generations", type=int, help="Override max improvement attempts.")
    return parser.parse_args()


def main() -> None:
    """Run the prompt evolver from the command line.

    Raises:
        FileNotFoundError: If prompt template files are missing.
    """

    args = _parse_args()
    if args.config:
        config = load_config(args.config)
    else:
        config = EvolverConfig()
    execution_template = Path(args.execution_prompt).read_text(encoding="utf-8")
    improvement_template = Path(args.improvement_prompt).read_text(encoding="utf-8")
    evaluation_template = Path(args.evaluation_prompt).read_text(encoding="utf-8")
    run_pipeline(
        Path(args.prompts),
        Path(args.texts),
        Path(args.tasks),
        Path(args.output),
        config=config,
        execution_prompt_template=execution_template,
        improvement_prompt_template=improvement_template,
        evaluation_prompt_template=evaluation_template,
        max_generations=args.max_generations,
    )


if __name__ == "__main__":
    main()
