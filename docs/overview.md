# Prompt Evolver Overview

Prompt Evolver improves task prompts by running them against test texts,
evaluating the outputs, and rewriting prompts when results fall short.
It is designed for reproducible prompt iteration with guardrails against
data leakage and runaway prompt growth.

Who it's for:
- Product teams iterating on LLM prompts with measurable targets
- Analysts who want repeatable evaluation against expected outputs
- Engineers building CSV-based prompt pipelines with audit trails

When to use it:
- You have known "expected outputs" for a set of tasks
- You want to reduce prompt cost without losing quality
- You need leakage guardrails and reproducible comparisons

Non-goals:
- It does not fine-tune models.
- It does not replace human evaluation for complex tasks.
- It does not auto-generate datasets from scratch.

Core flow:
1) Load prompts, texts, and tasks from CSV files.
2) Execute each task prompt against its test text.
3) Evaluate the output against expected results with a structured evaluator.
4) Improve the prompt using evaluator feedback (no test text leakage).
5) Repeat until pass or stop conditions are met.

Flow diagram:
execute -> evaluate -> improve -> guardrails -> stop

Key features:
- Task-based execution with explicit expected outputs.
- LLM-agnostic execution/improvement/evaluation models.
- Leakage checks to prevent test text inclusion in prompts.
- Sanity checks to avoid overly long or malformed prompts.
- Structured logging for traceability.
