"""
Run the prompt-evolution loop for a set of tasks.

This module executes prompts against task texts, evaluates outputs against
expected results, and rewrites prompts when they fail. It is designed for
reproducible prompt iteration with safety checks (leakage detection) and
token-budget limits.

Public API:
- run_pipeline(...)
- evolve_task(...)

Notes:
- Side effects: reads CSV inputs and writes results CSV.
- Requires LLM backends configured in configs/config.yaml.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import json
import logging
import re
from pathlib import Path

from .config import EvolverConfig
from .io import build_tasks_frame, results_to_frame, to_task_records, write_results_csv
from .llm import LLMClient, create_llm_client
from .models import EvaluationFeedback, OutputScore, TaskRecord, TaskResult
from .tokenizer import SimpleTokenCounter, TokenCounter

try:
    import jsonschema
    from jsonschema import ValidationError as JSONSchemaValidationError

    JSONSCHEMA_AVAILABLE = True
except ImportError:
    JSONSCHEMA_AVAILABLE = False
    JSONSchemaValidationError = Exception  # type: ignore


@dataclass(frozen=True)
class CandidateEvaluation:
    """Stores evaluation details for a prompt candidate."""

    prompt: str
    tokens: int
    output: str
    output_score: OutputScore
    evaluation: EvaluationFeedback


@dataclass(frozen=True)
class LeakageResult:
    """Leakage check output."""

    detected: bool
    similarity: float
    ngram_overlap: float


# JSON schema for evaluator response validation
EVALUATION_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "properties": {
        "pass": {
            "type": "boolean",
            "description": "Whether the output meets the requirements",
        },
        "score": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "Numerical score between 0.0 and 1.0",
        },
        "issues": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of issues found in the output",
        },
        "suggestions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of suggestions for improvement",
        },
    },
    "required": ["pass", "score", "issues", "suggestions"],
    "additionalProperties": True,  # Allow extra fields for flexibility
}


def run_pipeline(
    prompts_path: str | Path,
    texts_path: str | Path,
    tasks_path: str | Path,
    output_path: str | Path,
    *,
    config: EvolverConfig,
    execution_prompt_template: str,
    improvement_prompt_template: str,
    evaluation_prompt_template: str,
    max_generations: int | None = None,
    execution_model: str | None = None,
    improvement_model: str | None = None,
    evaluation_model: str | None = None,
    execution_llm: LLMClient | None = None,
    improvement_llm: LLMClient | None = None,
    evaluation_llm: LLMClient | None = None,
    token_counter: TokenCounter | None = None,
):
    """Run the full pipeline and write results to CSV.

    Args:
        prompts_path: Path to prompts CSV.
        texts_path: Path to texts CSV.
        tasks_path: Path to tasks CSV.
        output_path: Path for the results CSV.
        config: Pipeline configuration.
        execution_prompt_template: Template for task execution prompts.
        improvement_prompt_template: Template for prompt improvement.
        evaluation_prompt_template: Template for evaluation feedback.
        max_generations: Optional override for max improvement iterations.
        execution_model: Model name for execution LLM (overrides config).
        improvement_model: Model name for improvement LLM (overrides config).
        evaluation_model: Model name for evaluation LLM (overrides config).
        execution_llm: Optional execution model client.
        improvement_llm: Optional improvement model client.
        evaluation_llm: Optional evaluation model client.
        token_counter: Optional token counter implementation.

    Returns:
        pd.DataFrame: Results dataframe that was written to CSV.

    Raises:
        ValueError: If task references are missing in prompts/texts.
    """

    _ensure_logging()
    logger = logging.getLogger("prompt_evolver")
    token_counter = token_counter or SimpleTokenCounter()

    # Create LLM configs with model overrides if provided
    from dataclasses import replace
    exec_config = config.llm_execution
    if execution_model:
        exec_config = replace(exec_config, model=execution_model)
    improve_config = config.llm_improvement
    if improvement_model:
        improve_config = replace(improve_config, model=improvement_model)
    eval_config = config.llm_evaluation
    if evaluation_model:
        eval_config = replace(eval_config, model=evaluation_model)

    execution_llm = execution_llm or create_llm_client(exec_config)
    improvement_llm = improvement_llm or create_llm_client(improve_config)
    evaluation_llm = evaluation_llm or create_llm_client(eval_config)
    max_generations = max_generations if max_generations is not None else config.max_generations
    max_generations = max(max_generations, config.min_improvement_attempts)
    logger.info(
        "Start pipeline prompts=%s texts=%s tasks=%s output=%s",
        prompts_path,
        texts_path,
        tasks_path,
        output_path,
    )
    tasks_frame = build_tasks_frame(prompts_path, texts_path, tasks_path)
    records = to_task_records(tasks_frame)
    total_tasks = len(records)
    logger.info(f"Processing {total_tasks} tasks...")

# [ ] FIX: PARALLEL_TASK_EXECUTION
# Problem: Pipeline executes tasks sequentially; add max_workers option with stable ordering.
    results = []
    for idx, record in enumerate(records, 1):
        progress_pct = int((idx / total_tasks) * 100)
        logger.info(f"Task {idx}/{total_tasks} ({progress_pct}%) - Processing task_id={record.task_id}")

        result = evolve_task(
            record,
            config=config,
            execution_prompt_template=execution_prompt_template,
            improvement_prompt_template=improvement_prompt_template,
            evaluation_prompt_template=evaluation_prompt_template,
            max_generations=max_generations,
            max_no_improve=config.max_no_improve,
            execution_llm=execution_llm,
            improvement_llm=improvement_llm,
            evaluation_llm=evaluation_llm,
            token_counter=token_counter,
        )
        results.append(result)

        logger.info(
            f"Task {idx}/{total_tasks} ({progress_pct}%) - Completed: "
            f"iterations={result.iterations_used}, "
            f"token_delta={result.tokens_delta:+d}, "
            f"leakage={'YES' if result.leakage_flag else 'NO'}"
        )

    logger.info("All tasks completed. Writing results...")
    write_results_csv(output_path, results)
    logger.info(f"Pipeline complete. Results written to: {output_path}")
    return results_to_frame(results)


def evolve_task(
    record: TaskRecord,
    *,
    config: EvolverConfig,
    execution_prompt_template: str,
    improvement_prompt_template: str,
    evaluation_prompt_template: str,
    max_generations: int,
    max_no_improve: int,
    execution_llm: LLMClient,
    improvement_llm: LLMClient,
    evaluation_llm: LLMClient,
    token_counter: TokenCounter,
) -> TaskResult:
    """Evolve the prompt for a single task.

    Args:
        record: Task record to process.
        config: Pipeline configuration.
        execution_prompt_template: Template for task execution prompts.
        improvement_prompt_template: Template for prompt improvement.
        evaluation_prompt_template: Template for evaluation feedback.
        max_generations: Maximum improvement iterations.
        max_no_improve: Maximum consecutive non-improvements.
        execution_llm: Execution model client.
        improvement_llm: Improvement model client.
        evaluation_llm: Evaluation model client.
        token_counter: Token counter implementation.

    Returns:
        TaskResult: Best prompt and evaluation data for the task.
    """

    logger = logging.getLogger("prompt_evolver")
    logger.info(
        "Start task id_task=%s id_prompt=%s id_text=%s type=%s",
        record.task_id,
        record.prompt_id,
        record.text_id,
        record.task_type,
    )
    original_prompt = record.prompt.strip()
    original_tokens = record.prompt_tokens or token_counter.count(original_prompt)
    original_output = _run_task(
        record,
        original_prompt,
        execution_prompt_template,
        execution_llm,
    )
    original_eval = _evaluate_output(
        output=original_output,
        record=record,
        evaluation_prompt_template=evaluation_prompt_template,
        evaluation_llm=evaluation_llm,
    )
    if original_eval.passed:
        logger.info("Eval passed task=%s score=%.2f", record.task_id, original_eval.score)
    else:
        logger.warning(
            "Eval failed task=%s score=%.2f issues=%s",
            record.task_id,
            original_eval.score,
            original_eval.issues,
        )
    original_score = _score_output(
        original_output,
        record.expected_output,
        token_counter=token_counter,
        similarity_weight=config.similarity_weight,
        length_weight=config.length_weight,
    )
    original_output_tokens = token_counter.count(original_output)
    best = (
        CandidateEvaluation(
            prompt=original_prompt,
            tokens=original_tokens,
            output=original_output,
            output_score=original_score,
            evaluation=original_eval,
        )
        if original_eval.passed
        else None
    )
    iterations_used = 0
    seen = {original_prompt}
    current_prompt = original_prompt
    no_improve = 0
    leakage_flag = False
    sanity_check_details = None
    last_feedback = original_eval
    stop_reason = None
    min_attempts = config.min_improvement_attempts
    for _ in range(max_generations):
        if stop_reason == "pass":
            break
        iterations_used += 1
        logger.info("Iteration start task=%s iter=%d", record.task_id, iterations_used)
        allowed_max_tokens = _allowed_prompt_tokens(
            original_tokens=original_tokens,
            max_prompt_tokens=config.max_prompt_tokens,
            max_increase_ratio=config.max_prompt_increase_ratio,
            max_increase_tokens=config.max_prompt_increase_tokens,
        )
        improvement_request = _render_improvement_prompt(
            improvement_prompt_template,
            prompt=current_prompt,
            task_type=record.task_type,
            format_requirements=record.format_requirements,
            issues=last_feedback.issues,
            suggestions=last_feedback.suggestions,
            original_tokens=original_tokens,
            max_allowed_tokens=allowed_max_tokens,
        )
        response = improvement_llm.generate(improvement_request, metadata=record.metadata)
        candidate_prompt = _extract_prompt(response)
        if not candidate_prompt or candidate_prompt in seen:
            continue
# [ ] FIX: GUARDRAIL_FALLBACK_REWRITE
# Problem: When sanity/leakage blocks all candidates, add a fallback rewrite heuristic.
        sanity_ok, sanity_reason = _sanity_check_prompt(
            candidate_prompt,
            original_tokens=original_tokens,
            token_counter=token_counter,
            min_tokens=config.min_prompt_tokens,
            max_tokens=config.max_prompt_tokens,
            max_increase_ratio=config.max_prompt_increase_ratio,
            max_increase_tokens=config.max_prompt_increase_tokens,
        )
        if not sanity_ok:
            sanity_check_details = sanity_reason
            logger.warning(
                "Sanity check failed task=%s iter=%d reason=%s",
                record.task_id,
                iterations_used,
                sanity_reason,
            )
            last_feedback = EvaluationFeedback(
                passed=False,
                score=0.0,
                issues=last_feedback.issues + [f"sanity_check_failed:{sanity_reason}"],
                suggestions=last_feedback.suggestions
                + ["Keep the prompt concise and instruction-focused."],
                raw=last_feedback.raw,
            )
            no_improve += 1
            if no_improve >= max_no_improve and iterations_used >= min_attempts:
                stop_reason = f"sanity_check:{sanity_reason}"
                break
            continue
        leakage = _detect_leakage(
            candidate_prompt,
            record.text,
            similarity_threshold=config.leakage_similarity_threshold,
            ngram_size=config.leakage_ngram_size,
            ngram_overlap_threshold=config.leakage_ngram_overlap_threshold,
        )
        if leakage.detected:
            leakage_flag = True
            sanity_check_details = "leakage"
            logger.error(
                "Leakage detected task=%s iter=%d overlap=%.2f similarity=%.2f",
                record.task_id,
                iterations_used,
                leakage.ngram_overlap,
                leakage.similarity,
            )
            last_feedback = EvaluationFeedback(
                passed=False,
                score=0.0,
                issues=last_feedback.issues + ["leakage_detected"],
                suggestions=last_feedback.suggestions
                + ["Do not include input text; keep prompt generalizable."],
                raw=last_feedback.raw,
            )
            no_improve += 1
            if no_improve >= max_no_improve and iterations_used >= min_attempts:
                stop_reason = "leakage"
                break
            continue
        seen.add(candidate_prompt)
        candidate_tokens = token_counter.count(candidate_prompt)
        candidate_output = _run_task(
            record,
            candidate_prompt,
            execution_prompt_template,
            execution_llm,
        )
        candidate_eval = _evaluate_output(
            output=candidate_output,
            record=record,
            evaluation_prompt_template=evaluation_prompt_template,
            evaluation_llm=evaluation_llm,
        )
        candidate_score = _score_output(
            candidate_output,
            record.expected_output,
            token_counter=token_counter,
            similarity_weight=config.similarity_weight,
            length_weight=config.length_weight,
        )
        candidate_eval_entry = CandidateEvaluation(
            prompt=candidate_prompt,
            tokens=candidate_tokens,
            output=candidate_output,
            output_score=candidate_score,
            evaluation=candidate_eval,
        )
        last_feedback = candidate_eval
        if candidate_eval.passed and (
            best is None or _is_better_candidate(candidate_eval_entry, best)
        ):
            best = candidate_eval_entry
            current_prompt = candidate_prompt
            no_improve = 0
            logger.info(
                "Accept prompt task=%s iter=%d score=%.2f tokens_delta=%d",
                record.task_id,
                iterations_used,
                best.evaluation.score,
                best.tokens - original_tokens,
            )
            if iterations_used >= min_attempts:
                stop_reason = "pass"
                break
        else:
            no_improve += 1
            logger.warning(
                "Reject prompt task=%s iter=%d score=%.2f issues=%s",
                record.task_id,
                iterations_used,
                candidate_eval.score,
                candidate_eval.issues,
            )
        if best is not None and best.evaluation.passed:
            stop_reason = "pass"
            break
        if no_improve >= max_no_improve and iterations_used >= min_attempts:
            stop_reason = "no_improvement"
            break
    if stop_reason is None:
        stop_reason = "max_iterations" if iterations_used >= max_generations else "stopped"
    logger.info(
        "End task id_task=%s status=%s iterations_used=%d",
        record.task_id,
        stop_reason,
        iterations_used,
    )
    final_prompt = best.prompt if best is not None else original_prompt
    final_output = best.output if best is not None else original_output
    final_score = best.output_score if best is not None else original_score
    final_eval = best.evaluation if best is not None else original_eval
    final_tokens = best.tokens if best is not None else original_tokens
# [ ] FIX: EVALUATOR_METADATA
# Problem: Evaluator metadata (model, temperature) is not stored with evaluation output.
    return TaskResult(
        record=record,
        prompt_improved=final_prompt,
        tokens_original=original_tokens,
        tokens_improved=final_tokens,
        tokens_delta=final_tokens - original_tokens,
        score_original=original_score,
        score_improved=final_score,
        output_original=original_output,
        output_improved=final_output,
        output_tokens_original=original_output_tokens,
        output_tokens_improved=token_counter.count(final_output),
        evaluation_original=original_eval,
        evaluation_improved=final_eval,
        model_task=config.llm_execution.model,
        model_improve=config.llm_improvement.model,
        model_eval=config.llm_evaluation.model,
        leakage_flag=leakage_flag,
        sanity_check_details=sanity_check_details,
        failure_reason=None if stop_reason == "pass" else stop_reason,
        iterations_used=iterations_used,
    )


def _run_task(
    record: TaskRecord,
    prompt: str,
    execution_prompt_template: str,
    execution_llm: LLMClient,
) -> str:
    """Run a single prompt + text task through the execution LLM.

    Args:
        record: Task record containing text and metadata.
        prompt: Prompt string to execute.
        execution_prompt_template: Template used to format the task prompt.
        execution_llm: Execution model client.

    Returns:
        str: Model output for the task.
    """

    model_prompt = _render_execution_prompt(
        execution_prompt_template,
        prompt=prompt,
        task_type=record.task_type,
        text=record.text,
        format_requirements=record.format_requirements,
    )
    return execution_llm.generate(
        model_prompt,
        expected_output=record.expected_output,
        metadata=record.metadata,
    )


def _evaluate_output(
    *,
    output: str,
    record: TaskRecord,
    evaluation_prompt_template: str,
    evaluation_llm: LLMClient,
) -> EvaluationFeedback:
    """Evaluate output with an LLM using structured feedback.

    Args:
        output: Model output to evaluate.
        record: Task record containing expected output and constraints.
        evaluation_prompt_template: Template for evaluation prompt.
        evaluation_llm: Evaluation model client.

    Returns:
        EvaluationFeedback: Structured evaluation feedback.
    """

    prompt = _render_evaluation_prompt(
        evaluation_prompt_template,
        task_type=record.task_type,
        output=output,
        expected_output=record.expected_output,
        format_requirements=record.format_requirements,
    )
    response = evaluation_llm.generate(prompt, metadata=record.metadata)
    feedback = _parse_evaluation_response(response)
    feedback = _sanitize_feedback(feedback, record.expected_output)
    return feedback


def _score_output(
    output: str,
    expected_output: str,
    *,
    token_counter: TokenCounter,
    similarity_weight: float,
    length_weight: float,
) -> OutputScore:
    """Score model output for similarity and length alignment.

    Args:
        output: Model output to score.
        expected_output: Target output for comparison.
        token_counter: Token counter implementation.
        similarity_weight: Weight for similarity score.
        length_weight: Weight for length alignment score.

    Returns:
        OutputScore: Aggregated scoring metrics.
    """

# [ ] FIX: SIMILARITY_BACKEND
# Problem: SequenceMatcher is brittle for semantic tasks; add embedding/F1 similarity with fallback.
    similarity = SequenceMatcher(None, output, expected_output).ratio()
    output_tokens = token_counter.count(output)
    expected_tokens = token_counter.count(expected_output)
    if expected_tokens == 0 and output_tokens == 0:
        length_score = 1.0
    elif expected_tokens == 0 or output_tokens == 0:
        length_score = 0.0
    else:
        length_score = min(output_tokens, expected_tokens) / max(output_tokens, expected_tokens)
    total = similarity_weight * similarity + length_weight * length_score
    return OutputScore(total_score=total, similarity=similarity, length_score=length_score)


def _is_better_candidate(candidate: CandidateEvaluation, best: CandidateEvaluation) -> bool:
    """Return True if candidate should replace the best candidate.

    Args:
        candidate: Candidate evaluation.
        best: Current best evaluation.

    Returns:
        bool: True if candidate is preferred.
    """

# [ ] FIX: ACCEPTANCE_POLICY
# Problem: Acceptance criteria are implicit across eval vs similarity; formalize in config/docs.
    if candidate.evaluation.passed != best.evaluation.passed:
        return candidate.evaluation.passed
    if candidate.evaluation.score != best.evaluation.score:
        return candidate.evaluation.score > best.evaluation.score
    if candidate.output_score.total_score > best.output_score.total_score:
        return True
    if candidate.output_score.total_score == best.output_score.total_score:
        return candidate.tokens < best.tokens
    return False


def _render_execution_prompt(
    template: str,
    *,
    prompt: str,
    task_type: str,
    text: str,
    format_requirements: str | None,
) -> str:
    """Render the prompt sent to the execution model.

    Args:
        template: Prompt template string.
        prompt: Base prompt text.
        task_type: Task type label.
        text: Task input text.
        format_requirements: Optional formatting constraints.

    Returns:
        str: Formatted execution prompt.
    """

    return template.format(
        prompt=prompt,
        task_type=task_type,
        text=text,
        format_requirements=format_requirements or "None",
    )


def _render_improvement_prompt(
    template: str,
    *,
    prompt: str,
    task_type: str,
    format_requirements: str | None,
    issues: list[str],
    suggestions: list[str],
    original_tokens: int,
    max_allowed_tokens: int,
) -> str:
    """Render the prompt sent to the improvement model.

    Args:
        template: Prompt template string.
        prompt: Current prompt text.
        task_type: Task type label.
        format_requirements: Optional formatting constraints.
        issues: Evaluator issues list.
        suggestions: Evaluator suggestions list.
        original_tokens: Token count for the original prompt.
        max_allowed_tokens: Maximum allowed token count.

    Returns:
        str: Formatted improvement prompt.
    """

    return template.format(
        prompt=prompt,
        task_type=task_type,
        format_requirements=format_requirements or "None",
        issues="\n".join(f"- {issue}" for issue in issues) or "None",
        suggestions="\n".join(f"- {suggestion}" for suggestion in suggestions) or "None",
        original_tokens=original_tokens,
        max_allowed_tokens=max_allowed_tokens,
    )


def _render_evaluation_prompt(
    template: str,
    *,
    task_type: str,
    output: str,
    expected_output: str,
    format_requirements: str | None,
) -> str:
    """Render the prompt sent to the evaluation model.

    Args:
        template: Prompt template string.
        task_type: Task type label.
        output: Model output to evaluate.
        expected_output: Target output for comparison.
        format_requirements: Optional formatting constraints.

    Returns:
        str: Formatted evaluation prompt.
    """

    return template.format(
        task_type=task_type,
        output=output,
        expected_output=expected_output,
        format_requirements=format_requirements or "None",
    )


def _extract_prompt(response: str) -> str:
    """Extract the improved prompt from a model response.

    Args:
        response: Model response containing an improved prompt.

    Returns:
        str: Extracted prompt string.
    """

    stripped = response.strip()
    if "```" not in stripped:
        return stripped
    parts = stripped.split("```")
    if len(parts) < 2:
        return stripped
    return parts[1].strip()


def _parse_evaluation_response(response: str) -> EvaluationFeedback:
    """Parse evaluator JSON response into structured feedback with schema validation.

    Args:
        response: Raw evaluator response string.

    Returns:
        EvaluationFeedback: Parsed feedback.

    Raises:
        ValueError: If the JSON response does not conform to the expected schema.
    """

    logger = logging.getLogger("prompt_evolver")
    payload = _load_json_from_response(response)

    # Validate against schema if jsonschema is available
    if JSONSCHEMA_AVAILABLE:
        try:
            jsonschema.validate(instance=payload, schema=EVALUATION_SCHEMA)
        except JSONSchemaValidationError as exc:
            error_msg = f"Evaluator response failed schema validation: {exc.message}"
            logger.error(error_msg)
            logger.debug(f"Invalid payload: {payload}")
            # Return a failure feedback with schema validation error
            return EvaluationFeedback(
                passed=False,
                score=0.0,
                issues=[f"schema_validation_error: {exc.message}"],
                suggestions=["Ensure evaluator returns valid JSON with required fields"],
                raw=payload,
            )
    else:
        logger.warning(
            "jsonschema not available - skipping schema validation. "
            "Install with: pip install jsonschema"
        )

    # Extract and validate field values
    passed = bool(payload.get("pass", False))
    score = float(payload.get("score", 0.0))

    # Validate score range
    if not 0.0 <= score <= 1.0:
        logger.warning(f"Score {score} outside valid range [0.0, 1.0], clamping")
        score = max(0.0, min(1.0, score))

    issues = [str(item) for item in payload.get("issues", []) if str(item).strip()]
    suggestions = [str(item) for item in payload.get("suggestions", []) if str(item).strip()]

    if not passed and not issues:
        issues = ["unspecified_issues"]

    return EvaluationFeedback(
        passed=passed,
        score=score,
        issues=issues,
        suggestions=suggestions,
        raw=payload,
    )


def _load_json_from_response(response: str) -> dict:
    """Extract a JSON object from an LLM response.

    Args:
        response: Raw response string.

    Returns:
        dict: Parsed JSON object or fallback structure.
    """

    try:
        return json.loads(response)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", response, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return {"pass": False, "score": 0.0, "issues": ["invalid_json"], "suggestions": []}


def _sanitize_feedback(feedback: EvaluationFeedback, expected_output: str) -> EvaluationFeedback:
    """Remove leakage from evaluator feedback to avoid test data exposure.

    Args:
        feedback: Evaluation feedback to sanitize.
        expected_output: Expected output to guard against.

    Returns:
        EvaluationFeedback: Sanitized feedback.
    """

    if not expected_output:
        return feedback
    expected_lower = expected_output.lower()
    issues = [issue for issue in feedback.issues if expected_lower not in issue.lower()]
    suggestions = [
        suggestion
        for suggestion in feedback.suggestions
        if expected_lower not in suggestion.lower()
    ]
    if len(issues) != len(feedback.issues) or len(suggestions) != len(feedback.suggestions):
        issues = issues + ["evaluator_leakage_filtered"]
        return EvaluationFeedback(
            passed=False,
            score=0.0,
            issues=issues,
            suggestions=suggestions,
            raw=feedback.raw,
        )
    return feedback


def _detect_leakage(
    prompt: str,
    text: str,
    *,
    similarity_threshold: float,
    ngram_size: int,
    ngram_overlap_threshold: float,
) -> LeakageResult:
    """Detect whether a prompt leaks the input text.

    Args:
        prompt: Candidate prompt text.
        text: Task input text.
        similarity_threshold: Similarity threshold for leakage.
        ngram_size: N-gram size for overlap detection.
        ngram_overlap_threshold: Overlap threshold for leakage.

    Returns:
        LeakageResult: Leakage detection metrics.
    """

    prompt_text = prompt.strip().lower()
    source_text = text.strip().lower()
    similarity = SequenceMatcher(None, prompt_text, source_text).ratio()
    if source_text and source_text in prompt_text:
        return LeakageResult(detected=True, similarity=similarity, ngram_overlap=1.0)
    prompt_ngrams = _ngrams(_tokenize(prompt_text), ngram_size)
    text_ngrams = _ngrams(_tokenize(source_text), ngram_size)
    if not text_ngrams:
        return LeakageResult(detected=False, similarity=similarity, ngram_overlap=0.0)
    overlap = len(prompt_ngrams & text_ngrams) / len(text_ngrams)
    detected = similarity >= similarity_threshold or overlap >= ngram_overlap_threshold
    return LeakageResult(detected=detected, similarity=similarity, ngram_overlap=overlap)


def _tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase words for overlap checks.

    Args:
        text: Input text to tokenize.

    Returns:
        list[str]: Tokenized word list.
    """

    return re.findall(r"[a-z0-9]+", text.lower())


def _ngrams(tokens: list[str], size: int) -> set[str]:
    """Build an n-gram set from token list.

    Args:
        tokens: Token list.
        size: N-gram size.

    Returns:
        set[str]: Set of n-grams.
    """

    if size <= 0 or len(tokens) < size:
        return set()
    return {" ".join(tokens[i : i + size]) for i in range(len(tokens) - size + 1)}


def _ensure_logging() -> None:
    """Configure logging if no handlers are configured.

    Returns:
        None
    """

    root_logger = logging.getLogger()
    if root_logger.handlers:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )




def _sanity_check_prompt(
    prompt: str,
    *,
    original_tokens: int,
    token_counter: TokenCounter,
    min_tokens: int,
    max_tokens: int,
    max_increase_ratio: float,
    max_increase_tokens: int,
) -> tuple[bool, str]:
    """Validate prompt shape before evaluation.

    Args:
        prompt: Candidate prompt to validate.
        original_tokens: Token count for the original prompt.
        token_counter: Token counter implementation.
        min_tokens: Minimum token threshold.
        max_tokens: Maximum token threshold.
        max_increase_ratio: Max allowed ratio vs original tokens.
        max_increase_tokens: Max allowed absolute increase in tokens.

    Returns:
        tuple[bool, str]: (is_valid, reason).
    """

    tokens = token_counter.count(prompt)
    if tokens < min_tokens:
        return False, "too_short"
    if tokens > max_tokens:
        return False, "too_long"
    if original_tokens > 0:
        max_by_ratio = int(original_tokens * max_increase_ratio)
        max_by_abs = original_tokens + max_increase_tokens
        allowed_max = max(max_by_ratio, max_by_abs)
        if tokens > allowed_max:
            return False, "too_large_increase"
    return True, "ok"


def _allowed_prompt_tokens(
    *,
    original_tokens: int,
    max_prompt_tokens: int,
    max_increase_ratio: float,
    max_increase_tokens: int,
) -> int:
    """Compute the maximum allowed token count for an improved prompt.

    Args:
        original_tokens: Token count for the original prompt.
        max_prompt_tokens: Absolute maximum token threshold.
        max_increase_ratio: Max allowed ratio vs original tokens.
        max_increase_tokens: Max allowed absolute increase in tokens.

    Returns:
        int: Maximum allowed token count.
    """

    if original_tokens <= 0:
        return max_prompt_tokens
    max_by_ratio = int(original_tokens * max_increase_ratio)
    max_by_abs = original_tokens + max_increase_tokens
    return min(max_prompt_tokens, max(max_by_ratio, max_by_abs))
