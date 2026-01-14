# Pipeline Details

The pipeline processes tasks sequentially and writes results to CSV.

Inputs:
- Prompts CSV: id, prompt, tokens (optional)
- Texts CSV: id, text, tokens (optional)
- Tasks CSV: id, id_text, id_prompt, task_type, expected_output, format_requirements (optional)

Execution loop per task:
1) Run task prompt against the test text with the execution model.
2) Evaluate output using the evaluation prompt template.
3) If evaluation fails, improve the prompt using evaluator feedback only.
4) Reject candidates that fail leakage or sanity checks.
5) Repeat until pass, max iterations, or no improvement threshold is reached.

Pseudocode:
```
for task in tasks:
    output = execute(prompt, text)
    feedback = evaluate(output, expected)
    best = prompt if feedback.pass else None
    attempts = 0
    while attempts < max_generations:
        attempts += 1
        candidate = improve(prompt, feedback)
        if not sanity(candidate) or leakage(candidate, text):
            continue
        output = execute(candidate, text)
        feedback = evaluate(output, expected)
        if feedback.pass and better(candidate, best):
            best = candidate
        if feedback.pass and attempts >= min_improvement_attempts:
            break
```

Acceptance policy:
- A candidate prompt must pass evaluation to be accepted.
- If multiple candidates pass, prefer higher evaluation score, then lower tokens.
- If nothing passes, the original prompt remains.

What "pass" means:
- The evaluator returns JSON with `pass: true`.
- The evaluator score is higher than failing alternatives.
- The candidate does not violate leakage or sanity checks.

Stop conditions:
- Pass achieved after at least min_improvement_attempts.
- Reached max iterations.
- No improvement after max_no_improve attempts.
- Sanity or leakage failures after min_improvement_attempts.

Why tasks are sequential:
- Tasks are evaluated deterministically to keep results reproducible.
- This makes CSV diffs meaningful for audits and review.
- If you need scale, parallelize at the task level with a job queue.
