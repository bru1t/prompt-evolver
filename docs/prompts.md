# Prompt Templates

Prompt templates live in configs/prompts/.

prompt.execute.md:
- Builds the model prompt from task_type, prompt, and text.

Template variables:
- {prompt}
- {task_type}
- {text}
- {format_requirements}

prompt.evaluation.md:
- Evaluates outputs against expected results.
- Must return strict JSON for parsing.

Template variables:
- {task_type}
- {output}
- {expected_output}
- {format_requirements}

prompt.improve.md:
- Uses evaluator feedback to rewrite the prompt.
- Must not include task text or examples (anti-leakage).
- Must respect the token budget placeholders.

Template variables:
- {prompt}
- {task_type}
- {format_requirements}
- {issues}
- {suggestions}
- {original_tokens}
- {max_allowed_tokens}

Do:
- Keep prompts generalizable with placeholders like {text}.
- Follow format requirements.
- Use evaluator issues and suggestions only.

Don't:
- Copy any task input text or expected output.
- Include real names, values, or test fixtures.
