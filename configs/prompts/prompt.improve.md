You are improving a prompt for a {task_type} task.

Original prompt:
{prompt}

Format requirements:
{format_requirements}

Evaluator issues:
{issues}

Evaluator suggestions:
{suggestions}

Rewrite the prompt to be shorter and clearer while preserving intent and matching the expected output.
Keep the improved prompt <= {max_allowed_tokens} tokens (original: {original_tokens}).
Do NOT include any task input text or example content. Keep it generalizable and use placeholders like {{text}}.
Return only the improved prompt and nothing else.
