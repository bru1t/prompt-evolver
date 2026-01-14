You are evaluating a model output for a {task_type} task.

Criteria:
- The output should match the expected meaning and required format.
- It must follow any format requirements.
- It should be concise and correct.

Task output:
{output}

Expected output (for comparison only; do not quote or reveal it):
{expected_output}

Format requirements:
{format_requirements}

Respond ONLY with valid JSON using this schema:
{{
  "pass": true/false,
  "score": 0.0-1.0,
  "issues": ["..."],
  "suggestions": ["..."]
}}

Do not include the expected output or any task input text in your response.
