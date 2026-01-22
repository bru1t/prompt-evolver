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
- Response must conform to the evaluation JSON schema (see below).

Template variables:
- {task_type}
- {output}
- {expected_output}
- {format_requirements}

**Evaluation Response Schema:**

The evaluator LLM must return a JSON object with the following structure:

```json
{
  "pass": boolean,
  "score": number (0.0 to 1.0),
  "issues": [string, ...],
  "suggestions": [string, ...]
}
```

**Field Descriptions:**

- `pass` (boolean, required): Whether the output meets the requirements
  - `true` if the output is acceptable
  - `false` if the output has issues

- `score` (number, required): Numerical quality score
  - Must be between 0.0 and 1.0 (inclusive)
  - 0.0 = completely incorrect
  - 1.0 = perfect output
  - Values outside this range will be clamped

- `issues` (array of strings, required): List of problems found
  - Each issue should be a clear, concise description
  - Can be empty array if `pass` is `true`
  - If `pass` is `false` and array is empty, "unspecified_issues" is added

- `suggestions` (array of strings, required): Improvement recommendations
  - Each suggestion should be actionable guidance
  - Can be empty array if no suggestions
  - Should NOT include expected output or test data

**Schema Validation:**

The pipeline validates evaluator responses against this schema using the `jsonschema` library. If validation fails:
- A warning is logged with the validation error details
- The response is treated as a failed evaluation
- Issues will include "schema_validation_error" with the specific error message

**Example Valid Response:**

```json
{
  "pass": false,
  "score": 0.6,
  "issues": [
    "Output is missing required field 'timestamp'",
    "Date format does not match ISO 8601"
  ],
  "suggestions": [
    "Add a timestamp field with ISO 8601 format",
    "Use YYYY-MM-DD format for dates"
  ]
}
```

**Example Invalid Responses:**

```json
// Missing required field "issues"
{"pass": true, "score": 1.0, "suggestions": []}

// Score out of range
{"pass": true, "score": 1.5, "issues": [], "suggestions": []}

// Wrong type for "pass"
{"pass": "yes", "score": 0.8, "issues": [], "suggestions": []}
```

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
