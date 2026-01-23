# Configuration

The main configuration lives in configs/config.yaml.

Key fields:
- max_generations: maximum improvement attempts
- min_improvement_attempts: minimum attempts before early stop
- max_no_improve: stop after N non-improving attempts
- similarity_weight, length_weight: scoring weights
- leakage_similarity_threshold, leakage_ngram_size, leakage_ngram_overlap_threshold
- min_prompt_tokens, max_prompt_tokens
- max_prompt_increase_ratio, max_prompt_increase_tokens
- min_improvement_attempts

Minimal config example:
```yaml
max_generations: 3
min_improvement_attempts: 2
similarity_weight: 0.8
length_weight: 0.2
llm_execution:
  mode: lmstudio  # or: openai_compatible, echo, local
  api_url: http://127.0.0.1:1234
  # Model name specified in run_pipeline() call
llm_improvement:
  mode: lmstudio
  api_url: http://127.0.0.1:1234
llm_evaluation:
  mode: lmstudio
  api_url: http://127.0.0.1:1234
```

**Model Configuration:**

Model names are NOT specified in the config file. Instead, pass them as parameters when calling `run_pipeline()`:

```python
run_pipeline(
    ...,
    config=config,
    execution_model="mistralai/ministral-3-3b",  # For Ollama/LM Studio
    improvement_model="mistralai/ministral-3-3b",
    evaluation_model="mistralai/ministral-3-3b",
)
```

For OpenAI API, use model names like:
```python
run_pipeline(
    ...,
    execution_model="gpt-4",
    improvement_model="gpt-4",
    evaluation_model="gpt-3.5-turbo",
)
```

Presets:

Fast iteration:
```yaml
max_generations: 2
min_improvement_attempts: 2
max_no_improve: 1
```

Strict leakage:
```yaml
leakage_similarity_threshold: 0.35
leakage_ngram_size: 3
leakage_ngram_overlap_threshold: 0.05
```

Token savings emphasis:
```yaml
max_prompt_increase_ratio: 1.5
max_prompt_increase_tokens: 20
```

LLM configuration:
- llm_execution: backend settings for task execution
- llm_improvement: backend settings for prompt rewriting
- llm_evaluation: backend settings for output evaluation

Each LLM config can set:
- mode: local, lmstudio, openai_compatible (required)
- api_url: base URL for API endpoint (required for lmstudio/openai_compatible)
- api_key_env: environment variable name for API key (optional, for OpenAI)
- timeout_seconds: HTTP timeout (default: 30.0)
- max_retries: maximum retry attempts for transient errors (default: 3)
- base_delay_seconds: initial delay before first retry (default: 1.0)
- max_delay_seconds: maximum delay between retries (default: 30.0)

**Note:** Model names are specified when calling `run_pipeline()`, not in the config file.

**Retry Logic:**

The pipeline includes automatic retry logic with exponential backoff for transient API failures:

- **Retryable Errors**: HTTP 429 (rate limit), 500 (server error), 503 (service unavailable)
- **Non-Retryable Errors**: HTTP 400 (bad request), 401 (unauthorized), 404 (not found)
- **Backoff Strategy**: Exponential with configurable base and maximum delays

Example retry configuration:
```yaml
llm_execution:
  mode: openai_compatible
  model: gpt-4
  api_url: https://api.openai.com/v1
  max_retries: 5           # Retry up to 5 times
  base_delay_seconds: 2.0  # Start with 2 second delay
  max_delay_seconds: 60.0  # Cap delay at 60 seconds
```

**Retry Delay Calculation:**
- Attempt 1: 2.0s
- Attempt 2: 4.0s
- Attempt 3: 8.0s
- Attempt 4: 16.0s
- Attempt 5: 32.0s
- Attempt 6+: 60.0s (capped)

Retry attempts and delays are logged for debugging.

Troubleshooting:
- Timeouts: increase `timeout_seconds` or use a faster local model.
- Rate limits: increase `max_retries` and `max_delay_seconds`.
- Bad URLs: confirm the API base URL includes the correct port.
- Missing API key: ensure `api_key_env` is set and exported in your shell.
