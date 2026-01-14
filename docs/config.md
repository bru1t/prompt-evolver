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
  mode: lmstudio
  model: mistralai/ministral-3-3b
  api_url: http://127.0.0.1:1234
llm_improvement:
  mode: lmstudio
  model: mistralai/ministral-3-3b
  api_url: http://127.0.0.1:1234
llm_evaluation:
  mode: lmstudio
  model: mistralai/ministral-3-3b
  api_url: http://127.0.0.1:1234
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
- llm_execution: model used to run tasks
- llm_improvement: model used to rewrite prompts
- llm_evaluation: model used to evaluate outputs

Each LLM config can set:
- mode: local, lmstudio, openai_compatible
- model: model identifier
- api_url: base URL
- api_key_env: environment variable name for API key
- timeout_seconds: HTTP timeout

Troubleshooting:
- Timeouts: increase `timeout_seconds` or use a faster local model.
- Bad URLs: confirm the API base URL includes the correct port.
- Missing API key: ensure `api_key_env` is set and exported in your shell.
