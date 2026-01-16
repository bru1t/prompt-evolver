<p align="center">
  <br/>
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/bru1t/prompt-evolver/refs/heads/main/docs/images/prompt_evolver_logo%26text_dark.png">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/bru1t/prompt-evolver/refs/heads/main/docs/images/prompt_evolver_logo%26text_light.png">
    <img alt="Prompt Evolver logo" src="https://raw.githubusercontent.com/bru1t/prompt-evolver/refs/heads/main/docs/images/prompt_evolver_logo%26text_light.png" width="220">
  </picture>
</p>
<p align="center"><strong>Prompt optimization for LLM workflows</strong></p>

## Overview

Prompt Evolver is a lightweight, repeatable way to improve prompts for LLM pipelines.
You give it a prompt, a few test inputs, and what “good output” should look like. It runs your prompt, checks whether the output matches your expectations, and (when it fails) asks an LLM to propose a better prompt.

It’s built for people shipping LLM workflows who want measurable improvements, protection against prompts “cheating” (copying test answers), and a simple CSV-based process that’s easy to version and review.

### What you provide → What you get

**You provide**
- **Prompts** (CSV): the prompts you want to improve
- **Texts** (CSV): the inputs you’ll test against
- **Tasks** (CSV): links `prompt + text + expected_output` (your “prompt unit tests”)

**You get**
- A **results CSV** with the best prompt found, outputs, pass/fail + scores, token deltas, and reasons why candidates were rejected (e.g., leakage or token growth limits).

### How utility can help?
- Turn “prompt tweaking” into a repeatable loop with pass/fail criteria
- Catch regressions by re-running the same tasks over time
- Reduce prompt token usage when quality is equal or better
- Keep iterations safe using leakage detection and prompt growth limits

### Key Features
- **CSV in / CSV out** workflow (prompts + texts + tasks → results)
- **LLM-agnostic**: works with local models (LM Studio / Ollama) or OpenAI-compatible APIs
- **Token-aware optimization**: prefers shorter prompts when quality is comparable
- **Loop protection**: max generations + “keep best only” policy
- **Prompt templates** for execution, evaluation, and improvement
- **Structured logging** for traceability

### Key Capabilities
- Execute each `(prompt, text)` task and compare with `expected_output`
- Evaluate outputs with structured criteria and machine-readable feedback
- Rewrite prompts based on evaluator feedback (without copying the test text)
- Reject candidates that violate leakage checks or sanity limits
- Produce a results CSV with original vs improved prompts + token deltas

> **Leakage guardrail (plain language):** sometimes a “better” prompt is just one that copies the test answer
> or memorizes examples. Prompt Evolver flags and rejects prompt candidates that look too similar to the evaluation data.

## Quickstart (VS Code + Jupyter)

### Prerequisites
- Python 3.10+
- VS Code + extensions:
  - Python
  - Jupyter

### 1. Clone
```bash
git clone https://github.com/bru1t/prompt-evolver.git
cd prompt-evolver
```

### 2. Create environment

#### Windows
```bash
python -m venv .venv
.venv\Scripts\activate
```

#### macOS/Linux
```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Installing dependencies
```bash
pip install -U pip
pip install -r requirements.txt
```

### 4. Configure your LLM connection

Edit `configs/config.yaml` to point at your local or API-backed model(s):

- `llm_execution`: runs the task
- `llm_improvement`: rewrites the prompt
- `llm_evaluation`: scores output + provides feedback

Example (OpenAI-compatible API style):
```yaml
llm_execution:
  mode: openai_compatible
  model: your-model-name
  api_url: http://localhost:1234/v1   # or your provider base URL
  api_key_env: OPENAI_API_KEY
  timeout_seconds: 120
```

Then open:
- `notebooks/PromptEvolver.ipynb`

Run cells from top to bottom. To try it quickly, use the included minimal examples:
- `data/example.prompts.csv`
- `data/example.texts.csv`
- `data/example.tasks.csv`

## Documentation

- [Overview](docs/overview.md) — what the project does and the core flow
- [Pipeline](docs/pipeline.md) — loop details, acceptance policy, stop conditions
- [Data model](docs/data-model.md) — input CSV schemas + results format
- [Config](docs/config.md) — configuration reference for `configs/config.yaml`
- [Prompts](docs/prompts.md) — prompt templates in `configs/prompts/`

## License

[MIT](LICENSE)