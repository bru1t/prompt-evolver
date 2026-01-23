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

### Input & Output

**Input** (CSV format, JSON/TSV planned)
- `prompts.csv` — prompts to optimize
- `texts.csv` — test inputs
- `tasks.csv` — prompt + text + expected output

**Output**
- `results.csv` — optimized prompts, pass/fail scores, token deltas

### Features

- **CSV in / CSV out** — prompts + texts + tasks → optimized results
- **LLM-agnostic** — works with local models (LM Studio, Ollama) or OpenAI-compatible APIs
- **Token-aware** — prefers shorter prompts when quality is comparable
- **Leakage detection** — rejects prompts that copy test data
- **Iterative improvement** — evaluates, rewrites, and re-evaluates until pass or max iterations
- **Structured logging** — full traceability of each optimization step

## Quickstart

### Prerequisites
- Python 3.10+
- (Optional) VS Code with Python and Jupyter extensions

### Easy Setup (Recommended)

The easiest way to get started is using the automated setup script:

#### Unix/macOS/Linux:
```bash
git clone https://github.com/bru1t/prompt-evolver.git
cd prompt-evolver
./setup.sh
```

#### Windows:
```bash
git clone https://github.com/bru1t/prompt-evolver.git
cd prompt-evolver
setup.bat
```

The setup script will:
- Check your Python installation
- Create and activate a virtual environment
- Install all dependencies
- Guide you through LLM backend configuration
- Validate your setup with a test

### VS Code Setup (Alternative)

1. Clone and open in VS Code:
   ```bash
   git clone https://github.com/bru1t/prompt-evolver.git
   code prompt-evolver
   ```

2. VS Code will detect `requirements.txt` and prompt to create a virtual environment — click **Yes**

3. Open `notebooks/PromptEvolver.ipynb` and select the `.venv` kernel when prompted

For detailed manual setup, see [Installation Guide](docs/installation.md).

### Running Your First Optimization

1. Open the notebook `notebooks/PromptEvolver.ipynb`

2. Select the `.venv` Python interpreter as your Jupyter kernel:
   - Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on macOS)
   - Type "Python: Select Interpreter"
   - Choose the `.venv` environment

3. Run cells from top to bottom

Example data is included for quick testing:
- `data/example.prompts.csv`
- `data/example.texts.csv`
- `data/example.tasks.csv`

## Documentation

- [Overview](docs/overview.md) — what the project does and the core flow
- [Pipeline](docs/pipeline.md) — loop details, acceptance policy, stop conditions
- [Data model](docs/data-model.md) — input CSV schemas + results format
- [Config](docs/config.md) — configuration reference for `configs/config.yaml`
- [Prompts](docs/prompts.md) — prompt templates in `configs/prompts/`
- [Installation](docs/installation.md) — manual setup guide for developers

## License

[MIT](LICENSE)