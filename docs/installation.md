# Manual Installation

This guide covers manual installation for developers who prefer to set up the environment step by step.

> **Note:** For most users, the [automated setup script](../README.md#easy-setup-recommended) is recommended.

## Prerequisites

- Python 3.10+
- (Optional) VS Code with Python and Jupyter extensions

## Steps

### 1. Clone the repository

```bash
git clone https://github.com/bru1t/prompt-evolver.git
cd prompt-evolver
```

### 2. Create virtual environment

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS/Linux:**
```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -U pip
pip install -r requirements.txt
```

### 4. Configure your LLM connection

Edit `configs/config.yaml` to configure your API backend:

```yaml
llm_execution:
  mode: lmstudio  # or: openai_compatible, echo, local
  api_url: http://127.0.0.1:1234
  timeout_seconds: 30
```

**Model names** are specified in the notebook, not in the config file. This makes it easy to switch models without editing YAML.

Open `notebooks/PromptEvolver.ipynb` and set your model names:

```python
# For Ollama/LM Studio
EXECUTION_MODEL = "mistralai/ministral-3-3b"
IMPROVEMENT_MODEL = "mistralai/ministral-3-3b"
EVALUATION_MODEL = "mistralai/ministral-3-3b"

# For OpenAI API (set api_key_env in config.yaml)
# EXECUTION_MODEL = "gpt-4"
# IMPROVEMENT_MODEL = "gpt-4"
# EVALUATION_MODEL = "gpt-3.5-turbo"
```

## Running Your First Optimization

Open VS Code and the Jupyter notebook:

```bash
code .
# Open: notebooks/PromptEvolver.ipynb
```

Run cells from top to bottom. To try it quickly, use the included examples:
- `data/example.prompts.csv`
- `data/example.texts.csv`
- `data/example.tasks.csv`

## Troubleshooting

If you encounter issues:
1. Ensure your virtual environment is activated
2. Check that your LLM backend is running (LM Studio, Ollama, etc.)
3. Verify `configs/config.yaml` has the correct API URL
4. See [Config documentation](config.md) for detailed configuration options
