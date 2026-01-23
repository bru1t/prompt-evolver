# Future Implementation Plans

This document outlines planned improvements for **Prompt Evolver** (CSV-in / CSV-out prompt optimization for LLM workflows).

## Distribution & Packaging

### Python Packaging & PyPI Distribution
**What needs to be done**: Ship Prompt Evolver as a standard installable Python package with an official CLI entry point.

- [ ] Add `pyproject.toml` (build system, metadata, dependencies, optional extras)
- [ ] Move/import package code under `src/prompt_evolver/` (src-layout)
- [ ] Add `prompt_evolver/__version__.py` and single-source versioning
- [ ] Add console script entry point: `prompt-evolver`
- [ ] Ensure `pip install -e .` works for development
- [ ] Include LICENSE metadata and package classifiers

**Expected result**: Users can install Prompt Evolver via pip (editable or normal), run `prompt-evolver` from terminal, and the project is ready for publishing to PyPI.

## Performance & Scalability

### Parallel Task Execution (Threaded)
**What needs to be done**: Speed up pipeline runtime by running independent tasks concurrently (safe + deterministic).

- [ ] Add `max_workers` to configuration (e.g. `EvolverConfig`)
- [ ] Implement `ThreadPoolExecutor` in `pipeline.py` for task-level concurrency
- [ ] Ensure deterministic ordering in final outputs (stable sort by `task_id`)
- [ ] Add thread-safe logging that includes `task_id` context
- [ ] Add config default like `max_workers: 4`

**Expected result**: Pipeline executes significantly faster for I/O-bound LLM calls while producing consistent, reproducible results.

### Async LLM Execution (HTTP)
**What needs to be done**: Improve concurrency efficiency by supporting async LLM calls (better than threads for HTTP workloads).

- [ ] Implement `AsyncOpenAICompatibleClient` (e.g. using `aiohttp`)
- [ ] Add async pipeline methods (e.g. `run_async()` / `evolve_task_async()`)
- [ ] Use `asyncio.gather()` to execute tasks concurrently with controlled limits
- [ ] Add optional dependency group (e.g. `prompt-evolver[async]`)
- [ ] Benchmark async vs threads and document trade-offs

**Expected result**: Faster total execution time and better scalability when running many tasks against remote LLM endpoints.

### LLM Response Caching
**What needs to be done**: Avoid redundant LLM calls by caching identical requests during development/testing runs.

- [ ] Cache key based on `(model, prompt_template, prompt, input_text, config_hash)`
- [ ] Support `memory` cache (dict/LRU) and `disk` cache (SQLite)
- [ ] Add TTL + max size configuration and eviction strategy
- [ ] Log cache hit/miss stats and cache effectiveness summary
- [ ] Add invalidation strategy when model/config changes

**Expected result**: Repeated runs become much faster and cheaper, especially when iterating on scoring thresholds or evaluation logic.

### Rate Limiting & Backpressure
**What needs to be done**: Respect provider limits and prevent request bursts that trigger throttling.

- [ ] Add rate limiter (token bucket or leaky bucket)
- [ ] Configure limits for requests/minute and tokens/minute
- [ ] Apply limiter in the LLM client layer (single enforcement point)
- [ ] Add logging for rate-limit waits and throttling events

**Expected result**: Stable operation under real API constraints with fewer 429 errors and smoother throughput.

### Performance Benchmarks & Regression Tracking
**What needs to be done**: Measure performance consistently and prevent slowdowns from creeping in.

- [ ] Add benchmark suite for 10 / 100 / 1000 tasks
- [ ] Track latency per stage (execution, evaluation, improvement)
- [ ] Track memory usage for large runs
- [ ] Add optional CI threshold checks (e.g. fail if >10% slower)

**Expected result**: Performance changes become visible and measurable, and regressions are caught early.

## Reliability & Data Integrity

### Input Validation & User-Friendly Errors (Completed)
**What needs to be done**: Make common input problems understandable and actionable.

- [x] Improve CSV validation errors with available vs missing columns
- [x] Improve foreign key validation errors with missing ID lists
- [x] Add progress tracking feedback during pipeline execution

**Expected result**: Users see exactly what to fix in their inputs and can recover quickly without guessing.

### Input Validation Enhancements (Completed)
**What needs to be done**: Make CSV ingestion stricter and safer to prevent subtle data errors from producing misleading results.

- [x] Validate uniqueness of IDs (`prompts`, `texts`, `tasks`)
- [x] Enforce ID dtype normalization (always string; reject floats like `1.0`)
- [x] Validate `task_type` against a config-driven whitelist
- [x] Normalize NaN/whitespace in required fields (strip + reject empty)
- [x] Keep errors actionable with file + row context

**Expected result**: Bad inputs fail fast with clear feedback, preventing corrupted results and hard-to-debug behavior later in the pipeline.

### Additional Input Formats (JSON, TSV, Excel)
**What needs to be done**: Support additional data file formats beyond CSV to make the tool more flexible for different workflows.

- [ ] Add TSV input support (tab-separated values, simple delimiter change)
- [ ] Add JSON input support (array of objects format)
- [ ] Add JSONL input support (JSON Lines, one object per line)
- [ ] Add Excel/XLSX input support using `openpyxl`
- [ ] Implement format auto-detection based on file extension
- [ ] Add `--input-format` CLI flag to override auto-detection
- [ ] Update `io.py` with unified `load_data()` function that handles all formats
- [ ] Document supported formats and their schemas in `docs/data-model.md`

**Expected result**: Users can provide input data in their preferred format (CSV, TSV, JSON, JSONL, Excel) without manual conversion.

### Additional Output Formats (JSON, Excel)
**What needs to be done**: Support additional output formats beyond CSV for results export.

- [ ] Add JSON output support (structured results as JSON array)
- [ ] Add JSONL output support (one result per line)
- [ ] Add Excel/XLSX output support with formatting (color-coded improvements)
- [ ] Add `--output-format` CLI flag (csv, json, jsonl, xlsx)
- [ ] Include metadata in JSON outputs (config used, timestamps, versions)
- [ ] Ensure backward compatibility with existing CSV output as default

**Expected result**: Users can export results in format best suited for their downstream tools (spreadsheets, databases, APIs).

### Single-Entry API (Programmatic Use)
**What needs to be done**: Provide a simple function for optimizing a single prompt programmatically without CSV files.

- [ ] Add `optimize_prompt(prompt, text, expected_output, config)` wrapper function
- [ ] Return structured result object (not CSV row)
- [ ] Support both sync and async variants
- [ ] Add examples to documentation showing programmatic usage
- [ ] Ensure same evaluation and improvement logic as batch pipeline

**Expected result**: Developers can integrate prompt optimization into their code without managing CSV files, enabling use in APIs, scripts, and notebooks.

### Progress Persistence (Checkpoint & Resume)
**What needs to be done**: Make runs recoverable and resumable after failures or interruptions.

- [ ] Add `--checkpoint` option to save results after each task (JSONL)
- [ ] Add `--resume` option to continue from checkpoint
- [ ] Skip tasks already completed in checkpoint
- [ ] Ensure checkpoints are versioned and schema-safe

**Expected result**: Users can interrupt long runs safely and resume without rerunning completed work.

## Quality of Evaluation

### Evaluator JSON Schema Validation (Completed)
**What needs to be done**: Guarantee evaluator responses are machine-readable and safe to parse.

- [x] Validate evaluator output against JSON Schema
- [x] Provide clear error messages for schema violations
- [x] Document schema expectations in `docs/prompts.md`

**Expected result**: Evaluator output is predictable and pipeline behavior becomes more robust and debuggable.

### Semantic Similarity Backend (Embeddings)
**What needs to be done**: Replace character-based similarity with semantic embeddings for more accurate leakage detection.

- [ ] Introduce `SimilarityBackend` interface/protocol
- [ ] Keep current `SequenceMatcher` backend as default fallback
- [ ] Add embedding backend (e.g. sentence-transformers)
- [ ] Add config switch (e.g. `similarity_backend: embedding|sequence_matcher`)
- [ ] Add optional dependency group (e.g. `prompt-evolver[ml]`)

**Expected result**: Leakage detection and similarity scoring reflect meaning (not just surface text), improving candidate filtering accuracy.

### Proper Token Counting (Model-Aware)
**What needs to be done**: Track tokens accurately using real tokenizers per model family.

- [ ] Add `tiktoken` counter for OpenAI-compatible models
- [ ] Add HuggingFace tokenizer counter for open-source models
- [ ] Make tokenizer selection configurable (`tiktoken|huggingface|regex`)
- [ ] Cache tokenizer instances to avoid repeated loading overhead
- [ ] Add optional dependencies for tokenizers

**Expected result**: Token usage becomes accurate and consistent, enabling trustworthy cost estimation and token-aware optimization.

### Alternative Evaluation Metrics (Task-Specific)
**What needs to be done**: Support metrics beyond similarity for more correct scoring per task type.

- [ ] Add `Metric` interface/protocol and metric registry
- [ ] Implement `ExactMatch`, `BLEU`, `ROUGE` (as optional extras)
- [ ] Map `task_type -> metric` via config
- [ ] Record metric-specific outputs in results CSV

**Expected result**: Scoring matches real task intent (classification vs generation), producing more meaningful “pass/fail” decisions.

## Observability & Operations

### LLM Retry Logic (Completed)
**What needs to be done**: Make LLM calls resilient to transient failures and provider instability.

- [x] Implement exponential backoff retry wrapper
- [x] Handle common HTTP error classes (429 / 5xx)
- [x] Make retry behavior configurable and well-logged

**Expected result**: Runs fail less often due to transient API/network errors and recover automatically when safe.

### Structured Logging (JSON + Traceability)
**What needs to be done**: Improve debugging and monitoring by using structured logs with consistent fields.

- [ ] Add logging configuration module (text + JSON formatter)
- [ ] Include `trace_id`, `task_id`, `iteration`, and model identifiers
- [ ] Track per-call latency + token usage metadata in logs
- [ ] Configure log level and format via YAML config

**Expected result**: Logs become machine-readable and easier to search, enabling faster debugging and better monitoring in real runs.

### Usage Tracking & Cost Estimation
**What needs to be done**: Provide visibility into token usage and approximate cost for budgeting.

- [ ] Track input/output tokens per call and per task
- [ ] Add configurable cost rates per model family
- [ ] Export usage summary to results CSV
- [ ] Add `--budget` option to stop when cost exceeds threshold
- [ ] Log cumulative cost and token totals during execution

**Expected result**: Users can estimate cost, compare optimization strategies, and prevent accidental overspending.

## User Experience & Interfaces

### Automated Setup Script (Completed)
**What needs to be done**: Provide guided installation and initial configuration without manual setup steps.

- [x] Add `setup.sh` (Unix/macOS/Linux) and `setup.bat` (Windows)
- [x] Create venv, install deps, guide backend config, validate run
- [x] Generate starter `config.yaml` and sample assets

**Expected result**: New users can run a single script to install and validate Prompt Evolver with minimal friction.

### Web UI (Streamlit)
**What needs to be done**: Provide a simple GUI workflow for users who don’t want to use notebooks or edit YAML.

- [ ] Add Streamlit app (upload CSVs + run pipeline)
- [ ] Add config editor UI (no manual YAML required)
- [ ] Add live progress reporting and task status display
- [ ] Add before/after prompt comparison view
- [ ] Add one-click download for results CSV

**Expected result**: Prompt Evolver becomes usable for non-developers and faster to demo, with clear visibility into improvements and outcomes.

### CLI Configuration Wizard
**What needs to be done**: Guide users through setup interactively to reduce configuration errors.

- [ ] Add `prompt-evolver init` wizard command
- [ ] Prompt for backend mode + base URL + timeouts + env keys
- [ ] Validate backend connectivity before writing config
- [ ] Generate sample CSV templates for quick onboarding
- [ ] Add presets (local, openai-compatible, ollama/lmstudio)

**Expected result**: First-time setup becomes smooth and reliable, with fewer YAML mistakes and faster time-to-first-run.

### Interactive Jupyter Widgets
**What needs to be done**: Enhance existing Jupyter notebook with interactive widgets for users who prefer notebook environment.

- [ ] Add `ipywidgets` dependency
- [ ] Replace hard-coded model name cells with dropdown widget
- [ ] Add file upload widget for CSVs (no path editing required)
- [ ] Add progress bar widget (visual, not text-based)
- [ ] Add interactive before/after comparison with slider
- [ ] Add "Export to PDF" button widget
- [ ] Add configuration preset dropdown

**Expected result**: Jupyter notebook becomes interactive and requires minimal code editing, making it more accessible to non-programmers.

### One-Click Browser Launch Script
**What needs to be done**: Add simple script that launches the web UI (when implemented) with one command or double-click.

- [ ] Create `start_web_ui.sh` (Unix/macOS/Linux)
- [ ] Create `start_web_ui.bat` (Windows)
- [ ] Detect if virtual environment exists and activate it
- [ ] Launch Streamlit app on port 8501
- [ ] Open browser automatically to `http://localhost:8501`
- [ ] Add error handling for common issues (port in use, missing deps)

**Expected result**: Users can launch the web UI with one double-click without terminal commands.

### Model Presets & Quick Configuration
**What needs to be done**: Provide one-click configuration presets for popular LLM providers.

- [ ] Create preset configurations for common providers (OpenAI, Anthropic, Local LLMs, HuggingFace)
- [ ] Add preset selector in web UI and notebook
- [ ] Auto-fill API URL, model name, and recommended settings
- [ ] Validate API connectivity before saving preset
- [ ] Allow users to save custom presets
- [ ] Include presets for: OpenAI GPT-4/3.5, Local LM Studio, Ollama

**Expected result**: Users can switch between LLM providers in seconds without editing config files.

### CSV Template Generator
**What needs to be done**: Provide tool to generate blank CSV templates with correct columns.

- [ ] Add "Generate CSV Templates" feature in web UI
- [ ] Create ZIP file containing templates for prompts, texts, and tasks CSVs
- [ ] Include column headers and example rows
- [ ] Add README with instructions on filling out CSVs
- [ ] Add data validation rules in Excel-compatible format

**Expected result**: Users can generate correct CSV templates without reading documentation.

### Visual Before/After Prompt Comparison (Terminal)
**What needs to be done**: Create visual diff view in terminal/CLI showing original vs improved prompts with highlighting.

- [ ] Add terminal-based diff output using color codes
- [ ] Highlight changes (additions in green, deletions in red)
- [ ] Show token count changes prominently
- [ ] Show evaluation score changes (original vs improved)
- [ ] Add option to export comparison as HTML or markdown
- [ ] Include in pipeline output as optional flag `--show-diff`

**Expected result**: Users can visually see what changed in prompts directly in terminal without opening CSV files.

### Results Dashboard with Charts (Terminal)
**What needs to be done**: Create visual dashboard in terminal showing optimization metrics and trends.

- [ ] Add ASCII/Unicode charts for token savings per task
- [ ] Add summary statistics display (total tasks, success rate, token savings)
- [ ] Use terminal colors and formatting for readability
- [ ] Add optional flag `--show-dashboard` to display after pipeline run
- [ ] Include top N most improved prompts summary
- [ ] Export dashboard as HTML report option

**Expected result**: Users can understand optimization results at a glance in terminal with visual charts.

### Export Results to Excel
**What needs to be done**: Allow users to export results directly as Excel file with formatting.

- [ ] Add `openpyxl` dependency for Excel export
- [ ] Create formatted Excel output with multiple sheets (summary, details, comparisons)
- [ ] Add color coding (green for improvements, red for regressions)
- [ ] Include charts embedded in Excel file
- [ ] Add `--output-format excel` CLI option
- [ ] Preserve all CSV data but with better presentation

**Expected result**: Non-technical users can work with results in familiar Excel format with visual formatting.

## Developer Experience & Maintainability

### Configuration Validation with Pydantic
**What needs to be done**: Replace manual validation with typed config models and automated constraints.

- [ ] Implement Pydantic config models (replace dataclasses where beneficial)
- [ ] Add validators for thresholds, positive ints, enums, and weight sums
- [ ] Improve error messages with path + expected types
- [ ] Add support for config overrides / inheritance patterns

**Expected result**: Configuration errors become clear, early, and hard to misuse, reducing runtime surprises and improving maintainability.

### Test Coverage Expansion
**What needs to be done**: Increase confidence by covering core behavior and edge cases.

- [ ] Add tests for config parsing and validation
- [ ] Add tests for CSV validation edge cases and type normalization
- [ ] Add tests for leakage detection thresholds and candidate rejection logic
- [ ] Track coverage target and enforce a reasonable minimum

**Expected result**: Refactors become safer, regressions are caught earlier, and new features can be added with confidence.

## Documentation & Support

### Comprehensive Troubleshooting Guide
**What needs to be done**: Reduce support burden by documenting common failures and fixes.

- [ ] Add `docs/troubleshooting.md` with common error patterns + solutions
- [ ] Include platform-specific setup issues (Windows/macOS/Linux)
- [ ] Add a lightweight FAQ to README
- [ ] Link error messages to doc sections where possible

**Expected result**: Users can self-diagnose issues quickly, reducing repetitive support and improving onboarding success rate.

## Contributing Workflow

When implementing items from this plan:
1. Mark the item as completed here
2. Add/extend tests for the new behavior
3. Update `docs/config.md` and any relevant documentation
4. Keep CSV schemas and results format backward compatible when possible
5. Prefer single-responsibility modules (client, pipeline, metrics, validation)
