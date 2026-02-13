# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Multi-agent Python framework that uses LLMs to collaboratively design, implement, and test Python applications in Docker sandboxes. Three specialized agents (Architect, Software Engineer, Test Engineer) work in an iterative feedback loop until tests pass a configurable threshold.

## Setup and Commands

```bash
# Install dependencies
uv sync

# Activate virtual environment
source .venv/bin/activate

# Run the main agent flow (interactive mode)
python agents_from_scratch_docker.py

# Run with specific model and problem file
python agents_from_scratch_docker.py --description-file problem.txt --model anthropic/claude-opus-4-6

# All CLI options
python agents_from_scratch_docker.py \
  --description-file <path> \
  --model <provider/model> \
  --max-iterations <n>       # default: 2 \
  --pass-threshold <percent> # default: 90.0 \
  --max-tokens <n>           # default: 64000 \
  --output-dir <path>        # default: src \
  --search <true|false>      # default: true (Tavily web search for package validation)

# Code metrics
./metrics.sh <directory> <coverage> <ux_score> <csv_file>
```

**Required environment**: Docker, Python 3.12+, `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`. Optional: `TAVILY_API_KEY` for web search package validation.

## Architecture

### Source Files

- **`agents_from_scratch_docker.py`** — Main entry point and all agent logic (~1500 lines). Contains `Agent` base class, `ArchitectAgent`, `SoftwareEngineerAgent`, `TestEngineerAgent`, and `AgenticFlow` orchestrator.
- **`docker_test.py`** — `PythonPackageAnalyzer` class. Extracts imports via AST, generates Dockerfiles, builds/runs Docker containers for testing.
- **`pypi_validator.py`** — `PyPIValidator` and `TavilySearchHelper`. Three-tier package validation: PyPI lookup, known aliases, web search with LLM parsing. Persistent cache at `~/.pypi_validator_cache.json`.

### Agent Flow

```
Problem Description → Architect Agent → Architecture Plan
  → Software Engineer Agent → Implementation
  → Test Engineer Agent → Docker sandbox test execution
  → [pass threshold met?] YES → done / NO → feedback loop back to Engineer
```

### Key Patterns

- **LLM calls**: All go through `litellm.completion()` via `build_completion_params()`, which handles model-specific parameter differences (GPT-5/5.2/5.2-codex use `max_completion_tokens` and no temperature; Claude and GPT-4.x use `max_tokens` + `temperature`). Supported models include `openai/gpt-5.2`, `openai/gpt-5.2-codex`, and `anthropic/claude-opus-4-6`.
- **Streaming**: All LLM responses are streamed with chunk iteration and retry logic (3 retries, exponential backoff).
- **Conversation state**: Each agent maintains a `history` list of `{"role", "content"}` dicts.
- **Code extraction**: Regex-based extraction of Python code from markdown code blocks in LLM responses.
- **Test execution**: Tests run via pytest inside Docker containers (Python 3.12-slim base). Results parsed from pytest summary output.
- **Results storage**: Full run artifacts saved to `src/agentic_flow_results.json` (architecture plan, implementation history, test reports).

### Output Directories

Generated projects go into directories like `calculator_dir/`, `weather_api/`, `src/`, each containing `main.py`, `test_main.py`, `conftest.py`, `Dockerfile`, `requirements.txt`, and `agentic_flow_results.json`. The `src/` directory is gitignored as it's the default working output.
