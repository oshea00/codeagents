# Agents from scratch

This is an experiment using Claude Code to design and create an "agentic framework" using only barebones dependencies and raw openai (or openai-compatible) API calls.

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/oshea00/codeagents)

# Example of creating a calculator

The description given to the program for the calculator program:
'Create a cli scientific calculator using sympy that runs in a REPL loop and also accepts an expression from the command line'

The resulting code is in the ```calculator_dir```

## Build and run example
```bash
cd calculator_dir
docker build -t calc .
docker run -it calc python main.py
```

# Example of creating a weather API

The description given to the program for the api program is in ```problem.txt```
The resulting code is in the ```weather_api```

## Build and run Example
```bash
cd weather_api
docker build -t weather .
docker run -it -p8000:8000 weather uvicorn main:app --host 0.0.0.0 --port 8000
```
Open browser at http://localhost:8000/docs

# Instructions for use

## Prerequisites
- Docker
- OPENAI_API_KEY or ANTHROPIC_API_KEY (depending on provider)
- Python environment with `litellm` package
- `uv` utility

## Running the script

From the root of the project directory:
```bash
uv sync
source .venv/bin/activate
python agents_from_scratch_docker.py [OPTIONS]
```

## Command Line Options

- `--description-file <path>` - Path to a file containing the problem description
- `--max-iterations <n>` - Maximum number of test-fix iterations (default: 2)
- `--pass-threshold <percent>` - Minimum percentage of tests that must pass to succeed (default: 90.0)
- `--model <model>` - Model name to use (e.g., `openai/gpt-5-mini`, `anthropic/claude-sonnet-4-5-20250929`). If not provided, you'll be prompted to choose a provider.
- `--max-tokens <n>` - Maximum tokens to generate (default: 64000)

## Examples

### Interactive mode with default settings:
```bash
python agents_from_scratch_docker.py
```

### Specify model directly:
```bash
python agents_from_scratch_docker.py --model openai/gpt-5-mini
```

### Use a problem description file with custom settings:
```bash
python agents_from_scratch_docker.py \
  --description-file problem.txt \
  --model anthropic/claude-sonnet-4-5-20250929 \
  --max-iterations 3 \
  --pass-threshold 95.0 \
  --max-tokens 32000
```

## Supported Models

The system uses `litellm` to support multiple LLM providers:

### OpenAI models:
- `openai/gpt-4.1` - Supports `temperature` and `max_tokens`
- `openai/gpt-4o` - Supports `temperature` and `max_tokens`
- `openai/gpt-5-mini` - Uses `max_completion_tokens`, no `temperature` support

### Anthropic models:
- `anthropic/claude-sonnet-4-5-20250929` - Supports `temperature` and `max_tokens`

The system automatically handles parameter differences between models.

