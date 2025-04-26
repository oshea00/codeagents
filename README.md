# Agents from scratch

This is an experiment using Claude 3.7 to design and create an "agentic framework" using only barebones dependencies and raw openai (or openai-compatible) API calls.

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
- OPENAI_API_KEY
- Python environment has ```openai``` package.
- ```uv``` utility.

How to run the script (from root of project directory):
```
uv sync
source .venv/bin/activate
python agents_from_scratch_docker.py
```

Answer the prompts.

