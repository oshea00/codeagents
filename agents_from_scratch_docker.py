import os
import json
import time
import re
import warnings
from typing import List, Dict, Any, Optional
import argparse
from dotenv import load_dotenv
from docker_test import PythonPackageAnalyzer
from litellm import completion

# Suppress known litellm Pydantic serialization warnings for Responses API models
# (e.g. gpt-5.2-codex). See https://github.com/BerriAI/litellm/issues/17631
warnings.filterwarnings("ignore", message="Pydantic serializer warnings")


def build_completion_params(
    model: str,
    temperature: float = 0,
    max_tokens: int = 64000,
    stream: bool = True,
    timeout: int = 120,
    **kwargs,
) -> dict:
    """
    Build completion parameters based on the model being used.

    OpenAI models:
    - gpt-4.1, gpt-4o: support temperature and max_tokens
    - gpt-5, gpt-5.2, gpt-5.2-codex: do not support temperature,
      use max_completion_tokens instead of max_tokens (128K max output)

    Anthropic models:
    - claude-sonnet-4-5-20250929: support temperature and max_tokens
    - claude-opus-4-6: support temperature and max_tokens (128K max output)

    Note: gpt-5.2-codex uses OpenAI's Responses API; litellm >= 1.81.6
    auto-routes completion() calls to it.

    Args:
        model: The model identifier (e.g., 'openai/gpt-5.2', 'anthropic/claude-opus-4-6')
        temperature: Temperature setting (ignored for gpt-5 models)
        max_tokens: Maximum tokens to generate
        stream: Whether to stream the response
        timeout: Request timeout in seconds
        **kwargs: Additional parameters to pass through

    Returns:
        Dictionary of parameters to pass to litellm.completion()
    """
    params = {
        "model": model,
        "stream": stream,
        "timeout": timeout,
        "request_timeout": timeout,
        **kwargs,
    }

    # Check if this is a GPT-5 model (doesn't support temperature, uses max_completion_tokens)
    model_lower = model.lower()
    is_gpt5 = "gpt-5" in model_lower

    if is_gpt5:
        # GPT-5 models: no temperature, use max_completion_tokens
        params["max_completion_tokens"] = max_tokens
    else:
        # All other models (gpt-4.1, gpt-4o, claude-*): use temperature and max_tokens
        params["temperature"] = temperature
        params["max_tokens"] = max_tokens

    return params


def extract_code_from_response(response, debug_file=None):
    """Extract Python code from LLM response, preferring the last match.

    Returns (code, confidence) where confidence is one of:
    'matched_python_block', 'matched_generic_block', 'line_based_extraction', 'raw_fallback'
    """
    if debug_file:
        with open(debug_file, "w") as f:
            f.write(response)

    # Find all python-tagged code blocks, use the last one
    python_blocks = list(re.finditer(r"```python\s*([\s\S]*?)\s*```", response))
    if python_blocks:
        code = python_blocks[-1].group(1)
        print(f"✓ Matched python code block ({len(python_blocks)} found, using last)")
        return code, "matched_python_block"

    # Find all generic code blocks, filter out known non-Python languages
    non_python_langs = {
        "json", "bash", "sh", "shell", "yaml", "yml", "xml", "html",
        "css", "javascript", "js", "typescript", "ts", "sql", "toml",
        "ini", "dockerfile", "text", "txt", "markdown", "md",
    }
    generic_blocks = []
    for match in re.finditer(r"```(\w*)\s*([\s\S]*?)\s*```", response):
        lang = match.group(1).lower()
        if lang not in non_python_langs:
            generic_blocks.append(match.group(2))

    if generic_blocks:
        code = generic_blocks[-1]
        print(f"✓ Matched generic code block ({len(generic_blocks)} found, using last)")
        return code, "matched_generic_block"

    # Line-based extraction for responses starting with backticks
    if response.strip().startswith("```"):
        print("Found backticks at start, attempting line-based extraction...")
        lines = response.strip().split("\n")
        start_idx = 1 if lines[0].startswith("```") else 0
        end_idx = len(lines)
        for i in range(len(lines) - 1, 0, -1):
            if lines[i].strip() == "```":
                end_idx = i
                break
        code = "\n".join(lines[start_idx:end_idx])
        print(f"Extracted {len(code)} characters of code by line-based parsing")
        return code, "line_based_extraction"

    print("Warning: No code block found. Treating entire response as code")
    return response, "raw_fallback"


def extract_json_from_response(text):
    """Extract JSON from text that might contain other content."""
    # Try markdown-fenced JSON blocks first
    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if json_match:
        return json_match.group(1)

    # Use raw_decode to find the first structurally valid JSON object
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch == '{':
            try:
                _, end = decoder.raw_decode(text, i)
                return text[i:end]
            except json.JSONDecodeError:
                continue

    return text


def validate_code_syntax(code, label="<string>"):
    """Validate that code has correct Python syntax.

    Returns (is_valid, error_message).
    """
    try:
        compile(code, label, "exec")
        return True, ""
    except SyntaxError as e:
        error_msg = f"Syntax error at line {e.lineno}: {e.msg}"
        if e.text:
            error_msg += f"\n  {e.text.strip()}"
        return False, error_msg


class Agent:
    """Base agent class with core functionality."""

    def __init__(
        self, name: str, system_prompt: str, model: str, temperature: float = 0, max_tokens: int = 64000
    ):
        self.name = name
        self.system_prompt = system_prompt
        self.history: List[Dict[str, str]] = []
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def add_to_history(self, role: str, content: str):
        """Add a message to the conversation history."""
        self.history.append({"role": role, "content": content})

    def clear_history(self):
        """Clear the conversation history."""
        self.history = []

    def get_messages(self) -> List[Dict[str, str]]:
        """Get the messages for the API call, including system prompt."""
        return [{"role": "system", "content": self.system_prompt}] + self.history

    def query_llm(
        self,
        messages: List[Dict[str, str]] = None,
        max_retries: int = 3,
        timeout: int = 120,
    ) -> str:
        """Query the LLM with the current messages, allowing override
        of messages given in the function call, and return the response.

        Args:
            messages: Optional override for messages to send
            max_retries: Maximum number of retry attempts (default: 3)
            timeout: Timeout in seconds for the entire request (default: 120)
        """
        use_messages = messages if messages else self.get_messages()

        for attempt in range(max_retries):
            try:
                # Use streaming with timeout
                # Note: litellm timeout may not work properly with all providers
                print(
                    f"   Calling LLM API (timeout: {timeout}s, max_tokens: {self.max_tokens})...",
                    end="",
                    flush=True,
                )

                # Build parameters using helper function
                completion_params = build_completion_params(
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    stream=True,
                    timeout=timeout,
                    messages=use_messages,
                )

                response = completion(**completion_params)

                response_text = ""
                chunk_count = 0
                for chunk in response:
                    if chunk.choices and chunk.choices[0].delta.content:
                        response_text += chunk.choices[0].delta.content
                        chunk_count += 1
                        # Print a dot every 50 chunks as progress indicator
                        if chunk_count % 50 == 0:
                            print(".", end="", flush=True)

                print(" ✓")  # Success indicator
                return response_text

            except Exception as e:
                error_str = str(e).lower()
                # Check if it's a timeout or connection error
                is_timeout = any(
                    keyword in error_str
                    for keyword in [
                        "timeout",
                        "timed out",
                        "connection",
                        "read timeout",
                    ]
                )

                if is_timeout:
                    print(f"\n⚠️ LLM API timeout (attempt {attempt + 1}/{max_retries})")
                else:
                    print(
                        f"\n⚠️ LLM API error (attempt {attempt + 1}/{max_retries}): {str(e)[:100]}"
                    )

                if attempt < max_retries - 1:
                    print(f"   Retrying in 3 seconds...")
                    time.sleep(3)
                else:
                    print(f"   All retries exhausted. Failing.")
                    raise Exception(f"LLM API failed after {max_retries} attempts: {e}")

        raise Exception("LLM API call failed unexpectedly")

    def process(self, input_text: str) -> str:
        """Query the LLM with a prompt and return the response."""
        self.add_to_history("user", input_text)
        response_text = self.query_llm()
        self.add_to_history("assistant", response_text)
        return response_text


class ArchitectAgent(Agent):
    """Agent specialized in designing software architecture based on problem descriptions."""

    def __init__(self, model: str, temperature: float = 0, max_tokens: int = 64000):
        system_prompt = """You are an expert software architect. Your job is to:
1. Analyze the problem description that is provided to you
2. Break down the requirements into clear, actionable items
3. Design a software architecture that addresses the requirements
4. Create a development plan with specific tasks, dependencies, and estimated effort
5. Recommend technologies and frameworks appropriate for the solution
6. Identify potential challenges and risks

Format your response in a structured way with clear sections for:
- Problem Analysis
- Requirements
- Architecture Design
- Development Plan
- Technology Stack
- Risks and Mitigations

Be specific, practical, and focus on creating a plan that developers can follow to implement the solution."""

        super().__init__(
            "Architect", system_prompt, model=model, temperature=temperature, max_tokens=max_tokens
        )


class SoftwareEngineerAgent(Agent):
    """Agent specialized in implementing Python code based on architecture plans."""

    def __init__(self, model: str, temperature: float = 0, max_tokens: int = 64000):
        system_prompt = """You are an expert Python software engineer. Your job is to implement Python code based on the architecture and development plan provided to you.

For each part of the system you're asked to implement:
1. Write clean, efficient, and well-documented Python code
2. Include comprehensive docstrings and comments explaining your implementation choices
3. Follow PEP 8 style guidelines and Python best practices
4. Handle error cases and edge conditions appropriately
5. Consider performance, security, and maintainability
6. Create class structures, database models, and API endpoints as needed
7. Implement appropriate design patterns for the problem domain

If you receive a test report indicating failures or issues with your code:
- Carefully analyze the issues reported
- Fix each reported problem systematically
- Incorporate all the suggested improvements
- Make sure your revised code addresses all the feedback
- Explain the changes you've made to fix the issues

Your response must be formatted as follows:
1. Wrap ALL of your Python code in a markdown code block using triple backticks with the 'python' language identifier
2. Include all commentary and explanations as proper multi-line or single-line comments WITHIN the code
3. All imports should be at the top of the file
4. Any class or function definitions should be clearly separated
5. The code should be a complete, runnable implementation of the system

The response format should be:
```python
# Brief overview of implementation approach as a comment
# ... your complete Python implementation here ...
```

IMPORTANT:
- Wrap your code in ```python ... ``` markdown code blocks
- Do NOT include any explanatory text outside the code block
- All explanations must be Python comments inside the code block
- Ensure the code is COMPLETE - do not truncate or abbreviate any part

"""

        super().__init__(
            "Software Engineer", system_prompt, model=model, temperature=temperature, max_tokens=max_tokens
        )


class TestEngineerAgent(Agent):
    """Agent specialized in testing and evaluating Python code implementations."""

    def __init__(
        self, model: str, temperature: float = 0, pass_threshold: float = 90.0, max_tokens: int = 64000, enable_web_search: bool = False
    ):
        self.pass_threshold = pass_threshold
        self.enable_web_search = enable_web_search
        system_prompt = f"""You are an expert Python test engineer. Your job is to analyze, compile, and test Python code implementations.

Your responsibilities include:
1. Evaluating if the code compiles correctly
2. Creating and running appropriate tests for the functionality
3. Identifying bugs, errors, or inefficiencies
4. Providing detailed feedback about issues found
5. Suggesting specific fixes for any problems
6. The python file to be tested is named 'main.py' and should be imported in the test code.

You will receive actual test results from a Docker sandbox environment where the code was executed.
Use these results to provide an accurate assessment and specific suggestions for improvement.

Your response MUST be a JSON object with the following structure:
{{{{
    "passed": boolean,  // Whether the code meets the {pass_threshold}% pass threshold and compiles successfully
    "results": {{{{
        "compilation_success": boolean,  // Whether the code compiles without syntax errors
        "test_results": [
            {{{{
                "test_name": string,
                "passed": boolean,
                "description": string  // Description of the test and what it verifies
            }}}}
        ],
        "issues": [
            {{{{
                "type": string,  // "syntax", "logical", "performance", "security", etc.
                "severity": string,  // "critical", "major", "minor", "suggestion"
                "description": string,
                "location": string,  // The function, class, or line where the issue occurs
                "fix_suggestion": string  // Specific code or approach to fix the issue
            }}}}
        ],
        "test_coverage": string,  // Percentage of code covered by tests
        "overall_assessment": string,  // General assessment of the code quality and functionality
        "pass_percentage": number,  // Percentage of tests that passed (0-100)
        "passed_count": number,  // Number of tests that passed
        "failed_count": number,  // Number of tests that failed
        "total_count": number  // Total number of tests run
    }}}}
}}}}"""

        super().__init__(
            "Test Engineer", system_prompt, model=model, temperature=temperature, max_tokens=max_tokens
        )

    def process(self, input_text: str) -> str:
        """Process input with the agent, run code in Docker sandbox, and return the response."""
        self.add_to_history("user", input_text)
        # Extract code from input
        code_to_test, _ = extract_code_from_response(input_text)

        # Extract architecture plan for context
        arch_match = re.search(
            r"This code is intended to implement the following architecture plan:\s*([\s\S]*?)(?:\n\n|$)",
            input_text,
        )
        architecture_plan = arch_match.group(1).strip() if arch_match else ""

        # Generate test code based on the implementation
        test_code = self._generate_test_code(code_to_test, architecture_plan)

        # Run in Docker sandbox
        test_results = self._run_in_docker_sandbox(code_to_test, test_code)

        # Compress test results - only keep critical information
        compressed_results = self._compress_sandbox_results(test_results)

        # Enhance the user input with the compressed test results
        enhanced_input = f"""{input_text}

SANDBOX TEST RESULTS:
{compressed_results}

Based on these actual test results from executing the code in a Docker sandbox,
provide a detailed JSON test report following the format specified in your instructions.
Focus on providing actionable feedback and specific fixes for any issues found."""

        # Update the history with the enhanced input
        self.history[-1]["content"] = enhanced_input

        # Let the GPT model analyze the results and generate the report
        response_text = self.query_llm()
        self.add_to_history("assistant", response_text)

        # Try to ensure the response is valid JSON
        try:
            response_text = extract_json_from_response(response_text)
            # Validate by parsing
            json.loads(response_text)
        except:
            # If not valid JSON, create a basic report
            # Use 90% threshold from test_results if available
            passed = test_results.get(
                "success", False
            )  # This already reflects 90% threshold
            basic_report = {
                "passed": passed,
                "results": {
                    "compilation_success": "Traceback"
                    not in test_results.get("stderr", ""),
                    "test_results": [],
                    "issues": (
                        [
                            {
                                "type": (
                                    "execution"
                                    if "Traceback" in test_results.get("stderr", "")
                                    else "unknown"
                                ),
                                "severity": (
                                    "critical"
                                    if test_results.get("stderr", "")
                                    else "minor"
                                ),
                                "description": test_results.get(
                                    "stderr", "Unknown error occurred"
                                ),
                                "location": "unknown",
                                "fix_suggestion": "Review the error trace above",
                            }
                        ]
                        if test_results.get("stderr", "")
                        else []
                    ),
                    "overall_assessment": (
                        "Execution failed with errors"
                        if test_results.get("stderr", "")
                        else "Code executed but analysis could not be completed"
                    ),
                    "pass_percentage": test_results.get("pass_percentage", 0.0),
                    "passed_count": test_results.get("passed_count", 0),
                    "failed_count": test_results.get("failed_count", 0),
                    "total_count": test_results.get("total_count", 0),
                },
            }
            response_text = json.dumps(basic_report, indent=2)

        return response_text

    def _generate_test_code(self, code_to_test: str, architecture_plan: str) -> str:
        """Generate test code for the implementation using GPT."""
        # Prepare prompt for generating test code
        test_generation_prompt = f"""You are an expert Python test engineer.
Generate comprehensive unit tests for the following Python code.

CODE TO TEST:
```python
{code_to_test}
```

ARCHITECTURE/REQUIREMENTS:
{architecture_plan}

Create a complete test suite that:
1. Tests all main functionality
2. Includes edge cases and error conditions
3. Uses pytest or unittest framework
4. Can be executed directly
5. Includes proper assertions and validations
6. When comparing floating point numbers to numeric literals, use floating point literals.
7. The test suite should import components to test from the 'main.py' file.

IMPORTANT: Return ONLY the raw Python test code. Do NOT wrap it in markdown code blocks or backticks. Do NOT include explanations. Just return the pure Python code that can be written directly to a .py file."""

        # Call the API to generate test code
        prompt_messages = [
            {
                "role": "system",
                "content": "You are an expert in writing Python unit tests. Output only raw Python code without markdown formatting.",
            },
            {"role": "user", "content": test_generation_prompt},
        ]

        test_code_raw = self.query_llm(prompt_messages)

        # Extract test code from response
        test_code, _ = extract_code_from_response(test_code_raw)
        test_code = test_code.strip()

        # Validate that the test code has valid Python syntax
        is_valid, error_msg = validate_code_syntax(test_code, "<test_code>")
        if not is_valid:
            print(f"Warning: Generated test code has syntax error: {error_msg}")
            print("Attempting to generate simpler fallback tests...")
            # Generate a minimal fallback test
            test_code = f"""import unittest
from main import *

class TestBasic(unittest.TestCase):
    def test_import(self):
        \"\"\"Test that the module imports successfully.\"\"\"
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
"""

        return test_code

    def _run_in_docker_sandbox(self, code_to_test: str, test_code: str) -> dict:
        """Run the code and tests in a Docker sandbox environment."""
        import subprocess
        import tempfile
        import os

        # Create a temporary directory for the files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create implementation file
            impl_file = os.path.join(temp_dir, "main.py")
            with open(impl_file, "w") as f:
                f.write(code_to_test)

            # Create test file
            test_file = os.path.join(temp_dir, "test_main.py")
            with open(test_file, "w") as f:
                f.write(test_code)

            # Create a basic pytest configuration to capture output
            conftest_file = os.path.join(temp_dir, "conftest.py")
            with open(conftest_file, "w") as f:
                f.write(
                    """
import pytest

def pytest_configure(config):
    config.option.asyncio_mode = "auto"
    config._inicache["asyncio_default_fixture_loop_scope"] = "function"

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.when == "call":
        print(f"TEST: {item.name} - {'PASSED' if report.passed else 'FAILED'}")
"""
                )
            # copy the contents of the temp directory to the current directory for debugging purposes
            import shutil

            if os.path.exists("src"):
                shutil.rmtree("src")
            shutil.copytree(temp_dir, "src")

            # Check if Docker is available and responsive
            try:
                result = subprocess.run(
                    ["docker", "--version"],
                    check=True,
                    capture_output=True,
                    timeout=5,  # 5 second timeout for version check
                )
            except subprocess.TimeoutExpired:
                return {
                    "success": False,
                    "stdout": "",
                    "stderr": "Docker daemon is not responding. Please restart Docker Desktop.",
                    "return_code": -1,
                    "invalid_imports": [],
                }
            except (subprocess.SubprocessError, FileNotFoundError):
                return {
                    "success": False,
                    "stdout": "",
                    "stderr": "Docker is not available on this system",
                    "return_code": -1,
                    "invalid_imports": [],
                }

            # Additional check: try to list images to verify daemon is working
            try:
                subprocess.run(
                    ["docker", "images"],
                    check=True,
                    capture_output=True,
                    timeout=5,  # 5 second timeout
                )
            except subprocess.TimeoutExpired:
                return {
                    "success": False,
                    "stdout": "",
                    "stderr": "Docker daemon is not responding. Please restart Docker Desktop and try again.",
                    "return_code": -1,
                    "invalid_imports": [],
                }
            except subprocess.SubprocessError:
                return {
                    "success": False,
                    "stdout": "",
                    "stderr": "Docker daemon is not accessible. Please check Docker Desktop is running.",
                    "return_code": -1,
                    "invalid_imports": [],
                }

            try:
                analyzer = PythonPackageAnalyzer(src_dir="src", enable_web_search=self.enable_web_search, model=self.model)
                # Analyze code and get required packages
                required_packages = analyzer.analyze()
                print(f"Found {len(required_packages)} required packages:")
                for package in sorted(required_packages):
                    print(f"  - {package}")

                # Check for invalid imports
                invalid_imports = analyzer.get_invalid_imports()
                if invalid_imports:
                    print(f"\n⚠️  Found {len(invalid_imports)} invalid/unresolved imports:")
                    for import_name, error_msg in invalid_imports:
                        print(f"  ✗ {import_name}: {error_msg}")

                # Generate Dockerfile
                dockerfile = analyzer.generate_dockerfile(output_file="Dockerfile")
                print(f"\nGenerated Dockerfile and requirements.txt")
                print(f"Python version detected: {analyzer.python_version}")

                # Print excluded local modules
                if hasattr(analyzer, "src_dir") and analyzer.files:
                    local_modules = {file_path.stem for file_path in analyzer.files}
                    local_packages = set()
                    for file_path in analyzer.files:
                        parent_dir = file_path.parent
                        if (parent_dir / "__init__.py").exists():
                            local_packages.add(parent_dir.name)

                    print("\nExcluded local modules/packages:")
                    for module in sorted(local_modules.union(local_packages)):
                        print(f"  - {module}")

                # Build container using the generated Dockerfile
                print("\nBuilding Docker container (timeout: 120s)...")
                import time

                start_time = time.time()

                build_result = subprocess.run(
                    ["docker", "build", "-t", "test", "."],
                    capture_output=True,
                    text=True,
                    timeout=120,  # 2 minute timeout for build
                )

                elapsed = time.time() - start_time
                print(f"✓ Build completed in {elapsed:.1f}s")

                if build_result.returncode != 0:
                    print(f"✗ Docker build failed!")
                    print(
                        f"Build stderr (last 500 chars):\n{build_result.stderr[-500:]}"
                    )
                    return {
                        "success": False,
                        "stdout": build_result.stdout,
                        "stderr": f"Docker build failed:\n{build_result.stderr}",
                        "return_code": build_result.returncode,
                        "invalid_imports": invalid_imports,
                    }

                # Use Docker to run the tests in the "test" container
                print("Running tests in Docker container...")
                test_start = time.time()

                result = subprocess.run(
                    ["docker", "run", "test"],
                    capture_output=True,
                    text=True,
                    timeout=30,  # 30 second timeout
                    encoding="utf-8",
                )

                test_elapsed = time.time() - test_start
                print(f"✓ Tests completed in {test_elapsed:.1f}s")

                # Parse test results to calculate pass percentage
                passed_count = 0
                failed_count = 0
                total_count = 0
                lines = result.stdout.split("\n")

                # Look for pytest summary line like "2 failed, 94 passed in 0.31s"
                for line in lines:
                    if " passed" in line or " failed" in line:
                        # Extract numbers from summary line
                        passed_match = re.search(r"(\d+) passed", line)
                        failed_match = re.search(r"(\d+) failed", line)
                        if passed_match:
                            passed_count = int(passed_match.group(1))
                        if failed_match:
                            failed_count = int(failed_match.group(1))

                total_count = passed_count + failed_count

                # Calculate pass percentage
                if total_count > 0:
                    pass_percentage = (passed_count / total_count) * 100
                    print(
                        f"📊 Test Results: {passed_count}/{total_count} passed ({pass_percentage:.1f}%)"
                    )

                    # Use configurable passing grade threshold
                    meets_threshold = pass_percentage >= self.pass_threshold
                else:
                    meets_threshold = result.returncode == 0  # Fallback to exit code
                    pass_percentage = 100.0 if meets_threshold else 0.0

                # Show test summary
                if result.returncode == 0:
                    print("✓ All tests passed!")
                elif meets_threshold:
                    print(
                        f"✓ Passing grade achieved ({pass_percentage:.1f}% ≥ {self.pass_threshold}%)"
                    )
                else:
                    print(
                        f"✗ Below passing grade ({pass_percentage:.1f}% < {self.pass_threshold}%)"
                    )
                    # Extract and show failure summary
                    for i, line in enumerate(lines):
                        if "FAILED" in line or "ERROR" in line:
                            print(f"  {line}")
                        elif "short test summary" in line.lower():
                            # Print summary section
                            print("\n" + "\n".join(lines[i : min(i + 10, len(lines))]))
                            break

                return {
                    "success": meets_threshold,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "return_code": result.returncode,
                    "pass_percentage": pass_percentage,
                    "passed_count": passed_count,
                    "failed_count": failed_count,
                    "total_count": total_count,
                    "invalid_imports": invalid_imports,
                }

            except subprocess.TimeoutExpired as e:
                timeout_msg = (
                    "Docker build" if "build" in str(e.cmd) else "Test execution"
                )
                return {
                    "success": False,
                    "stdout": "",
                    "stderr": f"{timeout_msg} timed out",
                    "return_code": -1,
                    "invalid_imports": [],
                }
            except Exception as e:
                return {
                    "success": False,
                    "stdout": "",
                    "stderr": f"Error running tests: {str(e)}",
                    "return_code": -1,
                    "invalid_imports": [],
                }

    def _compress_sandbox_results(self, test_results: dict) -> str:
        """Compress sandbox results to only critical information."""
        stdout = test_results.get("stdout", "")
        stderr = test_results.get("stderr", "")
        return_code = test_results.get("return_code", -1)
        pass_percentage = test_results.get("pass_percentage", 0.0)
        passed_count = test_results.get("passed_count", 0)
        failed_count = test_results.get("failed_count", 0)
        total_count = test_results.get("total_count", 0)
        invalid_imports = test_results.get("invalid_imports", [])

        compressed = []

        # Add invalid imports warning first (critical issue)
        if invalid_imports:
            compressed.append("⚠️  INVALID/UNRESOLVED IMPORTS:")
            compressed.append("The following imports could not be resolved on PyPI and are likely hallucinated or misspelled:")
            for import_name, error_msg in invalid_imports:
                compressed.append(f"  ✗ {import_name}")
                compressed.append(f"    {error_msg}")
            compressed.append("\nPlease fix these imports by either:")
            compressed.append("  1. Correcting the import name to match the actual PyPI package")
            compressed.append("  2. Using standard library modules instead")
            compressed.append("  3. Removing the import if it's not needed\n")

        # Add test results summary with pass percentage
        if total_count > 0:
            status = (
                f"PASS (≥{self.pass_threshold}%)"
                if pass_percentage >= self.pass_threshold
                else f"FAIL (<{self.pass_threshold}%)"
            )
            compressed.append(
                f"📊 Test Results: {passed_count}/{total_count} passed ({pass_percentage:.1f}%) - {status}"
            )

        # Add return code status
        if return_code == 0:
            compressed.append("✓ Exit Code: 0 (SUCCESS)")
        else:
            compressed.append(f"✗ Exit Code: {return_code} (FAILURE)")

        # Extract only failed tests from stdout
        if stdout:
            failed_tests = [
                line
                for line in stdout.split("\n")
                if "FAILED" in line or "ERROR" in line
            ]
            if failed_tests:
                compressed.append("\nFailed Tests:")
                compressed.extend(
                    [f"  {test}" for test in failed_tests[:10]]
                )  # Limit to 10
            else:
                # Check for passed test count
                passed = [
                    line
                    for line in stdout.split("\n")
                    if "PASSED" in line or "passed" in line
                ]
                if passed:
                    compressed.append(f"\n✓ Tests passed: {len(passed)}")

        # Only include critical errors from stderr
        if stderr:
            # Extract only the last error or traceback
            lines = stderr.strip().split("\n")
            if "Traceback" in stderr:
                # Find the last traceback
                traceback_start = -1
                for i in range(len(lines) - 1, -1, -1):
                    if "Traceback" in lines[i]:
                        traceback_start = i
                        break
                if traceback_start >= 0:
                    compressed.append("\nCritical Error:")
                    compressed.append("  " + "\n  ".join(lines[traceback_start:]))
            elif lines:
                # Just show the last few error lines
                compressed.append("\nError Output:")
                compressed.append("  " + "\n  ".join(lines[-5:]))

        return "\n".join(compressed)



class AgenticFlow:
    """Manages the flow of information between agents."""

    def __init__(
        self, model: str, max_iterations=3, temperature=0, pass_threshold=90.0, max_tokens=64000, enable_web_search=False
    ):
        self.architect = ArchitectAgent(model=model, temperature=temperature, max_tokens=max_tokens)
        self.software_engineer = SoftwareEngineerAgent(
            model=model, temperature=temperature, max_tokens=max_tokens
        )
        self.test_engineer = TestEngineerAgent(
            model=model, temperature=temperature, pass_threshold=pass_threshold, max_tokens=max_tokens, enable_web_search=enable_web_search
        )
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.model = model
        self.pass_threshold = pass_threshold
        self.max_tokens = max_tokens
        self.results = {
            "architecture_plan": None,
            "implementation_history": [],
            "test_reports": [],
            "final_implementation": None,
            "final_test_report": None,
            "iterations_required": 0,
            "success": False,
        }

    def _compress_test_report_to_todos(self, test_report: dict) -> str:
        """Extract actionable todos from a test report."""
        todos = []

        if isinstance(test_report, dict) and "results" in test_report:
            results = test_report["results"]

            # Add compilation issues if any
            if not results.get("compilation_success", True):
                todos.append("- Fix compilation/syntax errors")

            # Add failed tests
            if "test_results" in results:
                for test in results["test_results"]:
                    if not test.get("passed", False):
                        todos.append(f"- Fix test: {test.get('test_name', 'unknown')}")

            # Add critical and major issues
            if "issues" in results:
                for issue in results["issues"]:
                    severity = issue.get("severity", "")
                    if severity in ["critical", "major"]:
                        location = issue.get("location", "unknown location")
                        description = issue.get("description", "")
                        fix = issue.get("fix_suggestion", "")
                        todos.append(
                            f"- [{severity.upper()}] {location}: {description[:100]}... Fix: {fix[:100]}"
                        )

        return "\n".join(todos) if todos else "No critical issues found"

    def _summarize_architecture_plan(self, architecture_plan: str) -> str:
        """Summarize the architecture plan into key requirements using LLM."""
        prompt = f"""Summarize the following architecture plan into a concise list of key requirements and constraints (maximum 10 bullet points):

{architecture_plan}

Provide only the essential technical requirements, constraints, and design decisions that a developer needs to keep in mind while implementing. Be concise."""

        messages = [
            {
                "role": "system",
                "content": "You are an expert at summarizing technical documents concisely.",
            },
            {"role": "user", "content": prompt},
        ]

        # Build parameters using helper function
        completion_params = build_completion_params(
            model=self.model,
            temperature=0,
            max_tokens=self.max_tokens,
            stream=True,
            timeout=120,
            messages=messages,
        )

        response = completion(**completion_params)

        summary = ""
        for chunk in response:
            if (
                hasattr(chunk, "choices")
                and chunk.choices
                and hasattr(chunk.choices[0], "delta")
                and chunk.choices[0].delta.content
            ):
                summary += chunk.choices[0].delta.content

        return summary.strip()

    def run(self, problem_description: str) -> Dict[str, Any]:
        """Execute the full agentic workflow."""
        print(f"🏛️ Starting Architect Agent...")
        architecture_plan = self.architect.process(problem_description)
        self.results["architecture_plan"] = architecture_plan

        # Initial implementation
        print(f"\n👩‍💻 Starting Software Engineer Agent (Iteration 1)...")
        implementation_prompt = f"""Based on the following architecture plan, implement the core Python code for this system. 
Focus on creating the main components, data models, and essential functionality.

ARCHITECTURE PLAN:
{architecture_plan}

Please implement a working Python prototype that demonstrates the key functionality described in the architecture.
Include proper error handling, documentation, and follow best practices for Python development."""

        implementation = self.software_engineer.process(implementation_prompt)

        # Extract and validate the generated code
        extracted_code, _ = extract_code_from_response(implementation, debug_file="debug_response.txt")
        is_valid, error_msg = validate_code_syntax(extracted_code)

        if not is_valid:
            print(f"⚠️ Warning: Generated code has syntax errors:")
            print(f"  {error_msg}")
            print("  The test phase will report these errors.")

        self.results["implementation_history"].append(implementation)

        # Test-fix loop
        iteration = 1
        success = False
        architecture_summary = None  # Will be created after first iteration

        while iteration <= self.max_iterations:
            print(f"\n🧪 Starting Test Engineer Agent (Iteration {iteration})...")
            print(f"   Pass threshold: {self.pass_threshold}%")

            # Extract just the code from the implementation (remove markdown if present)
            implementation_code, _ = extract_code_from_response(implementation)
            is_valid, error_msg = validate_code_syntax(implementation_code)
            if not is_valid:
                print(f"⚠️ Warning: Code has syntax errors before testing:")
                print(f"  {error_msg}")

            # Use full architecture plan for first iteration, summary for subsequent ones
            architecture_context = (
                architecture_plan if iteration == 1 else architecture_summary
            )

            test_prompt = f"""Please analyze, compile, and test the following Python implementation:

```python
{implementation_code}
```

This code is intended to implement the following architecture plan:

{architecture_context}

Provide a comprehensive test report including compilation status, test results, and any issues found."""

            test_report_str = self.test_engineer.process(test_prompt)

            # Try to parse the test report as JSON
            try:
                # Extract JSON from the response if it's not pure JSON
                test_report_str = extract_json_from_response(test_report_str)
                test_report = json.loads(test_report_str)
                self.results["test_reports"].append(test_report)

                # Check if we meet the configured threshold for passing
                results = test_report.get("results", {})
                pass_percentage = results.get("pass_percentage", 0.0)
                meets_threshold = pass_percentage >= self.pass_threshold

                if meets_threshold:
                    print(
                        f"✅ Tests passed! Implementation successful after {iteration} iteration(s) ({pass_percentage:.1f}% ≥ {self.pass_threshold}%)."
                    )
                    success = True
                    break

                # Create architecture summary after first iteration
                if iteration == 1 and architecture_summary is None:
                    print(
                        "📝 Summarizing architecture plan for subsequent iterations..."
                    )
                    architecture_summary = self._summarize_architecture_plan(
                        architecture_plan
                    )

                # If tests failed and we haven't reached max iterations, try to fix
                if iteration < self.max_iterations:
                    print(
                        f"\n🔄 Implementation failed tests. Starting revision {iteration + 1}..."
                    )

                    # Compress test report to actionable todos
                    todos = self._compress_test_report_to_todos(test_report)

                    # Use summary for subsequent iterations
                    arch_context = (
                        architecture_plan if iteration == 1 else architecture_summary
                    )

                    # Extract clean code without markdown
                    prev_code, _ = extract_code_from_response(implementation)

                    revision_prompt = f"""Your previous code implementation had some issues. Please revise your implementation to address the following:

ISSUES TO FIX:
{todos}

ARCHITECTURE REQUIREMENTS:
{arch_context}

YOUR PREVIOUS IMPLEMENTATION:
```python
{prev_code}
```

Please provide a complete revised implementation that addresses all the issues mentioned above."""

                    implementation = self.software_engineer.process(revision_prompt)

                    # Validate the revised code
                    extracted_code, _ = extract_code_from_response(implementation)
                    is_valid, error_msg = validate_code_syntax(extracted_code)

                    if not is_valid:
                        print(f"⚠️ Warning: Revised code still has syntax errors:")
                        print(f"  {error_msg}")

                    self.results["implementation_history"].append(implementation)
                else:
                    print(
                        "❌ Maximum iterations reached. Implementation still has issues."
                    )

            except json.JSONDecodeError:
                print("⚠️ Could not parse test report as JSON. Using the report as-is.")
                self.results["test_reports"].append({"raw_report": test_report_str})

                # Create architecture summary after first iteration
                if iteration == 1 and architecture_summary is None:
                    print(
                        "📝 Summarizing architecture plan for subsequent iterations..."
                    )
                    architecture_summary = self._summarize_architecture_plan(
                        architecture_plan
                    )

                if iteration < self.max_iterations:
                    # Use summary for subsequent iterations
                    arch_context = (
                        architecture_plan if iteration == 1 else architecture_summary
                    )

                    # Truncate test report if too long
                    compressed_report = (
                        test_report_str[:1000] + "..."
                        if len(test_report_str) > 1000
                        else test_report_str
                    )

                    # Extract clean code without markdown
                    prev_code, _ = extract_code_from_response(implementation)

                    revision_prompt = f"""Your previous code implementation had some issues. Please revise your implementation based on the following test report:

TEST REPORT (summary):
{compressed_report}

ARCHITECTURE REQUIREMENTS:
{arch_context}

YOUR PREVIOUS IMPLEMENTATION:
```python
{prev_code}
```

Please provide a complete revised implementation that addresses all the issues mentioned in the test report."""

                    implementation = self.software_engineer.process(revision_prompt)

                    # Validate the revised code
                    extracted_code, _ = extract_code_from_response(implementation)
                    is_valid, error_msg = validate_code_syntax(extracted_code)

                    if not is_valid:
                        print(f"⚠️ Warning: Revised code still has syntax errors:")
                        print(f"  {error_msg}")

                    self.results["implementation_history"].append(implementation)

            iteration += 1

        # Store final results
        self.results["final_implementation"] = implementation
        if len(self.results["test_reports"]) > 0:
            self.results["final_test_report"] = self.results["test_reports"][-1]
        self.results["iterations_required"] = iteration
        self.results["success"] = success

        return self.results

    def save_results(self, filename: str = "agentic_flow_results.json"):
        """Save the results to a JSON file in a pretty format."""
        src_dir = "src"
        filename = os.path.join(src_dir, filename)
        with open(filename, "w") as f:
            json.dump(self.results, f, indent=4, sort_keys=False)
        print(f"Results saved to {filename}")


def check_docker_availability():
    """Check if Docker is available and responsive before starting."""
    import subprocess

    print("🐳 Checking Docker availability...")

    try:
        # Check if Docker command exists
        result = subprocess.run(
            ["docker", "--version"],
            check=True,
            capture_output=True,
            timeout=5,
            text=True,
        )
        print(f"✓ Docker found: {result.stdout.strip()}")
    except FileNotFoundError:
        print("❌ Docker command not found!")
        print(
            "   Please install Docker Desktop from https://www.docker.com/products/docker-desktop"
        )
        return False
    except subprocess.TimeoutExpired:
        print("❌ Docker is not responding (timeout after 5 seconds)")
        print("   Please restart Docker Desktop and try again")
        return False
    except subprocess.SubprocessError as e:
        print(f"❌ Error checking Docker: {e}")
        return False

    # Check if Docker daemon is running
    try:
        result = subprocess.run(
            ["docker", "ps"], check=True, capture_output=True, timeout=5, text=True
        )
        print("✓ Docker daemon is running")
        return True
    except subprocess.TimeoutExpired:
        print("❌ Docker daemon is not responding")
        print("   Please start Docker Desktop and try again")
        return False
    except subprocess.CalledProcessError:
        print("❌ Docker daemon is not accessible")
        print("   Please start Docker Desktop and ensure it's running properly")
        return False
    except subprocess.SubprocessError as e:
        print(f"❌ Error accessing Docker daemon: {e}")
        return False


# Example usage
if __name__ == "__main__":
    print("🤖 Agentic Flow - Software Development System")
    print("--------------------------------------------")

    # Load environment variables from .env file
    load_dotenv()

    # Check Docker availability first
    if not check_docker_availability():
        print(
            "\n⚠️  Cannot proceed without Docker. Please fix the Docker issue and try again."
        )
        exit(1)

    print()  # Add blank line for readability

    # Argument parser setup
    parser = argparse.ArgumentParser(description="Run the Agentic Flow system.")
    parser.add_argument(
        "--description-file",
        type=str,
        help="Path to a file containing the problem description.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=2,
        help="Maximum number of test-fix iterations (default: 2).",
    )
    parser.add_argument(
        "--pass-threshold",
        type=float,
        default=90.0,
        help="Minimum percentage of tests that must pass to succeed (default: 90.0).",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model name to use (e.g., openai/gpt-5.2, openai/gpt-5.2-codex, anthropic/claude-opus-4-6).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=64000,
        help="Maximum tokens to generate (default: 64000).",
    )
    parser.add_argument(
        "--search",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Enable Tavily web search for package validation (default: true).",
    )
    args = parser.parse_args()

    # Check for TAVILY_API_KEY if web search is enabled
    if args.search:
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        if not tavily_api_key:
            print("\n⚠️  WARNING: Web search is enabled (--search true) but TAVILY_API_KEY environment variable is not set!")
            print("   Web search will not work without the API key.")
            print("   Options:")
            print("   1. Set TAVILY_API_KEY in your .env file")
            print("   2. Export TAVILY_API_KEY in your shell: export TAVILY_API_KEY='your-key'")
            print("   3. Disable web search with --search false")

            response = input("\n   Continue anyway? (y/n): ").strip().lower()
            if response != 'y':
                print("\n   Exiting. Please set TAVILY_API_KEY or disable web search.")
                exit(1)
        else:
            print(f"✓ Tavily API key found (web search enabled)")

    # Setup correct client based on user input or command line argument
    if args.model:
        model = args.model
        print(f"Using model from command line: {model}")
    else:
        print(
            "\nℹ️  Tip: Use --model to specify a model directly, e.g.:"
        )
        print(
            "   --model openai/gpt-5.2-codex  (optimized for coding tasks)"
        )
        chosen_provider = (
            input("🌐 Choose LLM provider (openai, anthropic): ").strip().lower()
        )
        if chosen_provider == "anthropic":
            model = "anthropic/claude-opus-4-6"
        elif chosen_provider == "openai":
            model = "openai/gpt-5.2"
        else:
            print("Invalid provider selected. Defaulting to OpenAI gpt-5.2.")
            chosen_provider = "openai"
            model = "openai/gpt-5.2"

        print(f"Using model: {model}")
    # Load problem description
    if args.description_file:
        try:
            with open(args.description_file, "r") as file:
                problem_description = file.read().strip()
            print(f"📄 Loaded problem description from {args.description_file}")
        except Exception as e:
            print(f"❌ Error reading description file: {e}")
            exit(1)
    else:
        problem_description = input(
            "📝 Enter your problem description (or press Enter for a default example): "
        ).strip()
        if not problem_description:
            problem_description = """
            We need a system to manage inventory for a small retail store. 
            The store sells various products and needs to track stock levels, 
            sales, and purchases. The system should alert when items are running low 
            and generate reports on sales and inventory status. It should be easy 
            to use for staff who aren't very technical.
            """
            print("\n✨ Using default example problem:\n")
            print(problem_description)

    # Display configuration
    print("\n⚙️  Configuration:")
    print(f"   Max iterations: {args.max_iterations}")
    print(f"   Pass threshold: {args.pass_threshold}%")
    print(f"   Max tokens: {args.max_tokens}")

    start_time = time.time()
    flow = AgenticFlow(
        model=model,
        max_iterations=args.max_iterations,
        temperature=0,
        pass_threshold=args.pass_threshold,
        max_tokens=args.max_tokens,
        enable_web_search=args.search,
    )
    results = flow.run(problem_description)
    end_time = time.time()

    print("\n=== 📐 Architecture Plan ===")
    print(results["architecture_plan"])

    print("\n=== 💻 Final Python Implementation ===")
    print(results["final_implementation"])

    print("\n=== 🧪 Testing Summary ===")
    if results["success"]:
        print(
            f"✅ Implementation successful after {results['iterations_required']+1} iteration(s)"
        )
    else:
        print(
            f"❌ Implementation still has issues after {results['iterations_required']} iteration(s)"
        )

    # Show pass rate from final test report if available
    if results.get("final_test_report"):
        final_report = results["final_test_report"]
        if isinstance(final_report, dict) and "results" in final_report:
            test_results = final_report["results"]
            # Check if we have pass percentage info
            if "pass_percentage" in test_results:
                pass_pct = test_results["pass_percentage"]
                passed = test_results.get("passed_count", 0)
                total = test_results.get("total_count", 0)
                print(
                    f"📊 Final Test Results: {passed}/{total} passed ({pass_pct:.1f}%)"
                )
                if pass_pct >= args.pass_threshold:
                    print(f"✓ Meets passing grade (≥{args.pass_threshold}%)")
                else:
                    print(f"✗ Below passing grade (<{args.pass_threshold}%)")

    print(f"\n⏱️ Total execution time: {end_time - start_time:.2f} seconds")

    # Ask if user wants to see test reports
    show_reports = (
        input("\nDo you want to see the detailed test reports? (y/n): ")
        .lower()
        .startswith("y")
    )
    if show_reports and results["test_reports"]:
        print("\n=== 📋 Test Reports ===")
        for i, report in enumerate(results["test_reports"]):
            print(f"\nTest Report {i+1}:")
            print(json.dumps(report, indent=2))

    flow.save_results()
    print("\n✅ Process completed successfully!")
