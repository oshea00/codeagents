import os
import json
import time
import re
from typing import List, Dict, Any, Optional
import argparse  # Add argparse import
from docker_test import PythonPackageAnalyzer
from litellm import completion


class Agent:
    """Base agent class with core functionality."""
    
    def __init__(self, name: str, system_prompt: str, model: str = "gpt-4o"):
        self.name = name
        self.system_prompt = system_prompt
        self.history: List[Dict[str, str]] = []
        self.model = model
    
    def add_to_history(self, role: str, content: str):
        """Add a message to the conversation history."""
        self.history.append({"role": role, "content": content})
    
    def clear_history(self):
        """Clear the conversation history."""
        self.history = []
    
    def get_messages(self) -> List[Dict[str, str]]:
        """Get the messages for the API call, including system prompt."""
        return [{"role": "system", "content": self.system_prompt}] + self.history
    
    def process(self, input_text: str) -> str:
        """Process input with the agent and return the response."""
        self.add_to_history("user", input_text)
        
        response = completion(model=self.model,
            messages=self.get_messages(),
            stream=True
        )

        response_text = ""
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                response_text += chunk.choices[0].delta.content

        self.add_to_history("assistant", response_text)
        
        return response_text


class ArchitectAgent(Agent):
    """Agent specialized in designing software architecture based on problem descriptions."""
    
    def __init__(self,model: str = "gpt-4o"):
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
        
        super().__init__("Architect", system_prompt,model=model)


class SoftwareEngineerAgent(Agent):
    """Agent specialized in implementing Python code based on architecture plans."""
    
    def __init__(self,model: str = "gpt-4o"):
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

Format your response entirely as a single python syntax file name 'main.py' with all commentary and
explanations commented as proper multi-line or single-line comments. All imports should be at the top of the file,
and any class or function definitions should be clearly separated. The file should be
a complete, runnable implementation of the system as described in the architecture plan.

The response should include:
- Brief overview of implementation approach
- Python modules, classes, and functions you're implementing
- The actual Python code implementation with proper imports
- If revising code based on a test report, explain your fixes

Do not include any additional text or explanations outside of the Python code block.
Do not use markdown formatting or triple backticks to enclose the source code.

"""
        
        super().__init__("Software Engineer", system_prompt,model=model)

class TestEngineerAgent(Agent):
    """Agent specialized in testing and evaluating Python code implementations."""
    
    def __init__(self,model: str = "gpt-4o"):
        system_prompt = """You are an expert Python test engineer. Your job is to analyze, compile, and test Python code implementations.

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
{
    "passed": boolean,  // Whether the code compiles and passes basic functionality tests
    "results": {
        "compilation_success": boolean,  // Whether the code compiles without syntax errors
        "test_results": [
            {
                "test_name": string,
                "passed": boolean,
                "description": string  // Description of the test and what it verifies
            }
        ],
        "issues": [
            {
                "type": string,  // "syntax", "logical", "performance", "security", etc.
                "severity": string,  // "critical", "major", "minor", "suggestion"
                "description": string,
                "location": string,  // The function, class, or line where the issue occurs
                "fix_suggestion": string  // Specific code or approach to fix the issue
            }
        ],
        "overall_assessment": string  // General assessment of the code quality and functionality
    }
}"""
        
        super().__init__("Test Engineer", system_prompt,model=model)
        
    def process(self, input_text: str) -> str:
        """Process input with the agent, run code in Docker sandbox, and return the response."""
        self.add_to_history("user", input_text)
        
        # Extract code from input
        import re
        code_match = re.search(r'```python\s*([\s\S]*?)\s*```', input_text)
        
        if not code_match:
            # No code found, use regular processing without code execution
            response = completion(
                model=self.model,
                messages=self.get_messages(),
                stream=True
            )
            
            response_text = ""
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    response_text += chunk.choices[0].delta.content
            
            self.add_to_history("assistant", response_text)
            return response_text
        
        # Extract the code and architecture plan for context
        code_to_test = code_match.group(1)
        arch_match = re.search(r'This code is intended to implement the following architecture plan:\s*([\s\S]*?)(?:\n\n|$)', input_text)
        architecture_plan = arch_match.group(1).strip() if arch_match else ""
        
        # Generate test code based on the implementation
        test_code = self._generate_test_code(code_to_test, architecture_plan)
        
        # Run in Docker sandbox
        test_results = self._run_in_docker_sandbox(code_to_test, test_code)
        
        # Enhance the user input with the test results
        enhanced_input = f"""{input_text}

SANDBOX TEST RESULTS:
```
{test_results.get('stdout', '')}
```

ERROR OUTPUT:
```
{test_results.get('stderr', '')}
```

EXIT CODE: {test_results.get('return_code', -1)}

Based on these actual test results from executing the code in a Docker sandbox,
provide a detailed JSON test report following the format specified in your instructions.
Focus on providing actionable feedback and specific fixes for any issues found."""
        
        # Update the history with the enhanced input
        self.history[-1]["content"] = enhanced_input
        
        # Let the GPT model analyze the results and generate the report
        response = completion(
            model=self.model,
            messages=self.get_messages(),
            stream=True
        )
        
        response_text = ""
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                response_text += chunk.choices[0].delta.content

        self.add_to_history("assistant", response_text)
        
        # Try to ensure the response is valid JSON
        try:
            response_text = self._extract_json(response_text)
            # Validate by parsing
            json.loads(response_text)
        except:
            # If not valid JSON, create a basic report
            basic_report = {
                "passed": test_results.get("return_code", -1) == 0,
                "results": {
                    "compilation_success": "Traceback" not in test_results.get("stderr", ""),
                    "test_results": [],
                    "issues": [{
                        "type": "execution" if "Traceback" in test_results.get("stderr", "") else "unknown",
                        "severity": "critical" if test_results.get("stderr", "") else "minor",
                        "description": test_results.get("stderr", "Unknown error occurred"),
                        "location": "unknown",
                        "fix_suggestion": "Review the error trace above"
                    }] if test_results.get("stderr", "") else [],
                    "overall_assessment": "Execution failed with errors" if test_results.get("stderr", "") else 
                                         "Code executed but analysis could not be completed"
                }
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

Return ONLY the test code, with no explanations or other text."""

        # Call the API to generate test code
        response = completion(
            model=self.model,
            messages=[{"role": "system", "content": "You are an expert in writing Python unit tests."},
                      {"role": "user", "content": test_generation_prompt}],
            stream=True
        )
        
        test_code_raw = ""
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                test_code_raw += chunk.choices[0].delta.content

        # Extract just the code if it's in a code block
        test_code_match = re.search(r'```python\s*([\s\S]*?)\s*```', test_code_raw)
        if test_code_match:
            test_code = test_code_match.group(1)
        else:
            test_code = test_code_raw
        
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
                f.write("""
import pytest

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.when == "call":
        print(f"TEST: {item.name} - {'PASSED' if report.passed else 'FAILED'}")
    return report
""")
            # copy the contents of the temp directory to the current directory for debugging purposes
            import shutil
            if os.path.exists("src"):
                shutil.rmtree("src")
            shutil.copytree(temp_dir, "src")

            # Check if Docker is available
            try:
                subprocess.run(["docker", "--version"], check=True, capture_output=True)
            except (subprocess.SubprocessError, FileNotFoundError):
                return {
                    "success": False,
                    "stdout": "",
                    "stderr": "Docker is not available on this system",
                    "return_code": -1
                }
            

            try:
                analyzer = PythonPackageAnalyzer(src_dir="src")
                # Analyze code and get required packages
                required_packages = analyzer.analyze()
                print(f"Found {len(required_packages)} required packages:")
                for package in sorted(required_packages):
                    print(f"  - {package}")

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
                subprocess.run(["docker", "build", "-t", "test", "."])

                # Use Docker to run the tests in the "test" container
                result = subprocess.run(["docker", "run", "test"],
                    capture_output=True,
                    text=True,
                    timeout=30, # 30 second timeout
                    encoding="utf-8"
                )
                
                return {
                    "success": result.returncode == 0,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "return_code": result.returncode
                }
                
            except subprocess.TimeoutExpired:
                return {
                    "success": False,
                    "stdout": "",
                    "stderr": "Execution timed out after 30 seconds",
                    "return_code": -1
                }
            except Exception as e:
                return {
                    "success": False,
                    "stdout": "",
                    "stderr": f"Error running tests: {str(e)}",
                    "return_code": -1
                }
    
    def _extract_json(self, text):
        """Attempt to extract JSON from a text that might contain other content."""
        # Try to find JSON between triple backticks
        import re
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
        if json_match:
            return json_match.group(1)
        
        # If no backticks, look for text that starts with { and ends with }
        json_match = re.search(r'(\{[\s\S]*\})', text)
        if json_match:
            return json_match.group(1)
            
        # If all else fails, return the original text
        return text
    
class AgenticFlow:
    """Manages the flow of information between agents."""
    
    def __init__(self, max_iterations=3, model="gpt-4o"):
        self.architect = ArchitectAgent(model=model)
        self.software_engineer = SoftwareEngineerAgent(model=model)
        self.test_engineer = TestEngineerAgent(model=model)
        self.max_iterations = max_iterations
        self.results = {
            "architecture_plan": None,
            "implementation_history": [],
            "test_reports": [],
            "final_implementation": None,
            "final_test_report": None,
            "iterations_required": 0,
            "success": False
        }
    
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
        self.results["implementation_history"].append(implementation)
        
        # Test-fix loop
        iteration = 1
        success = False
        
        while iteration <= self.max_iterations:
            print(f"\n🧪 Starting Test Engineer Agent (Iteration {iteration})...")
            test_prompt = f"""Please analyze, compile, and test the following Python implementation:

```python
{implementation}
```

This code is intended to implement the following architecture plan:

{architecture_plan}

Provide a comprehensive test report including compilation status, test results, and any issues found."""
            
            test_report_str = self.test_engineer.process(test_prompt)
            
            # Try to parse the test report as JSON
            try:
                # Extract JSON from the response if it's not pure JSON
                test_report_str = self._extract_json(test_report_str)
                test_report = json.loads(test_report_str)
                self.results["test_reports"].append(test_report)
                
                if test_report.get("passed", False):
                    print(f"✅ Tests passed! Implementation successful after {iteration} iteration(s).")
                    success = True
                    break
                
                # If tests failed and we haven't reached max iterations, try to fix
                if iteration < self.max_iterations:
                    print(f"\n🔄 Implementation failed tests. Starting revision {iteration + 1}...")
                    
                    revision_prompt = f"""Your previous code implementation had some issues. Please revise your implementation based on the following test report:

TEST REPORT:
{json.dumps(test_report, indent=2)}

ARCHITECTURE PLAN:
{architecture_plan}

YOUR PREVIOUS IMPLEMENTATION:
```python
{implementation}
```

Please provide a complete revised implementation that addresses all the issues mentioned in the test report."""
                    
                    implementation = self.software_engineer.process(revision_prompt)
                    self.results["implementation_history"].append(implementation)
                else:
                    print("❌ Maximum iterations reached. Implementation still has issues.")
            
            except json.JSONDecodeError:
                print("⚠️ Could not parse test report as JSON. Using the report as-is.")
                self.results["test_reports"].append({"raw_report": test_report_str})
                if iteration < self.max_iterations:
                    revision_prompt = f"""Your previous code implementation had some issues. Please revise your implementation based on the following test report:

TEST REPORT:
{test_report_str}

ARCHITECTURE PLAN:
{architecture_plan}

YOUR PREVIOUS IMPLEMENTATION:
```python
{implementation}
```

Please provide a complete revised implementation that addresses all the issues mentioned in the test report."""
                    
                    implementation = self.software_engineer.process(revision_prompt)
                    self.results["implementation_history"].append(implementation)
            
            iteration += 1
        
        # Store final results
        self.results["final_implementation"] = implementation
        if len(self.results["test_reports"]) > 0:
            self.results["final_test_report"] = self.results["test_reports"][-1]
        self.results["iterations_required"] = iteration - 1
        self.results["success"] = success
        
        return self.results
    
    def _extract_json(self, text):
        """Attempt to extract JSON from a text that might contain other content."""
        # Try to find JSON between triple backticks
        import re
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
        if json_match:
            return json_match.group(1)
        
        # If no backticks, look for text that starts with { and ends with }
        json_match = re.search(r'(\{[\s\S]*\})', text)
        if json_match:
            return json_match.group(1)
            
        # If all else fails, return the original text
        return text
            
    def save_results(self, filename: str = "agentic_flow_results.json"):
        """Save the results to a JSON file."""
        # Create a serializable version of the results
        serializable_results = self.results.copy()
        
        with open(filename, "w") as f:
            json.dump(serializable_results, f, indent=2)
        print(f"Results saved to {filename}")
    
    def save_results(self, filename: str = "agentic_flow_results.json"):
        """Save the results to a JSON file."""
        with open(filename, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to {filename}")


# Example usage
if __name__ == "__main__":    
    print("🤖 Agentic Flow - Software Development System")
    print("--------------------------------------------")
    
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Run the Agentic Flow system.")
    parser.add_argument('--description-file', type=str, help='Path to a file containing the problem description.')
    parser.add_argument('--max-iterations', type=int, default=3, help='Maximum number of test-fix iterations (default: 3).')
    args = parser.parse_args()

    # Setup correct client based on user input
    chosen_provider = input("🌐 Choose LLM provider (openai, anthropic): ").strip().lower()
    if chosen_provider == "anthropic":
        model = "anthropic/claude-3-7-sonnet-20250219"
    elif chosen_provider == "openai":
        model = "openai/gpt-4o"
    else:
        print("Invalid provider selected. Defaulting to OpenAI.")
        chosen_provider = "openai"
        model = "openai/gpt-4o"

    print(f"Using LLM provider: {chosen_provider}")
    # Load problem description
    if args.description_file:
        try:
            with open(args.description_file, 'r') as file:
                problem_description = file.read().strip()
            print(f"📄 Loaded problem description from {args.description_file}")
        except Exception as e:
            print(f"❌ Error reading description file: {e}")
            exit(1)
    else:
        problem_description = input("📝 Enter your problem description (or press Enter for a default example): ").strip()
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
    
    start_time = time.time()
    flow = AgenticFlow(max_iterations=args.max_iterations,model=model)
    results = flow.run(problem_description)
    end_time = time.time()
    
    print("\n=== 📐 Architecture Plan ===")
    print(results["architecture_plan"])
    
    print("\n=== 💻 Final Python Implementation ===")
    print(results["final_implementation"])
    
    print("\n=== 🧪 Testing Summary ===")
    if results["success"]:
        print(f"✅ Implementation successful after {results['iterations_required']} iteration(s)")
    else:
        print(f"❌ Implementation still has issues after {results['iterations_required']} iteration(s)")
    
    print(f"\n⏱️ Total execution time: {end_time - start_time:.2f} seconds")
    
    # Ask if user wants to see test reports
    show_reports = input("\nDo you want to see the detailed test reports? (y/n): ").lower().startswith('y')
    if show_reports and results["test_reports"]:
        print("\n=== 📋 Test Reports ===")
        for i, report in enumerate(results["test_reports"]):
            print(f"\nTest Report {i+1}:")
            print(json.dumps(report, indent=2))
    
    flow.save_results()
    print("\n✅ Process completed successfully!")