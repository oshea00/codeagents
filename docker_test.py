import ast
import re
import os
import sys
import argparse
from pathlib import Path


class PythonPackageAnalyzer:
    """
    A class that analyzes Python code to determine required packages
    and generates a Dockerfile that installs these packages along with pytest.
    """

    def __init__(
        self,
        src_dir=None,
        python_code=None,
        filename=None,
        python_version="3.12",
    ):
        """
        Initialize the analyzer with a source directory, Python code as a string, or a Python file.

        Args:
            src_dir (str, optional): Directory containing Python files to analyze
            python_code (str, optional): Python code as a string
            filename (str, optional): Path to a Python file
        """
        self.src_dir = src_dir
        self.files = []

        if src_dir is not None:
            self.src_dir = Path(src_dir)
            self.files = list(self.src_dir.glob("**/*.py"))
            if not self.files:
                print(f"Warning: No Python files found in {src_dir}")
        elif python_code is not None:
            self.code = python_code
            self.source_filename = None
        elif filename is not None:
            with open(filename, "r") as f:
                self.code = f.read()
            self.source_filename = Path(filename).name
        else:
            raise ValueError(
                "Either src_dir, python_code, or filename must be provided"
            )

        self.required_packages = set()
        self.python_version = python_version
        self.package_aliases = {
            "dotenv": "python-dotenv",
            "sklearn": "scikit-learn",
            # Add more aliases as needed
        }

    def analyze(self):
        """
        Analyze Python files to identify imported packages.

        Returns:
            set: Set of required package names
        """
        # Get local module names to exclude them from required packages
        local_module_names = set()
        if hasattr(self, "src_dir") and self.files:
            for file_path in self.files:
                # Extract module name from the file path (without .py extension)
                module_name = file_path.stem
                local_module_names.add(module_name)

                # Also check for package names (directory containing __init__.py)
                parent_dir = file_path.parent
                if (parent_dir / "__init__.py").exists():
                    local_module_names.add(parent_dir.name)

        # Analyze Python files for imports
        if hasattr(self, "src_dir") and self.files:
            # Analyze all Python files in the source directory
            for file_path in self.files:
                try:
                    with open(
                        file_path,
                        "r",
                        encoding="utf-8",
                    ) as f:
                        code = f.read()
                    self._analyze_code(code)
                except Exception as e:
                    print(f"Error analyzing {file_path}: {e}")
        else:
            # Analyze a single code string
            self._analyze_code(self.code)

        # Look for potential Python version requirements
        self._find_python_version()

        # Remove local modules from required packages
        self.required_packages = self.required_packages - local_module_names

        # Apply package aliases
        self.required_packages = {
            self.package_aliases.get(pkg, pkg) for pkg in self.required_packages
        }

        return self.required_packages

    def _analyze_code(self, code):
        """
        Analyze a single Python code string for imports.

        Args:
            code (str): Python code to analyze
        """
        self.required_packages.add("pytest-asyncio")

        try:
            tree = ast.parse(code)

            # Find import statements
            for node in ast.walk(tree):
                # Handle 'import package' statements
                if isinstance(node, ast.Import):
                    for name in node.names:
                        # Extract the base package name (e.g., 'numpy' from 'numpy.random')
                        package = name.name.split(".")[0]
                        if not self._is_standard_library(package):
                            self.required_packages.add(package)

                # Handle 'from package import module' statements
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        # Extract the base package name
                        package = node.module.split(".")[0]
                        if not self._is_standard_library(package):
                            self.required_packages.add(package)
        except SyntaxError as e:
            print(f"Error: Could not parse Python code. Syntax error: {e}")
            return set()

    def _is_standard_library(self, package_name):
        """
        Check if a package is part of the Python standard library.
        This is a simplified check and may not be 100% accurate.

        Args:
            package_name (str): Package name to check

        Returns:
            bool: True if the package is likely part of the standard library
        """
        # List of common standard library modules
        std_libs = [n for n in sys.stdlib_module_names if n[0] != '_'] 
        return package_name in std_libs

    def _find_python_version(self):
        """
        Look for potential Python version requirements in the code.
        If analyzing multiple files, uses the highest version found.
        """
        if self.python_version == "":
            if hasattr(self, "src_dir") and self.files:
                for file_path in self.files:
                    try:
                        with open(
                            file_path,
                            "r",
                            encoding="utf-8",
                        ) as f:
                            code = f.read()
                        self._check_python_version_features(code)
                    except Exception as e:
                        print(f"Error checking Python version in {file_path}: {e}")
            else:
                self._check_python_version_features(self.code)

    def _check_python_version_features(self, code):
        """
        Check a single code string for Python version-specific features.

        Args:
            code (str): Python code to analyze
        """
        # Check for f-strings (Python 3.6+)
        if re.search(r'f["\']', code):
            self.python_version = max(self.python_version, "3.6")

        # Check for type annotations (common in Python 3.7+)
        if re.search(
            r": (?:str|int|float|bool|list|dict|set|tuple)",
            code,
        ):
            self.python_version = max(self.python_version, "3.7")

        # Check for walrus operator (Python 3.8+)
        if re.search(r":=", code):
            self.python_version = max(self.python_version, "3.8")

        # Check for pattern matching (Python 3.10+)
        if re.search(r"match .+:", code) and re.search(r"case .+:", code):
            self.python_version = max(self.python_version, "3.10")

    def generate_dockerfile(self, output_file="Dockerfile"):
        """
        Generate a Dockerfile that installs the required packages and pytest.
        It copies all Python files from the source directory to the Docker container.

        Args:
            output_file (str): Path to output Dockerfile

        Returns:
            str: Content of the generated Dockerfile
        """
        if not self.required_packages:
            self.analyze()

        # Format the package list for requirements.txt
        packages_list = "\n".join(sorted(self.required_packages))

        # Create Dockerfile content
        dockerfile = f"""# Use Python {self.python_version} as the base image
FROM python:{self.python_version}-slim

# Set working directory
WORKDIR /app

# Copy requirements file first (for better caching)
COPY requirements.txt .

# Install dependencies and pytest
RUN pip install --no-cache-dir -r requirements.txt pytest pytest-cov

# Copy all Python files from source directory
COPY {self.src_dir} /app/

# Default command to run tests
CMD ["python", "-m", "pytest","--cov=main", "--cov-report=term-missing"]
"""

        # Write the Dockerfile
        with open(output_file, "w") as f:
            f.write(dockerfile)

        # Write the requirements.txt file
        with open("requirements.txt", "w") as f:
            f.write(packages_list)

        return dockerfile

    def get_required_packages(self):
        """
        Return the list of required packages.

        Returns:
            list: List of required package names
        """
        if not self.required_packages:
            self.analyze()

        return sorted(list(self.required_packages))


if __name__ == "__main__":
    # Set up command line argument parser
    parser = argparse.ArgumentParser(
        description="Analyze Python code and generate a Dockerfile with required packages"
    )
    parser.add_argument(
        "src_dir",
        help="Directory containing Python files to analyze",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="Dockerfile",
        help="Output Dockerfile name (default: Dockerfile)",
    )
    parser.add_argument(
        "-p",
        "--python",
        help="Override Python version (e.g., 3.8)",
    )
    args = parser.parse_args()

    # Create analyzer with source directory
    analyzer = PythonPackageAnalyzer(
        src_dir=args.src_dir,
        python_version=(args.python if args.python else ""),
    )

    # Analyze code and get required packages
    required_packages = analyzer.analyze()
    print(f"Found {len(required_packages)} required packages:")
    for package in sorted(required_packages):
        print(f"  - {package}")

    # Generate Dockerfile
    dockerfile = analyzer.generate_dockerfile(output_file=args.output)
    print(f"\nGenerated {args.output} and requirements.txt")
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

    print("\nTo build the Docker image, run:")
    print(f"  docker build -t my-python-app -f {args.output} .")
