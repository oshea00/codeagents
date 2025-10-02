"""
PyPI Package Validator

This module validates Python imports against PyPI to detect hallucinated or invalid package names.
It uses the PyPI JSON API to check if a package exists and determine the correct distribution name
for requirements.txt entries.
"""

import requests
import re
import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from functools import lru_cache
from litellm import completion


class TavilySearchHelper:
    """
    Helper class to search the web using Tavily for package information.

    This is used when a package is not found on PyPI to search for references
    and documentation that might indicate the correct package name.
    """

    def __init__(self, model: str = "anthropic/claude-sonnet-4-5-20250929"):
        """
        Initialize the Tavily search helper.

        Args:
            model: LLM model to use for parsing search results
        """
        self.tavily_client = None
        self.model = model

    def _get_websearch_client(self):
        """Get or create Tavily client."""
        if self.tavily_client is None:
            try:
                from tavily import TavilyClient

                api_key = os.getenv("TAVILY_API_KEY")
                if not api_key:
                    print("Warning: TAVILY_API_KEY not found in environment")
                    return None
                self.tavily_client = TavilyClient(api_key=api_key)
            except ImportError:
                print("Warning: tavily-python package not installed")
                return None
        return self.tavily_client

    def _extract_package_candidates_with_llm(
        self, search_response: dict, import_name: str
    ) -> List[str]:
        """
        Use LLM to extract potential PyPI package names from search results.

        Args:
            search_response: The Tavily search response
            import_name: The original import name being searched

        Returns:
            List of candidate package names to validate
        """
        # Format search results for LLM
        search_context = f"Original import name: {import_name}\n\n"

        if search_response.get("answer"):
            search_context += f"AI Answer: {search_response['answer']}\n\n"

        search_context += "Search Results:\n"
        for i, result in enumerate(search_response.get("results", [])[:5], 1):
            search_context += f"\n{i}. Title: {result.get('title', 'N/A')}\n"
            search_context += f"   URL: {result.get('url', 'N/A')}\n"
            search_context += f"   Content: {result.get('content', 'N/A')[:300]}...\n"

        # Create LLM prompt
        prompt = f"""Analyze the following web search results to find PyPI package names related to the import "{import_name}".

{search_context}

Extract a list of potential PyPI package names that could be the correct package for this import.
Look for:
- Packages mentioned in "pip install" commands
- Package names in pypi.org URLs
- Package names mentioned in the content

Return ONLY a JSON object with this structure:
{{
    "candidates": ["package-name-1", "package-name-2", ...],
    "reasoning": "Brief explanation of why these are candidates"
}}

Be conservative - only include packages that are clearly related to the import name.
Return valid JSON only, no markdown formatting."""

        try:
            response = completion(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a Python package expert. Extract PyPI package names from search results and return valid JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=500,
            )

            response_text = response.choices[0].message.content.strip()

            # Try to extract JSON from response
            # Remove markdown code blocks if present
            if "```json" in response_text:
                response_text = re.search(
                    r"```json\s*(.*?)\s*```", response_text, re.DOTALL
                ).group(1)
            elif "```" in response_text:
                response_text = re.search(
                    r"```\s*(.*?)\s*```", response_text, re.DOTALL
                ).group(1)

            result = json.loads(response_text)
            candidates = result.get("candidates", [])
            reasoning = result.get("reasoning", "")

            if reasoning:
                print(f"   LLM reasoning: {reasoning}")

            return candidates

        except Exception as e:
            print(f"   Warning: LLM extraction failed: {e}")
            # Fallback to regex extraction
            return self._extract_package_candidates_regex(search_response, import_name)

    def _extract_package_candidates_regex(
        self, search_response: dict, import_name: str
    ) -> List[str]:
        """
        Fallback regex-based extraction of package names.

        Args:
            search_response: The Tavily search response
            import_name: The original import name being searched

        Returns:
            List of candidate package names
        """
        candidates = []

        # Extract from answer summary
        answer = search_response.get("answer", "")
        if answer:
            pip_matches = re.findall(r"pip install ([a-zA-Z0-9_-]+)", answer)
            candidates.extend(pip_matches)

        # Extract from URLs (pypi.org/project/<package>)
        for result in search_response.get("results", []):
            url = result.get("url", "")
            pypi_match = re.search(r"pypi\.org/project/([a-zA-Z0-9_-]+)", url)
            if pypi_match:
                candidates.append(pypi_match.group(1))

            content = result.get("content", "")
            pip_matches = re.findall(r"pip install ([a-zA-Z0-9_-]+)", content)
            candidates.extend(pip_matches)

        # Remove duplicates while preserving order
        seen = set()
        unique_candidates = []
        for candidate in candidates:
            if candidate not in seen:
                seen.add(candidate)
                unique_candidates.append(candidate)

        return unique_candidates

    def search_and_validate_package(
        self, import_name: str, max_results: int = 3
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Search the web for package information and validate candidates against PyPI.

        Args:
            import_name: The import name to search for
            max_results: Maximum number of search results to return

        Returns:
            Tuple of (valid_package_name, formatted_search_results)
            - valid_package_name: PyPI package name if found, None otherwise
            - formatted_search_results: Human-readable search summary
        """
        try:
            client = self._get_websearch_client()
            if client is None:
                return None, None

            # Construct search query
            query = f"Python package PyPI import {import_name} pip install"

            # Perform search
            response = client.search(
                query=query,
                max_results=max_results,
                search_depth="basic",
                include_answer=True,
            )

            # Extract and validate candidates using LLM
            candidates = self._extract_package_candidates_with_llm(
                response, import_name
            )

            # Try to validate each candidate against PyPI
            valid_package = None
            for candidate in candidates:
                # Use a simple direct check (not using the validator to avoid recursion)
                try:
                    url = f"https://pypi.org/pypi/{candidate}/json"
                    check_response = requests.get(url, timeout=5)
                    if check_response.status_code == 200:
                        valid_package = candidate
                        break
                except:
                    continue

            # Format results
            result_text = f"# Web Search Results for: {import_name}\n\n"

            if valid_package:
                result_text += f"## ✓ Found Valid Package\n"
                result_text += f"The correct package name is: **{valid_package}**\n"
                result_text += f"Install with: `pip install {valid_package}`\n\n"

            if response.get("answer"):
                result_text += f"## AI Summary\n{response['answer']}\n\n"

            if candidates:
                result_text += f"## Candidates Found\n"
                for candidate in candidates:
                    status = "✓ Valid" if candidate == valid_package else "○ Checked"
                    result_text += f"- {candidate} ({status})\n"
                result_text += "\n"

            result_text += (
                f"## Search Sources ({len(response.get('results', []))} results)\n"
            )
            for i, result in enumerate(response.get("results", [])[:3], 1):
                result_text += f"{i}. [{result.get('title', 'No title')}]({result.get('url', 'N/A')})\n"

            return valid_package, result_text

        except Exception as e:
            print(f"Warning: Web search failed for '{import_name}': {e}")
            return None, None


class PyPIValidator:
    """
    Validates Python import names against PyPI and resolves them to distribution names.

    This helps detect hallucinated packages and ensures requirements.txt entries are correct.
    """

    def __init__(
        self,
        enable_web_search: bool = False,
        model: str = "anthropic/claude-sonnet-4-5-20250929",
        cache_file: Optional[str] = None,
    ):
        self.base_url = "https://pypi.org/pypi"
        self.timeout = 5  # seconds
        # Cache for mapping import names to distribution names
        self.import_to_dist_cache: Dict[str, Optional[str]] = {}
        self.enable_web_search = enable_web_search
        self.web_search_helper = (
            TavilySearchHelper(model=model) if enable_web_search else None
        )

        # Set up persistent cache file
        if cache_file is None:
            cache_file = Path.home() / ".pypi_validator_cache.json"
        self.cache_file = Path(cache_file)
        self._load_cache()

    def _load_cache(self):
        """Load the persistent cache from disk if it exists."""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    cache_data = json.load(f)
                    # Validate cache format
                    if isinstance(cache_data, dict):
                        self.import_to_dist_cache = cache_data
                        print(
                            f"Loaded {len(cache_data)} cached entries from {self.cache_file}"
                        )
                    else:
                        print(
                            f"Warning: Invalid cache format in {self.cache_file}, starting fresh"
                        )
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load cache from {self.cache_file}: {e}")
            # Continue with empty cache

    def _save_cache(self):
        """Save the current cache to disk."""
        try:
            # Create directory if it doesn't exist
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)

            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.import_to_dist_cache, f, indent=2, sort_keys=True)
        except IOError as e:
            print(f"Warning: Could not save cache to {self.cache_file}: {e}")

    @lru_cache(maxsize=256)
    def check_distribution_exists(
        self, distribution_name: str
    ) -> Tuple[bool, Optional[Dict]]:
        """
        Check if a distribution exists on PyPI.

        Args:
            distribution_name: The distribution name to check (e.g., 'tavily-websearch')

        Returns:
            Tuple of (exists, metadata_dict or None)
        """
        try:
            url = f"{self.base_url}/{distribution_name}/json"
            response = requests.get(url, timeout=self.timeout)

            if response.status_code == 200:
                return True, response.json()
            elif response.status_code == 404:
                return False, None
            else:
                # Unexpected status code, treat as not found
                return False, None

        except requests.RequestException as e:
            # Network error or timeout - treat as unknown
            print(f"Warning: Could not check PyPI for '{distribution_name}': {e}")
            return False, None

    def generate_candidate_names(self, import_name: str) -> List[str]:
        """
        Generate candidate distribution names from an import name.

        Examples:
            'tavily_websearch' -> ['tavily_websearch', 'tavily-websearch', 'tavilywebsearch']
            'sklearn' -> ['sklearn', 'scikit-learn']

        Args:
            import_name: The import name from the code

        Returns:
            List of candidate distribution names to try
        """
        candidates = [import_name]

        # Convert underscores to hyphens
        if "_" in import_name:
            candidates.append(import_name.replace("_", "-"))
            # Also try without any separators
            candidates.append(import_name.replace("_", ""))

        # Convert hyphens to underscores
        if "-" in import_name:
            candidates.append(import_name.replace("-", "_"))
            # Also try without any separators
            candidates.append(import_name.replace("-", ""))

        # Try common variations for dots
        if "." in import_name:
            base = import_name.split(".")[0]
            candidates.append(base)
            candidates.append(base.replace(".", "-"))
            candidates.append(base.replace(".", "_"))

        # Remove duplicates while preserving order
        seen = set()
        unique_candidates = []
        for candidate in candidates:
            if candidate not in seen:
                seen.add(candidate)
                unique_candidates.append(candidate)

        return unique_candidates

    def validate_import(
        self, import_name: str, known_aliases: Dict[str, str] = None
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Validate an import name against PyPI and return the distribution name if found.

        Args:
            import_name: The import name from Python code (e.g., 'sklearn', 'tavily_websearch')
            known_aliases: Optional dict mapping import names to distribution names

        Returns:
            Tuple of (is_valid, distribution_name, error_message)
            - is_valid: True if the package was found on PyPI
            - distribution_name: The correct distribution name for requirements.txt, or None
            - error_message: Description of the issue if not valid, or None
        """
        # Check cache first
        if import_name in self.import_to_dist_cache:
            cached = self.import_to_dist_cache[import_name]
            if cached is not None:
                return True, cached, None
            else:
                return (
                    False,
                    None,
                    f"Import '{import_name}' not found on PyPI (cached result)",
                )

        # Check known aliases first
        if known_aliases and import_name in known_aliases:
            dist_name = known_aliases[import_name]
            exists, _ = self.check_distribution_exists(dist_name)
            if exists:
                self.import_to_dist_cache[import_name] = dist_name
                self._save_cache()
                return True, dist_name, None

        # Generate and try candidate distribution names
        candidates = self.generate_candidate_names(import_name)

        for candidate in candidates:
            exists, _ = self.check_distribution_exists(candidate)
            if exists:
                # Found it! Cache and return
                self.import_to_dist_cache[import_name] = candidate
                self._save_cache()
                return True, candidate, None

        # Not found on PyPI - try web search if enabled
        if self.enable_web_search and self.web_search_helper:
            print(f"Searching web for information about '{import_name}'...")
            valid_package, search_results = (
                self.web_search_helper.search_and_validate_package(import_name)
            )

            if valid_package:
                # Found a valid package through web search!
                print(f"✓ Web search found valid package: {valid_package}")
                self.import_to_dist_cache[import_name] = valid_package
                self._save_cache()
                return True, valid_package, None
            elif search_results:
                # Web search completed but no valid package found
                self.import_to_dist_cache[import_name] = None
                self._save_cache()
                error_msg = (
                    f"Import '{import_name}' not found on PyPI. Tried: {', '.join(candidates)}\n"
                    f"\nWeb search results:\n{search_results}"
                )
                return False, None, error_msg

        # Not found on PyPI
        self.import_to_dist_cache[import_name] = None
        self._save_cache()
        error_msg = (
            f"Import '{import_name}' not found on PyPI. Tried: {', '.join(candidates)}"
        )
        return False, None, error_msg

    def get_package_info(self, distribution_name: str) -> Optional[Dict]:
        """
        Get detailed package information from PyPI.

        Args:
            distribution_name: The distribution name on PyPI

        Returns:
            Dictionary with package info or None if not found
        """
        exists, metadata = self.check_distribution_exists(distribution_name)
        if not exists or not metadata:
            return None

        info = metadata.get("info", {})
        return {
            "name": info.get("name"),
            "version": info.get("version"),
            "summary": info.get("summary"),
            "description": info.get("description", "")[
                :200
            ],  # Truncate long descriptions
            "home_page": info.get("home_page"),
            "author": info.get("author"),
            "requires_python": info.get("requires_python"),
            "keywords": info.get("keywords", ""),
        }

    def validate_imports_batch(
        self, import_names: List[str], known_aliases: Dict[str, str] = None
    ) -> Dict[str, Tuple[bool, Optional[str], Optional[str]]]:
        """
        Validate multiple imports at once.

        Args:
            import_names: List of import names to validate
            known_aliases: Optional dict mapping import names to distribution names

        Returns:
            Dictionary mapping import_name -> (is_valid, distribution_name, error_message)
        """
        results = {}
        for import_name in import_names:
            results[import_name] = self.validate_import(import_name, known_aliases)
        return results


def validate_requirements(
    import_names: List[str],
    known_aliases: Dict[str, str] = None,
    cache_file: Optional[str] = None,
) -> Tuple[List[str], List[Tuple[str, str]]]:
    """
    Convenience function to validate a list of imports and generate requirements.txt entries.

    Args:
        import_names: List of import names from Python code
        known_aliases: Optional dict mapping import names to distribution names
        cache_file: Optional path to cache file for persistent storage

    Returns:
        Tuple of (valid_requirements, invalid_imports)
        - valid_requirements: List of distribution names for requirements.txt
        - invalid_imports: List of (import_name, error_message) tuples
    """
    validator = PyPIValidator(cache_file=cache_file)
    valid_requirements = []
    invalid_imports = []

    for import_name in import_names:
        is_valid, dist_name, error_msg = validator.validate_import(
            import_name, known_aliases
        )
        if is_valid and dist_name:
            valid_requirements.append(dist_name)
        else:
            invalid_imports.append((import_name, error_msg or "Unknown error"))

    return valid_requirements, invalid_imports


if __name__ == "__main__":
    # Example usage
    validator = PyPIValidator()

    test_imports = [
        "numpy",
        "sklearn",  # Should resolve to scikit-learn
        "tavily_websearch",  # Might be tavily-websearch or hallucinated
        "dotenv",  # Should resolve to python-dotenv
        "requests",
        "this_package_definitely_does_not_exist_12345",
    ]

    known_aliases = {
        "sklearn": "scikit-learn",
        "dotenv": "python-dotenv",
    }

    print("Validating imports against PyPI...\n")

    for import_name in test_imports:
        is_valid, dist_name, error_msg = validator.validate_import(
            import_name, known_aliases
        )
        if is_valid:
            print(f"✓ {import_name:30} -> {dist_name}")
            # Show package info
            info = validator.get_package_info(dist_name)
            if info and info.get("summary"):
                print(f"  {info['summary'][:80]}")
        else:
            print(f"✗ {import_name:30} -> NOT FOUND")
            print(f"  {error_msg}")
        print()
