#!/usr/bin/env python3
"""
Test script to demonstrate PyPI validation functionality.

This script shows how the PyPI validator detects hallucinated or invalid imports
and resolves import names to correct distribution names.
"""

from pypi_validator import PyPIValidator, validate_requirements


def test_pypi_validator():
    """Test the PyPI validator with various import scenarios."""
    print("=" * 80)
    print("PyPI Package Validator Test")
    print("=" * 80)
    print()

    # Enable web search to find information about packages not on PyPI
    validator = PyPIValidator(enable_web_search=True)

    # Test cases with valid and invalid imports
    test_cases = [
        ("numpy", "Standard scientific computing library"),
        ("requests", "HTTP library"),
        ("sklearn", "Alias for scikit-learn"),
        ("dotenv", "Alias for python-dotenv"),
        ("pytest", "Testing framework"),
        ("tavily", "Tavily web search client"),
        ("tavily_websearch", "Possibly hallucinated package"),
        ("this_is_definitely_fake_12345", "Obviously hallucinated package"),
        ("beautifulsoup4", "HTML/XML parser"),
        ("bs4", "Import name for beautifulsoup4"),
        ("tavily-websear5ch", "Import name for tavili web search"),
    ]

    known_aliases = {
        "sklearn": "scikit-learn",
        "dotenv": "python-dotenv",
        "bs4": "beautifulsoup4",
        "tavily": "tavily-python",
    }

    print("Testing individual imports:")
    print("-" * 80)

    for import_name, description in test_cases:
        print(f"\n{import_name:30} ({description})")
        is_valid, dist_name, error_msg = validator.validate_import(
            import_name, known_aliases
        )

        if is_valid:
            print(f"  ✓ VALID: {dist_name}")
            # Get package info
            info = validator.get_package_info(dist_name)
            if info and info.get("summary"):
                print(f"  Summary: {info['summary'][:70]}")
        else:
            print(f"  ✗ INVALID")
            print(f"  Error: {error_msg}")

    print("\n" + "=" * 80)
    print("\nBatch validation test:")
    print("-" * 80)

    imports_to_validate = [
        "numpy",
        "requests",
        "sklearn",
        "fake_package_xyz",
        "pandas",
        "pandas",
        "nonexistent_lib",
        "tavily",
    ]

    valid_reqs, invalid_imports = validate_requirements(
        imports_to_validate, known_aliases
    )

    print("\n✓ Valid requirements for requirements.txt:")
    for req in sorted(valid_reqs):
        print(f"  {req}")

    print("\n✗ Invalid/hallucinated imports:")
    for import_name, error_msg in invalid_imports:
        print(f"  {import_name}")
        print(f"    {error_msg}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    test_pypi_validator()
