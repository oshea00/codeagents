
import pytest

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.when == "call":
        print(f"TEST: {item.name} - {'PASSED' if report.passed else 'FAILED'}")
    return report
