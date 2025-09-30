
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
