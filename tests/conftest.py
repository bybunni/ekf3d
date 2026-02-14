import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--runperf",
        action="store_true",
        default=False,
        help="run performance benchmark tests",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if config.getoption("--runperf"):
        return

    skip_perf = pytest.mark.skip(reason="performance tests require --runperf")
    for item in items:
        if "perf" in item.keywords:
            item.add_marker(skip_perf)
