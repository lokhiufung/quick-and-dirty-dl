import pytest


def pytest_addoption(parser):
    parser.addoption(
        '--manifest_file', action='store', help="testing data manifest.json"
    )


@pytest.fixture
def manifest_file(request):
    return request.config.getoption('--manifest_file')
    