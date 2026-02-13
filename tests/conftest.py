import pytest
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

@pytest.fixture(scope="session")
def test_artifacts_dir(tmp_path_factory):
    """Create temporary artifacts directory for tests"""
    artifacts = tmp_path_factory.mktemp("test_artifacts")
    return str(artifacts)

@pytest.fixture(autouse=True)
def set_test_env(monkeypatch, test_artifacts_dir):
    """Set test environment variables"""
    monkeypatch.setenv("ARTIFACTS_ROOT", test_artifacts_dir)
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")