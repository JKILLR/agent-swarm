"""
Pytest configuration and shared fixtures for agent swarm testing.
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Any
import pytest
import yaml

from tests.utils.safety_validator import CodeValidator, RiskScorer
from tests.utils.backup_manager import BackupManager
from tests.utils.performance_tracker import PerformanceTracker


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def swarms_dir(project_root) -> Path:
    """Get the swarms directory."""
    return project_root / "swarms"


@pytest.fixture(scope="session")
def test_data_dir(project_root) -> Path:
    """Get the test data directory."""
    test_dir = project_root / "tests" / "data"
    test_dir.mkdir(parents=True, exist_ok=True)
    return test_dir


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace for testing."""
    temp_dir = tempfile.mkdtemp(prefix="agent_swarm_test_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_swarm_config() -> Dict[str, Any]:
    """Create a mock swarm configuration."""
    return {
        "name": "test_swarm",
        "description": "Test swarm for unit testing",
        "agents": [
            {
                "name": "test_agent",
                "role": "tester",
                "capabilities": ["test", "validate"]
            }
        ]
    }


@pytest.fixture
def code_validator():
    """Provide a code validator instance."""
    return CodeValidator()


@pytest.fixture
def risk_scorer():
    """Provide a risk scorer instance."""
    return RiskScorer()


@pytest.fixture
def backup_manager(temp_workspace):
    """Provide a backup manager instance."""
    backup_dir = temp_workspace / "backups"
    return BackupManager(backup_dir)


@pytest.fixture
def performance_tracker(temp_workspace):
    """Provide a performance tracker instance."""
    metrics_file = temp_workspace / "metrics.json"
    return PerformanceTracker(metrics_file)


@pytest.fixture
def sample_python_code():
    """Provide sample Python code for testing."""
    return '''
class TestAgent:
    """A test agent for validation."""

    def __init__(self, name: str):
        self.name = name

    def execute(self, task: str) -> str:
        """Execute a task."""
        return f"Executing {task}"

    def validate(self) -> bool:
        """Validate the agent."""
        return len(self.name) > 0
'''


@pytest.fixture
def unsafe_python_code():
    """Provide unsafe Python code for testing."""
    return '''
import os
import subprocess

def dangerous_operation():
    """This is unsafe code."""
    os.system("rm -rf /")
    subprocess.call(["sudo", "reboot"])
    eval(user_input)
    exec(malicious_code)
'''


@pytest.fixture
def sample_yaml_config():
    """Provide sample YAML configuration."""
    return {
        "swarm": {
            "name": "test_swarm",
            "version": "1.0.0",
            "agents": [
                {
                    "name": "agent1",
                    "type": "orchestrator"
                },
                {
                    "name": "agent2",
                    "type": "worker"
                }
            ]
        }
    }


@pytest.fixture
def create_test_file(temp_workspace):
    """Factory fixture to create test files."""
    def _create_file(filename: str, content: str) -> Path:
        file_path = temp_workspace / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        return file_path
    return _create_file


@pytest.fixture
def mock_agent_response():
    """Provide a mock agent response."""
    return {
        "status": "success",
        "message": "Task completed successfully",
        "data": {
            "files_modified": ["test.py"],
            "tests_passed": 10,
            "coverage": 85.5
        }
    }


@pytest.fixture(autouse=True)
def reset_environment():
    """Reset environment variables between tests."""
    original_env = os.environ.copy()
    yield
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def mock_swarm_structure(temp_workspace):
    """Create a mock swarm directory structure."""
    swarm_dir = temp_workspace / "swarms" / "test_swarm"
    swarm_dir.mkdir(parents=True)

    # Create swarm.yaml
    config = {
        "name": "test_swarm",
        "agents": ["orchestrator", "worker"]
    }
    (swarm_dir / "swarm.yaml").write_text(yaml.dump(config))

    # Create agents directory
    agents_dir = swarm_dir / "agents"
    agents_dir.mkdir()

    # Create agent files
    (agents_dir / "orchestrator.md").write_text("# Orchestrator Agent")
    (agents_dir / "worker.md").write_text("# Worker Agent")

    return swarm_dir


@pytest.fixture
def capture_metrics(performance_tracker):
    """Context manager for capturing performance metrics."""
    class MetricsCapture:
        def __init__(self, tracker):
            self.tracker = tracker
            self.metrics = []

        def record(self, operation: str, **kwargs):
            metric = self.tracker.start_operation(operation)
            self.metrics.append({
                "operation": operation,
                "metric": metric,
                **kwargs
            })
            return metric

    return MetricsCapture(performance_tracker)
