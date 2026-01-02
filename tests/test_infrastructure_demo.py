"""
Demonstration test to validate testing infrastructure deployment.
"""

import pytest
from tests.utils.safety_validator import CodeValidator, RiskLevel
from tests.utils.backup_manager import BackupManager
from tests.utils.performance_tracker import PerformanceTracker


@pytest.mark.unit
def test_safety_validator_safe_code():
    """Test that safe code is validated correctly."""
    validator = CodeValidator()

    safe_code = '''
def hello_world():
    """A safe function."""
    return "Hello, World!"
'''

    result = validator.validate_code(safe_code)
    assert result.is_safe is True
    assert result.risk_score == 0


@pytest.mark.unit
def test_safety_validator_dangerous_code():
    """Test that dangerous code is detected."""
    validator = CodeValidator()

    dangerous_code = '''
import os
os.system("rm -rf /")
'''

    result = validator.validate_code(dangerous_code)
    assert result.is_safe is False
    assert len(result.violations) > 0


@pytest.mark.unit
def test_backup_manager(temp_workspace, create_test_file):
    """Test backup manager functionality."""
    backup_mgr = BackupManager(temp_workspace / "backups")

    # Create a test file
    test_file = create_test_file("test.txt", "original content")

    # Create backup
    backup_path = backup_mgr.create_backup(test_file, operation="modify")
    assert backup_path is not None
    assert backup_path.exists()

    # Modify original
    test_file.write_text("modified content")

    # Restore from backup
    success = backup_mgr.restore_backup(test_file)
    assert success is True
    assert test_file.read_text() == "original content"


@pytest.mark.unit
def test_performance_tracker(temp_workspace):
    """Test performance tracker functionality."""
    tracker = PerformanceTracker(temp_workspace / "metrics.json")

    # Track an operation
    with tracker.track_operation("test_operation", test_param="value"):
        # Simulate work
        pass

    # Check stats
    stats = tracker.get_stats("test_operation")
    assert "test_operation" in stats
    assert stats["test_operation"].total_calls == 1
    assert stats["test_operation"].successful_calls == 1


@pytest.mark.integration
def test_full_workflow(temp_workspace, create_test_file):
    """Test complete workflow with all utilities."""
    validator = CodeValidator()
    backup_mgr = BackupManager(temp_workspace / "backups")
    tracker = PerformanceTracker(temp_workspace / "metrics.json")

    # Generate code
    code = '''
def calculate_sum(a: int, b: int) -> int:
    """Calculate sum of two numbers."""
    return a + b
'''

    with tracker.track_operation("code_validation"):
        # Validate code
        result = validator.validate_code(code)
        assert result.is_safe is True

    # Create file and backup
    code_file = create_test_file("calculator.py", code)
    backup_path = backup_mgr.create_backup(code_file)
    assert backup_path is not None

    # Check performance stats
    stats = tracker.get_stats("code_validation")
    assert stats["code_validation"].total_calls == 1
    assert stats["code_validation"].successful_calls == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
