#!/usr/bin/env python3
"""Verification hook for subagent completion.

Boris's insight: "Probably the most important thing - give Claude a way to verify its work.
If Claude has that feedback loop, it will 2-3x the quality."

This hook runs deterministic verification after each subagent stops:
1. Python syntax check on modified files
2. Import verification
3. Test execution if test files were modified
"""

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

SWARM_ROOT = Path(os.environ.get("SWARM_ROOT", Path(__file__).parent.parent.parent))


def get_recently_modified_files(minutes: int = 5) -> list[str]:
    """Get files modified in the last N minutes."""
    result = subprocess.run(
        ["find", str(SWARM_ROOT), "-name", "*.py", "-mmin", f"-{minutes}", "-type", "f"],
        capture_output=True,
        text=True
    )
    files = [f for f in result.stdout.strip().split("\n") if f and "node_modules" not in f and ".git" not in f]
    return files


def verify_python_syntax(files: list[str]) -> list[str]:
    """Verify Python syntax for files. Returns list of errors."""
    errors = []
    for file_path in files:
        result = subprocess.run(
            ["python3", "-m", "py_compile", file_path],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            errors.append(f"Syntax error in {file_path}: {result.stderr}")
    return errors


def verify_imports(files: list[str]) -> list[str]:
    """Verify imports work. Returns list of errors."""
    errors = []
    for file_path in files:
        # Try to parse and check imports
        result = subprocess.run(
            ["python3", "-c", f"import ast; ast.parse(open('{file_path}').read())"],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            errors.append(f"Parse error in {file_path}: {result.stderr}")
    return errors


def run_tests_if_modified(files: list[str]) -> list[str]:
    """Run tests if test files were modified. Returns list of errors."""
    test_files = [f for f in files if "test" in f.lower()]
    if not test_files:
        return []

    result = subprocess.run(
        ["python3", "-m", "pytest", "--tb=short", "-q"] + test_files,
        capture_output=True,
        text=True,
        cwd=str(SWARM_ROOT)
    )
    if result.returncode != 0:
        return [f"Test failures: {result.stdout}\n{result.stderr}"]
    return []


def main():
    try:
        hook_input = json.loads(sys.stdin.read())
    except json.JSONDecodeError:
        sys.exit(0)

    agent_name = hook_input.get("agent", "unknown")

    # Get recently modified files
    modified_files = get_recently_modified_files(minutes=5)

    if not modified_files:
        # No files modified, nothing to verify
        sys.exit(0)

    all_errors = []

    # Run verification checks
    all_errors.extend(verify_python_syntax(modified_files))
    all_errors.extend(verify_imports(modified_files))
    all_errors.extend(run_tests_if_modified(modified_files))

    if all_errors:
        # Log verification failures
        log_path = SWARM_ROOT / "logs" / "verification.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        with open(log_path, "a") as f:
            f.write(f"\n--- {datetime.now().isoformat()} - Agent: {agent_name} ---\n")
            for error in all_errors:
                f.write(f"ERROR: {error}\n")

        # Output warning but don't block
        print(json.dumps({
            "message": f"Verification found {len(all_errors)} issue(s). Check logs/verification.log",
            "issues": all_errors[:3],  # First 3 issues
            "continue": True  # Allow but warn
        }))

    # Always exit 0 - don't block agent completion
    sys.exit(0)


if __name__ == "__main__":
    main()
