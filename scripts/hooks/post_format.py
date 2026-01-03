#!/usr/bin/env python3
"""Post-write hook for auto-formatting.

Formats code after Write/Edit tool invocations.
Boris's pattern: "Claude usually generates well-formatted code, and the hook handles the last 10%."
"""

import json
import subprocess
import sys
from pathlib import Path


def format_python(file_path: str) -> bool:
    """Format Python file with black or ruff.

    Tries black first, falls back to ruff if black is not available.
    Returns True if formatting succeeded, False otherwise.
    """
    # Try black first
    try:
        result = subprocess.run(
            ["python3", "-m", "black", "--quiet", file_path],
            capture_output=True,
            timeout=30
        )
        if result.returncode == 0:
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    except Exception as e:
        print(f"black error: {e}", file=sys.stderr)

    # Fall back to ruff
    try:
        result = subprocess.run(
            ["ruff", "format", "--quiet", file_path],
            capture_output=True,
            timeout=30
        )
        if result.returncode == 0:
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    except Exception as e:
        print(f"ruff error: {e}", file=sys.stderr)

    return False


def format_json(file_path: str) -> bool:
    """Format JSON file using python's json.tool.

    Reads the file, formats it, and writes it back.
    Returns True if formatting succeeded, False otherwise.
    """
    try:
        # Read the file
        path = Path(file_path)
        if not path.exists():
            return False

        content = path.read_text()

        # Parse and re-format with indentation
        parsed = json.loads(content)
        formatted = json.dumps(parsed, indent=2, ensure_ascii=False)

        # Add trailing newline for consistency
        if not formatted.endswith("\n"):
            formatted += "\n"

        # Write back
        path.write_text(formatted)
        return True
    except json.JSONDecodeError as e:
        print(f"JSON parse error in {file_path}: {e}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"JSON format error: {e}", file=sys.stderr)
        return False


def format_js_ts(file_path: str) -> bool:
    """Format JS/TS files with prettier.

    Returns True if formatting succeeded, False otherwise.
    """
    try:
        result = subprocess.run(
            ["npx", "prettier", "--write", file_path],
            capture_output=True,
            timeout=60
        )
        if result.returncode == 0:
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    except Exception as e:
        print(f"prettier error: {e}", file=sys.stderr)

    return False


def main():
    """Main entry point for the post-format hook.

    Reads hook input from stdin, determines file type, and runs appropriate formatter.
    Always exits 0 - formatting is best-effort and should never block writes.
    """
    try:
        hook_input = json.loads(sys.stdin.read())
    except json.JSONDecodeError:
        # Invalid input, skip formatting
        sys.exit(0)

    # Extract file path from tool input
    tool_input = hook_input.get("tool_input", {})
    file_path = tool_input.get("file_path", "")

    if not file_path:
        sys.exit(0)

    # Check file exists
    path = Path(file_path)
    if not path.exists():
        sys.exit(0)

    # Format based on extension
    suffix = path.suffix.lower()

    if suffix == ".py":
        format_python(file_path)
    elif suffix == ".json":
        format_json(file_path)
    elif suffix in (".ts", ".tsx", ".js", ".jsx"):
        format_js_ts(file_path)
    # Other file types: skip formatting silently

    # Always succeed - formatting is best-effort
    sys.exit(0)


if __name__ == "__main__":
    main()
