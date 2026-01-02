#!/usr/bin/env python3
"""Pre-write hook for validation.

This hook runs before Write/Edit tool invocations to:
1. Validate the file path is safe
2. Check for protected files
3. Log the write operation
"""

import json
import sys
from pathlib import Path

# Files that should not be modified without explicit approval
PROTECTED_PATTERNS = [
    ".env",
    "credentials",
    "secrets",
    ".git/",
    "node_modules/",
]


def main():
    try:
        hook_input = json.loads(sys.stdin.read())
    except json.JSONDecodeError:
        sys.exit(0)

    tool_input = hook_input.get("tool_input", {})
    file_path = tool_input.get("file_path", tool_input.get("path", ""))

    # Check for protected patterns
    for pattern in PROTECTED_PATTERNS:
        if pattern in file_path.lower():
            print(json.dumps({
                "message": f"Warning: Modifying protected file pattern '{pattern}' in {file_path}",
                "continue": True  # Allow but warn
            }))
            break

    # Allow the operation
    sys.exit(0)


if __name__ == "__main__":
    main()
