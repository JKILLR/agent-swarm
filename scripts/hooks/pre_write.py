#!/usr/bin/env python3
"""Pre-write hook for validation and file organization enforcement.

This hook runs before Write/Edit tool invocations to:
1. Validate file path is safe (no protected files)
2. Enforce file organization standards
3. Warn about misplaced files
4. Remind about STATE.md updates

Operations Integration: This hook enforces swarm workspace standards.
"""

import json
import re
import sys
from pathlib import Path

# Files that should not be modified without explicit approval
PROTECTED_PATTERNS = [
    ".env",
    "credentials",
    "secrets",
    ".git/",
    "node_modules/",
    "__pycache__/",
]

# Valid top-level directories (files outside these get warnings)
VALID_TOP_LEVEL = [
    "backend",
    "frontend",
    "scripts",
    "swarms",
    "workspace",  # Project-level workspace
    "shared",
    "logs",
    "tests",
    ".claude",
    "docs",
]

# Files allowed at project root
ALLOWED_ROOT_FILES = [
    "README.md",
    "LICENSE",
    "package.json",
    "package-lock.json",
    "tsconfig.json",
    "pyproject.toml",
    "requirements.txt",
    ".gitignore",
    ".env.example",
    "Makefile",
    "docker-compose.yml",
    "Dockerfile",
]

# Swarm workspace required structure
SWARM_WORKSPACE_DIRS = [
    "sessions",
    "research",
    "decisions",
    "archive",
]


def validate_swarm_path(file_path: str, path: Path) -> list[str]:
    """Validate files within swarms/ follow organization standards."""
    warnings = []
    parts = path.parts

    # Find swarm name (swarms/{swarm_name}/...)
    if len(parts) < 2:
        return warnings

    swarm_name = parts[1]  # e.g., "swarm_dev", "operations"

    # Skip template
    if swarm_name == "_template":
        return warnings

    # Check if file is in workspace
    if len(parts) >= 3 and parts[2] == "workspace":
        # Good - file is in workspace
        # Check if it's organized in subdirectories for larger files
        if len(parts) == 4 and parts[3] not in ["STATE.md", "README.md", "CHANGELOG.md"]:
            # File directly in workspace/ but not a standard file
            filename = parts[3]
            if not filename.startswith("."):
                warnings.append(
                    f"Consider organizing '{filename}' into a subdirectory "
                    f"(sessions/, research/, decisions/, archive/)"
                )

    # Check if file is in agents/
    elif len(parts) >= 3 and parts[2] == "agents":
        # Good - agent definitions belong here
        pass

    # Check if file is in protocols/ (for operations)
    elif len(parts) >= 3 and parts[2] == "protocols":
        # Good - protocol docs belong here
        pass

    # File not in standard location
    elif len(parts) >= 3:
        location = parts[2]
        if location not in ["workspace", "agents", "protocols", "swarm.yaml"]:
            warnings.append(
                f"File in non-standard swarm location '{location}'. "
                f"Swarm files should be in: workspace/, agents/, or protocols/"
            )

    return warnings


def validate_naming(file_path: str, path: Path) -> list[str]:
    """Validate file naming conventions."""
    warnings = []
    filename = path.name
    suffix = path.suffix.lower()
    stem = path.stem

    # Python files should be snake_case
    if suffix == ".py":
        if not re.match(r'^[a-z][a-z0-9_]*$', stem) and stem != "__init__":
            warnings.append(
                f"Python file '{filename}' should use snake_case naming"
            )

    # Markdown files can be UPPER_CASE or Title Case
    # No validation needed for .md

    # TypeScript/JavaScript - camelCase or PascalCase for components
    if suffix in [".ts", ".tsx", ".js", ".jsx"]:
        # Allow various conventions for JS/TS
        pass

    return warnings


def check_state_md_reminder(file_path: str, path: Path) -> str | None:
    """Remind to update STATE.md when modifying workspace files."""
    parts = path.parts

    # Check if this is a swarm workspace file (not STATE.md itself)
    if "swarms" in parts and "workspace" in parts:
        if path.name != "STATE.md":
            swarm_idx = parts.index("swarms")
            if swarm_idx + 1 < len(parts):
                swarm_name = parts[swarm_idx + 1]
                return (
                    f"Remember to update swarms/{swarm_name}/workspace/STATE.md "
                    f"with this change"
                )

    return None


def validate_root_file(file_path: str, path: Path) -> list[str]:
    """Validate files at project root."""
    warnings = []

    # Get path relative to project
    # If path has only 1 part or first part is a file, it's at root
    if len(path.parts) == 1:
        filename = path.parts[0]
        if filename not in ALLOWED_ROOT_FILES and not filename.startswith("."):
            warnings.append(
                f"File '{filename}' at project root. "
                f"Consider moving to appropriate directory: {', '.join(VALID_TOP_LEVEL[:5])}"
            )

    return warnings


def validate_file_organization(file_path: str) -> tuple[list[str], list[str]]:
    """
    Validate file follows organization standards.

    Returns:
        Tuple of (errors, warnings)
        - errors: Issues that should block the write
        - warnings: Issues to inform the user about
    """
    errors = []
    warnings = []

    # Normalize path
    path = Path(file_path)

    # Make path relative if absolute
    try:
        # Try to make relative to common roots
        if path.is_absolute():
            for root in ["/home/user/agent-swarm", Path.cwd()]:
                try:
                    path = path.relative_to(root)
                    break
                except ValueError:
                    continue
    except Exception:
        pass

    parts = path.parts
    if not parts:
        return errors, warnings

    first_part = parts[0]

    # Check if in valid top-level directory
    if first_part not in VALID_TOP_LEVEL:
        # Could be a root file
        if len(parts) == 1:
            warnings.extend(validate_root_file(file_path, path))
        else:
            warnings.append(
                f"File in unrecognized directory '{first_part}'. "
                f"Standard directories: {', '.join(VALID_TOP_LEVEL[:5])}"
            )

    # Swarm-specific validation
    if first_part == "swarms":
        warnings.extend(validate_swarm_path(file_path, path))

    # Naming convention validation
    warnings.extend(validate_naming(file_path, path))

    # STATE.md reminder
    reminder = check_state_md_reminder(file_path, path)
    if reminder:
        warnings.append(reminder)

    return errors, warnings


def main():
    try:
        hook_input = json.loads(sys.stdin.read())
    except json.JSONDecodeError:
        sys.exit(0)

    tool_input = hook_input.get("tool_input", {})
    file_path = tool_input.get("file_path", tool_input.get("path", ""))

    if not file_path:
        sys.exit(0)

    messages = []
    should_block = False

    # Check for protected patterns
    for pattern in PROTECTED_PATTERNS:
        if pattern in file_path.lower():
            messages.append(f"⚠️  Protected file pattern '{pattern}' detected")
            break

    # Validate file organization
    errors, warnings = validate_file_organization(file_path)

    # Errors block the operation
    if errors:
        for error in errors:
            messages.append(f"❌ {error}")
        should_block = True

    # Warnings inform but don't block
    for warning in warnings:
        messages.append(f"📁 {warning}")

    # Output result
    if messages:
        print(json.dumps({
            "message": "\n".join(messages),
            "continue": not should_block
        }))

    sys.exit(0 if not should_block else 1)


if __name__ == "__main__":
    main()
