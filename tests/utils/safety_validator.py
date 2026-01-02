"""
Safety validation utilities for agent-generated code.

Provides AST-based analysis to detect dangerous patterns and validate code safety.
"""

import ast
import re
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum


class RiskLevel(Enum):
    """Risk levels for code operations."""
    SAFE = 0
    LOW = 25
    MEDIUM = 50
    HIGH = 75
    CRITICAL = 100


@dataclass
class SafetyViolation:
    """Represents a safety violation in code."""
    line: int
    column: int
    severity: RiskLevel
    message: str
    code_snippet: str


@dataclass
class ValidationResult:
    """Result of code validation."""
    is_safe: bool
    risk_score: int
    violations: List[SafetyViolation]
    warnings: List[str]


class CodeValidator:
    """Validates Python code for safety concerns."""

    # Dangerous imports that should be flagged
    DANGEROUS_IMPORTS = {
        'os.system', 'subprocess.call', 'subprocess.run', 'subprocess.Popen',
        'eval', 'exec', 'compile', '__import__',
        'pickle.loads', 'pickle.load',  # Arbitrary code execution
        'yaml.load',  # Without safe_load
        'shelve',  # Arbitrary code execution
    }

    # Dangerous patterns (regex)
    DANGEROUS_PATTERNS = [
        (r'rm\s+-rf', 'Destructive file deletion command'),
        (r'os\.system\s*\(', 'Direct system command execution'),
        (r'eval\s*\(', 'Dynamic code evaluation'),
        (r'exec\s*\(', 'Dynamic code execution'),
        (r'__import__\s*\(', 'Dynamic import'),
        (r'open\s*\([^)]*["\']w["\']', 'File write operation'),
        (r'shutil\.rmtree', 'Recursive directory deletion'),
        (r'subprocess\.', 'Subprocess execution'),
    ]

    # File operation patterns that need approval
    FILE_OPERATIONS = {
        'write', 'delete', 'remove', 'unlink', 'rmdir', 'rmtree',
        'move', 'rename', 'chmod', 'chown'
    }

    def __init__(self, max_risk_score: int = 75):
        """
        Initialize validator.

        Args:
            max_risk_score: Maximum acceptable risk score (0-100)
        """
        self.max_risk_score = max_risk_score

    def validate_code(self, code: str) -> ValidationResult:
        """
        Validate Python code for safety.

        Args:
            code: Python source code to validate

        Returns:
            ValidationResult with safety assessment
        """
        violations = []
        warnings = []

        # Check syntax validity
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return ValidationResult(
                is_safe=False,
                risk_score=100,
                violations=[SafetyViolation(
                    line=e.lineno or 0,
                    column=e.offset or 0,
                    severity=RiskLevel.CRITICAL,
                    message=f"Syntax error: {e.msg}",
                    code_snippet=e.text or ""
                )],
                warnings=[]
            )

        # AST-based analysis
        violations.extend(self._check_dangerous_calls(tree, code))
        violations.extend(self._check_dangerous_imports(tree, code))
        violations.extend(self._check_file_operations(tree, code))

        # Pattern-based analysis
        pattern_violations, pattern_warnings = self._check_dangerous_patterns(code)
        violations.extend(pattern_violations)
        warnings.extend(pattern_warnings)

        # Calculate risk score
        risk_score = self._calculate_risk_score(violations)

        return ValidationResult(
            is_safe=risk_score <= self.max_risk_score,
            risk_score=risk_score,
            violations=violations,
            warnings=warnings
        )

    def _check_dangerous_calls(self, tree: ast.AST, code: str) -> List[SafetyViolation]:
        """Check for dangerous function calls."""
        violations = []
        lines = code.split('\n')

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func_name = self._get_function_name(node.func)

                if func_name in ['eval', 'exec', 'compile']:
                    violations.append(SafetyViolation(
                        line=node.lineno,
                        column=node.col_offset,
                        severity=RiskLevel.CRITICAL,
                        message=f"Dangerous function call: {func_name}()",
                        code_snippet=lines[node.lineno - 1] if node.lineno <= len(lines) else ""
                    ))

                elif func_name == '__import__':
                    violations.append(SafetyViolation(
                        line=node.lineno,
                        column=node.col_offset,
                        severity=RiskLevel.HIGH,
                        message="Dynamic import detected",
                        code_snippet=lines[node.lineno - 1] if node.lineno <= len(lines) else ""
                    ))

        return violations

    def _check_dangerous_imports(self, tree: ast.AST, code: str) -> List[SafetyViolation]:
        """Check for dangerous imports."""
        violations = []
        lines = code.split('\n')

        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                module_name = node.module if isinstance(node, ast.ImportFrom) else None

                for alias in node.names:
                    import_path = f"{module_name}.{alias.name}" if module_name else alias.name

                    if any(danger in import_path for danger in ['os', 'subprocess', 'sys']):
                        violations.append(SafetyViolation(
                            line=node.lineno,
                            column=node.col_offset,
                            severity=RiskLevel.MEDIUM,
                            message=f"Potentially dangerous import: {import_path}",
                            code_snippet=lines[node.lineno - 1] if node.lineno <= len(lines) else ""
                        ))

        return violations

    def _check_file_operations(self, tree: ast.AST, code: str) -> List[SafetyViolation]:
        """Check for file operations that need approval."""
        violations = []
        lines = code.split('\n')

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func_name = self._get_function_name(node.func)

                if any(op in func_name.lower() for op in self.FILE_OPERATIONS):
                    violations.append(SafetyViolation(
                        line=node.lineno,
                        column=node.col_offset,
                        severity=RiskLevel.MEDIUM,
                        message=f"File operation requires approval: {func_name}",
                        code_snippet=lines[node.lineno - 1] if node.lineno <= len(lines) else ""
                    ))

        return violations

    def _check_dangerous_patterns(self, code: str) -> Tuple[List[SafetyViolation], List[str]]:
        """Check for dangerous patterns using regex."""
        violations = []
        warnings = []
        lines = code.split('\n')

        for i, line in enumerate(lines, 1):
            for pattern, message in self.DANGEROUS_PATTERNS:
                if re.search(pattern, line, re.IGNORECASE):
                    violations.append(SafetyViolation(
                        line=i,
                        column=0,
                        severity=RiskLevel.HIGH,
                        message=message,
                        code_snippet=line.strip()
                    ))

        return violations, warnings

    def _get_function_name(self, node: ast.AST) -> str:
        """Extract function name from AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            value = self._get_function_name(node.value)
            return f"{value}.{node.attr}" if value else node.attr
        return ""

    def _calculate_risk_score(self, violations: List[SafetyViolation]) -> int:
        """Calculate overall risk score from violations."""
        if not violations:
            return 0

        max_severity = max(v.severity.value for v in violations)
        violation_count = len(violations)

        # Base score on highest severity
        base_score = max_severity

        # Add penalty for multiple violations
        penalty = min(25, violation_count * 5)

        return min(100, base_score + penalty)


class RiskScorer:
    """Scores risk level for different operations."""

    # Critical system files
    CRITICAL_FILES = {
        'swarms/core/', 'swarms/agents/',
        '.git/', '.github/',
        'requirements.txt', 'setup.py', 'pyproject.toml'
    }

    def score_file_operation(self, file_path: str, operation: str) -> int:
        """
        Score risk of a file operation.

        Args:
            file_path: Path to file
            operation: Type of operation (read, write, delete, etc.)

        Returns:
            Risk score (0-100)
        """
        base_scores = {
            'read': 0,
            'write': 25,
            'edit': 25,
            'delete': 75,
            'move': 50,
            'chmod': 50,
        }

        score = base_scores.get(operation.lower(), 50)

        # Increase score for critical files
        if any(critical in file_path for critical in self.CRITICAL_FILES):
            score = min(100, score + 25)

        # Increase score for system files
        if file_path.startswith(('/etc/', '/sys/', '/usr/')):
            score = 100

        return score

    def requires_approval(self, risk_score: int) -> bool:
        """Determine if operation requires human approval."""
        return risk_score >= 50
