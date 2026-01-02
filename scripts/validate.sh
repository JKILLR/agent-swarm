#!/bin/bash
# Validation script - run before committing changes
# Usage: ./scripts/validate.sh [--fix]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "  Agent Swarm Validation"
echo "=========================================="
echo ""

ERRORS=0

# Check if ruff is installed
if ! command -v ruff &> /dev/null; then
    echo -e "${YELLOW}Installing ruff...${NC}"
    pip install ruff --quiet
fi

# 1. Linting with ruff
echo -e "${YELLOW}[1/3] Running linter (ruff)...${NC}"
if [ "$1" = "--fix" ]; then
    echo "  (Auto-fixing issues...)"
    ruff check . --fix || true
    ruff format . || true
fi

if ruff check . --output-format=concise; then
    echo -e "${GREEN}  ✓ Linting passed${NC}"
else
    echo -e "${RED}  ✗ Linting failed${NC}"
    echo "    Run './scripts/validate.sh --fix' to auto-fix issues"
    ERRORS=$((ERRORS + 1))
fi
echo ""

# 2. Format check (if not fixing)
if [ "$1" != "--fix" ]; then
    echo -e "${YELLOW}[2/3] Checking code formatting...${NC}"
    if ruff format . --check --quiet 2>/dev/null; then
        echo -e "${GREEN}  ✓ Formatting check passed${NC}"
    else
        echo -e "${RED}  ✗ Formatting issues found${NC}"
        echo "    Run './scripts/validate.sh --fix' to auto-fix"
        ERRORS=$((ERRORS + 1))
    fi
else
    echo -e "${YELLOW}[2/3] Formatting applied${NC}"
    echo -e "${GREEN}  ✓ Code formatted${NC}"
fi
echo ""

# 3. Run tests
echo -e "${YELLOW}[3/3] Running tests...${NC}"
if python -m pytest tests/ -v --tb=short; then
    echo -e "${GREEN}  ✓ All tests passed${NC}"
else
    echo -e "${RED}  ✗ Tests failed${NC}"
    ERRORS=$((ERRORS + 1))
fi
echo ""

# Summary
echo "=========================================="
if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}  ✓ All validation checks passed!${NC}"
    echo "    Safe to commit."
    exit 0
else
    echo -e "${RED}  ✗ $ERRORS validation check(s) failed${NC}"
    echo "    Please fix issues before committing."
    exit 1
fi
