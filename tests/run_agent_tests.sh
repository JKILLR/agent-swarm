#!/bin/bash
#
# Agent Independence Test Runner
#
# This script runs the agent independence tests to verify that:
# 1. The COO properly delegates to agents
# 2. Agents execute independently
# 3. Tool execution is properly attributed
# 4. The system handles errors gracefully
#
# Usage:
#   ./run_agent_tests.sh [options]
#
# Options:
#   --unit       Run unit tests only
#   --live       Run live integration tests only
#   --all        Run all tests (default)
#   --verbose    Verbose output
#   --coverage   Generate coverage report
#   -h, --help   Show this help message
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default settings
RUN_UNIT=true
RUN_LIVE=false
VERBOSE=""
COVERAGE=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --unit)
            RUN_UNIT=true
            RUN_LIVE=false
            shift
            ;;
        --live)
            RUN_UNIT=false
            RUN_LIVE=true
            shift
            ;;
        --all)
            RUN_UNIT=true
            RUN_LIVE=true
            shift
            ;;
        --verbose|-v)
            VERBOSE="-v"
            shift
            ;;
        --coverage)
            COVERAGE="--cov=backend --cov-report=html --cov-report=term"
            shift
            ;;
        -h|--help)
            echo "Agent Independence Test Runner"
            echo ""
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --unit       Run unit tests only"
            echo "  --live       Run live integration tests only"
            echo "  --all        Run all tests (default)"
            echo "  --verbose    Verbose output"
            echo "  --coverage   Generate coverage report"
            echo "  -h, --help   Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}  Agent Independence Test Suite${NC}"
echo -e "${BLUE}======================================${NC}"
echo ""

# Change to project root
cd "$PROJECT_ROOT"

# Check for pytest
if ! command -v pytest &> /dev/null; then
    echo -e "${RED}Error: pytest not found. Install with: pip install pytest${NC}"
    exit 1
fi

# Check for backend dependencies
echo -e "${YELLOW}Checking dependencies...${NC}"
if ! python -c "import fastapi" 2>/dev/null; then
    echo -e "${RED}Warning: FastAPI not installed. Some tests may fail.${NC}"
fi

if ! python -c "import anthropic" 2>/dev/null; then
    echo -e "${YELLOW}Note: Anthropic SDK not installed. API tests will use CLI fallback.${NC}"
fi

echo ""

# Run unit tests
if [ "$RUN_UNIT" = true ]; then
    echo -e "${GREEN}Running Unit Tests...${NC}"
    echo "-----------------------------------------"

    pytest tests/test_agent_independence.py $VERBOSE $COVERAGE --tb=short \
        -x --ignore=tests/test_live_agents.py 2>&1 || {
        echo -e "${RED}Unit tests failed!${NC}"
        UNIT_RESULT=1
    }
    UNIT_RESULT=${UNIT_RESULT:-0}

    echo ""
fi

# Run live tests
if [ "$RUN_LIVE" = true ]; then
    echo -e "${GREEN}Running Live Integration Tests...${NC}"
    echo "-----------------------------------------"

    # Check if backend is running
    if ! curl -s http://localhost:8000/api/status > /dev/null 2>&1; then
        echo -e "${YELLOW}Warning: Backend not running. Starting check...${NC}"
        echo -e "${YELLOW}Please start backend with: ./run.sh${NC}"
        echo -e "${YELLOW}Skipping live tests.${NC}"
        LIVE_RESULT=2  # Skipped
    else
        pytest tests/test_live_agents.py $VERBOSE --tb=short -x 2>&1 || {
            echo -e "${RED}Live tests failed!${NC}"
            LIVE_RESULT=1
        }
        LIVE_RESULT=${LIVE_RESULT:-0}
    fi

    echo ""
fi

# Summary
echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}  Test Summary${NC}"
echo -e "${BLUE}======================================${NC}"
echo ""

if [ "$RUN_UNIT" = true ]; then
    if [ "${UNIT_RESULT:-0}" -eq 0 ]; then
        echo -e "Unit Tests:        ${GREEN}PASSED${NC}"
    else
        echo -e "Unit Tests:        ${RED}FAILED${NC}"
    fi
fi

if [ "$RUN_LIVE" = true ]; then
    if [ "${LIVE_RESULT:-0}" -eq 0 ]; then
        echo -e "Live Tests:        ${GREEN}PASSED${NC}"
    elif [ "${LIVE_RESULT:-0}" -eq 2 ]; then
        echo -e "Live Tests:        ${YELLOW}SKIPPED${NC}"
    else
        echo -e "Live Tests:        ${RED}FAILED${NC}"
    fi
fi

echo ""

# Coverage report location
if [ -n "$COVERAGE" ]; then
    echo -e "${BLUE}Coverage report: htmlcov/index.html${NC}"
    echo ""
fi

# Overall result
if [ "${UNIT_RESULT:-0}" -eq 0 ] && [ "${LIVE_RESULT:-0}" -ne 1 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed.${NC}"
    exit 1
fi
