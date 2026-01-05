#!/bin/bash
# Run both backend and frontend for Agent Swarm web interface

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo -e "${GREEN}Starting Agent Swarm Web Interface${NC}"
echo ""

# Check if we're in the right directory
if [ ! -f "$SCRIPT_DIR/main.py" ]; then
    echo -e "${RED}Error: main.py not found in $SCRIPT_DIR${NC}"
    exit 1
fi

# Find Python 3.10+ (required for type union syntax)
PYTHON=""
for py in python3.12 python3.11 python3.10; do
    if command -v $py &> /dev/null; then
        PYTHON=$py
        break
    fi
done

if [ -z "$PYTHON" ]; then
    # Check if default python3 is 3.10+
    if command -v python3 &> /dev/null; then
        PY_VERSION=$(python3 -c 'import sys; print(sys.version_info.minor)')
        if [ "$PY_VERSION" -ge 10 ]; then
            PYTHON=python3
        fi
    fi
fi

if [ -z "$PYTHON" ]; then
    echo -e "${RED}Error: Python 3.10+ is required but not found${NC}"
    echo -e "${YELLOW}Please install Python 3.10 or later${NC}"
    exit 1
fi

echo -e "${GREEN}Using Python: $PYTHON${NC}"

# Function to cleanup background processes
cleanup() {
    echo ""
    echo -e "${YELLOW}Shutting down...${NC}"
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    exit 0
}

trap cleanup SIGINT SIGTERM

# Start backend
echo -e "${GREEN}Starting backend on http://localhost:8000${NC}"
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"
cd "$SCRIPT_DIR/backend"
$PYTHON -m uvicorn main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Wait a bit for backend to start
sleep 2

# Start frontend
echo -e "${GREEN}Starting frontend on http://localhost:3000${NC}"
cd "$SCRIPT_DIR/frontend"
npm run dev -- -H 0.0.0.0 &
FRONTEND_PID=$!

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Agent Swarm Web Interface Running${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "  Dashboard: ${YELLOW}http://localhost:3000${NC}"
echo -e "  API:       ${YELLOW}http://localhost:8000${NC}"
echo -e "  API Docs:  ${YELLOW}http://localhost:8000/docs${NC}"
echo ""
echo -e "Press Ctrl+C to stop"
echo ""

# Wait for processes
wait
