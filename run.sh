#!/bin/bash
# Run both backend and frontend for Agent Swarm web interface

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting Agent Swarm Web Interface${NC}"
echo ""

# Check if we're in the right directory
if [ ! -f "main.py" ]; then
    echo -e "${RED}Error: Run this script from the agent-swarm root directory${NC}"
    exit 1
fi

# Function to cleanup background processes
cleanup() {
    echo ""
    echo -e "${YELLOW}Shutting down...${NC}"
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    exit 0
}

trap cleanup SIGINT SIGTERM

# Install backend dependencies if needed
if [ ! -d "backend/.venv" ] && [ ! -d "venv" ]; then
    echo -e "${YELLOW}Installing backend dependencies...${NC}"
    pip install -r backend/requirements.txt
fi

# Install frontend dependencies if needed
if [ ! -d "frontend/node_modules" ]; then
    echo -e "${YELLOW}Installing frontend dependencies...${NC}"
    cd frontend && npm install && cd ..
fi

# Start backend
echo -e "${GREEN}Starting backend on http://localhost:8000${NC}"
cd backend && python -m uvicorn main:app --reload --port 8000 &
BACKEND_PID=$!
cd ..

# Wait a bit for backend to start
sleep 2

# Start frontend
echo -e "${GREEN}Starting frontend on http://localhost:3000${NC}"
cd frontend && npm run dev &
FRONTEND_PID=$!
cd ..

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
