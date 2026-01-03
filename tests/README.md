# Agent Swarm Test Suite

This directory contains tests to verify that the agent swarm system is working correctly,
with a focus on **agent independence** - ensuring that the COO (Supreme Orchestrator)
properly delegates work to specialized agents rather than implementing tasks directly.

## Test Files Overview

| File | Description |
|------|-------------|
| `test_agent_independence.py` | Unit tests for delegation patterns, tool execution, and session management |
| `test_live_agents.py` | Live integration tests that require a running backend |
| `test_memory.py` | Tests for the memory management system |
| `test_tools.py` | Tests for tool execution and error handling |
| `run_agent_tests.sh` | Shell script to run the test suite |
| `manual/AGENT_INDEPENDENCE_TEST.md` | Manual test procedures with checklists |

## What These Tests Verify

### Agent Independence Tests

1. **Delegation Patterns**
   - COO uses Task tool to spawn agents for implementation work
   - Agents have proper capabilities and tools available
   - Multi-agent tasks are coordinated correctly

2. **Workspace Isolation**
   - Each swarm has its own workspace directory
   - Agents operate within their designated workspace
   - No cross-swarm contamination of files

3. **Tool Attribution**
   - Tool usage is properly attributed to the correct agent
   - Activity feed shows which agent is using which tool

4. **Session Management**
   - Conversation context is preserved across messages
   - Session summaries are saved correctly
   - Context is loaded appropriately for agents

5. **Error Handling**
   - Agent failures are caught and reported
   - Timeouts are handled gracefully
   - Invalid inputs produce helpful error messages

## How to Run Tests

### Prerequisites

1. **Install test dependencies**:
   ```bash
   pip install pytest pytest-asyncio pytest-cov aiohttp websockets
   ```

2. **Ensure backend can import correctly**:
   ```bash
   pip install -r requirements.txt  # from project root
   ```

### Running Unit Tests

Unit tests do not require a running backend:

```bash
# Run all unit tests
pytest tests/test_agent_independence.py -v

# Run with coverage
pytest tests/test_agent_independence.py -v --cov=backend --cov-report=html

# Run specific test class
pytest tests/test_agent_independence.py::TestOrchestratorDelegation -v

# Run specific test
pytest tests/test_agent_independence.py::TestToolExecutor::test_required_tools_available -v
```

### Running Live Integration Tests

Live tests require a running backend:

1. **Start the backend**:
   ```bash
   ./run.sh
   # or
   python -m uvicorn backend.main:app --port 8000
   ```

2. **Run live tests**:
   ```bash
   pytest tests/test_live_agents.py -v
   ```

### Using the Test Runner Script

The `run_agent_tests.sh` script provides a convenient way to run tests:

```bash
# Make it executable (first time only)
chmod +x tests/run_agent_tests.sh

# Run unit tests only
./tests/run_agent_tests.sh --unit

# Run live tests only (requires running backend)
./tests/run_agent_tests.sh --live

# Run all tests
./tests/run_agent_tests.sh --all

# Run with verbose output
./tests/run_agent_tests.sh --verbose

# Run with coverage report
./tests/run_agent_tests.sh --coverage

# See all options
./tests/run_agent_tests.sh --help
```

### Running Manual Tests

For manual verification of agent behavior:

1. Open `tests/manual/AGENT_INDEPENDENCE_TEST.md`
2. Follow the test procedures step by step
3. Mark each checklist item as passed/failed
4. Record any issues found

## Interpreting Results

### Unit Test Output

```
tests/test_agent_independence.py::TestOrchestratorDelegation::test_orchestrator_discovers_swarms PASSED
tests/test_agent_independence.py::TestOrchestratorDelegation::test_orchestrator_has_agents PASSED
tests/test_agent_independence.py::TestToolExecutor::test_required_tools_available PASSED
...
```

- **PASSED**: Test passed successfully
- **FAILED**: Test failed - see error message for details
- **SKIPPED**: Test was skipped (usually due to missing prerequisites)
- **ERROR**: Test had an error during execution

### Live Test Output

Live tests may show additional skip reasons:
- "Backend not available" - Start the backend server
- "websockets not installed" - Install: `pip install websockets`
- "aiohttp not installed" - Install: `pip install aiohttp`

### Coverage Report

If running with `--coverage`, a coverage report is generated:
- Terminal summary shows percentage coverage
- HTML report at `htmlcov/index.html` for detailed view

### Common Issues

1. **Import errors**: Ensure you're running from project root
   ```bash
   cd /path/to/agent-swarm
   pytest tests/test_agent_independence.py -v
   ```

2. **Backend connection refused**: Start the backend first
   ```bash
   ./run.sh
   ```

3. **WebSocket tests timeout**: May indicate slow network or overloaded system

4. **Claude CLI not found**: Install with
   ```bash
   npm install -g @anthropic-ai/claude-cli
   ```

## Adding New Tests

When adding new tests, follow these patterns:

### Unit Test Pattern

```python
class TestNewFeature:
    """Tests for new feature."""

    @pytest.fixture
    def setup_data(self):
        """Create test data."""
        return {"key": "value"}

    def test_feature_behavior(self, setup_data):
        """Test that feature behaves correctly."""
        result = feature_function(setup_data)
        assert result == expected_value
```

### Live Test Pattern

```python
@pytest.mark.skipif(not backend_available(), reason="Backend not available")
class TestLiveFeature:
    """Live tests for feature."""

    def test_api_endpoint(self):
        """Test API endpoint."""
        import urllib.request
        req = urllib.request.Request(f"{BACKEND_URL}/api/endpoint")
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read())
            assert "expected_field" in data
```

## Test Categories

Tests are organized by category:

| Category | Tests | Purpose |
|----------|-------|---------|
| `TestOrchestratorDelegation` | 4 | Verify swarm/agent discovery |
| `TestToolExecutor` | 4 | Verify tool definitions and execution |
| `TestAgentWorkspaceIsolation` | 2 | Verify workspace separation |
| `TestSessionContext` | 3 | Verify session/context management |
| `TestAgentTypeMatching` | 3 | Verify task-to-agent mapping |
| `TestParallelAgentExecution` | 2 | Verify parallel task patterns |
| `TestErrorPropagation` | 2 | Verify error handling |
| `TestSwarmConfiguration` | 2 | Verify swarm config loading |
| `TestBackendHealth` | 2 | Verify backend is running |
| `TestChatSessions` | 3 | Verify session CRUD |
| `TestWebSocketChat` | 2 | Verify WebSocket communication |
| `TestAgentDelegationPatterns` | 1 | Verify delegation in practice |

## CI/CD Integration

For CI/CD pipelines, use:

```yaml
# Example GitHub Actions step
- name: Run Agent Tests
  run: |
    pip install pytest pytest-asyncio pytest-cov
    pytest tests/test_agent_independence.py -v --tb=short
```

For full integration testing (requires backend):

```yaml
- name: Run Live Tests
  run: |
    ./run.sh &
    sleep 10  # Wait for backend to start
    pytest tests/test_live_agents.py -v --tb=short
```
