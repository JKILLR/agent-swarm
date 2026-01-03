# Agent Independence Test Procedure

This document provides manual test procedures to verify that agents in the swarm system
are working independently and that the COO (Supreme Orchestrator) correctly delegates
work rather than implementing it directly.

## Prerequisites

Before running these tests, ensure:

1. Backend is running: `./run.sh` or `python -m uvicorn backend.main:app --port 8000`
2. Frontend is running: `cd frontend && npm run dev`
3. Claude CLI is installed and authenticated: `claude --version`
4. At least one swarm exists with agents defined

## Test Categories

### Category 1: Agent Spawning Verification

#### Test 1.1: Basic Agent Delegation
**Objective**: Verify COO delegates implementation work to agents

**Steps**:
1. Open the chat interface at http://localhost:3000/chat
2. Send message: "Create a simple Python script that prints 'Hello World' in the swarm_dev workspace"
3. Watch the Activity Feed for agent spawning

**Expected Results**:
- [ ] Activity Feed shows "Task" tool being used
- [ ] Activity shows agent being spawned (e.g., "implementer")
- [ ] COO response references delegating to an agent
- [ ] Final response includes synthesized results from agent

**Failure Indicators**:
- COO writes the code itself without spawning an agent
- No Task tool usage visible in activity feed
- Response says "I've implemented..." rather than "The implementer agent..."

---

#### Test 1.2: Research Delegation
**Objective**: Verify research tasks are delegated to researcher agents

**Steps**:
1. Open chat interface
2. Send: "Research the current architecture of this project and summarize the key components"
3. Monitor Activity Feed

**Expected Results**:
- [ ] Task tool spawns a "researcher" agent
- [ ] Agent uses Read/Glob tools to explore codebase
- [ ] COO synthesizes agent's findings in response

---

#### Test 1.3: Multi-Agent Task
**Objective**: Verify COO can spawn multiple agents for complex tasks

**Steps**:
1. Open chat interface
2. Send: "Review the backend code and suggest improvements. Have one agent analyze the code and another suggest improvements."
3. Monitor Activity Feed

**Expected Results**:
- [ ] Multiple Task tool calls visible
- [ ] Different agent types spawned (researcher, critic, architect, etc.)
- [ ] COO coordinates and synthesizes multiple agent outputs

---

### Category 2: Agent Isolation

#### Test 2.1: Agent Workspace Isolation
**Objective**: Verify agents operate in correct workspace directories

**Steps**:
1. Send: "Create a test file called 'isolation_test.txt' in the trading_bots workspace"
2. Verify file location

**Expected Results**:
- [ ] File created at `swarms/trading_bots/workspace/isolation_test.txt`
- [ ] File NOT created in project root or other swarm workspaces
- [ ] Agent correctly uses the swarm's workspace directory

---

#### Test 2.2: No Cross-Swarm Contamination
**Objective**: Verify agents don't modify files outside their swarm

**Steps**:
1. Create test files in asa_research workspace
2. Send: "Using the swarm_dev agents, modify the test files in asa_research"
3. Verify behavior

**Expected Results**:
- [ ] Agent correctly refuses or asks for clarification
- [ ] Or agent is given explicit permission via COO coordination
- [ ] No unauthorized cross-swarm file modifications

---

### Category 3: Tool Usage Independence

#### Test 3.1: Agent Tool Access
**Objective**: Verify spawned agents have proper tool access

**Steps**:
1. Send: "Have an implementer create a Python file with some utility functions"
2. Watch Activity Feed for tool usage

**Expected Results**:
- [ ] Agent uses Read tool to understand existing patterns
- [ ] Agent uses Write/Edit tools to create file
- [ ] Agent may use Bash for testing/validation
- [ ] Tools shown in Activity Feed with agent attribution

---

#### Test 3.2: COO Tool Limitation
**Objective**: Verify COO primarily uses Task tool for implementation work

**Steps**:
1. Review Activity Feed across multiple conversations
2. Note which tools COO uses directly vs. through agents

**Expected Results**:
- [ ] COO uses Read/Glob for context gathering
- [ ] COO uses Task for actual implementation
- [ ] COO rarely uses Write/Edit directly (delegates instead)

---

### Category 4: Error Handling

#### Test 4.1: Agent Failure Recovery
**Objective**: Verify system handles agent failures gracefully

**Steps**:
1. Send: "Have an implementer create a file in a nonexistent directory /invalid/path/file.txt"
2. Observe error handling

**Expected Results**:
- [ ] Error is caught and reported
- [ ] System doesn't crash
- [ ] COO provides helpful error message to user
- [ ] Activity Feed shows failed operation

---

#### Test 4.2: Timeout Handling
**Objective**: Verify long-running tasks are handled appropriately

**Steps**:
1. Send a complex task that may take several minutes
2. Monitor for timeout behavior

**Expected Results**:
- [ ] Progress updates visible during execution
- [ ] System doesn't hang indefinitely
- [ ] 15-minute timeout triggers if needed
- [ ] User informed of status

---

### Category 5: Session Continuity

#### Test 5.1: Context Preservation
**Objective**: Verify conversation context persists across messages

**Steps**:
1. Send: "Remember the number 42 for this conversation"
2. Send several unrelated messages
3. Send: "What number did I ask you to remember?"

**Expected Results**:
- [ ] COO recalls the number correctly
- [ ] Context loaded from session history
- [ ] Recent messages visible in conversation context

---

#### Test 5.2: Agent Context Isolation
**Objective**: Verify agents get appropriate context without full history pollution

**Steps**:
1. Have a long conversation with multiple topics
2. Send: "Have an implementer create a simple config file"
3. Verify agent doesn't reference unrelated conversation topics

**Expected Results**:
- [ ] Agent focuses on immediate task
- [ ] Agent doesn't reference old conversation topics
- [ ] Agent has access to necessary codebase context

---

## Test Results Template

### Test Run Information
- **Date**: ____________________
- **Tester**: ____________________
- **Backend Version**: ____________________
- **Frontend Version**: ____________________
- **Claude CLI Version**: ____________________

### Summary

| Category | Pass | Fail | Skip |
|----------|------|------|------|
| Agent Spawning | | | |
| Agent Isolation | | | |
| Tool Usage | | | |
| Error Handling | | | |
| Session Continuity | | | |
| **TOTAL** | | | |

### Issues Found

1. Issue: ____________________
   - Severity: Critical / Major / Minor
   - Steps to reproduce: ____________________
   - Expected: ____________________
   - Actual: ____________________

### Notes

_Additional observations, suggestions, or concerns from testing._

---

## Automated Test Companion

For automated verification of some of these behaviors, see:
- `tests/test_agent_independence.py` - Unit tests
- `tests/test_live_agents.py` - Live integration tests
- `tests/run_agent_tests.sh` - Test runner script
