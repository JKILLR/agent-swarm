# Agent Swarm Architecture

## Overview

This is a hierarchical multi-agent system where a human (CEO) directs work through an AI orchestrator (COO) that manages multiple project teams (Swarms), each containing specialized AI agents.

```
┌─────────────────────────────────────────────────────────────────┐
│                         CEO (Human)                              │
│                    Web UI Chat Interface                         │
└─────────────────────────┬───────────────────────────────────────┘
                          │ WebSocket /ws/chat
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                   COO (Supreme Orchestrator)                     │
│                    supreme/orchestrator.py                       │
│  • Routes requests to appropriate swarms                         │
│  • Spawns parallel agents via Task tool                          │
│  • Maintains awareness of all swarms and agents                  │
└─────────────────────────┬───────────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┬───────────────┐
          ▼               ▼               ▼               ▼
┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│ ASA Research│   │  Swarm Dev  │   │  Operations │   │  MYND App   │
│   Swarm     │   │    Swarm    │   │    Swarm    │   │   Swarm     │
└──────┬──────┘   └──────┬──────┘   └──────┬──────┘   └──────┬──────┘
       │                 │                 │                 │
       ▼                 ▼                 ▼                 ▼
┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│ • researcher│   │ • architect │   │ • ops_lead  │   │ • researcher│
│ • implementr│   │ • implementr│   │ • monitor   │   │ • implementr│
│ • critic    │   │ • reviewer  │   │             │   │ • critic    │
│ • lead      │   │ • tester    │   │             │   │             │
└─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘
```

## Directory Structure

```
agent-swarm/
├── backend/                    # FastAPI backend (central hub)
│   ├── main.py                # API endpoints, WebSocket, orchestration
│   ├── jobs.py                # Background job queue (SQLite)
│   ├── tools.py               # Tool definitions and executor
│   └── memory.py              # Session and context management
│
├── supreme/                    # Supreme Orchestrator (COO)
│   ├── orchestrator.py        # Swarm coordination and routing
│   └── agents/
│       └── supreme.md         # COO system prompt
│
├── shared/                     # Shared libraries
│   ├── agent_base.py          # Base agent class
│   ├── agent_definitions.py   # Agent configuration dataclass
│   ├── agent_executor.py      # Claude CLI execution
│   ├── swarm_interface.py     # Swarm base class
│   └── consensus.py           # Multi-agent voting (not yet integrated)
│
├── swarms/                     # Project teams
│   ├── _template/             # Template for new swarms
│   ├── asa_research/          # ASA Research swarm
│   ├── swarm_dev/             # Swarm Dev swarm
│   ├── operations/            # Operations swarm
│   └── mynd_app/              # MYND App swarm
│
├── frontend/                   # Next.js web UI
│   ├── app/
│   │   ├── page.tsx           # Dashboard
│   │   ├── chat/page.tsx      # Chat interface
│   │   └── swarm/[name]/      # Swarm detail pages
│   ├── components/            # React components
│   └── lib/
│       ├── api.ts             # REST API client
│       └── websocket.ts       # WebSocket client
│
└── logs/                       # Runtime data
    └── sessions/              # Chat session JSON files
```

## Data Flows

### 1. Chat Flow (User → Response)

```
User types message in chat UI
         │
         ▼
WebSocket connection to /ws/chat
         │
         ▼
backend/main.py: websocket_chat_endpoint()
         │
         ▼
stream_claude_response() → Spawns Claude CLI process
         │
         ▼
parse_claude_stream() → Parses JSON stream events
         │
         ├──► tool_start event → Frontend shows activity
         ├──► agent_delta event → Frontend shows streaming text
         ├──► tool_complete event → Frontend updates activity
         └──► agent_complete event → Message complete
         │
         ▼
Session saved to logs/sessions/{id}.json
```

### 2. Tool Execution Flow

```
Agent requests tool (e.g., Task, Read, Bash)
         │
         ▼
ToolExecutor.execute(tool_name, tool_input)
         │
         ├──► Read: Read file contents
         ├──► Write: Write file
         ├──► Bash: Execute shell command
         ├──► Glob: Find files by pattern
         ├──► Grep: Search file contents
         ├──► Task: Spawn subagent ──────────────────┐
         ├──► GitStatus/GitSync: Git operations      │
         └──► ListSwarms/GetSwarmStatus: Swarm info  │
                                                     │
         ┌───────────────────────────────────────────┘
         ▼
Task tool spawns new Claude CLI process
for specified agent (e.g., "swarm_dev/implementer")
         │
         ▼
Subagent executes with own tools
         │
         ▼
Result returned to parent agent
```

### 3. Agent Delegation Pattern

The COO delegates work using the Task tool:

```python
# COO spawns researcher in parallel
Task(agent="asa_research/researcher", prompt="Research transformer attention...")

# COO spawns implementer in parallel
Task(agent="asa_research/implementer", prompt="Implement sparse attention...")

# COO spawns critic to review
Task(agent="asa_research/critic", prompt="Review the implementation...")
```

### 4. Background Jobs Flow

```
Job created via POST /api/jobs
         │
         ▼
JobManager.submit_job() → Added to queue
         │
         ▼
JobManager._process_queue() → Executes up to 3 concurrent jobs
         │
         ▼
Job status broadcast via WebSocket /ws/jobs
         │
         ▼
JobsPanel component shows live updates
```

## Key Components

### Backend (backend/main.py)

The central hub that connects everything:

- **REST API**: Swarm CRUD, file operations, chat sessions, jobs
- **WebSocket**: Real-time chat streaming, job updates
- **Orchestrator Integration**: Routes requests through SupremeOrchestrator
- **Tool Execution**: ToolExecutor handles agent tool calls

### Supreme Orchestrator (supreme/orchestrator.py)

Manages the swarm hierarchy:

- **discover_swarms()**: Loads all swarms from /swarms directory
- **route_request()**: Analyzes user input, delegates to appropriate swarm
- **send_directive()**: Sends instructions to specific swarm
- **run_parallel_on_swarm()**: Executes multiple agents in parallel
- **create_swarm()**: Creates new swarm from template

### Swarm Interface (shared/swarm_interface.py)

Base class for all swarms:

- **agents**: Dict of agent name → BaseAgent
- **agent_definitions**: Dict of agent name → AgentDefinition
- **receive_directive()**: Handles incoming directives
- **run_parallel()**: Executes multiple agents concurrently
- **get_status()**: Returns swarm health and agent info

### Tool Executor (backend/tools.py)

Executes tools requested by agents:

- File operations: Read, Write, Edit, Glob, Grep
- Shell: Bash command execution
- Agent spawning: Task (spawns subagents)
- Git: GitStatus, GitSync, GitCommit
- Swarm info: ListSwarms, GetSwarmStatus

### Memory Manager (backend/memory.py)

Handles persistence:

- **Chat sessions**: Stored in logs/sessions/{id}.json
- **Swarm context**: Each swarm has context.md for persistent memory
- **Session restoration**: Loads previous messages when resuming chat

## Swarm Configuration

Each swarm is defined by `swarm.yaml`:

```yaml
name: "Swarm Dev"
description: "Development team for the agent-swarm system"
status: active  # active, paused, archived
version: "0.1.0"

priorities:
  - "Claude Agent SDK integration"
  - "Web UI enhancements"
  - "Testing infrastructure"

agents:
  - name: architect
    type: architect
    model: opus
    background: false

  - name: implementer
    type: implementer
    model: sonnet
    background: true
```

## Agent Definition

Each agent has a markdown file in `swarms/{name}/agents/{agent}.md`:

```markdown
# Agent: implementer

## Role
Senior software engineer focused on implementation.

## Capabilities
- Write clean, tested code
- Follow existing patterns
- Use appropriate tools

## Tools
- Read, Write, Edit
- Bash, Glob, Grep
- Task (for delegation)
```

## Frontend Components

| Component | Purpose |
|-----------|---------|
| `ChatPage` | Main chat interface with COO |
| `ActivityFeed` | Shows real-time tool usage |
| `JobsPanel` | Background job monitoring |
| `Sidebar` | Navigation and swarm list |
| `SwarmCard` | Swarm summary on dashboard |
| `OrgChart` | Visual agent hierarchy |
| `CeoTodoPanel` | CEO's task list |

## API Endpoints

### Swarms
- `GET /api/swarms` - List all swarms
- `GET /api/swarms/{name}` - Get swarm details
- `POST /api/swarms` - Create new swarm
- `GET /api/swarms/{name}/agents` - List swarm agents

### Chat
- `WS /ws/chat` - WebSocket for chat streaming
- `GET /api/chat/sessions` - List chat sessions
- `POST /api/chat/sessions` - Create new session
- `GET /api/chat/sessions/{id}` - Get session with messages

### Jobs
- `GET /api/jobs` - List background jobs
- `POST /api/jobs` - Create new job
- `DELETE /api/jobs/{id}` - Cancel job
- `WS /ws/jobs` - WebSocket for job updates

### Files
- `GET /api/files` - List files in directory
- `GET /api/files/content` - Read file content
- `POST /api/files/content` - Write file content

## Future Improvements

1. **Consensus Integration**: The `shared/consensus.py` module implements multi-agent voting but isn't wired into the orchestration flow yet.

2. **Task Tracking**: Infrastructure exists in `swarm_interface.py` (`add_task`, `complete_task`, `report_progress`) but isn't used.

3. **Memory Summarization**: Methods in `memory.py` for conversation summarization are scaffolded but not integrated.

4. **Cross-Swarm Communication**: Agents can only communicate through the COO; direct swarm-to-swarm messaging could improve efficiency.

## Running the System

```bash
# Start backend
cd agent-swarm
./run.sh

# Or manually:
python3 -m uvicorn backend.main:app --reload --port 8000

# Frontend runs on port 3000
# Access at http://localhost:3000
```

## Review Questions

When reviewing this architecture, consider:

1. Is the agent hierarchy well-structured for autonomous operation?
2. Are there bottlenecks in the data flow?
3. Is the tool execution pattern extensible?
4. How well does information flow between agents/swarms?
5. Are there missing feedback loops for agent coordination?
6. Is the system resilient to failures at each layer?
7. How could the consensus mechanism be better integrated?
8. Are there opportunities for better parallel execution?
