# Main.py Modular Refactoring Plan

## Overview

**File**: `/Users/jellingson/agent-swarm/backend/main.py`
**Current Size**: 2823 lines
**Status**: CRITICAL - Needs immediate refactoring for maintainability

This document provides an actionable plan to split `main.py` into logical modules following FastAPI best practices.

---

## Current Structure Analysis

### Line-by-Line Breakdown

| Lines | Section | Description | Proposed Module |
|-------|---------|-------------|-----------------|
| 1-80 | Imports & Setup | FastAPI app init, CORS, logging | `app.py` |
| 82-160 | Startup Events | DB init, job manager, workspace/pool init | `app.py` |
| 162-188 | Orchestrator | Supreme Orchestrator singleton | `services/orchestrator_service.py` |
| 190-241 | Pydantic Models | Request/Response models | `models/` |
| 243-361 | Chat History | ChatHistoryManager class | `services/chat_history.py` |
| 364-457 | Job Routes | `/api/jobs` endpoints | `routes/jobs.py` |
| 459-593 | Work Ledger Routes | `/api/work` endpoints | `routes/work.py` |
| 596-750 | Mailbox Routes | `/api/mailbox` endpoints | `routes/mailbox.py` |
| 752-878 | Escalation Routes | `/api/escalations` endpoints | `routes/escalations.py` |
| 881-1000 | Workflow Routes | `/api/swarms/{name}/workflow` endpoints | `routes/workflows.py` |
| 1003-1244 | Executor Pool Routes | `/api/agents` + WebSocket | `routes/agents.py` |
| 1247-1331 | Swarm Routes | `/api/swarms` CRUD endpoints | `routes/swarms.py` |
| 1333-1347 | Chat Route | `/api/chat` (non-streaming) | `routes/chat.py` |
| 1349-1625 | File Routes | `/api/swarms/{name}/files` endpoints | `routes/files.py` |
| 1627-1698 | Chat Session Routes | `/api/chat/sessions` endpoints | `routes/chat.py` |
| 1700-1818 | Web Search Routes | `/api/search`, `/api/fetch` endpoints | `routes/web.py` |
| 1821-1858 | ConnectionManager | WebSocket connection manager | `websocket/connection_manager.py` |
| 1861-2022 | Claude CLI | `stream_claude_response`, `parse_claude_stream` | `services/claude_service.py` |
| 2024-2045 | Tool Utils | `_get_tool_description`, `_get_file_info` | `utils/tool_helpers.py` |
| 2060-2441 | CLI Event Handler | `_process_cli_event` (381 lines!) | `services/event_processor.py` |
| 2444-2531 | Job WebSocket | `/ws/jobs` subscriber management | `websocket/job_updates.py` |
| 2534-2816 | Chat WebSocket | `/ws/chat` main chat endpoint | `websocket/chat_handler.py` |
| 2819-2823 | Main Block | uvicorn runner | `app.py` |

---

## Proposed Directory Structure

```
backend/
    __init__.py
    app.py                          # FastAPI app factory, startup/shutdown

    models/
        __init__.py
        requests.py                 # Request Pydantic models
        responses.py                # Response Pydantic models
        chat.py                     # Chat-specific models

    routes/
        __init__.py
        agents.py                   # /api/agents/* endpoints
        chat.py                     # /api/chat/* endpoints
        escalations.py              # /api/escalations/* endpoints
        files.py                    # /api/swarms/{name}/files/* endpoints
        jobs.py                     # /api/jobs/* endpoints
        mailbox.py                  # /api/mailbox/* endpoints
        swarms.py                   # /api/swarms/* endpoints
        web.py                      # /api/search, /api/fetch endpoints
        work.py                     # /api/work/* endpoints
        workflows.py                # /api/swarms/{name}/workflow/* endpoints

    services/
        __init__.py
        chat_history.py             # ChatHistoryManager class
        claude_service.py           # Claude CLI execution
        event_processor.py          # _process_cli_event (refactored)
        orchestrator_service.py     # Supreme Orchestrator management

    websocket/
        __init__.py
        connection_manager.py       # ConnectionManager class
        chat_handler.py             # /ws/chat handler
        job_updates.py              # /ws/jobs handler
        executor_pool.py            # /ws/executor-pool handler

    utils/
        __init__.py
        tool_helpers.py             # _get_tool_description, _get_file_info
        constants.py                # Magic numbers -> named constants

    # Keep existing (already modular)
    memory.py
    session_manager.py
    jobs.py                         # Already separate, good
```

---

## Detailed Module Specifications

### 1. `app.py` - Application Factory

**Purpose**: FastAPI app creation, middleware, startup/shutdown events

**Contents**:
- FastAPI app initialization
- CORS middleware configuration
- Startup event handler (DB init, manager initialization)
- Shutdown event handler (cleanup)
- Router registration
- Main uvicorn block

**Dependencies**:
- All route modules (imports routers)
- `services/orchestrator_service.py`
- Existing: `shared/workspace_manager.py`, `shared/agent_executor_pool.py`

**Exports**:
- `app` - FastAPI application instance
- `PROJECT_ROOT` - Path constant
- `get_orchestrator()` - Orchestrator getter

**Estimated Size**: ~150 lines

---

### 2. `models/requests.py`

**Purpose**: Pydantic models for API request bodies

**Contents**:
```python
class SwarmCreate(BaseModel):
    name: str
    description: str = ""
    template: str = "_template"

class ChatMessage(BaseModel):
    message: str
    swarm: str | None = None

class JobCreate(BaseModel):
    type: str = "chat"
    prompt: str
    swarm: str | None = None
    session_id: str | None = None

class WorkCreateRequest(BaseModel):
    title: str
    description: str
    work_type: str = "task"
    priority: str = "medium"
    parent_id: str | None = None
    swarm_name: str | None = None
    context: dict | None = None

class MessageSendRequest(BaseModel):
    from_agent: str
    to_agent: str
    subject: str
    body: str
    message_type: str = "request"
    priority: str = "normal"
    # ... rest

class HandoffRequest(BaseModel):
    # ... fields

class EscalationCreateRequest(BaseModel):
    # ... fields

class AgentExecuteRequest(BaseModel):
    swarm: str
    agent: str
    prompt: str
    max_turns: int = 25
    timeout: float = 600.0

class SearchRequest(BaseModel):
    query: str
    num_results: int = 5

class FetchRequest(BaseModel):
    url: str
    extract_text: bool = True
```

**Estimated Size**: ~100 lines

---

### 3. `models/responses.py`

**Purpose**: Pydantic models for API responses

**Contents**:
```python
class SwarmResponse(BaseModel):
    name: str
    description: str
    status: str
    agent_count: int
    priorities: list[str]

class AgentInfo(BaseModel):
    name: str
    type: str
    model: str
    background: bool
    description: str

class HealthResponse(BaseModel):
    status: str
    swarm_count: int
    agent_count: int
```

**Estimated Size**: ~40 lines

---

### 4. `models/chat.py`

**Purpose**: Chat-specific Pydantic models

**Contents**:
```python
class ChatMessageModel(BaseModel):
    id: str
    role: str
    content: str
    timestamp: str
    agent: str | None = None
    thinking: str | None = None

class ChatSession(BaseModel):
    id: str
    title: str
    swarm: str | None = None
    created_at: str
    updated_at: str
    messages: list[ChatMessageModel] = []
```

**Estimated Size**: ~30 lines

---

### 5. `services/chat_history.py`

**Purpose**: Chat history persistence management

**Contents**:
- `ChatHistoryManager` class (lines 243-361)
- `get_chat_history()` singleton getter

**Dependencies**:
- `models/chat.py`
- Standard library: json, uuid, datetime, pathlib

**Exports**:
- `ChatHistoryManager`
- `get_chat_history()`

**Estimated Size**: ~130 lines

---

### 6. `services/orchestrator_service.py`

**Purpose**: Supreme Orchestrator management

**Contents**:
```python
from supreme.orchestrator import SupremeOrchestrator

_orchestrator: SupremeOrchestrator | None = None

def get_orchestrator() -> SupremeOrchestrator:
    """Get or create the Supreme Orchestrator."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = SupremeOrchestrator(
            base_path=PROJECT_ROOT,
            config_path=PROJECT_ROOT / "config.yaml",
            logs_dir=PROJECT_ROOT / "logs",
        )
    return _orchestrator
```

**Dependencies**:
- `supreme/orchestrator.py`
- `app.py` (for PROJECT_ROOT)

**Estimated Size**: ~30 lines

---

### 7. `services/claude_service.py`

**Purpose**: Claude CLI execution and streaming

**Contents**:
- `stream_claude_response()` function (lines 1861-1924)
- `parse_claude_stream()` function (lines 1927-2022)
- Related helper functions

**Dependencies**:
- `utils/tool_helpers.py`
- `services/event_processor.py`
- `session_manager.py`
- Standard library: asyncio, json, os

**Exports**:
- `stream_claude_response()`
- `parse_claude_stream()`

**Estimated Size**: ~170 lines

---

### 8. `services/event_processor.py`

**Purpose**: Process Claude CLI streaming events

**Critical Refactor**: The `_process_cli_event` function is 381 lines with 7 levels of nesting. This MUST be refactored into a class with separate methods.

**Proposed Structure**:
```python
class CLIEventProcessor:
    """Process streaming events from Claude CLI."""

    def __init__(self, websocket: WebSocket, manager: ConnectionManager):
        self.websocket = websocket
        self.manager = manager
        self.context = {
            "full_response": "",
            "full_thinking": "",
            "current_block_type": None,
            "session_id": None,
            "agent_stack": ["COO"],
            "pending_tasks": {},
            "subagent_tools": {},
        }

    async def process(self, event: dict, session_mgr=None, chat_id: str = None):
        """Main entry point - dispatch to specific handlers."""
        event_type = event.get("type", "")

        handlers = {
            "assistant": self._handle_assistant,
            "content_block_start": self._handle_block_start,
            "content_block_delta": self._handle_block_delta,
            "content_block_stop": self._handle_block_stop,
            "result": self._handle_result,
            "tool_result": self._handle_tool_result,
            "user": self._handle_user,
        }

        handler = handlers.get(event_type)
        if handler:
            await handler(event, session_mgr, chat_id)

    async def _handle_assistant(self, event: dict, session_mgr, chat_id):
        """Handle assistant message events."""
        # Lines 2087-2165 - assistant message handling
        pass

    async def _handle_block_start(self, event: dict, session_mgr, chat_id):
        """Handle content_block_start events."""
        # Lines 2167-2205 - block start handling
        pass

    async def _handle_block_delta(self, event: dict, session_mgr, chat_id):
        """Handle content_block_delta events."""
        # Lines 2207-2273 - delta handling (thinking, text, input_json)
        pass

    async def _handle_block_stop(self, event: dict, session_mgr, chat_id):
        """Handle content_block_stop events."""
        # Lines 2275-2322 - block stop handling
        pass

    async def _handle_result(self, event: dict, session_mgr, chat_id):
        """Handle final result events."""
        # Lines 2324-2339 - result handling
        pass

    async def _handle_tool_result(self, event: dict, session_mgr, chat_id):
        """Handle tool_result events."""
        # Lines 2341-2354 - tool result handling
        pass

    async def _handle_user(self, event: dict, session_mgr, chat_id):
        """Handle user message events (subagent tool results)."""
        # Lines 2356-2438 - user/subagent handling
        pass

    def get_current_agent(self) -> str:
        """Get the currently active agent from the stack."""
        return self.context["agent_stack"][-1] if self.context["agent_stack"] else "COO"

    def get_result(self) -> dict:
        """Get the accumulated response and thinking."""
        return {
            "response": self.context["full_response"],
            "thinking": self.context["full_thinking"],
        }
```

**Benefits of Class-Based Approach**:
- Eliminates deeply nested conditionals
- Each handler method is focused and testable
- Context management is cleaner (instance state)
- Easier to extend with new event types

**Estimated Size**: ~300 lines (same logic, better organized)

---

### 9. `utils/tool_helpers.py`

**Purpose**: Shared tool-related utility functions

**Contents**:
```python
def _get_tool_description(tool_name: str, tool_input: dict) -> str:
    """Generate human-readable description for a tool call."""
    # Lines 2024-2044
    pass

def _get_file_info(tool_name: str, tool_input: dict) -> tuple[str | None, str | None]:
    """Extract file path and operation type from tool input."""
    # Lines 2047-2057
    pass
```

**NOTE**: This fixes the DRY violation - `_get_tool_description` also exists in `shared/agent_executor_pool.py:597-625`. After creating this module, update `agent_executor_pool.py` to import from here.

**Estimated Size**: ~40 lines

---

### 10. `utils/constants.py`

**Purpose**: Named constants for magic numbers

**Contents**:
```python
# Chat history
MAX_RECENT_MESSAGES = 2
MAX_CONTENT_LENGTH = 1000

# Claude CLI
CLI_TIMEOUT_HOURS = 1
CLI_TIMEOUT_SECONDS = 3600

# Web fetch
MAX_FETCH_CONTENT_LENGTH = 10000
WEB_REQUEST_TIMEOUT = 15

# Search
MAX_SEARCH_RESULTS = 10

# Executor pool
DEFAULT_MAX_CONCURRENT = 5
DEFAULT_AGENT_TIMEOUT = 600.0
DEFAULT_MAX_TURNS = 25
```

**Estimated Size**: ~30 lines

---

### 11. `websocket/connection_manager.py`

**Purpose**: WebSocket connection lifecycle management

**Contents**:
```python
class ConnectionManager:
    """Manage WebSocket connections."""

    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        # Lines 1828-1831
        pass

    def disconnect(self, websocket: WebSocket):
        # Lines 1833-1838
        pass

    async def send_event(self, websocket: WebSocket, event_type: str, data: dict):
        # Lines 1840-1855
        pass

# Global singleton
manager = ConnectionManager()
```

**Exports**:
- `ConnectionManager`
- `manager` (singleton instance)

**Estimated Size**: ~50 lines

---

### 12. `websocket/chat_handler.py`

**Purpose**: Main chat WebSocket endpoint

**Contents**:
- `websocket_chat()` handler (lines 2534-2816)
- COO system prompt building
- Image attachment handling

**Dependencies**:
- `services/claude_service.py`
- `services/chat_history.py`
- `services/orchestrator_service.py`
- `websocket/connection_manager.py`
- `memory.py`

**Estimated Size**: ~300 lines

---

### 13. `websocket/job_updates.py`

**Purpose**: Job update WebSocket endpoint and broadcasting

**Contents**:
- `job_update_subscribers` dict
- `websocket_jobs()` handler (lines 2448-2509)
- `broadcast_job_update()` function (lines 2512-2531)

**Estimated Size**: ~100 lines

---

### 14. `websocket/executor_pool.py`

**Purpose**: Executor pool events WebSocket

**Contents**:
- `executor_pool_subscribers` list
- `broadcast_executor_pool_event()` function (lines 1117-1172)
- `websocket_executor_pool()` handler (lines 1175-1244)

**Estimated Size**: ~130 lines

---

### 15. Route Modules

Each route module follows the same pattern:

```python
# routes/jobs.py
from fastapi import APIRouter, HTTPException
from models.requests import JobCreate
from jobs import get_job_queue, get_job_manager

router = APIRouter(prefix="/api/jobs", tags=["jobs"])

@router.get("")
async def list_jobs(...):
    ...

@router.post("")
async def create_job(data: JobCreate):
    ...

@router.get("/{job_id}")
async def get_job(job_id: str):
    ...
```

**Route Module Sizes**:
| Module | Endpoints | Estimated Lines |
|--------|-----------|-----------------|
| `routes/jobs.py` | 5 | ~80 |
| `routes/work.py` | 8 | ~120 |
| `routes/mailbox.py` | 9 | ~150 |
| `routes/escalations.py` | 8 | ~120 |
| `routes/workflows.py` | 6 | ~100 |
| `routes/agents.py` | 2 | ~80 |
| `routes/swarms.py` | 4 | ~100 |
| `routes/chat.py` | 6 | ~100 |
| `routes/files.py` | 5 | ~200 |
| `routes/web.py` | 4 | ~120 |

---

## Implementation Plan

### Phase 1: Foundation (Day 1)

**Goal**: Create directory structure and move non-breaking pieces

1. Create directory structure:
   ```bash
   mkdir -p backend/{models,routes,services,websocket,utils}
   touch backend/{models,routes,services,websocket,utils}/__init__.py
   ```

2. Create `utils/constants.py` with magic numbers

3. Create `utils/tool_helpers.py` with shared functions

4. Create `models/` modules (requests, responses, chat)

5. **Test**: Import all models, verify no circular dependencies

### Phase 2: Services (Day 2)

**Goal**: Extract service classes

1. Create `services/chat_history.py`
   - Move `ChatHistoryManager` class
   - Move `get_chat_history()` singleton

2. Create `services/orchestrator_service.py`
   - Move `get_orchestrator()` function

3. Create `services/event_processor.py`
   - **Refactor** `_process_cli_event` into `CLIEventProcessor` class
   - This is the most critical refactor

4. Create `services/claude_service.py`
   - Move `stream_claude_response()` and `parse_claude_stream()`
   - Update to use `CLIEventProcessor`

5. **Test**: All services import correctly

### Phase 3: WebSocket (Day 3)

**Goal**: Extract WebSocket handlers

1. Create `websocket/connection_manager.py`
   - Move `ConnectionManager` class and `manager` instance

2. Create `websocket/job_updates.py`
   - Move job subscriber management

3. Create `websocket/executor_pool.py`
   - Move executor pool subscriber management

4. Create `websocket/chat_handler.py`
   - Move `websocket_chat()` handler
   - **NOTE**: This depends on all services being available

5. **Test**: WebSocket connections work

### Phase 4: Routes (Day 4)

**Goal**: Extract route handlers

1. Create each route module:
   - `routes/jobs.py`
   - `routes/work.py`
   - `routes/mailbox.py`
   - `routes/escalations.py`
   - `routes/workflows.py`
   - `routes/agents.py`
   - `routes/swarms.py`
   - `routes/chat.py`
   - `routes/files.py`
   - `routes/web.py`

2. Each module defines `router = APIRouter(prefix=..., tags=[...])`

3. **Test**: All API endpoints respond correctly

### Phase 5: App Assembly (Day 5)

**Goal**: Create final `app.py` and remove old `main.py`

1. Create new `app.py`:
   ```python
   from fastapi import FastAPI
   from fastapi.middleware.cors import CORSMiddleware

   from routes import (
       jobs, work, mailbox, escalations, workflows,
       agents, swarms, chat, files, web
   )
   from websocket import chat_handler, job_updates, executor_pool

   app = FastAPI(title="Agent Swarm API", ...)

   # Middleware
   app.add_middleware(CORSMiddleware, ...)

   # Route registration
   app.include_router(jobs.router)
   app.include_router(work.router)
   # ... etc

   # WebSocket registration
   app.add_websocket_route("/ws/chat", chat_handler.websocket_chat)
   app.add_websocket_route("/ws/jobs", job_updates.websocket_jobs)
   app.add_websocket_route("/ws/executor-pool", executor_pool.websocket_executor_pool)

   @app.on_event("startup")
   async def startup_event():
       # ... initialization

   if __name__ == "__main__":
       import uvicorn
       uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
   ```

2. Rename `main.py` to `main.py.bak` (keep for reference)

3. **Full Integration Test**: All functionality works

4. Remove `main.py.bak` after verification

---

## Shared Dependencies to Handle Carefully

### 1. `PROJECT_ROOT`

Currently defined in `main.py`. Must be:
- Defined in `app.py`
- Exported for other modules to import

```python
# app.py
BACKEND_DIR = Path(__file__).parent
PROJECT_ROOT = BACKEND_DIR.parent

# Other modules import:
from app import PROJECT_ROOT
```

### 2. `manager` (ConnectionManager instance)

Currently global in `main.py`. Must be:
- Defined in `websocket/connection_manager.py`
- Imported by modules that need it

### 3. Singleton Getters

These functions return singletons with lazy initialization:
- `get_orchestrator()` -> `services/orchestrator_service.py`
- `get_chat_history()` -> `services/chat_history.py`
- `get_work_ledger()` -> Already in `shared/work_ledger.py`
- `get_mailbox_manager()` -> Already in `shared/agent_mailbox.py`
- `get_escalation_manager()` -> Already in `shared/escalation_protocol.py`

### 4. Startup Initialization Order

The startup event must initialize in correct order:
1. Database (coordination.db)
2. Job manager
3. Workspace manager
4. Executor pool
5. Work ledger
6. Mailbox manager
7. Escalation manager

---

## Risk Mitigation

### 1. Circular Import Prevention

**Strategy**:
- Services import from `shared/` modules
- Routes import from services
- App imports routes
- No cross-route imports
- Models are leaf modules (no imports from backend/)

### 2. Test Coverage

Before starting, create minimal tests for:
- Health endpoint returns 200
- WebSocket chat connects
- File listing works
- Job creation works

After each phase, run these smoke tests.

### 3. Rollback Plan

Keep `main.py` as `main.py.bak` until full verification.

If issues arise:
```bash
mv backend/main.py.bak backend/main.py
# Remove new modules
rm -rf backend/{models,routes,services,websocket,utils}
```

---

## Post-Refactor Benefits

1. **Maintainability**: Each module is <200 lines and focused
2. **Testability**: Services can be unit tested in isolation
3. **Extensibility**: New endpoints just add a route file
4. **Clarity**: Finding code is easier (file name = purpose)
5. **Team Development**: Multiple developers can work on different modules
6. **DRY**: Shared utilities in one place (`tool_helpers.py`)

---

## Related Fixes to Include

While refactoring, also fix these issues identified in reviews:

1. **DRY Violation**: Move `_get_tool_description` to `utils/tool_helpers.py`, update `shared/agent_executor_pool.py` to import from there

2. **Magic Numbers**: Replace all magic numbers with constants from `utils/constants.py`

3. **Import Placement**: All imports at top of each new module (fix the threading import issue)

4. **Thread Safety**: Add locking to singleton getters that need it

---

## Verification Checklist

After refactoring is complete:

- [ ] `python -m py_compile backend/app.py` passes
- [ ] All route modules compile without errors
- [ ] `uvicorn backend.app:app` starts without errors
- [ ] GET `/api/status` returns healthy
- [ ] WebSocket `/ws/chat` connects and responds
- [ ] File browser works in frontend
- [ ] Jobs can be created and monitored
- [ ] No console errors in frontend
- [ ] All existing tests pass

---

## Estimated Effort

| Phase | Duration | Risk Level |
|-------|----------|------------|
| Phase 1: Foundation | 2-3 hours | Low |
| Phase 2: Services | 4-6 hours | Medium |
| Phase 3: WebSocket | 2-3 hours | Medium |
| Phase 4: Routes | 3-4 hours | Low |
| Phase 5: Assembly | 2-3 hours | High |

**Total**: 2-3 days with testing

---

## Appendix: Full Module Dependency Graph

```
app.py
    |-- routes/*.py (all routers)
    |-- websocket/*.py (all handlers)
    |-- services/orchestrator_service.py

routes/jobs.py
    |-- models/requests.py
    |-- jobs.py (existing)

routes/chat.py
    |-- models/requests.py
    |-- models/chat.py
    |-- services/chat_history.py
    |-- services/orchestrator_service.py

services/claude_service.py
    |-- services/event_processor.py
    |-- utils/tool_helpers.py
    |-- session_manager.py (existing)

services/event_processor.py
    |-- websocket/connection_manager.py
    |-- utils/tool_helpers.py

websocket/chat_handler.py
    |-- services/claude_service.py
    |-- services/chat_history.py
    |-- services/orchestrator_service.py
    |-- websocket/connection_manager.py
    |-- memory.py (existing)
```

---

**Author**: System Architect
**Date**: 2026-01-03
**Status**: DESIGN COMPLETE - Ready for Implementation
