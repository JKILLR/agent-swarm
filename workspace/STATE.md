# Agent Swarm - State Management

## Latest Work: Local Neural Net Brain Architecture
**Architect**: System Architect
**Researcher**: Research Specialist
**Date**: 2026-01-03

**Design Documents:**
- `/workspace/LOCAL_NEURAL_BRAIN_DESIGN.md` - Research and model selection
- See ADR-002 below for complete architectural specification

## Current Objectives
- [COMPLETE] Fix activity panel state persistence when navigating away from the chat page
- [COMPLETE] Ensure agent/tool activities persist across navigation
- [COMPLETE] Move ActivityPanel to global sidebar for visibility on ALL pages

## Architecture Decisions
- Using React Context (AgentActivityContext) for global state management
- Extended the existing AgentActivityProvider to include panel-specific activities
- No external state management library (no Zustand) - using React's built-in useState/useContext
- ActivityPanel is now rendered in the global Sidebar component, visible on all pages

## Key Files
- `/frontend/lib/AgentActivityContext.tsx` - Global context for activity state
- `/frontend/app/chat/page.tsx` - Chat page component (MODIFIED - mobile responsive, bottom sheet)
- `/frontend/components/ActivityPanel.tsx` - Activity panel display component (MODIFIED - mobile touch targets)
- `/frontend/components/Sidebar.tsx` - Global sidebar (MODIFIED - mobile onNavigate, touch targets)
- `/frontend/components/MobileLayout.tsx` - NEW: Mobile-responsive layout wrapper with hamburger menu
- `/frontend/components/ChatInput.tsx` - Chat input (MODIFIED - mobile touch targets, 16px font)
- `/frontend/components/Providers.tsx` - App-level provider wrapper
- `/frontend/app/layout.tsx` - Root layout (MODIFIED - viewport meta, MobileLayout wrapper)
- `/frontend/app/globals.css` - Global styles (MODIFIED - mobile utilities, animations)

## Progress Log

### 2026-01-02 - Move ActivityPanel to Global Sidebar
**Problem:** The ActivityPanel was only visible on the chat page. Users navigating to /dashboard or other pages could not see agent activity, even though the state was persisted in the global context.

**Solution Implemented:**
1. Added `ActivityPanel` import to `Sidebar.tsx`
2. Added context hooks to Sidebar to access panel activities:
   - `panelAgentActivities`
   - `panelToolActivities`
   - `clearPanelActivities`
3. Added computed values for display logic:
   - `hasActivity` - true if any activities exist
   - `isProcessing` - true if any agent is still working
4. Rendered ActivityPanel in sidebar between the Swarms list and CEO Todo Panel
5. Removed ActivityPanel rendering from `chat/page.tsx`
6. Removed unused `clearPanelActivities` import from chat page

**Files Modified:**
- `frontend/components/Sidebar.tsx`: Added ActivityPanel with global context data
- `frontend/app/chat/page.tsx`: Removed ActivityPanel rendering (kept WebSocket handlers)

**Behavior:**
- The ActivityPanel now appears in the sidebar when there is any activity
- Activity is visible whether on /dashboard, /chat, /swarm/*, or any other page
- The chat page still handles WebSocket events and updates the global context
- The Clear History button in the panel works from any page

### 2026-01-02 - Activity Panel State Persistence
**Problem:** When navigating away from the chat page, the activity panel state (agentActivities, toolActivities) was lost. The backend work continued but the frontend lost the WebSocket connection state.

**Solution Implemented:**
1. Extended `AgentActivityContext` to include new types and state:
   - Added `PanelAgentActivity` and `PanelToolActivity` interfaces
   - Added `panelAgentActivities` and `panelToolActivities` state arrays
   - Added `setPanelAgentActivities`, `setPanelToolActivities`, and `clearPanelActivities` functions

2. Updated `ActivityPanel.tsx`:
   - Imported types from `AgentActivityContext` instead of defining locally
   - Re-exported types for backwards compatibility

3. Updated `chat/page.tsx`:
   - Replaced local `useState` for agentActivities/toolActivities with context hooks
   - Using `useAgentActivity()` to get/set activities from global context
   - Activities now persist across component unmount/remount

**Files Modified:**
- `frontend/lib/AgentActivityContext.tsx`: Added panel activity types and state
- `frontend/app/chat/page.tsx`: Use context instead of local state
- `frontend/components/ActivityPanel.tsx`: Import types from context

## Known Issues

None currently tracked.

## Progress Log (continued)

### 2026-01-02 - WebSocket Connection Leak Fix
**Problem:** WebSocket connections were leaking in `/backend/main.py`:
- "Total: N" counter accumulated without proper cleanup
- Error: "Cannot call 'send' once a close message has been sent"
- Connections not removed from `active_connections` list

**Root Causes Identified:**
1. Missing `finally` block - cleanup only in `except` blocks
2. `disconnect()` failed with `ValueError` if item already removed
3. `send_event()` had no exception handling for closed sockets
4. Race condition - error handler tried to send after client disconnect

**Solution Implemented:**

**Fix 1: ConnectionManager.disconnect() (line 1122-1127)**
Added try/except to handle cases where websocket already removed:
```python
def disconnect(self, websocket: WebSocket):
    try:
        self.active_connections.remove(websocket)
    except ValueError:
        pass  # Already removed
    logger.info(f"WebSocket disconnected. Total: {len(self.active_connections)}")
```

**Fix 2: ConnectionManager.send_event() (lines 1129-1144)**
Added connection check and exception handling:
```python
async def send_event(self, websocket: WebSocket, event_type: str, data: dict[str, Any]):
    try:
        if websocket not in self.active_connections:
            return  # Connection already closed
        await websocket.send_json(...)
    except (RuntimeError, Exception) as e:
        if "close message" in str(e).lower():
            logger.debug(f"Skipped send to closed WebSocket: {e}")
        else:
            logger.error(f"Error sending event: {e}")
```

**Fix 3: websocket_chat() - finally block (lines 1932-1937)**
Changed from separate `except` blocks to unified `finally`:
```python
except WebSocketDisconnect:
    logger.info("WebSocket client disconnected")
except Exception as e:
    logger.error(f"WebSocket error: {e}")
finally:
    manager.disconnect(websocket)
```

**Fix 4: Error-path sends (lines 1911-1930)**
Wrapped error response sends in try/except:
```python
try:
    await manager.send_event(websocket, "agent_complete", {...})
    await manager.send_event(websocket, "chat_complete", {...})
except Exception as send_err:
    logger.debug(f"Failed to send error response (client may have disconnected): {send_err}")
```

**Files Modified:**
- `/backend/main.py`: ConnectionManager class and websocket_chat function

### 2026-01-02 - Escalation Protocol Code Review
**Reviewer**: Quality Critic
**Verdict**: NEEDS_CHANGES

**Files Reviewed**:
- `/Users/jellingson/agent-swarm/shared/escalation_protocol.py`
- `/Users/jellingson/agent-swarm/swarms/swarm_dev/workspace/DESIGN_ESCALATION_PROTOCOL.md`

**Critical Issues Found** (Must Fix):
1. **Thread Safety - Singleton Race Condition** (line 459-474): `get_escalation_manager()` is not thread-safe. Two threads could create separate instances.
2. **ID Collision After Restart** (line 198-204): `_id_counter` always starts at 0, not restored from loaded escalations.
3. **Non-Atomic File Writes** (line 367-373): Direct file writes could corrupt data on crash.

**High Issues Found** (Should Fix):
4. **Dict Mutation During Iteration** (line 334-365): Query methods not thread-safe.
5. **Missing Hierarchy Validation** (line 206-265): No validation that escalations follow AGENT->COO->CEO.
6. **Missing `__init__.py` Exports**: Escalation protocol not exported from shared module.

**Positive Observations**:
- Implementation matches design specification well
- Follows codebase patterns (similar to consensus.py)
- Good type hints and documentation
- Clean separation of concerns

## Known Issues

None currently tracked - escalation protocol issues have been fixed.

## Next Steps
- Test mobile UI on actual mobile device
- Verify hamburger menu opens/closes sidebar correctly
- Test bottom sheet for chat history on mobile
- Verify touch targets are at least 44px
- Test safe area insets on notched devices (iPhone X+)
- Run production build to verify no TypeScript errors

## Progress Log (continued)

### 2026-01-02 - Mobile UI Optimization
**Problem:** The Agent Swarm UI was not mobile-friendly. Issues included:
- Sidebar and main content overlapping/cramped on mobile
- Input area squeezed at the bottom
- Desktop layout forced on mobile screens
- Activity panel didn't work well on small screens
- No mobile-friendly navigation patterns

**Solution Implemented:**

**New Component: MobileLayout.tsx**
- Created responsive layout wrapper that detects mobile viewport (< 768px)
- Hamburger menu header on mobile with toggle button
- Sidebar slides in/out with overlay backdrop on mobile
- Prevents body scroll when sidebar is open
- Context provider (`useMobileLayout`) for child components to access mobile state

**Updated layout.tsx:**
- Added proper viewport meta tags (width, initialScale, maximumScale, userScalable)
- Wrapped content in MobileLayout component
- Moved Sidebar rendering into MobileLayout for responsive control

**Updated Sidebar.tsx:**
- Added `onNavigate` prop to close sidebar on navigation (mobile)
- Increased touch targets to 44px minimum on mobile
- Responsive icon sizes (w-5/h-5 on mobile, w-4/h-4 on desktop)
- Added `active:` states for touch feedback
- Hidden logo on mobile (shown in MobileLayout header instead)

**Updated chat/page.tsx:**
- Hidden desktop sidebar on mobile, replaced with bottom sheet
- Mobile history bottom sheet with slide-up animation
- Responsive header with mobile-specific history toggle
- Connection status text hidden on very small screens
- Responsive padding and spacing throughout
- Touch-friendly quick suggestion buttons
- Safe area padding for input on notched devices

**Updated ActivityPanel.tsx:**
- Mobile viewport detection
- Increased touch targets for all buttons
- Responsive max-height (48 on mobile, 64 on desktop)
- Added active states for touch feedback

**Updated ChatInput.tsx:**
- Touch-friendly buttons with 44px minimum size
- 16px font size to prevent iOS zoom on focus
- Responsive attachment preview sizing
- Hidden hint text on mobile to save space

**Updated globals.css:**
- Added touch-manipulation utility class
- Safe area inset utilities (pb-safe, pt-safe)
- Bottom sheet slide-up animation
- Better scrolling on iOS (-webkit-overflow-scrolling: touch)

**Files Modified:**
- `/frontend/app/layout.tsx`
- `/frontend/components/MobileLayout.tsx` (NEW)
- `/frontend/components/Sidebar.tsx`
- `/frontend/app/chat/page.tsx`
- `/frontend/components/ActivityPanel.tsx`
- `/frontend/components/ChatInput.tsx`
- `/frontend/app/globals.css`

### 2026-01-02 - Escalation Protocol Critical/High Fixes
**Problem:** Code review identified critical and high priority issues in `/shared/escalation_protocol.py`:
1. Thread-unsafe singleton pattern (race condition)
2. ID collision after restart (_id_counter not restored)
3. Non-atomic file writes (data corruption risk)
4. Thread-unsafe query methods (dict mutation during iteration)
5. Missing hierarchy validation (could create invalid escalation paths)

**Solution Implemented:**

**Fix 1: Thread-safe singleton** (lines 492-517)
- Added `_singleton_lock = threading.Lock()` at module level
- Implemented double-checked locking pattern in `get_escalation_manager()`

**Fix 2: ID collision prevention** (lines 405-421)
- Modified `load_from_disk()` to extract counter from existing escalation IDs
- After loading all files, sets `_id_counter = max_counter`
- Parses ID format ESC-TIMESTAMP-NNNN to find highest counter

**Fix 3: Atomic file writes** (lines 389-403)
- Write to temp file first (`.json.tmp` suffix)
- Use `temp_path.rename(filepath)` for atomic operation on POSIX
- Clean up temp file on error

**Fix 4: Thread-safe queries** (lines 200, 256, 300, 331, 353, 375, 383)
- Added `self._lock = threading.RLock()` in `__init__`
- Wrapped `create_escalation`, `resolve_escalation`, `update_status` with lock
- Wrapped `get_pending`, `get_by_swarm`, `get_blocked_work` with lock

**Fix 5: Hierarchy validation** (lines 245-254)
- Added validation in `create_escalation` before creating escalation
- Valid paths: AGENT->COO, COO->CEO only
- Raises `ValueError` for invalid paths

**Files Modified:**
- `/Users/jellingson/agent-swarm/shared/escalation_protocol.py`

**Testing:**
- Syntax verified by reading completed file
- Run `python3 -m py_compile shared/escalation_protocol.py` to verify

### 2026-01-03 - Claude CLI-Style Frontend Restyling
**Goal:** Restyle the Agent Swarm frontend to match Claude CLI's aesthetic - minimal, terminal-style dark theme with orange accents.

**Design Reference:**
- Deep black background: #0d0d0d (nearly pure black)
- Orange accent color: #ea580c / #f97316
- Monospace font: JetBrains Mono
- Minimal UI with clean lines, no gradients
- Subtle borders: zinc-800/50 separators
- White/light gray text for content

**Changes Implemented:**

**1. globals.css:**
- Added JetBrains Mono font import from Google Fonts
- Changed background from gradient to pure black (#0d0d0d)
- Updated CSS variables for new color scheme
- Changed agent-orchestrator color to orange (#ea580c)
- Added terminal cursor blink animation
- Added .terminal-prompt CSS class
- Simplified scrollbar styling (more minimal)
- Added fadeIn animation

**2. layout.tsx:**
- Switched font from Inter to JetBrains_Mono
- Updated body background to bg-[#0d0d0d]

**3. chat/page.tsx:**
- Replaced Bot icon with Terminal icon
- Changed purple-* colors to orange-* throughout
- Updated all bg-zinc-900 to bg-[#0d0d0d]
- Updated border colors to border-zinc-800/50 (more subtle)
- Added terminal-style prompt (>) in quick suggestions
- Made empty state use Terminal icon with orange tint

**4. ChatInput.tsx:**
- Added terminal prompt ">" character before textarea
- Changed send button from blue to orange
- Updated all backgrounds to #0d0d0d
- Changed hover states to orange accents
- Updated border styling to match terminal aesthetic

**5. ChatMessage.tsx:**
- Changed user avatar from blue to orange
- Updated "You" label to text-orange-500
- Changed text attachment icon to orange
- Updated backgrounds to pure black/subtle zinc

**6. AgentResponse.tsx:**
- Made orchestrator/supreme agent types use orange color
- Changed thinking indicator from purple to orange
- Updated CEO decision highlights to orange theme
- Changed thinking section styling to orange tints
- Made loading dots smaller and more subtle

**7. Sidebar.tsx:**
- Changed Bot logo to Terminal icon with orange color
- Updated active nav state to use orange accent
- Changed all blue-* references to orange-*
- Updated active swarm indicators to orange

**8. MobileLayout.tsx:**
- Added Terminal icon with orange accent in mobile header
- Updated overlay to pure black background
- Changed all background references to #0d0d0d

**9. ActivityPanel.tsx:**
- Changed all blue/purple accents to orange
- Updated status indicators (thinking, working, delegating) to orange
- Changed background to #0d0d0d
- Updated agent bot icons to orange

**Color Mappings Applied:**
- purple-* -> orange-*
- blue-* -> orange-*
- bg-zinc-950/bg-zinc-900 -> bg-[#0d0d0d]
- Primary accent: #ea580c

**Files Modified:**
- `/frontend/app/globals.css`
- `/frontend/app/layout.tsx`
- `/frontend/app/chat/page.tsx`
- `/frontend/components/ChatInput.tsx`
- `/frontend/components/ChatMessage.tsx`
- `/frontend/components/AgentResponse.tsx`
- `/frontend/components/Sidebar.tsx`
- `/frontend/components/MobileLayout.tsx`
- `/frontend/components/ActivityPanel.tsx`

### 2026-01-03 - Purple Accent Addition
**Goal:** Add subtle purple accents to complement the existing dark terminal theme with orange primary color.

**Purple Color Used:** #8B5CF6 (violet-500 in Tailwind)

**Changes Implemented:**

**1. globals.css:**
- Added purple CSS variables: `--accent-purple`, `--accent-purple-light`, `--accent-purple-dark`
- Added scrollbar hover state to purple
- Added `.purple-glow`, `.purple-glow-hover`, `.purple-glow-focus` utility classes
- Added `.border-purple-accent` and `.text-purple-accent` utility classes

**2. Sidebar.tsx:**
- Nav item hover: subtle purple background tint (`hover:bg-violet-500/5`)
- Active swarm border on working: purple tint (`border-violet-500/30`)
- Settings button hover: purple tint

**3. ChatInput.tsx:**
- Input focus: purple glow effect (`purple-glow-focus` class)
- Input border on focus: violet border (`focus-within:border-violet-500/50`)
- Attach button hover: purple accent (`hover:text-violet-400`, purple shadow)

**4. ActivityPanel.tsx:**
- Thinking status: violet color (`text-violet-400`)
- Delegating status: violet color
- Active badge: violet background/text
- Header hover: purple tint

**5. AgentResponse.tsx:**
- Thinking indicator: violet color
- Thinking section button hover: purple tint
- Thinking content: violet tinted background with left border

**6. FileBrowser.tsx:**
- Image file icon: violet color
- Selected file: violet background tint with left border
- Drag-over state: violet border/background
- Input focus states: violet border
- Create File button: violet background

**7. JobsPanel.tsx:**
- Running job status: violet color
- Running badge: violet background/text
- Progress bar: violet fill

**8. chat/page.tsx:**
- Quick suggestion buttons: purple hover effects
- Connection status dot: subtle green glow

**9. MobileLayout.tsx:**
- Mobile header: subtle purple shadow
- Hamburger menu button: purple hover

**Design Philosophy:**
- Purple is used for "thinking/processing" states (AI working)
- Purple hover effects add depth without changing the core orange accent
- Purple glow effects are subtle (15-20% opacity)
- Orange remains the primary action color (send button, terminal prompt)

**Files Modified:**
- `/frontend/app/globals.css`
- `/frontend/app/chat/page.tsx`
- `/frontend/components/Sidebar.tsx`
- `/frontend/components/ChatInput.tsx`
- `/frontend/components/ActivityPanel.tsx`
- `/frontend/components/AgentResponse.tsx`
- `/frontend/components/FileBrowser.tsx`
- `/frontend/components/JobsPanel.tsx`
- `/frontend/components/MobileLayout.tsx`

### 2026-01-03 - CeoTodoPanel localStorage Race Condition Fix
**Problem:** Todos were not persisting between page reloads due to a race condition in the useEffect hooks:
1. Initial state: `useState<Todo[]>([])` - empty array
2. First useEffect (load): Loads todos from localStorage
3. Second useEffect (save): Saves todos whenever they change

The save useEffect ran immediately on mount with the empty initial state, overwriting localStorage BEFORE the load useEffect had a chance to restore saved data.

**Solution Implemented:**
1. Added `useRef` to imports (line 3)
2. Added `const hasLoaded = useRef(false)` after state declarations (line 20)
3. Set `hasLoaded.current = true` after loading from localStorage completes (line 32)
4. Added early return in save useEffect if `hasLoaded.current` is false (line 37)

**Files Modified:**
- `/frontend/components/CeoTodoPanel.tsx`

## Next Steps
- Review and approve Local Neural Net Brain design (ADR-002)
- Set up Ollama locally for Phase 1 testing
- Begin Phase 1 implementation (basic inference integration)
- Continue Smart Context Injection development (ADR-001)

---

## Architecture Decisions

### ADR-002: Local Neural Net Brain Integration

**Date:** 2026-01-03
**Status:** PROPOSED
**Author:** System Architect
**Co-Author:** Research Specialist (model research)

---

#### Context

The agent-swarm system currently relies exclusively on cloud-based Claude API for all inference. This presents challenges:

1. **Cost**: Every query, regardless of complexity, incurs API costs
2. **Latency**: Round-trip to cloud adds 500ms-2s per request
3. **Privacy**: All user interactions sent to external servers
4. **Personalization**: No learning from user patterns and preferences
5. **Dependency**: System requires internet connectivity

**Goal:** Integrate a local neural network "brain" that can:
- Handle simple queries locally to reduce costs
- Learn from user interactions to personalize responses
- Store persistent behavioral patterns and preferences
- Intelligently route between local and cloud models

---

#### Decision

Implement a **Local Neural Net Brain** using **Ollama** as the inference server with **LoRA fine-tuning** for personalization.

**Architecture Overview:**

```
                                   +------------------+
                                   |   User Request   |
                                   +--------+---------+
                                            |
                                            v
+------------------+    route     +-------------------+
|  Query Router    |<-------------|  Confidence       |
|  (classifier)    |              |  Threshold (0.85) |
+--------+---------+              +-------------------+
         |
    +----+----+
    |         |
    v         v
+-------+  +-------+
| Local |  | Cloud |
| Brain |  | Claude|
+---+---+  +---+---+
    |          |
    +----+-----+
         |
         v
+------------------+              +-------------------+
|    Response      |------------->|  Interaction      |
|    Merger        |              |  Logger           |
+------------------+              +-------------------+
                                            |
                                            v
                                  +-------------------+
                                  |  Training Data    |
                                  |  Collector        |
                                  +-------------------+
```

---

#### 1. Integration Points

##### 1.1 Primary Hook: `backend/main.py` - `websocket_chat()` (Lines 1656-1937)

```python
# After receiving message, before Claude processing
from local_brain import get_local_brain

local_brain = get_local_brain()

# Step 1: Check if local brain can handle this
routing_decision = await local_brain.route_query(
    prompt=user_message,
    context=conversation_history,
    session_id=session_id,
)

if routing_decision.use_local:
    # Local inference via Ollama
    local_response = await local_brain.generate(
        prompt=user_message,
        context=conversation_history,
        max_tokens=1024,
    )

    # Send response with local indicator
    await manager.send_event(websocket, "agent_start", {
        "agent": "Local Brain",
        "agent_type": "local",
        "model": local_brain.model_name,
    })

    await manager.send_event(websocket, "agent_delta", {
        "agent": "Local Brain",
        "delta": local_response.text,
    })

    # Log for learning
    await local_brain.log_interaction(
        prompt=user_message,
        response=local_response.text,
        routing_decision=routing_decision,
        session_id=session_id,
    )
else:
    # Continue with existing Claude CLI flow (lines 1832-1881)
    # After response, log for learning
    await local_brain.log_interaction(
        prompt=user_message,
        response=final_content,
        routing_decision=routing_decision,
        cloud_used=True,
        session_id=session_id,
    )
```

##### 1.2 Agent Executor Pool Integration

Modify `shared/agent_executor_pool.py`:

```python
async def execute(
    self,
    context: AgentExecutionContext,
    prompt: str,
    system_prompt: str | None = None,
    on_event: Callable[[dict], None] | None = None,
    prefer_local: bool = False,  # NEW PARAMETER
) -> AsyncIterator[dict[str, Any]]:

    if prefer_local and self._local_brain_available():
        async for event in self._execute_local(context, prompt):
            yield event
    else:
        async for event in self._run_agent(...):
            yield event
```

---

#### 2. Data Flow

##### 2.1 Training Data Sources

| Source | Path | Data Extracted |
|--------|------|----------------|
| Chat sessions | `logs/chat/*.json` | User prompts + agent responses |
| Session summaries | `memory/sessions/*.md` | Interaction patterns |
| Swarm context | `swarms/*/workspace/STATE.md` | Domain knowledge |
| Decisions | `memory/core/decisions.md` | Preferences |

##### 2.2 Interaction Schema

```python
@dataclass
class TrainingInteraction:
    id: str
    timestamp: datetime
    session_id: str

    # Input
    prompt: str
    context: list[dict]

    # Output
    response: str
    agent_used: str
    tools_used: list[str]

    # Routing
    local_handled: bool
    local_confidence: float

    # Feedback
    feedback_type: str | None  # "approve", "reject", "edit"
    edited_response: str | None
    feedback_timestamp: datetime | None

    # Metadata
    swarm_name: str | None
    response_time_ms: int
    token_count: int
```

##### 2.3 User Feedback Capture

**Frontend (AgentResponse.tsx):**
- Add thumbs up/down buttons
- Add edit capability for corrections
- POST to `/api/feedback` endpoint

**Backend Endpoint:**
```python
@app.post("/api/feedback")
async def submit_feedback(request: FeedbackRequest):
    local_brain = get_local_brain()
    await local_brain.record_feedback(
        message_id=request.message_id,
        feedback_type=request.type,
        edited_content=request.edited_content,
        session_id=request.session_id,
    )
    return {"success": True}
```

##### 2.4 Storage Locations

| Data Type | Location | Format |
|-----------|----------|--------|
| Training interactions | `local_brain/data/interactions/` | JSONL (daily rotation) |
| Model weights | `local_brain/models/` | Safetensors |
| LoRA adapters | `local_brain/models/lora/` | Safetensors |
| User preferences | `local_brain/data/preferences.json` | JSON |
| Routing statistics | `local_brain/data/routing_stats.json` | JSON |

---

#### 3. Architecture Components

##### 3.1 Local Inference Server: Ollama

**Why Ollama:**
- Simple installation and model management
- REST API compatible with OpenAI format
- Efficient memory management (Apple Silicon optimized)
- Built-in model caching
- Active development

**Client Implementation:**

```python
# local_brain/inference/ollama_client.py
@dataclass
class OllamaConfig:
    base_url: str = "http://localhost:11434"
    model: str = "llama3.2:3b"  # or "qwen2.5:7b"
    timeout: float = 60.0

class OllamaClient:
    async def generate(
        self,
        prompt: str,
        system: str | None = None,
        context: list[dict] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        if context:
            messages.extend(context)
        messages.append({"role": "user", "content": prompt})

        response = await self._client.post("/api/chat", json={
            "model": self.config.model,
            "messages": messages,
            "options": {"num_predict": max_tokens, "temperature": temperature},
            "stream": False,
        })
        return response.json()["message"]["content"]

    async def is_available(self) -> bool:
        try:
            return (await self._client.get("/api/tags")).status_code == 200
        except Exception:
            return False
```

##### 3.2 Query Router

```python
# local_brain/router/query_router.py
class QueryComplexity(Enum):
    SIMPLE = "simple"      # Greetings, yes/no
    MODERATE = "moderate"  # Single-step tasks
    COMPLEX = "complex"    # Multi-step, tool usage

class QueryIntent(Enum):
    GREETING = "greeting"
    CLARIFICATION = "clarification"
    SIMPLE_QUESTION = "simple_question"
    CODE_GENERATION = "code_generation"
    FILE_OPERATION = "file_operation"
    RESEARCH = "research"
    ORCHESTRATION = "orchestration"

@dataclass
class RoutingDecision:
    use_local: bool
    confidence: float
    intent: QueryIntent
    complexity: QueryComplexity
    reasoning: str

class QueryRouter:
    LOCAL_CAPABLE_INTENTS = {
        QueryIntent.GREETING,
        QueryIntent.CLARIFICATION,
        QueryIntent.SIMPLE_QUESTION,
    }

    CLOUD_REQUIRED_PATTERNS = [
        r"write.*code",
        r"implement.*",
        r"create.*file",
        r"fix.*bug",
        r"run.*command",
        r"Task\(",  # Delegation
    ]

    def __init__(self, confidence_threshold: float = 0.85):
        self.confidence_threshold = confidence_threshold

    async def analyze(self, prompt: str, history: list | None = None) -> RoutingDecision:
        # Pattern match for cloud-required
        if self._requires_cloud(prompt):
            return RoutingDecision(use_local=False, confidence=1.0, ...)

        # Classify intent and complexity
        intent = await self._classify_intent(prompt)
        complexity = self._assess_complexity(prompt, history)
        confidence = self._calculate_confidence(intent, complexity, prompt)

        use_local = (
            intent in self.LOCAL_CAPABLE_INTENTS and
            complexity == QueryComplexity.SIMPLE and
            confidence >= self.confidence_threshold
        )

        return RoutingDecision(use_local=use_local, confidence=confidence, ...)
```

##### 3.3 Training Pipeline (LoRA)

```python
# local_brain/training/lora_trainer.py
@dataclass
class TrainingConfig:
    base_model: str = "meta-llama/Llama-3.2-3B"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    learning_rate: float = 2e-4
    batch_size: int = 4
    max_steps: int = 500

class LoRATrainer:
    def prepare_training_data(self, interactions_dir: Path) -> list[dict]:
        """Convert logged interactions to training format."""
        training_data = []
        for file in interactions_dir.glob("*.jsonl"):
            for line in file.read_text().splitlines():
                interaction = json.loads(line)
                if interaction.get("feedback_type") == "approve":
                    training_data.append({
                        "prompt": interaction["prompt"],
                        "response": interaction["response"],
                    })
                elif interaction.get("feedback_type") == "edit":
                    training_data.append({
                        "prompt": interaction["prompt"],
                        "response": interaction["edited_response"],
                    })
        return training_data

    async def train(self, training_data: list[dict]) -> Path:
        """Run LoRA fine-tuning using PEFT."""
        # Implementation using transformers + peft
        # Returns path to saved adapter
```

##### 3.4 Memory Integration

Extends existing `backend/memory.py`:

```python
# local_brain/memory/preference_memory.py
class PreferenceMemory:
    """Persistent storage for learned user preferences."""

    PREFERENCE_CATEGORIES = [
        "response_style",      # concise, detailed, technical
        "code_preferences",    # formatting, languages
        "interaction_patterns", # usage patterns
        "domain_expertise",    # frequent topics
    ]

    def learn_from_interaction(self, interaction: dict):
        """Extract and update preferences from interaction."""
        if interaction.get("feedback_type") == "edit":
            self._learn_from_edit(
                original=interaction["response"],
                edited=interaction["edited_response"],
            )

    def get_context_for_prompt(self, prompt: str) -> str:
        """Generate preference context to inject into prompts."""
        relevant_prefs = [
            f"- {pref.key}: {pref.value}"
            for pref in self.preferences.values()
            if pref.confidence > 0.7
        ]
        if relevant_prefs:
            return "User preferences:\n" + "\n".join(relevant_prefs)
        return ""
```

---

#### 4. File Structure

```
local_brain/
├── __init__.py                    # Package exports, get_local_brain()
├── config.py                      # Configuration dataclasses
├── brain.py                       # Main LocalBrain orchestrator
│
├── inference/                     # Local model inference
│   ├── __init__.py
│   ├── ollama_client.py          # Ollama API client
│   ├── base_client.py            # Abstract interface
│   └── model_manager.py          # Model download/management
│
├── router/                        # Query routing
│   ├── __init__.py
│   ├── query_router.py           # Main routing logic
│   ├── intent_classifier.py      # Intent classification
│   ├── complexity_scorer.py      # Complexity assessment
│   └── patterns.py               # Regex patterns
│
├── training/                      # Fine-tuning
│   ├── __init__.py
│   ├── lora_trainer.py           # LoRA implementation
│   ├── data_processor.py         # Training data prep
│   └── evaluation.py             # Metrics
│
├── memory/                        # Learning storage
│   ├── __init__.py
│   ├── interaction_logger.py     # Log interactions
│   ├── preference_memory.py      # User preferences
│   └── feedback_handler.py       # Process feedback
│
├── data/                          # Data storage (gitignored)
│   ├── interactions/             # JSONL files
│   ├── preferences.json
│   ├── routing_stats.json
│   └── embeddings/
│
├── models/                        # Weights (gitignored)
│   ├── lora/
│   └── cache/
│
└── tests/
    ├── test_router.py
    ├── test_inference.py
    ├── test_training.py
    └── test_memory.py
```

**Files to Modify:**

| File | Changes |
|------|---------|
| `backend/main.py` | LocalBrain integration in websocket_chat() |
| `config.yaml` | Add local_brain configuration section |
| `shared/agent_executor_pool.py` | Add prefer_local option |
| `frontend/components/AgentResponse.tsx` | Add feedback buttons |
| `frontend/lib/api.ts` | Add feedback endpoint |
| `frontend/components/ActivityPanel.tsx` | Show local vs cloud indicator |

---

#### 5. Implementation Phases

##### Phase 1: MVP - Basic Local Inference (1-2 weeks)
1. Create `local_brain/` directory structure
2. Implement OllamaClient with health checks
3. Create basic QueryRouter with pattern matching
4. Add routing hook in backend/main.py
5. Frontend indicator for local vs cloud
6. Basic interaction logging

**Success Criteria:**
- Greetings handled locally
- Clear UI indication of local processing
- Seamless fallback to cloud
- All interactions logged

##### Phase 2: Feedback Loop (1 week)
1. Add feedback buttons to AgentResponse.tsx
2. Create /api/feedback endpoint
3. Implement FeedbackHandler
4. Store feedback with interaction links

##### Phase 3: Preference Learning (1-2 weeks)
1. Implement PreferenceMemory
2. Extract preferences from feedback
3. Inject preferences into prompts
4. Add preference display in settings

##### Phase 4: LoRA Fine-tuning (2-3 weeks)
1. Implement LoRATrainer
2. Create training data processor
3. Add training job to queue system
4. Model evaluation and adapter management

##### Phase 5: Advanced Routing (1-2 weeks)
1. Train intent classifier
2. Implement hybrid mode
3. Routing confidence calibration
4. Adaptive threshold adjustment

##### Phase 6: Production Hardening (1 week)
1. Comprehensive error handling
2. Graceful degradation
3. Telemetry and metrics
4. Documentation

---

#### Trade-offs

**Pros:**
- 50-90% cost reduction for simple queries
- 5-10x latency improvement for local inference
- Personalization improves over time
- Works offline for supported queries
- Privacy for local processing

**Cons:**
- Additional system complexity
- Requires local compute (8GB+ RAM)
- Training needs significant interactions (1000+)
- Lower quality than Claude for complex tasks

**Mitigations:**
- Conservative routing (high confidence threshold)
- Clear UI indicators
- Easy disable option
- Graceful fallback on errors

---

#### Hardware Requirements

**Minimum:** CPU quad-core, 8GB RAM, 10GB storage
**Recommended:** 8+ cores, 16GB+ RAM, 50GB storage
**macOS:** M1/M2/M3 with 16GB unified memory (ideal)

---

#### Dependencies

**Required:**
- Ollama (local inference)
- httpx (async HTTP)

**Training (Phase 4+):**
- transformers
- peft
- torch
- bitsandbytes (8-bit training)

---

#### Success Metrics

1. API calls reduced by 30%+ for simple queries
2. Response latency under 200ms for local
3. Routing accuracy above 90%
4. User feedback approval rate above 80%
5. System uptime above 99%

---

#### Configuration

```yaml
# config.yaml addition
local_brain:
  enabled: true
  inference:
    provider: "ollama"
    base_url: "http://localhost:11434"
    model: "llama3.2:3b"
    timeout: 30
  router:
    confidence_threshold: 0.85
    local_capable_intents:
      - greeting
      - clarification
      - simple_question
    cloud_required_patterns:
      - "write.*code"
      - "implement"
      - "create.*file"
      - "Task\\("
  training:
    enabled: false  # Enable after Phase 4
    schedule: "weekly"
    min_interactions: 100
```

---

**Next Steps:**
1. Review and approve this design
2. Install Ollama: `brew install ollama && ollama pull llama3.2:3b`
3. Implementer builds Phase 1
4. Critic review before integration

---

### ADR-001: Smart Context Injection System

**Date:** 2026-01-03
**Status:** PROPOSED
**Author:** System Architect

---

#### Context

The current agent swarm system requires agents to manually read relevant files before they can work on a task. When a user asks about "auth" or "websocket", the agent must first discover which files are relevant, then read them. This leads to:

1. **Wasted turns** - Agents spend turns exploring the codebase before doing useful work
2. **Missed context** - Agents may not find all relevant files
3. **Inconsistent quality** - Success depends on agent's exploration strategy
4. **Slower responses** - Extra round-trips to discover and read files

**Goal:** Auto-detect relevant files for each task and inject them into the agent's context, reducing manual exploration.

---

#### Decision

Implement a **Smart Context Injection** system with the following architecture:

```
+------------------+     +-------------------+     +------------------+
|   User Prompt    | --> | Context Analyzer  | --> | File Relevance   |
|                  |     | (Topic Detection) |     | Scorer           |
+------------------+     +-------------------+     +------------------+
                                                           |
                                                           v
+------------------+     +-------------------+     +------------------+
| Agent Execution  | <-- | Context Injector  | <-- | Codebase Index   |
| (with context)   |     | (Prompt Builder)  |     | (Topic -> Files) |
+------------------+     +-------------------+     +------------------+
```

**Core Components:**

1. **CodebaseIndexer** - Builds and maintains a topic-to-file mapping
2. **ContextAnalyzer** - Extracts topics/keywords from user prompts
3. **RelevanceScorer** - Ranks files by relevance to detected topics
4. **ContextInjector** - Builds enhanced prompts with injected file context

---

#### Component Design

##### 1. CodebaseIndexer (`shared/context_indexer.py`)

Builds an index mapping topics/concepts to relevant files.

**Indexing Strategies:**
- **File path analysis**: `auth.py` -> ["auth", "authentication"]
- **Function/class extraction**: Parse AST for public symbols
- **Comment/docstring analysis**: Extract semantic meaning
- **Import graph**: Files that import auth-related modules
- **Keyword extraction**: TF-IDF or simple frequency analysis

**Index Structure:**
```python
{
    "auth": {
        "files": [
            {"path": "backend/auth.py", "score": 1.0, "type": "primary"},
            {"path": "backend/middleware/auth_middleware.py", "score": 0.8, "type": "related"},
        ],
        "keywords": ["login", "jwt", "token", "session", "permission"]
    },
    "websocket": {
        "files": [
            {"path": "backend/main.py", "score": 0.9, "sections": ["ConnectionManager", "websocket_chat"]},
            {"path": "frontend/lib/websocket.ts", "score": 1.0, "type": "primary"},
        ],
        "keywords": ["ws", "socket", "real-time", "streaming"]
    }
}
```

**Persistence:** JSON file at `.claude/context_index.json`, rebuilt on demand or on file changes.

**Update Strategy:**
- Full rebuild: On startup or manual trigger
- Incremental: On file save (via file watcher or git hook)
- Lazy: Only index files when first queried

##### 2. ContextAnalyzer (`shared/context_analyzer.py`)

Extracts topics and intent from user prompts.

**Analysis Methods:**
- **Keyword extraction**: Match against known topic vocabulary
- **Intent classification**: "fix", "implement", "explain", "review"
- **Entity extraction**: File paths, function names, error messages
- **Semantic similarity**: Embed prompt and compare to topic embeddings (optional, heavier)

**Output:**
```python
{
    "topics": ["websocket", "connection", "error"],
    "intent": "fix",
    "entities": {
        "files": [],
        "functions": ["websocket_chat"],
        "errors": ["WebSocketDisconnect"]
    },
    "confidence": 0.85
}
```

##### 3. RelevanceScorer (`shared/relevance_scorer.py`)

Ranks and filters files based on detected topics.

**Scoring Factors:**
- **Topic match score**: Direct match = 1.0, related = 0.5-0.8
- **Recency**: Recently modified files get a boost
- **Centrality**: Files imported by many others rank higher
- **Size penalty**: Very large files get lower scores (harder to process)
- **STATE.md bonus**: Swarm STATE.md files always included

**Filtering:**
- **Max files**: Default 5-10 relevant files
- **Max tokens**: Stay within ~50k token budget for context
- **Min score threshold**: Exclude files below 0.3 relevance

##### 4. ContextInjector (`shared/context_injector.py`)

Builds the enhanced prompt with injected context.

**Injection Modes:**
- **File summaries**: Just file paths + brief descriptions
- **Key sections**: Relevant functions/classes extracted
- **Full content**: Complete file contents (for small files)
- **Hybrid**: Mix based on file size and relevance

**Prompt Template:**
```
## Relevant Context (auto-detected)

The following files appear relevant to your request:

### backend/main.py (WebSocket handling)
Lines 1110-1200: ConnectionManager class
Lines 1656-1937: websocket_chat endpoint
[Full content or summary]

### frontend/lib/websocket.ts
[Full content - 150 lines]

---

## Your Request

{original_prompt}

---

Note: Relevant files have been pre-loaded. Use Read tool for additional files if needed.
```

---

#### Integration Points

##### Backend Integration (`backend/main.py`)

Modify `websocket_chat()` at lines 1756-1826 where the user prompt is processed:

```python
# After receiving message, before building prompt
from shared.context_injector import inject_context

# Analyze and inject relevant context
enhanced_prompt, injected_files = await inject_context(
    prompt=user_message,
    workspace=PROJECT_ROOT,
    max_files=8,
    max_tokens=50000,
)

# Use enhanced_prompt instead of user_message
user_prompt = enhanced_prompt
```

##### Agent Executor Integration (`shared/agent_executor_pool.py`)

Add optional context injection in `execute()` method:

```python
async def execute(
    self,
    context: AgentExecutionContext,
    prompt: str,
    system_prompt: str | None = None,
    inject_context: bool = True,  # New parameter
    ...
):
    if inject_context:
        prompt = await self._inject_relevant_context(prompt, context.workspace)
    ...
```

##### CLI Agent Spawning (`stream_claude_response()`)

Add context injection before spawning Claude CLI:

```python
# Before building cmd
if inject_context:
    from shared.context_injector import inject_context
    prompt = await inject_context(prompt, workspace)
```

---

#### Implementation Plan

**Phase 1: Core Infrastructure (2-3 days)**
1. Create `shared/context_indexer.py` with basic file-path-based indexing
2. Create `shared/context_analyzer.py` with keyword extraction
3. Create `shared/relevance_scorer.py` with simple scoring
4. Create `shared/context_injector.py` with basic injection
5. Unit tests for each component

**Phase 2: Backend Integration (1-2 days)**
1. Integrate into `websocket_chat()` in `backend/main.py`
2. Add configuration options (enable/disable, max files, etc.)
3. Add WebSocket event for showing injected context to frontend
4. Test with real prompts

**Phase 3: Advanced Features (2-3 days)**
1. AST-based indexing for Python files
2. Import graph analysis for related files
3. Section extraction (inject relevant functions, not whole files)
4. Caching layer for index
5. Incremental index updates

**Phase 4: Observability (1 day)**
1. Logging of injected context
2. Frontend indicator showing "N files auto-loaded"
3. Metrics on context injection effectiveness

---

#### Files to Create

| File | Purpose |
|------|---------|
| `shared/context_indexer.py` | Build and maintain topic -> file mapping |
| `shared/context_analyzer.py` | Extract topics from user prompts |
| `shared/relevance_scorer.py` | Score and rank relevant files |
| `shared/context_injector.py` | Build enhanced prompts with context |
| `shared/context_config.py` | Configuration for context injection |
| `.claude/context_index.json` | Persistent index (generated) |
| `tests/test_context_injection.py` | Unit tests |

---

#### Files to Modify

| File | Changes |
|------|---------|
| `backend/main.py` | Add context injection in `websocket_chat()` |
| `shared/agent_executor_pool.py` | Optional context injection parameter |
| `frontend/components/ActivityPanel.tsx` | Show injected files indicator |

---

#### Trade-offs and Considerations

**Pros:**
- Reduces agent exploration time significantly
- Improves first-response quality
- Consistent context loading
- Agents can focus on the actual task

**Cons:**
- Additional complexity in the system
- Index may become stale
- May inject irrelevant files (false positives)
- Context window consumption
- Performance overhead for indexing

**Mitigations:**
- Keep index simple initially (file-path based)
- Set conservative defaults (5-8 files max)
- Allow agents to request more context via Read tool
- Show injected files to user for transparency
- Make injection optional/configurable

---

#### Performance Considerations

1. **Index Build Time**: Initial full index ~1-5 seconds for typical codebase
2. **Query Time**: Topic matching should be <50ms
3. **Context Injection**: File reading parallelized, <200ms for 10 files
4. **Token Budget**: Stay under 50k tokens to leave room for agent work
5. **Caching**: Index cached in memory, rebuilt on significant changes

---

#### Alternatives Considered

1. **RAG with embeddings**: More accurate but requires embedding infrastructure
2. **LLM-based file selection**: Send file list to LLM to pick relevant ones - expensive and slow
3. **Manual hints**: User tags files - doesn't solve the problem
4. **Full codebase in context**: Too large, expensive, noisy

**Chosen approach** balances simplicity, speed, and effectiveness.

---

#### Success Metrics

1. **Reduction in exploration turns**: Measure average turns before useful work
2. **First-response quality**: Does the agent's first response show understanding?
3. **File hit rate**: Are injected files actually used by agents?
4. **User satisfaction**: Fewer "read this file first" instructions needed

---

#### Dependencies

- No external dependencies for basic implementation
- Optional: `tree-sitter` for AST parsing (Phase 3)
- Optional: Sentence embeddings for semantic matching (future enhancement)

---

**Next Steps:**
1. Review and approve this design
2. Implementer to build Phase 1 components
3. Critic review before integration
4. Test with real swarm tasks
