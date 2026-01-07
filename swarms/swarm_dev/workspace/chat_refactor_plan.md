# Chat Page Refactoring Plan

**File:** `frontend/app/chat/page.tsx` (~1070 lines)
**Created:** 2026-01-06
**Status:** Planning

---

## Executive Summary

The chat page has grown into a monolithic component handling:
- **WebSocket event handling** (40+ event types in one 400-line switch statement)
- **Session CRUD operations** (create, load, delete, list)
- **Message state management** (streaming, thinking, completion)
- **Multiple UI concerns** (sidebar, mobile sheets, chat area, activity panel)

This plan breaks the refactoring into 6 incremental phases, each producing a testable, working state.

---

## Phase 1: Extract `useWebSocketEvents` Hook

**Priority:** High
**Risk:** Medium
**Lines affected:** 292-723

### Current Problems
1. The WebSocket event handler (lines 296-688) is a 400-line switch statement embedded in a useEffect
2. Event handlers directly mutate multiple pieces of state (`messages`, `agentActivities`, `toolActivities`, `isLoading`, etc.)
3. Tight coupling between event parsing and state updates makes testing impossible

### Target Interface

```typescript
// frontend/hooks/useWebSocketEvents.ts

interface UseWebSocketEventsOptions {
  onChatStart: () => void
  onChatComplete: () => void
  onAgentStart: (agent: string, agentType: string) => void
  onAgentDelta: (delta: string) => void
  onAgentComplete: (content: string, agent: string, thinking?: string) => void
  onThinkingStart: () => void
  onThinkingDelta: (delta: string) => void
  onThinkingComplete: (thinking: string) => void
  onToolStart: (tool: string, description: string, agentName?: string) => void
  onToolComplete: (tool: string, success: boolean, summary?: string) => void
  onAgentSpawn: (agent: string, parentAgent: string, description?: string) => void
  onAgentSubComplete: (agent: string) => void
  onError: (message: string) => void
}

interface UseWebSocketEventsReturn {
  isConnected: boolean
  send: (message: string, options?: SendOptions) => void
}

function useWebSocketEvents(options: UseWebSocketEventsOptions): UseWebSocketEventsReturn
```

### Implementation Steps

1. **Create hook file** at `frontend/hooks/useWebSocketEvents.ts`
2. **Extract event type handlers** into individual functions:
   ```typescript
   const handleChatStart = useCallback(() => { ... }, [deps])
   const handleToolStart = useCallback((event) => { ... }, [deps])
   // etc.
   ```
3. **Create event router** that maps event types to handlers
4. **Move WebSocket connection logic** from chat page to hook
5. **Update chat page** to use the new hook with callbacks

### Testing Strategy
- Create mock WebSocket for unit testing
- Test each event handler in isolation
- Integration test: verify event → state update flow

### Rollback Plan
- Keep old implementation commented for 1 PR cycle
- Feature flag: `USE_NEW_WEBSOCKET_HOOK=true`

---

## Phase 2: Extract `useChatSessions` Hook

**Priority:** High
**Risk:** Low
**Lines affected:** 161-228, 248-290

### Current Problems
1. Session CRUD scattered across multiple `useCallback` definitions
2. Session loading duplicated in initialization (lines 253-283) and `loadSession` (lines 172-201)
3. No clear separation between session state and message state

### Target Interface

```typescript
// frontend/hooks/useChatSessions.ts

interface UseChatSessionsOptions {
  onSessionLoad?: (messages: Message[]) => void
  onSessionCreate?: (sessionId: string) => void
}

interface UseChatSessionsReturn {
  sessions: ChatSessionSummary[]
  sessionId: string | null
  isLoading: boolean
  loadSessions: () => Promise<void>
  loadSession: (id: string) => Promise<void>
  createSession: () => Promise<string>
  deleteSession: (id: string) => Promise<void>
  saveMessage: (role: 'user' | 'assistant', content: string, agent?: string, thinking?: string) => Promise<void>
}

function useChatSessions(options?: UseChatSessionsOptions): UseChatSessionsReturn
```

### Implementation Steps

1. **Create hook file** at `frontend/hooks/useChatSessions.ts`
2. **Extract session state**: `sessions`, `sessionId`
3. **Consolidate message deduplication logic** (currently duplicated at lines 189-196 and 272-279)
4. **Fix initialization race condition** (see Phase 5)
5. **Update chat page** to use the hook

### Deduplication Logic (to consolidate)

```typescript
// Current duplicated logic (lines 189-196, 272-279)
const dedupeMessages = (messages: Message[]): Message[] => {
  const seen = new Set<string>()
  return messages.filter(m => {
    const key = `${m.type}:${m.content.substring(0, 200)}`
    if (seen.has(key)) return false
    seen.add(key)
    return true
  })
}
```

### Testing Strategy
- Mock API calls with MSW
- Test session CRUD operations
- Test message deduplication

---

## Phase 3: Extract `useMessageState` Hook

**Priority:** Medium
**Risk:** Medium
**Lines affected:** 35, 452-656

### Current Problems
1. Message state updates are scattered throughout WebSocket handlers
2. Complex logic for finding/updating "thinking" messages
3. Streaming message synchronization with context is fragile (lines 88-102)

### Target Interface

```typescript
// frontend/hooks/useMessageState.ts

interface Message {
  id: string
  type: 'user' | 'agent'
  content: string
  agent?: string
  agentType?: string
  status?: 'thinking' | 'complete'
  timestamp: Date
  attachments?: Attachment[]
  thinking?: string
  isThinking?: boolean
}

interface UseMessageStateReturn {
  messages: Message[]

  // User messages
  addUserMessage: (content: string, attachments?: Attachment[]) => void

  // Agent messages
  startAgentMessage: (agent: string, agentType: string) => string // returns message ID
  updateAgentContent: (delta: string) => void
  startThinking: () => void
  updateThinking: (delta: string) => void
  completeThinking: (thinking: string) => void
  completeAgentMessage: (content: string, agent: string, thinking?: string) => void

  // Bulk operations
  setMessages: React.Dispatch<React.SetStateAction<Message[]>>
  markAllComplete: () => void

  // Streaming sync (for context persistence)
  syncWithStreamingContext: (streamingMessage: StreamingMessage | null) => void
}

function useMessageState(
  streamingMessage: StreamingMessage | null,
  setStreamingMessage: (msg: StreamingMessage | null) => void
): UseMessageStateReturn
```

### Implementation Steps

1. **Create hook file** at `frontend/hooks/useMessageState.ts`
2. **Extract message finding logic**:
   ```typescript
   // Current pattern (repeated 5+ times)
   const thinkingIdx = prev.findIndex(m => m.type === 'agent' && m.status === 'thinking')
   ```
3. **Encapsulate streaming message synchronization** (lines 70-102)
4. **Create atomic update functions** instead of inline setMessages calls
5. **Update chat page** to use the hook

### Key Refactoring: Message Finding

Current (scattered throughout):
```typescript
setMessages((prev) => {
  const thinkingIdx = prev.findIndex(m => m.type === 'agent' && m.status === 'thinking')
  if (thinkingIdx !== -1) {
    const updated = [...prev]
    updated[thinkingIdx] = { ...prev[thinkingIdx], content: ... }
    return updated
  }
  return prev
})
```

Target (encapsulated):
```typescript
updateAgentContent(delta)  // Handles finding + updating internally
```

---

## Phase 4: Extract UI Components

**Priority:** Low
**Risk:** Low
**Lines affected:** 812-1103

### Components to Extract

#### 4.1 `SessionSidebar` Component
**Lines:** 815-856
**Props:**
```typescript
interface SessionSidebarProps {
  sessions: ChatSessionSummary[]
  currentSessionId: string | null
  showSidebar: boolean
  onToggleSidebar: () => void
  onSessionClick: (id: string) => void
  onNewSession: () => void
  onDeleteSession: (id: string, e: React.MouseEvent) => void
}
```

#### 4.2 `MobileHistorySheet` Component
**Lines:** 858-928
**Props:**
```typescript
interface MobileHistorySheetProps {
  isOpen: boolean
  onClose: () => void
  sessions: ChatSessionSummary[]
  currentSessionId: string | null
  onSessionClick: (id: string) => void
  onNewSession: () => void
  onDeleteSession: (id: string, e: React.MouseEvent) => void
}
```

#### 4.3 `MobileActivitySheet` Component
**Lines:** 930-967
**Props:**
```typescript
interface MobileActivitySheetProps {
  isOpen: boolean
  onClose: () => void
  agents: PanelAgentActivity[]
  tools: PanelToolActivity[]
  isProcessing: boolean
  onClear: () => void
}
```

#### 4.4 `ChatHeader` Component
**Lines:** 972-1029
**Props:**
```typescript
interface ChatHeaderProps {
  isConnected: boolean
  isMobile: boolean
  showSidebar: boolean
  hasActivity: boolean
  isActivityProcessing: boolean
  onToggleSidebar: () => void
  onShowHistory: () => void
  onShowActivity: () => void
}
```

#### 4.5 `EmptyState` Component
**Lines:** 1033-1060
**Props:**
```typescript
interface EmptyStateProps {
  onQuickSend: (message: string) => void
}
```

### Implementation Steps

1. Create `frontend/components/chat/` directory
2. Extract each component to its own file
3. Keep styling inline (Tailwind) to match current pattern
4. Update imports in chat page

---

## Phase 5: Fix Race Condition in Session Creation

**Priority:** Critical
**Risk:** High
**Lines affected:** 731-742

### Current Problem

```typescript
// Lines 731-742 - RACE CONDITION
let currentSessionId = sessionId  // Captures stale closure value
if (!currentSessionId) {
  try {
    const session = await createChatSession()
    setSessionId(session.id)         // React state update (async)
    currentSessionId = session.id    // Local variable update (sync)
    await loadSessions()             // Another async operation
  } catch (e) {
    console.error('Failed to create session:', e)
    return
  }
}
// Problem: If user sends two messages quickly before session exists:
// 1. First message: currentSessionId is null, creates session A
// 2. Second message: currentSessionId is STILL null (state not updated yet)
// 3. Second message creates session B
// Result: Two sessions created, messages split
```

### Root Causes

1. **Stale closure**: `sessionId` captured at callback creation time
2. **No guard against concurrent session creation**
3. **State update is async**: `setSessionId` doesn't immediately update `sessionId`

### Solution: Session Creation Lock

```typescript
// frontend/hooks/useChatSessions.ts

interface UseChatSessionsReturn {
  // ... existing fields
  ensureSession: () => Promise<string>  // Returns session ID, creates if needed
}

function useChatSessions(): UseChatSessionsReturn {
  const [sessionId, setSessionId] = useState<string | null>(null)
  const sessionCreationLock = useRef<Promise<string> | null>(null)

  const ensureSession = useCallback(async (): Promise<string> => {
    // Fast path: session already exists
    if (sessionId) return sessionId

    // Check if creation is already in progress (prevents race condition)
    if (sessionCreationLock.current) {
      return sessionCreationLock.current
    }

    // Create session with lock
    sessionCreationLock.current = (async () => {
      try {
        const session = await createChatSession()
        setSessionId(session.id)
        await loadSessions()
        return session.id
      } finally {
        sessionCreationLock.current = null
      }
    })()

    return sessionCreationLock.current
  }, [sessionId, loadSessions])

  return { /* ... */ ensureSession }
}
```

### Updated `handleSend`

```typescript
const handleSend = useCallback(async (content: string, attachments: Attachment[]) => {
  // Clear old activities
  setAgentActivities(prev => prev.filter(a => a.status !== 'complete' && a.status !== 'error'))
  setToolActivities(prev => prev.filter(t => t.status === 'running'))

  // Get or create session (with race condition protection)
  let currentSessionId: string
  try {
    currentSessionId = await ensureSession()
  } catch (e) {
    console.error('Failed to ensure session:', e)
    return
  }

  // ... rest of handleSend
}, [ensureSession, /* other deps */])
```

### Testing Strategy

1. **Unit test**: Call `ensureSession()` twice simultaneously, verify only one session created
2. **Integration test**: Rapidly send two messages, verify both go to same session
3. **Stress test**: 10 concurrent sends, verify single session

---

## Phase 6: Fix Missing useEffect Dependency

**Priority:** Medium
**Risk:** Low
**Lines affected:** 290

### Current Problem

```typescript
// Line 290
}, []) // eslint-disable-line react-hooks/exhaustive-deps
```

The initialization useEffect has missing dependencies and uses an eslint-disable comment.

### Dependencies Analysis

The effect uses:
- `getChatSessions` - API function (stable, no dep needed)
- `getChatSession` - API function (stable, no dep needed)
- `setSessions` - State setter (stable, no dep needed)
- `setSessionId` - State setter (stable, no dep needed)
- `setMessages` - State setter (stable, no dep needed)
- `createNewSession` - **Callback that changes** (defined at line 204)
- `getDisplayName` - Helper function (stable)

### The Real Issue

The `createNewSession` dependency is intentionally excluded because:
1. `createNewSession` depends on `loadSessions`
2. `loadSessions` is created with `useCallback`
3. Including it would cause the effect to re-run when `loadSessions` changes

### Solution: Extract Initialization Logic

```typescript
// Move initialization to useChatSessions hook
function useChatSessions(): UseChatSessionsReturn {
  const [sessions, setSessions] = useState<ChatSessionSummary[]>([])
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [messages, setMessages] = useState<Message[]>([])
  const initRef = useRef(false)

  // Initialize on mount - self-contained, no external deps
  useEffect(() => {
    if (initRef.current) return
    initRef.current = true

    const init = async () => {
      const sessionList = await getChatSessions()
      setSessions(sessionList)

      if (sessionList.length > 0) {
        const session = await getChatSession(sessionList[0].id)
        setSessionId(sessionList[0].id)
        setMessages(dedupeMessages(convertMessages(session.messages)))
      } else {
        const session = await createChatSession()
        setSessionId(session.id)
        setMessages([])
        setSessions([session])  // Add to local state directly
      }
    }

    init()
  }, [])  // Empty deps is now correct - all operations are self-contained

  // ...
}
```

### Key Insight

By moving the initialization into `useChatSessions`, we:
1. Remove the dependency on `createNewSession` callback
2. Make the initialization self-contained
3. Remove the need for `eslint-disable-line`
4. Fix the actual React Strict Mode double-init (which `initRef` handles)

---

## File Structure After Refactoring

```
frontend/
├── app/
│   └── chat/
│       └── page.tsx              # ~200 lines (down from 1070)
├── components/
│   └── chat/
│       ├── SessionSidebar.tsx    # ~50 lines
│       ├── MobileHistorySheet.tsx # ~80 lines
│       ├── MobileActivitySheet.tsx # ~50 lines
│       ├── ChatHeader.tsx        # ~70 lines
│       └── EmptyState.tsx        # ~40 lines
├── hooks/
│   ├── useWebSocketEvents.ts     # ~200 lines
│   ├── useChatSessions.ts        # ~150 lines
│   └── useMessageState.ts        # ~150 lines
└── lib/
    ├── websocket.ts              # (existing, unchanged)
    └── api.ts                    # (existing, unchanged)
```

---

## Implementation Order & Dependencies

```
Phase 5 (Race Condition Fix)
    │
    └──► Phase 2 (useChatSessions)
              │
              └──► Phase 6 (useEffect Deps)
                        │
Phase 3 (useMessageState) ◄───┘
    │
    └──► Phase 1 (useWebSocketEvents)
              │
              └──► Phase 4 (UI Components)
```

### Recommended Execution Order

1. **Phase 5** - Fix race condition (critical bug, standalone fix possible)
2. **Phase 2** - Extract `useChatSessions` (includes Phase 5 solution properly)
3. **Phase 6** - Fix useEffect deps (naturally fixed by Phase 2)
4. **Phase 3** - Extract `useMessageState` (independent of sessions)
5. **Phase 1** - Extract `useWebSocketEvents` (depends on message state interface)
6. **Phase 4** - Extract UI components (purely presentational, lowest risk)

---

## Testing Checklist

### For Each Phase

- [ ] Unit tests for new hook/component
- [ ] Integration test with existing code
- [ ] Manual smoke test: send message, receive response
- [ ] Manual smoke test: navigate away and back during streaming
- [ ] Manual smoke test: rapid message sending
- [ ] No console errors/warnings

### End-to-End Scenarios

- [ ] New user: first session created automatically
- [ ] Returning user: most recent session loaded
- [ ] Session switching: messages update correctly
- [ ] Delete current session: clears messages
- [ ] Streaming response: content updates in real-time
- [ ] Thinking indicator: shows/hides correctly
- [ ] Tool activity: displays in activity panel
- [ ] Connection loss: reconnects automatically
- [ ] Mobile: bottom sheets work correctly

---

## Risk Mitigation

### Rollback Strategy

Each phase should be a separate PR with:
1. Feature flag for gradual rollout
2. Old code preserved (commented) until next phase
3. Clear revert path via git

### Code Freeze Coordination

- Notify team before starting Phase 1 or Phase 3 (highest risk)
- Don't refactor during active development of new WebSocket events
- Keep changes to `websocket.ts` and `AgentActivityContext.tsx` minimal

### Monitoring

- Log any unexpected event types
- Track session creation count per user
- Alert on duplicate session creation

---

## Appendix: Current Code Reference

### Event Types Handled (lines 300-688)

| Event Type | Handler Lines | State Updated |
|------------|---------------|---------------|
| `chat_start` | 302-315 | `isLoading`, `isStreaming`, `agentActivities`, `toolActivities` |
| `tool_start` | 317-339 | `toolActivities`, `agentActivities` |
| `tool_complete` | 341-376 | `toolActivities`, `agentActivities` |
| `agent_spawn` | 378-421 | `agentActivities` |
| `agent_complete_subagent` | 424-448 | `agentActivities` |
| `agent_start` | 450-472 | `messages`, `streamingMessage` |
| `thinking_start` | 474-491 | `messages`, `streamingMessage` |
| `thinking_delta` | 494-511 | `messages`, `streamingMessage` |
| `thinking_complete` | 514-531 | `messages`, `streamingMessage` |
| `agent_delta` | 534-561 | `messages`, `streamingMessage` |
| `agent_complete` | 563-657 | `messages`, `isLoading`, `isStreaming`, `streamingMessage` |
| `chat_complete` | 659-674 | `isLoading`, `isStreaming`, `streamingMessage`, `agentActivities` |
| `error` | 676-687 | `isLoading`, `agentActivities` |

### State Dependencies

```
sessionId ──► saveMessage
    │
sessions ◄── loadSessions
    │
messages ◄── loadSession, streaming events
    │
isLoading ◄── chat_start, agent_complete, chat_complete, error
    │
agentActivities ◄── most events
    │
toolActivities ◄── tool_start, tool_complete
    │
streamingMessage ◄── agent_start, thinking_*, agent_delta, agent_complete
```

---

## Notes

- The `LOADING_TIMEOUT_MS` constant (5 minutes) at line 65 should be moved to a config file
- The `getDisplayName` helper (lines 105-108) should move to a utils file
- Consider moving `Message` interface to a shared types file
- The debug console.logs (lines 569-576, 593-599, etc.) should be removed or gated behind a debug flag
