# Chat Page Refactoring Plan v2

**File:** `frontend/app/chat/page.tsx` (~1105 lines)
**Created:** 2026-01-06
**Updated:** 2026-01-06 (Addresses all critic review concerns)
**Status:** Ready for Implementation

---

## Executive Summary

The chat page has grown into a monolithic component handling:
- **WebSocket event handling** (40+ event types in one 400-line switch statement)
- **Session CRUD operations** (create, load, delete, list)
- **Message state management** (streaming, thinking, completion)
- **Multiple UI concerns** (sidebar, mobile sheets, chat area, activity panel)

This plan breaks the refactoring into **7 incremental phases**, each producing a testable, working state.

### Changes from v1 (Addressing Critic Review)

| Issue | Resolution |
|-------|------------|
| Race condition fix has stale closure bug | Use `sessionIdRef` pattern instead of closure |
| Context double-handler problem | Explicit event ownership table and cleanup plan |
| Missing rapid session switching handling | Add load sequence guard |
| WebSocket reconnection during streaming | Explicit handling in Phase 1 |
| Phase dependency order wrong | Corrected to: Phase 5 → 2 → 6 → 1 → 3 → 4.5 → 4 |
| Missing Phase 4.5 for bottom sheets | Added with exact line numbers |
| Missing exact line numbers | Added for every extraction |
| Missing rollback strategy per phase | Added explicit rollback for each phase |

---

## Event Ownership Table (Critical)

**Problem:** Both `AgentActivityContext` (line 208) and `ChatPage` (line 691) have `ws.on('*', handleEvent)`.

| Event Type | Current Owner | After Refactor Owner | Notes |
|------------|---------------|---------------------|-------|
| `agent_start` | Both | `useWebSocketEvents` | Context tracks for org chart only |
| `agent_complete` | Both | `useWebSocketEvents` | Context tracks for org chart only |
| `agent_spawn` | Both | `useWebSocketEvents` | Context tracks for org chart only |
| `agent_complete_subagent` | Both | `useWebSocketEvents` | Context tracks for org chart only |
| `tool_start` | ChatPage only | `useWebSocketEvents` | Panel activities |
| `tool_complete` | ChatPage only | `useWebSocketEvents` | Panel activities |
| `thinking_*` | ChatPage only | `useWebSocketEvents` | Message state |
| `agent_delta` | ChatPage only | `useWebSocketEvents` | Message state |
| `chat_start` | ChatPage only | `useWebSocketEvents` | Loading state |
| `chat_complete` | Both | `useWebSocketEvents` | Context only marks idle |
| `error` | ChatPage only | `useWebSocketEvents` | Error handling |

**Resolution:** `AgentActivityContext` keeps its handler for org chart (`activities` state). Chat hooks handle panel activities and message state. No overlap in state mutations.

---

## Phase 5: Fix Race Condition in Session Creation

**Priority:** Critical
**Risk:** Medium (isolated fix)
**Lines affected:** 725-799 (handleSend function)

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
```

**Bug:** If user sends two messages quickly:
1. First message: `sessionId` is null, creates session A
2. Second message: `sessionId` is STILL null (state not updated yet)
3. Second message creates session B
4. Result: Two sessions, messages split

### Solution: sessionIdRef Pattern (Fixes Critic Issue 1.1)

```typescript
// Add at component level (after line 38)
const sessionIdRef = useRef<string | null>(null)
const sessionCreationLock = useRef<Promise<string> | null>(null)

// Sync ref with state (add after line 244)
useEffect(() => {
  sessionIdRef.current = sessionId
}, [sessionId])

// New ensureSession helper (add before handleSend)
const ensureSession = useCallback(async (): Promise<string> => {
  // Use ref to avoid stale closure
  const currentId = sessionIdRef.current
  if (currentId) return currentId

  // Check if creation is already in progress
  if (sessionCreationLock.current) {
    return sessionCreationLock.current
  }

  // Create session with lock
  sessionCreationLock.current = (async () => {
    try {
      const session = await createChatSession()
      sessionIdRef.current = session.id  // Update ref immediately
      setSessionId(session.id)           // Then update state
      await loadSessions()
      return session.id
    } finally {
      sessionCreationLock.current = null
    }
  })()

  return sessionCreationLock.current
}, [loadSessions])  // Note: NO sessionId dependency - we use ref
```

### Updated handleSend (lines 725-799)

```typescript
const handleSend = useCallback(async (content: string, attachments: Attachment[]) => {
  // Clear old activities
  setAgentActivities((prev) => prev.filter(a => a.status !== 'complete' && a.status !== 'error'))
  setToolActivities((prev) => prev.filter(t => t.status === 'running'))

  // Get or create session (race condition protected)
  let currentSessionId: string
  try {
    currentSessionId = await ensureSession()
  } catch (e) {
    console.error('Failed to ensure session:', e)
    return
  }

  // ... rest unchanged
}, [ensureSession, /* other deps */])
```

### Exact Changes

1. **Add ref declarations** after line 38:
   ```typescript
   const sessionIdRef = useRef<string | null>(null)
   const sessionCreationLock = useRef<Promise<string> | null>(null)
   ```

2. **Add sync effect** after line 244:
   ```typescript
   useEffect(() => {
     sessionIdRef.current = sessionId
   }, [sessionId])
   ```

3. **Add ensureSession** before line 725:
   - Full implementation as shown above

4. **Replace lines 731-742** with:
   ```typescript
   let currentSessionId: string
   try {
     currentSessionId = await ensureSession()
   } catch (e) {
     console.error('Failed to ensure session:', e)
     return
   }
   ```

### Testing Strategy

1. **Unit test:** Call `ensureSession()` twice simultaneously, verify only one session created
2. **Integration test:** Rapidly send two messages, verify both go to same session
3. **Stress test:** 10 concurrent sends, verify single session

### Rollback Strategy

- **Revert:** Single commit with all ref additions
- **Feature flag:** Not needed - changes are internal implementation
- **Verification:** Check session count before/after rapid sends
- **Rollback time:** < 5 minutes (single git revert)

---

## Phase 2: Extract `useChatSessions` Hook

**Priority:** High
**Risk:** Low
**Lines affected:** 161-290

### Lines to Extract

| Code | Current Lines | Description |
|------|---------------|-------------|
| `loadSessions` | 162-169 | Load session list |
| `loadSession` | 172-201 | Load specific session |
| `createNewSession` | 204-213 | Create new session |
| `handleDeleteSession` | 216-228 | Delete session |
| `saveMessage` | 231-239 | Save message to backend |
| `initRef` + init effect | 248-290 | Initialization logic |

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
  sessionIdRef: React.RefObject<string | null>  // For race condition fix
  isInitialized: boolean
  loadSessions: () => Promise<void>
  loadSession: (id: string) => Promise<void>
  createSession: () => Promise<string>
  deleteSession: (id: string, e: React.MouseEvent) => Promise<void>
  saveMessage: (role: 'user' | 'assistant', content: string, agent?: string, thinking?: string) => Promise<void>
  ensureSession: () => Promise<string>  // Race condition protected
}
```

### Rapid Session Switching Guard (Fixes Critic Issue 2.2)

```typescript
const loadSequenceRef = useRef(0)

const loadSession = useCallback(async (id: string) => {
  const loadSequence = ++loadSequenceRef.current

  try {
    const session = await getChatSession(id)

    // Check if this load is still current
    if (loadSequence !== loadSequenceRef.current) {
      return // User switched to different session
    }

    setSessionId(id)
    sessionIdRef.current = id
    // Call onSessionLoad callback with messages
    onSessionLoad?.(dedupeMessages(convertMessages(session.messages)))
  } catch (e) {
    console.error('Failed to load session:', e)
  }
}, [onSessionLoad])
```

### Deduplication Logic (Consolidated)

```typescript
// Improved key generation (Fixes Critic Issue 3.2)
const dedupeMessages = (messages: Message[]): Message[] => {
  const seen = new Set<string>()
  return messages.filter(m => {
    // Include agent and timestamp for better uniqueness
    const key = `${m.type}:${m.agent || ''}:${m.content.substring(0, 200)}:${m.timestamp?.getTime() || ''}`
    if (seen.has(key)) return false
    seen.add(key)
    return true
  })
}
```

### Implementation Steps

1. Create `frontend/hooks/useChatSessions.ts`
2. Move state: `sessionId`, `sessions` (NOT messages - that stays in page)
3. Move all session CRUD functions
4. Add `sessionIdRef` and `ensureSession` from Phase 5
5. Add `loadSequenceRef` for rapid switching protection
6. Consolidate `dedupeMessages` helper
7. Export `UseChatSessionsReturn` interface

### Rollback Strategy

- **Revert:** Single commit creating the hook
- **Feature flag:** `localStorage.getItem('USE_NEW_SESSIONS_HOOK') === 'true'`
  ```typescript
  const useNewHook = typeof window !== 'undefined' &&
    localStorage.getItem('USE_NEW_SESSIONS_HOOK') === 'true'

  // In ChatPage:
  const sessionsHook = useNewHook ? useChatSessions(options) : useOldSessionsLogic()
  ```
- **Rollback time:** < 5 minutes
- **Preserve old code:** Keep old implementation in `page.tsx.bak` for one PR cycle

---

## Phase 6: Fix useEffect Dependency

**Priority:** Medium
**Risk:** Low (auto-fixed by Phase 2)
**Lines affected:** 290

### Current Problem

```typescript
// Line 290
}, []) // eslint-disable-line react-hooks/exhaustive-deps
```

### Solution

This is **automatically resolved** by Phase 2. The initialization logic moves into `useChatSessions`, making the empty deps array correct because:
- All operations are self-contained in the hook
- No external callbacks needed
- `initRef` prevents double-init in Strict Mode

### Rollback Strategy

- N/A - this is fixed by Phase 2's implementation
- If Phase 2 is reverted, this issue returns but is not critical

---

## Phase 1: Extract `useWebSocketEvents` Hook

**Priority:** High
**Risk:** Medium
**Lines affected:** 292-723

### Lines to Extract

| Code | Current Lines | Description |
|------|---------------|-------------|
| WebSocket setup | 293-295 | Get WebSocket reference |
| `handleEvent` switch | 296-688 | All event handling |
| Event subscription | 691 | `ws.on('*', handleEvent)` |
| Disconnect handler | 694-701 | Clean up on disconnect |
| Connection logic | 704-712 | `ws.connect()` |
| Cleanup | 714-722 | Unsubscribe and clear timeout |

### Target Interface

```typescript
// frontend/hooks/useWebSocketEvents.ts

interface UseWebSocketEventsOptions {
  // Chat lifecycle
  onChatStart: () => void
  onChatComplete: () => void
  onError: (message: string) => void

  // Agent messages
  onAgentStart: (agent: string, agentType: string) => void
  onAgentDelta: (delta: string) => void
  onAgentComplete: (content: string, agent: string, thinking?: string) => void

  // Thinking
  onThinkingStart: () => void
  onThinkingDelta: (delta: string) => void
  onThinkingComplete: (thinking: string) => void

  // Tools (for panel activities)
  onToolStart: (tool: string, description: string, agentName?: string) => void
  onToolComplete: (tool: string, success: boolean, summary?: string, agentName?: string) => void

  // Agent hierarchy (for panel activities)
  onAgentSpawn: (agent: string, parentAgent: string, description?: string) => void
  onAgentSubComplete: (agent: string) => void

  // Connection (Fixes Critic Issue 2.3)
  onDisconnect?: () => void
  onReconnect?: () => void
}

interface UseWebSocketEventsReturn {
  isConnected: boolean
  connectionState: 'disconnected' | 'connecting' | 'connected' | 'reconnecting'
  send: (message: string, options?: SendOptions) => void
  reconnect: () => void  // Manual reconnect trigger
}

function useWebSocketEvents(
  options: UseWebSocketEventsOptions,
  ws?: ChatWebSocket  // Optional for testing (Fixes Critic 6.1)
): UseWebSocketEventsReturn
```

### Disconnect Handling (Fixes Critic Issue 2.3)

```typescript
const handleDisconnect = useCallback(() => {
  setIsConnected(false)
  setConnectionState('disconnected')
  // Call the onDisconnect callback so parent can clean up streaming state
  options.onDisconnect?.()
}, [options])
```

In ChatPage, wire it up:
```typescript
const wsEvents = useWebSocketEvents({
  // ... other callbacks
  onDisconnect: () => {
    setIsLoading(false)
    setIsStreaming(false)
    setStreamingMessage(null)
  },
})
```

### Event Type Mapping

| Event Type | Handler Lines | Callback |
|------------|---------------|----------|
| `chat_start` | 302-315 | `onChatStart` |
| `tool_start` | 317-339 | `onToolStart` |
| `tool_complete` | 341-376 | `onToolComplete` |
| `agent_spawn` | 378-421 | `onAgentSpawn` |
| `agent_complete_subagent` | 424-448 | `onAgentSubComplete` |
| `agent_start` | 450-472 | `onAgentStart` |
| `thinking_start` | 474-491 | `onThinkingStart` |
| `thinking_delta` | 494-511 | `onThinkingDelta` |
| `thinking_complete` | 514-531 | `onThinkingComplete` |
| `agent_delta` | 534-561 | `onAgentDelta` |
| `agent_complete` | 563-657 | `onAgentComplete` |
| `chat_complete` | 659-674 | `onChatComplete` |
| `error` | 676-687 | `onError` |

### Implementation Steps

1. Create `frontend/hooks/useWebSocketEvents.ts`
2. Define callback interface (no state in hook)
3. Move connection logic
4. Create event router that calls appropriate callbacks
5. **Do NOT handle state updates in hook** - parent provides callbacks
6. Move debug logger calls to use debug utility (Critic 3.4)

### Debug Logger Utility (Fixes Critic Issue 3.4)

```typescript
// frontend/lib/debug.ts
const DEBUG = typeof window !== 'undefined' &&
  localStorage.getItem('DEBUG_WEBSOCKET') === 'true'

export const wsDebug = {
  log: (...args: unknown[]) => DEBUG && console.log('[WS]', ...args),
  warn: (...args: unknown[]) => DEBUG && console.warn('[WS]', ...args),
  error: (...args: unknown[]) => console.error('[WS]', ...args),  // Always log errors
}
```

### Rollback Strategy

- **Revert:** Single commit creating the hook
- **Feature flag:** `localStorage.getItem('USE_NEW_WEBSOCKET_HOOK') === 'true'`
- **Rollback time:** < 10 minutes (most complex hook)
- **Testing:** Verify all 13 event types still work correctly
- **Preserve old code:** Keep switch statement commented in `page.tsx.bak`

---

## Phase 3: Extract `useMessageState` Hook

**Priority:** Medium
**Risk:** Medium
**Lines affected:** 35, 70-102, portions of 450-657

### Lines to Extract

| Code | Current Lines | Description |
|------|---------------|-------------|
| `messages` state | 35 | `useState<Message[]>([])` |
| Restore streaming effect | 70-83 | Restore message from context |
| Sync streaming effect | 88-102 | Keep messages synced with context |
| Message finding pattern | 536-560, 586-650 | Find thinking message, update |

### Target Interface

```typescript
// frontend/hooks/useMessageState.ts

interface UseMessageStateOptions {
  streamingMessage: StreamingMessage | null
  setStreamingMessage: (msg: StreamingMessage | null) => void
  isStreaming: boolean
}

interface UseMessageStateReturn {
  messages: Message[]

  // User messages
  addUserMessage: (content: string, attachments?: Attachment[]) => string  // returns ID

  // Agent messages
  startAgentMessage: (agent: string, agentType: string) => string  // returns ID
  updateAgentContent: (delta: string) => void
  startThinking: () => void
  updateThinking: (delta: string) => void
  completeThinking: (thinking: string) => void
  completeAgentMessage: (content: string, agent: string, thinking?: string) => void

  // Bulk operations
  setMessages: React.Dispatch<React.SetStateAction<Message[]>>
  markAllComplete: () => void

  // Internal - for session loading
  loadMessages: (messages: Message[]) => void
}

function useMessageState(options: UseMessageStateOptions): UseMessageStateReturn
```

### ID Generation (Fixes Critic Issue 5.3)

```typescript
// Use crypto.randomUUID() for all IDs
// These are client-side temporary IDs, not database IDs
const generateMessageId = () => crypto.randomUUID()
```

### Implementation Steps

1. Create `frontend/hooks/useMessageState.ts`
2. Move `messages` state
3. Move streaming restore/sync effects
4. Create atomic update functions that encapsulate the "find thinking message" pattern
5. Wire up to `useWebSocketEvents` callbacks

### Rollback Strategy

- **Revert:** Single commit creating the hook
- **Feature flag:** Not recommended (too tightly coupled to Phase 1)
- **Rollback time:** < 10 minutes
- **If Phase 1 reverted:** This phase must also be reverted
- **Testing:** Verify streaming, thinking, completion all work

---

## Phase 4.5: Extract Bottom Sheet Components (NEW)

**Priority:** Low
**Risk:** Low
**Lines affected:** 858-967

### Components to Extract

#### 4.5.1 `MobileHistorySheet` Component
**Lines:** 858-928

```typescript
// frontend/components/chat/MobileHistorySheet.tsx

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

**Exact lines to move:**
- Lines 858-866: Conditional render + overlay
- Lines 867-885: Bottom sheet container + header
- Lines 886-925: Session list
- Lines 926-928: Closing tags

#### 4.5.2 `MobileActivitySheet` Component
**Lines:** 930-967

```typescript
// frontend/components/chat/MobileActivitySheet.tsx

interface MobileActivitySheetProps {
  isOpen: boolean
  onClose: () => void
  agents: PanelAgentActivity[]
  tools: PanelToolActivity[]
  isProcessing: boolean
  onClear: () => void
}
```

**Exact lines to move:**
- Lines 930-936: Conditional render + overlay
- Lines 937-956: Bottom sheet container + header
- Lines 957-964: ActivityPanel embed
- Lines 965-967: Closing tags

### Implementation Steps

1. Create `frontend/components/chat/MobileHistorySheet.tsx`
2. Create `frontend/components/chat/MobileActivitySheet.tsx`
3. Move exact lines as specified
4. Add imports in `page.tsx`
5. Replace inline JSX with component usage

### Rollback Strategy

- **Revert:** Two small commits (one per component)
- **Feature flag:** Not needed - purely presentational
- **Rollback time:** < 5 minutes
- **Risk:** Very low - no logic changes, just moving JSX

---

## Phase 4: Extract Remaining UI Components

**Priority:** Low
**Risk:** Low
**Lines affected:** 814-856, 972-1060

### Components to Extract

#### 4.1 `SessionSidebar` Component
**Lines:** 814-856

```typescript
// frontend/components/chat/SessionSidebar.tsx

interface SessionSidebarProps {
  sessions: ChatSessionSummary[]
  currentSessionId: string | null
  showSidebar: boolean
  onSessionClick: (id: string) => void
  onNewSession: () => void
  onDeleteSession: (id: string, e: React.MouseEvent) => void
}
```

**Exact lines to move:**
- Lines 814-825: Container + header with new button
- Lines 826-855: Session list
- Line 856: Closing div

#### 4.2 `ChatHeader` Component
**Lines:** 972-1029

```typescript
// frontend/components/chat/ChatHeader.tsx

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

**Exact lines to move:**
- Lines 972-1000: Left side (toggles, title)
- Lines 1001-1028: Right side (activity button, connection status)
- Line 1029: Closing div

#### 4.3 `EmptyState` Component
**Lines:** 1033-1060

```typescript
// frontend/components/chat/EmptyState.tsx

interface EmptyStateProps {
  onQuickSend: (message: string) => void
}
```

**Exact lines to move:**
- Lines 1034-1042: Header and description
- Lines 1043-1058: Quick send buttons
- Lines 1059-1060: Closing divs

### Implementation Steps

1. Create `frontend/components/chat/` directory (if not exists)
2. Extract each component to its own file
3. Keep Tailwind classes inline
4. Update imports in chat page

### Rollback Strategy

- **Revert:** One commit per component
- **Feature flag:** Not needed - purely presentational
- **Rollback time:** < 5 minutes per component
- **Risk:** Very low - no logic changes

---

## Implementation Order & Dependencies

```
Phase 5 (Race Condition Fix)        [STANDALONE - CAN SHIP]
    │
    └──► Phase 2 (useChatSessions)  [DEPENDS ON 5]
              │
              └──► Phase 6 (useEffect Deps) [AUTO-FIXED BY 2]
                        │
Phase 1 (useWebSocketEvents) ◄──────┘  [INDEPENDENT]
    │
    └──► Phase 3 (useMessageState)     [DEPENDS ON 1]
              │
              └──► Phase 4.5 (Bottom Sheets) [INDEPENDENT]
                        │
                        └──► Phase 4 (Remaining UI) [INDEPENDENT]
```

### Recommended Execution Order

1. **Phase 5** - Fix race condition (critical bug, standalone PR)
2. **Phase 2** - Extract `useChatSessions` (includes Phase 5 solution)
3. **Phase 6** - Auto-fixed by Phase 2
4. **Phase 1** - Extract `useWebSocketEvents` (largest extraction)
5. **Phase 3** - Extract `useMessageState` (depends on Phase 1 interface)
6. **Phase 4.5** - Extract bottom sheet components
7. **Phase 4** - Extract remaining UI components

---

## File Structure After Refactoring

```
frontend/
├── app/
│   └── chat/
│       └── page.tsx              # ~250 lines (down from 1105)
├── components/
│   └── chat/
│       ├── SessionSidebar.tsx    # ~50 lines
│       ├── MobileHistorySheet.tsx # ~80 lines
│       ├── MobileActivitySheet.tsx # ~45 lines
│       ├── ChatHeader.tsx        # ~70 lines
│       └── EmptyState.tsx        # ~35 lines
├── hooks/
│   ├── useWebSocketEvents.ts     # ~250 lines
│   ├── useChatSessions.ts        # ~180 lines
│   └── useMessageState.ts        # ~150 lines
└── lib/
    ├── websocket.ts              # (existing, unchanged)
    ├── api.ts                    # (existing, unchanged)
    └── debug.ts                  # ~20 lines (new)
```

---

## Testing Checklist

### Per-Phase Testing

#### Phase 5 (Race Condition)
- [ ] Unit test: `ensureSession()` called twice simultaneously → one session
- [ ] Integration: Rapid double-send → both messages in same session
- [ ] Stress: 10 concurrent sends → single session
- [ ] Manual: Click send button rapidly → no duplicate sessions

#### Phase 2 (Sessions Hook)
- [ ] Unit: Session CRUD operations work
- [ ] Unit: `dedupeMessages` handles edge cases
- [ ] Integration: Rapid session switching → correct messages shown
- [ ] Manual: Create/load/delete sessions works

#### Phase 1 (WebSocket Hook)
- [ ] Unit: All 13 event types call correct callbacks
- [ ] Unit: Disconnect clears streaming state
- [ ] Integration: Full chat flow works
- [ ] Manual: Navigate away during streaming, come back

#### Phase 3 (Message State)
- [ ] Unit: `startAgentMessage` creates correct message
- [ ] Unit: `updateAgentContent` finds and updates thinking message
- [ ] Integration: Full streaming flow works
- [ ] Manual: Thinking indicator shows/hides

#### Phase 4.5 & 4 (UI Components)
- [ ] Visual: Components look identical to before
- [ ] Manual: All interactions work on mobile
- [ ] Manual: All interactions work on desktop

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
- [ ] Navigate away during streaming: state preserved
- [ ] Navigate back after streaming complete: completion visible

---

## Risk Mitigation

### Rollback Strategy Summary

| Phase | Rollback Method | Time | Feature Flag |
|-------|----------------|------|--------------|
| 5 | Git revert | < 5 min | No |
| 2 | Git revert + flag | < 5 min | Yes |
| 1 | Git revert + flag | < 10 min | Yes |
| 3 | Git revert (with Phase 1) | < 10 min | No |
| 4.5 | Git revert | < 5 min | No |
| 4 | Git revert per component | < 5 min | No |

### Code Freeze Coordination

- Notify team before starting Phase 1 or Phase 3 (highest risk)
- Don't refactor during active development of new WebSocket events
- Keep changes to `websocket.ts` and `AgentActivityContext.tsx` minimal

### Monitoring

After each phase ships:
- [ ] Log any unexpected event types
- [ ] Track session creation count per user
- [ ] Alert on duplicate session creation
- [ ] Monitor for console errors in production

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
agentActivities ◄── most events (panel) + context (org chart)
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
- Debug console.logs should use the new debug utility with feature flag
- Context's `activities` state (org chart) remains separate from panel activities

---

*Plan v2 created: 2026-01-06*
*Addresses all issues from Critic Review*
*Safe for incremental implementation*
