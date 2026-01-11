# Frontend Streaming Debug Review

## Executive Summary

After reviewing the frontend streaming and display code, I identified **5 critical issues** that can cause messages to appear then disappear, streaming content not persisting, and agent status getting stuck on "working".

---

## File Analysis

### Files Reviewed
1. `frontend/app/chat/page.tsx` - Main chat page with WebSocket handling
2. `frontend/lib/websocket.ts` - WebSocket client singleton
3. `frontend/lib/AgentActivityContext.tsx` - Global activity state
4. `frontend/components/AgentResponse.tsx` - Agent message display
5. `frontend/components/ActivityPanel.tsx` - Tool/agent activity display

---

## Critical Issues Found

### Issue 1: Duplicate Event Handlers Causing State Race Conditions

**Location:** `frontend/app/chat/page.tsx:479` and `frontend/lib/AgentActivityContext.tsx:183`

**Problem:** Both `ChatPage` and `AgentActivityContext` listen to the same WebSocket wildcard (`'*'`) events. This creates race conditions where:
- Both handlers process `agent_complete` and update different state slices
- State updates from one handler can overwrite updates from another
- The `chat_complete` event in `AgentActivityContext:169` clears swarm activities while `ChatPage` is still updating messages

**Evidence:**
```typescript
// AgentActivityContext.tsx:183
ws.on('*', handleEvent)

// chat/page.tsx:479
ws.on('*', handleEvent)
```

**Impact:** Messages can appear briefly then disappear when state updates conflict.

---

### Issue 2: Message Status Matching Race Condition

**Location:** `frontend/app/chat/page.tsx:396-423` (agent_complete handler)

**Problem:** The `agent_complete` handler searches for ANY message with `status === 'thinking'`:
```typescript
const thinkingIdx = prev.findIndex(
  (m) => m.type === 'agent' && m.status === 'thinking'
)
```

If a new `agent_start` event arrives between `agent_delta` events and `agent_complete`, the handler may:
1. Find the wrong "thinking" message
2. Overwrite content in the wrong message
3. Leave the original message stuck in "thinking" state

**Impact:** Streaming content not persisting; agent status stuck on "thinking".

---

### Issue 3: `isLoading` State Can Get Stuck

**Location:** `frontend/app/chat/page.tsx:443, 447, 465`

**Problem:** Multiple events try to set `isLoading=false`:
- Line 443: `agent_complete` sets it as fallback
- Line 447: `chat_complete` sets it
- Line 465: `error` sets it

However, `isLoading` can get stuck `true` if:
1. WebSocket disconnects during streaming (before `chat_complete`)
2. Backend crashes mid-response
3. The `chat_complete` event is lost due to network issues

The loading timeout at line 56-59 is defined but **never actually implemented**:
```typescript
const LOADING_TIMEOUT_MS = 5 * 60 * 1000  // Defined but unused!
const loadingTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)
// But no code ever sets this timeout!
```

**Impact:** Agent status gets stuck on "working" indefinitely.

---

### Issue 4: Inconsistent `isThinking` State Transitions

**Location:** `frontend/app/chat/page.tsx:316-366`

**Problem:** The thinking state machine has gaps:

```
thinking_start -> sets isThinking=true, thinking=''
thinking_delta -> appends to thinking (only if isThinking=true)
thinking_complete -> sets isThinking=false
```

But `agent_delta` (line 368-383) only updates content if `status === 'thinking'`, NOT based on `isThinking`:
```typescript
if (lastMessage && lastMessage.type === 'agent' && lastMessage.status === 'thinking') {
  // Updates content
}
```

If `thinking_complete` arrives before any `agent_delta` events, and then `agent_delta` arrives, the delta content is **silently dropped** because the message state may no longer match expectations.

**Impact:** Streaming content not persisting; thought process display is inconsistent.

---

### Issue 5: AgentActivityPanel Status Never Transitions from "working"

**Location:** `frontend/app/chat/page.tsx:197-204`

**Problem:** When `tool_start` arrives, COO is set to "working":
```typescript
if (cooIdx !== -1 && updated[cooIdx].status === 'thinking') {
  updated[cooIdx] = { ...updated[cooIdx], status: 'working' }
}
```

But there's **no event that transitions COO from "working" to "thinking"** if the agent starts working on a new task. The only way COO becomes "complete" is via `chat_complete` (line 448-455).

If tools are used but no `chat_complete` arrives, COO stays "working" forever.

**Impact:** Agent status stuck on "working".

---

## Data Flow Trace

```
SSE Event (Backend)
    |
    v
WebSocket (frontend/lib/websocket.ts)
    |
    +---> ChatPage handler (frontend/app/chat/page.tsx:165-476)
    |         |
    |         +---> setMessages() - Updates message array
    |         +---> setAgentActivities() - Updates panel agents (via context)
    |         +---> setToolActivities() - Updates panel tools (via context)
    |         +---> setIsLoading() - Updates loading state
    |
    +---> AgentActivityContext handler (frontend/lib/AgentActivityContext.tsx:70-178)
              |
              +---> setActivities() - Updates swarm activity map
              (Note: Does NOT update panelAgentActivities/panelToolActivities)
```

**Key Finding:** The comment at `AgentActivityContext.tsx:167-168` explicitly acknowledges this design:
```typescript
// NOTE: Panel activities (panelAgentActivities, panelToolActivities) are managed
// by ChatPage to avoid double event handlers updating the same state
```

This separation is intentional but creates complexity where the two handlers can still race on shared state (like the WebSocket connection events).

---

## Recommendations

### Fix 1: Remove Duplicate Wildcard Listener
Either:
- Have `AgentActivityContext` NOT listen to `*` events, only specific ones it needs
- OR have `ChatPage` delegate all activity updates to the context

### Fix 2: Use Message IDs for Matching
Instead of finding "any thinking message", use a correlation ID from the backend to match events to specific messages.

### Fix 3: Implement Loading Timeout
Add the actual timeout implementation:
```typescript
useEffect(() => {
  if (isLoading) {
    loadingTimeoutRef.current = setTimeout(() => {
      setIsLoading(false)
      // Also mark agents as error
    }, LOADING_TIMEOUT_MS)
  } else if (loadingTimeoutRef.current) {
    clearTimeout(loadingTimeoutRef.current)
    loadingTimeoutRef.current = null
  }
  return () => {
    if (loadingTimeoutRef.current) clearTimeout(loadingTimeoutRef.current)
  }
}, [isLoading])
```

### Fix 4: Defensive State Updates
Always check current state before updating. Use functional setState patterns consistently:
```typescript
setMessages((prev) => {
  // Guard: Check if update is still valid
  if (!prev.some(m => m.status === 'thinking')) return prev
  // ... rest of update
})
```

### Fix 5: Backend Correlation IDs
Have backend include a `message_id` or `request_id` in all events so frontend can definitively match events to messages.

---

## Quick Win

The fastest fix is to implement the loading timeout (Fix 3). This prevents the "working forever" issue even if other bugs persist. Users will at least see the system recover after 5 minutes.

---

## Testing Recommendations

1. **Simulate WebSocket disconnect mid-stream** - Should recover gracefully
2. **Send rapid sequential messages** - Should not cause state corruption
3. **Long-running agent tasks** - Should not get stuck
4. **Multiple agent spawns** - Activity panel should show correct hierarchy
5. **Backend error mid-response** - Should show error state, not "working"

---

*Generated: 2026-01-06*
