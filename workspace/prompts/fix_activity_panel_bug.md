# Bug Fix: Activity Panel Shows "Working" Indefinitely After Chat Completes

## Bug Description

The Activity Panel in the sidebar shows "Working" with a spinner and gets stuck in that state even after the COO's response is complete. The user has to navigate away from chat and return to see the response, but the Activity Panel still shows "Working" status. The spinner never stops.

## Root Cause Analysis

The bug is a **state synchronization issue** between two disconnected state sources:

1. **Sidebar.tsx** calculates `isProcessing` locally by checking if any `panelAgentActivities` has a status that is NOT 'complete' or 'error':
   ```typescript
   // Sidebar.tsx:31-33
   const isProcessing = panelAgentActivities.some(
     (a) => a.status !== 'complete' && a.status !== 'error'
   )
   ```

2. **chat/page.tsx** handles the `chat_complete` WebSocket event and updates `panelAgentActivities` to mark agents as complete:
   ```typescript
   // chat/page.tsx:432-441
   case 'chat_complete':
     setIsLoading(false)
     setAgentActivities((prev) =>
       prev.map((a) => ({
         ...a,
         status: 'complete' as const,
         endTime: a.endTime || new Date(),
       }))
     )
   ```

3. **The Problem**: There's a race condition. The `chat_complete` event fires on the WebSocket, and `chat/page.tsx` processes it. However:
   - If the user navigates away from `/chat` before `chat_complete` fires, the handler in `chat/page.tsx` never runs (component unmounted)
   - The `AgentActivityContext` listens for `chat_complete` but only updates `activities` (used for swarm-level tracking), NOT `panelAgentActivities`
   - `panelAgentActivities` state only gets updated when the chat page's WebSocket handler runs

4. **Critical Gap**: The `AgentActivityContext.tsx` handles `chat_complete` (line 167-176) but only updates the `activities` object, NOT the `panelAgentActivities` array that Sidebar uses.

## Files Involved

| File | Role | Line Numbers |
|------|------|--------------|
| `frontend/components/Sidebar.tsx` | Calculates `isProcessing` from `panelAgentActivities` | Lines 31-33, 132-139 |
| `frontend/app/chat/page.tsx` | Handles `chat_complete` and updates `panelAgentActivities` | Lines 432-441 |
| `frontend/lib/AgentActivityContext.tsx` | Global context, provides `panelAgentActivities`, handles some WebSocket events | Lines 41-44, 62-64, 166-176 |
| `frontend/components/ActivityPanel.tsx` | Displays activity, receives `isProcessing` as prop | Line 38, 193 |

## The Fix

The `AgentActivityContext` should also update `panelAgentActivities` when `chat_complete` fires. This ensures the state is synchronized regardless of which page the user is on.

### Required Change in `frontend/lib/AgentActivityContext.tsx`

In the `handleEvent` function (around line 166), modify the `chat_complete` handler to ALSO update `panelAgentActivities`:

```typescript
// Clear activities when chat completes
if (event.type === 'chat_complete') {
  // Mark all swarm activities as idle (existing code)
  setActivities((prev) => {
    const updated = { ...prev }
    for (const key in updated) {
      updated[key] = { ...updated[key], status: 'idle' }
    }
    return updated
  })

  // NEW: Also mark all panel agent activities as complete
  setPanelAgentActivities((prev) =>
    prev.map((a) => ({
      ...a,
      status: 'complete' as const,
      endTime: a.endTime || new Date(),
    }))
  )

  // NEW: Also mark all panel tool activities as complete
  setPanelToolActivities((prev) =>
    prev.map((t) => ({
      ...t,
      status: 'complete' as const,
      endTime: t.endTime || new Date(),
    }))
  )
}
```

### Why This Fixes the Bug

1. The `AgentActivityContext` is mounted at the app level (via Providers), so it's ALWAYS listening to WebSocket events
2. When `chat_complete` fires, the context will update `panelAgentActivities` regardless of whether `/chat` is mounted
3. Sidebar reads `panelAgentActivities` from the context, so it will immediately reflect the completed state
4. The spinner will stop because `isProcessing` will become `false` when all agents have `status: 'complete'`

### Optional Cleanup in `chat/page.tsx`

After implementing the context fix, the `chat_complete` handler in `chat/page.tsx` becomes redundant for updating agent activities (the context handles it now). You can simplify it:

```typescript
case 'chat_complete':
  setIsLoading(false)
  // Agent activity updates now handled by AgentActivityContext
  // Save the assistant message to backend
  if (pendingMessageRef.current && saveMessageRef.current) {
    const msg = pendingMessageRef.current
    saveMessageRef.current('assistant', msg.content, msg.agent, msg.thinking)
    pendingMessageRef.current = null
  }
  break
```

However, leaving the duplicate update is harmless (React will batch the state updates), so this cleanup is optional.

## Testing the Fix

1. Start the app and navigate to `/chat`
2. Send a message to the COO
3. Wait for the response to complete
4. Verify the Activity Panel spinner stops and shows "Done" or "Complete"
5. Try navigating away from `/chat` mid-response, then returning - verify state is still correct
6. Try staying on dashboard during COO response - verify Activity Panel correctly shows completion

## Additional Context

- The Activity Panel is rendered in `Sidebar.tsx` (line 132-141)
- `panelAgentActivities` and `panelToolActivities` are global state in `AgentActivityContext`
- The WebSocket singleton is shared between the context and chat page
- The context already listens for `chat_complete` events, it just doesn't update the panel-specific state
