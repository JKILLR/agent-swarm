# Frontend Chat Response Flow Bug Fixes

**File:** `frontend/app/chat/page.tsx`
**Date:** 2026-01-06

## Summary

Fixed 4 critical bugs in the frontend chat system that caused response flow issues:

## Bug 1: Loading Timeout Never Implemented (Lines 69-100)

**Problem:** `LOADING_TIMEOUT_MS` and `loadingTimeoutRef` were defined but never used.

**Fix:** Added a new `useEffect` hook that:
- Starts a 5-minute timeout when `isLoading` becomes true
- When timeout fires: sets `isLoading=false` and marks all non-complete agents as error
- Clears the timeout when `isLoading` becomes false
- Cleans up on unmount

```typescript
useEffect(() => {
  if (isLoading) {
    loadingTimeoutRef.current = setTimeout(() => {
      console.warn('Loading timeout reached, resetting state')
      setIsLoading(false)
      setAgentActivities((prev) =>
        prev.map((a) =>
          a.status !== 'complete'
            ? { ...a, status: 'error' as const, endTime: new Date() }
            : a
        )
      )
    }, LOADING_TIMEOUT_MS)
  } else {
    if (loadingTimeoutRef.current) {
      clearTimeout(loadingTimeoutRef.current)
      loadingTimeoutRef.current = null
    }
  }
  return () => {
    if (loadingTimeoutRef.current) {
      clearTimeout(loadingTimeoutRef.current)
      loadingTimeoutRef.current = null
    }
  }
}, [isLoading, setAgentActivities, LOADING_TIMEOUT_MS])
```

## Bug 2: COO Status Stuck on Working (Lines 258-274)

**Problem:** The `tool_start` handler set COO to "working" status, but nothing transitioned it back from "delegating" status.

**Fix:** Added logic in `tool_complete` handler to check if COO should go back to "working" (from "delegating") when no subagents are actively working:

```typescript
setAgentActivities((prev) => {
  const cooIdx = prev.findIndex(a => a.name.includes('COO'))
  if (cooIdx !== -1 && prev[cooIdx].status === 'delegating') {
    const hasActiveSubagents = prev.some(a =>
      !a.name.includes('COO') &&
      (a.status === 'working' || a.status === 'thinking')
    )
    if (!hasActiveSubagents) {
      const updated = [...prev]
      updated[cooIdx] = { ...updated[cooIdx], status: 'working' }
      return updated
    }
  }
  return prev
})
```

## Bug 3: agent_delta Handler Too Strict (Lines 418-435)

**Problem:** The `agent_delta` handler only checked the LAST message for `status='thinking'`. If multiple messages existed, it might miss the correct one.

**Fix:** Changed to use `findIndex` to find ANY message with `status='thinking'`:

```typescript
setMessages((prev) => {
  const thinkingIdx = prev.findIndex(
    (m) => m.type === 'agent' && m.status === 'thinking'
  )
  if (thinkingIdx !== -1) {
    const updated = [...prev]
    updated[thinkingIdx] = {
      ...updated[thinkingIdx],
      content: updated[thinkingIdx].content + (event.delta || ''),
    }
    return updated
  }
  return prev
})
```

## Bug 4: Cleanup Doesn't Clear Timeout (Lines 554-563)

**Problem:** The WebSocket useEffect cleanup function didn't clear `loadingTimeoutRef`.

**Fix:** Added timeout cleanup to the return function:

```typescript
return () => {
  mounted = false
  ws.off('*', handleEvent)
  ws.off('disconnected', handleDisconnect)
  // Clear loading timeout on cleanup
  if (loadingTimeoutRef.current) {
    clearTimeout(loadingTimeoutRef.current)
    loadingTimeoutRef.current = null
  }
}
```

## Testing Notes

- Test loading timeout by disconnecting WebSocket during a request
- Test COO status transitions by watching Activity Panel during agent delegation
- Test agent_delta by sending multiple concurrent messages
- Test cleanup by navigating away during an active request
