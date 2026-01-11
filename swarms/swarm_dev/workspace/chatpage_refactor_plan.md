# ChatPage Refactoring Plan

## Executive Summary

`frontend/app/chat/page.tsx` is currently ~1070 lines with multiple code smells that make it difficult to maintain, test, and extend. This plan breaks the refactor into 6 incremental phases, each delivering working code that can be tested before moving to the next phase.

---

## Current Issues Analysis

### Issue 1: Component Size (~1070 lines)
The component handles too many responsibilities:
- WebSocket connection management
- Session management (CRUD operations)
- Message state management
- UI rendering for desktop and mobile
- Activity panel state coordination

### Issue 2: Massive WebSocket Event Handler (lines 296-688)
The `handleEvent` function contains a 400-line switch statement handling 13+ event types:
- `chat_start`, `tool_start`, `tool_complete`, `agent_spawn`
- `agent_complete_subagent`, `agent_start`, `thinking_start`
- `thinking_delta`, `thinking_complete`, `agent_delta`
- `agent_complete`, `chat_complete`, `error`

Each case has complex state update logic that's difficult to test in isolation.

### Issue 3: Race Condition in Session Creation (lines 731-742)
```typescript
let currentSessionId = sessionId
if (!currentSessionId) {
  try {
    const session = await createChatSession()
    setSessionId(session.id)        // React state update is async
    currentSessionId = session.id   // But this is used immediately
    await loadSessions()
  } catch (e) { ... }
}
```
**Problem**: Multiple rapid sends could trigger multiple session creations before React state updates. The local variable mitigation works for the single call, but doesn't prevent concurrent `handleSend` calls from racing.

### Issue 4: Missing useEffect Dependency (line 290)
```typescript
useEffect(() => {
  // Uses createNewSession and loadSession implicitly via init()
}, []) // eslint-disable-line react-hooks/exhaustive-deps
```
The eslint disable masks that `createNewSession` should be in deps (it's used inside `init()`).

---

## Target Architecture

```
frontend/
├── app/chat/
│   └── page.tsx                     # Slim orchestrator (~200 lines)
├── components/chat/
│   ├── ChatPageLayout.tsx           # Desktop/mobile responsive layout
│   ├── SessionSidebar.tsx           # Desktop session sidebar
│   ├── MobileHistorySheet.tsx       # Mobile history bottom sheet
│   ├── MobileActivitySheet.tsx      # Mobile activity bottom sheet
│   ├── ChatHeader.tsx               # Header with connection status
│   ├── ChatMessagesArea.tsx         # Message list and empty state
│   └── ChatInputArea.tsx            # Input wrapper with context
├── hooks/
│   ├── useChatWebSocket.ts          # WebSocket connection + event handling
│   ├── useChatSessions.ts           # Session CRUD operations
│   ├── useChatMessages.ts           # Message state management
│   └── useLoadingTimeout.ts         # Loading state with timeout
└── types/
    └── chat.ts                      # Shared chat types
```

---

## Phase 1: Extract Types and Utilities

**Goal**: Create shared types file and utility functions without changing any behavior.

### Files to Create

#### `frontend/types/chat.ts`
```typescript
import { type Attachment } from '@/components/ChatInput'

export interface Message {
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

// Re-export types that other modules need
export type { Attachment } from '@/components/ChatInput'
export type {
  PanelAgentActivity,
  PanelToolActivity,
  StreamingMessage
} from '@/lib/AgentActivityContext'
```

#### `frontend/lib/chatUtils.ts`
```typescript
// Agent name display mapping
export function getDisplayName(name: string): string {
  if (name === 'Supreme Orchestrator') return 'Axel'
  return name
}

// Generate unique message IDs
export function generateMessageId(): string {
  return crypto.randomUUID()
}

// Deduplicate messages by content+type (handles DB duplicates)
export function deduplicateMessages<T extends { type: string; content: string }>(
  messages: T[]
): T[] {
  const seen = new Set<string>()
  return messages.filter(m => {
    const key = `${m.type}:${m.content.substring(0, 200)}`
    if (seen.has(key)) return false
    seen.add(key)
    return true
  })
}
```

### Changes to page.tsx
1. Import `Message` from `@/types/chat`
2. Import `getDisplayName`, `deduplicateMessages` from `@/lib/chatUtils`
3. Remove the local `Message` interface definition
4. Remove the local `getDisplayName` function

### Testing Phase 1
```bash
# Run the dev server
npm run dev

# Test:
# 1. Open chat page - should load normally
# 2. Send a message - should work
# 3. Check console for any type errors
# 4. Verify agent names display correctly ("Axel" instead of "Supreme Orchestrator")
```

---

## Phase 2: Extract Loading Timeout Hook

**Goal**: Extract the loading timeout logic into a reusable hook.

### Files to Create

#### `frontend/hooks/useLoadingTimeout.ts`
```typescript
import { useEffect, useRef } from 'react'
import { type PanelAgentActivity } from '@/lib/AgentActivityContext'

interface UseLoadingTimeoutOptions {
  timeoutMs?: number
  onTimeout?: () => void
}

interface UseLoadingTimeoutReturn {
  isLoading: boolean
  setIsLoading: (loading: boolean) => void
}

export function useLoadingTimeout(
  setAgentActivities: React.Dispatch<React.SetStateAction<PanelAgentActivity[]>>,
  options: UseLoadingTimeoutOptions = {}
): UseLoadingTimeoutReturn {
  const { timeoutMs = 5 * 60 * 1000 } = options
  const [isLoading, setIsLoadingState] = useState(false)
  const loadingTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  useEffect(() => {
    if (isLoading) {
      loadingTimeoutRef.current = setTimeout(() => {
        console.warn('Loading timeout reached, resetting state')
        setIsLoadingState(false)
        setAgentActivities((prev) =>
          prev.map((a) =>
            a.status !== 'complete'
              ? { ...a, status: 'error' as const, endTime: new Date() }
              : a
          )
        )
      }, timeoutMs)
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
  }, [isLoading, setAgentActivities, timeoutMs])

  return { isLoading, setIsLoading: setIsLoadingState }
}
```

### Changes to page.tsx
1. Import `useLoadingTimeout`
2. Replace the local `isLoading` state and timeout effect with the hook
3. Remove lines 37, 61, 64-65, 129-159

### Testing Phase 2
```bash
# Test:
# 1. Send a message and verify loading state activates
# 2. Wait for response - loading should clear
# 3. (Optional) Temporarily set timeout to 5s and test timeout triggers error state
```

---

## Phase 3: Extract Session Management Hook

**Goal**: Extract all session-related logic into a dedicated hook with race condition fix.

### Files to Create

#### `frontend/hooks/useChatSessions.ts`
```typescript
import { useState, useCallback, useRef, useEffect } from 'react'
import {
  getChatSessions,
  getChatSession,
  createChatSession,
  deleteChatSession,
  addChatMessage,
  type ChatSessionSummary,
  type ChatSession,
} from '@/lib/api'
import { getDisplayName, deduplicateMessages } from '@/lib/chatUtils'
import { type Message } from '@/types/chat'

interface UseChatSessionsReturn {
  sessionId: string | null
  sessions: ChatSessionSummary[]
  loadSessions: () => Promise<void>
  loadSession: (id: string) => Promise<Message[]>
  createNewSession: () => Promise<string | null>
  deleteSession: (id: string, e: React.MouseEvent) => Promise<void>
  saveMessage: (role: 'user' | 'assistant', content: string, agent?: string, thinking?: string) => Promise<void>
  getOrCreateSessionId: () => Promise<string | null>
  isInitialized: boolean
}

export function useChatSessions(): UseChatSessionsReturn {
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [sessions, setSessions] = useState<ChatSessionSummary[]>([])
  const [isInitialized, setIsInitialized] = useState(false)

  // Mutex to prevent concurrent session creation
  const sessionCreationLock = useRef<Promise<string | null> | null>(null)
  const initRef = useRef(false)

  const loadSessions = useCallback(async () => {
    try {
      const sessionList = await getChatSessions()
      setSessions(sessionList)
    } catch (e) {
      console.error('Failed to load sessions:', e)
    }
  }, [])

  const loadSession = useCallback(async (id: string): Promise<Message[]> => {
    try {
      const session = await getChatSession(id)
      setSessionId(id)

      const loadedMessages: Message[] = session.messages.map((m) => ({
        id: m.id,
        type: m.role === 'user' ? 'user' : 'agent',
        content: m.content,
        agent: getDisplayName(m.agent || 'Claude'),
        agentType: 'assistant',
        status: 'complete' as const,
        timestamp: new Date(m.timestamp),
        thinking: m.thinking,
      }))

      return deduplicateMessages(loadedMessages)
    } catch (e) {
      console.error('Failed to load session:', e)
      return []
    }
  }, [])

  const createNewSession = useCallback(async (): Promise<string | null> => {
    try {
      const session = await createChatSession()
      setSessionId(session.id)
      await loadSessions()
      return session.id
    } catch (e) {
      console.error('Failed to create session:', e)
      return null
    }
  }, [loadSessions])

  const deleteSession = useCallback(async (id: string, e: React.MouseEvent) => {
    e.stopPropagation()
    try {
      await deleteChatSession(id)
      if (sessionId === id) {
        setSessionId(null)
      }
      await loadSessions()
    } catch (e) {
      console.error('Failed to delete session:', e)
    }
  }, [sessionId, loadSessions])

  const saveMessage = useCallback(async (
    role: 'user' | 'assistant',
    content: string,
    agent?: string,
    thinking?: string
  ) => {
    if (!sessionId) return
    try {
      await addChatMessage(sessionId, role, content, agent, thinking)
      await loadSessions()
    } catch (e) {
      console.error('Failed to save message:', e)
    }
  }, [sessionId, loadSessions])

  // Race-condition-safe session getter
  // Uses a mutex pattern to ensure only one session creation happens
  const getOrCreateSessionId = useCallback(async (): Promise<string | null> => {
    // Fast path: session already exists
    if (sessionId) return sessionId

    // If another call is already creating a session, wait for it
    if (sessionCreationLock.current) {
      return sessionCreationLock.current
    }

    // Create the session with a lock
    sessionCreationLock.current = (async () => {
      try {
        const session = await createChatSession()
        setSessionId(session.id)
        await loadSessions()
        return session.id
      } catch (e) {
        console.error('Failed to create session:', e)
        return null
      } finally {
        sessionCreationLock.current = null
      }
    })()

    return sessionCreationLock.current
  }, [sessionId, loadSessions])

  // Initialize on mount
  useEffect(() => {
    if (initRef.current) return
    initRef.current = true

    const init = async () => {
      const sessionList = await getChatSessions()
      setSessions(sessionList)

      if (sessionList.length > 0) {
        setSessionId(sessionList[0].id)
      } else {
        await createNewSession()
      }
      setIsInitialized(true)
    }
    init()
  }, [createNewSession])

  return {
    sessionId,
    sessions,
    loadSessions,
    loadSession,
    createNewSession,
    deleteSession,
    saveMessage,
    getOrCreateSessionId,
    isInitialized,
  }
}
```

### Changes to page.tsx
1. Import `useChatSessions`
2. Replace session-related state and callbacks with hook
3. Update `handleSend` to use `getOrCreateSessionId()` instead of manual session creation
4. Remove lines 38-39, 162-244, 248-290 (session-related code)

### Testing Phase 3
```bash
# Test race condition fix:
# 1. Open browser dev tools, Network tab
# 2. Open chat with no existing sessions (clear localStorage/DB if needed)
# 3. Rapidly click send twice in quick succession
# 4. Verify only ONE session is created (check Network tab for POST requests)
# 5. Both messages should appear in the same session

# Test normal flow:
# 1. Create new session - should work
# 2. Switch sessions - messages should load correctly
# 3. Delete session - should work, UI should update
```

---

## Phase 4: Extract WebSocket Event Handler Hook

**Goal**: Extract the massive WebSocket event handling into a dedicated hook with clear event handler separation.

### Files to Create

#### `frontend/hooks/useChatWebSocket.ts`
```typescript
import { useEffect, useRef, useCallback } from 'react'
import { getChatWebSocket, type WebSocketEvent } from '@/lib/websocket'
import { getDisplayName, generateMessageId } from '@/lib/chatUtils'
import {
  type PanelAgentActivity,
  type PanelToolActivity,
  type StreamingMessage
} from '@/lib/AgentActivityContext'
import { type Message } from '@/types/chat'

// Event handler types for each event category
interface ChatEventHandlers {
  onChatStart: () => void
  onChatComplete: () => void
  onError: (message: string) => void
}

interface AgentEventHandlers {
  onAgentStart: (agent: string, agentType: string) => Message
  onAgentDelta: (delta: string) => void
  onAgentComplete: (content: string, agent: string, thinking: string, agentType?: string) => void
  onAgentSpawn: (agent: string, parentAgent: string, description?: string) => void
  onAgentCompleteSubagent: (agent: string) => void
}

interface ThinkingEventHandlers {
  onThinkingStart: () => void
  onThinkingDelta: (delta: string) => void
  onThinkingComplete: (thinking: string) => void
}

interface ToolEventHandlers {
  onToolStart: (tool: string, description: string, agentName?: string) => void
  onToolComplete: (tool: string, success: boolean, summary?: string, agentName?: string) => void
}

interface UseChatWebSocketOptions {
  setMessages: React.Dispatch<React.SetStateAction<Message[]>>
  setAgentActivities: React.Dispatch<React.SetStateAction<PanelAgentActivity[]>>
  setToolActivities: React.Dispatch<React.SetStateAction<PanelToolActivity[]>>
  setStreamingMessage: React.Dispatch<React.SetStateAction<StreamingMessage | null>>
  setIsLoading: (loading: boolean) => void
  setIsStreaming: React.Dispatch<React.SetStateAction<boolean>>
}

interface UseChatWebSocketReturn {
  isConnected: boolean
  send: (message: string, options?: { session_id?: string; attachments?: Array<{type: string; name: string; content: string; mimeType?: string}> }) => void
}

export function useChatWebSocket(options: UseChatWebSocketOptions): UseChatWebSocketReturn {
  const {
    setMessages,
    setAgentActivities,
    setToolActivities,
    setStreamingMessage,
    setIsLoading,
    setIsStreaming,
  } = options

  const [isConnected, setIsConnected] = useState(false)
  const wsRef = useRef(getChatWebSocket())
  const pendingMessageRef = useRef<{ content: string; agent?: string; thinking?: string } | null>(null)

  // ===== CHAT EVENT HANDLERS =====
  const handleChatStart = useCallback(() => {
    setIsLoading(true)
    setIsStreaming(true)
    setAgentActivities([])
    setToolActivities([])
    setAgentActivities([{
      id: generateMessageId(),
      name: 'Axel',
      status: 'thinking',
      startTime: new Date(),
    }])
  }, [setIsLoading, setIsStreaming, setAgentActivities, setToolActivities])

  const handleChatComplete = useCallback(() => {
    setIsLoading(false)
    setIsStreaming(false)
    setStreamingMessage(null)
    setAgentActivities((prev) =>
      prev.map((a) => ({
        ...a,
        status: 'complete' as const,
        endTime: a.endTime || new Date(),
      }))
    )
    pendingMessageRef.current = null
  }, [setIsLoading, setIsStreaming, setStreamingMessage, setAgentActivities])

  const handleError = useCallback((message: string) => {
    setIsLoading(false)
    pendingMessageRef.current = null
    setAgentActivities((prev) =>
      prev.map((a) =>
        a.status !== 'complete' ? { ...a, status: 'error' as const, endTime: new Date() } : a
      )
    )
    console.error('Chat error:', message)
  }, [setIsLoading, setAgentActivities])

  // ===== TOOL EVENT HANDLERS =====
  const handleToolStart = useCallback((tool: string, description: string, agentName?: string) => {
    setToolActivities((prev) => [
      ...prev,
      {
        id: generateMessageId(),
        tool,
        description,
        status: 'running',
        timestamp: new Date(),
        agentName,
      },
    ])
    // Update COO status to working when tools are used
    setAgentActivities((prev) => {
      const updated = [...prev]
      const cooIdx = updated.findIndex(a => a.name.includes('COO'))
      if (cooIdx !== -1 && updated[cooIdx].status === 'thinking') {
        updated[cooIdx] = { ...updated[cooIdx], status: 'working' }
      }
      return updated
    })
  }, [setToolActivities, setAgentActivities])

  const handleToolComplete = useCallback((
    tool: string,
    success: boolean,
    summary?: string,
    agentName?: string
  ) => {
    setToolActivities((prev) => {
      const idx = [...prev].reverse().findIndex(
        (t) => t.tool === tool && t.status === 'running'
      )
      if (idx === -1) return prev
      const actualIdx = prev.length - 1 - idx
      const updated = [...prev]
      updated[actualIdx] = {
        ...updated[actualIdx],
        status: success ? 'complete' : 'error',
        summary,
        endTime: new Date(),
        agentName: agentName || updated[actualIdx].agentName,
      }
      return updated
    })
    // Check if COO should go back to working
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
  }, [setToolActivities, setAgentActivities])

  // ===== AGENT EVENT HANDLERS =====
  const handleAgentSpawn = useCallback((
    agent: string,
    parentAgent: string,
    description?: string
  ) => {
    setAgentActivities((prev) => {
      const agentName = getDisplayName(agent)
      const parentName = getDisplayName(parentAgent || 'COO')

      const existingIdx = prev.findIndex(a => a.name === agentName)

      let updated = prev.map(a =>
        a.name === parentName || (parentName === 'COO' && a.name.includes('COO'))
          ? { ...a, status: 'delegating' as const }
          : a
      )

      if (existingIdx !== -1) {
        updated = updated.map((a, idx) =>
          idx === existingIdx
            ? {
                ...a,
                status: 'working' as const,
                description: description?.substring(0, 100),
                startTime: new Date(),
                endTime: undefined,
              }
            : a
        )
        return updated
      }

      return [
        ...updated,
        {
          id: generateMessageId(),
          name: agentName,
          status: 'working' as const,
          description: description?.substring(0, 100),
          startTime: new Date(),
        },
      ]
    })
  }, [setAgentActivities])

  const handleAgentCompleteSubagent = useCallback((agent: string) => {
    setAgentActivities((prev) => {
      const agentName = getDisplayName(agent)
      return prev.map(a => {
        if (a.name === agentName) {
          return { ...a, status: 'complete' as const, endTime: new Date() }
        }
        if (a.status === 'delegating') {
          const otherActive = prev.some(other =>
            other.name !== agentName &&
            other.status !== 'complete' &&
            other.status !== 'error' &&
            !other.name.includes('COO')
          )
          if (!otherActive) {
            return { ...a, status: 'working' as const }
          }
        }
        return a
      })
    })
  }, [setAgentActivities])

  const handleAgentStart = useCallback((agent: string, agentType: string) => {
    const newStreamingMsg: StreamingMessage = {
      id: generateMessageId(),
      type: 'agent',
      content: '',
      agent: getDisplayName(agent),
      agentType: agentType || 'worker',
      status: 'thinking',
      timestamp: new Date(),
    }

    setMessages((prev) => {
      const cleaned = prev.map((m) =>
        m.type === 'agent' && m.status === 'thinking'
          ? { ...m, status: 'complete' as const }
          : m
      )
      return [...cleaned, newStreamingMsg]
    })
    setStreamingMessage(newStreamingMsg)
  }, [setMessages, setStreamingMessage])

  const handleAgentDelta = useCallback((delta: string) => {
    setMessages((prev) => {
      const thinkingIdx = prev.findIndex(
        (m) => m.type === 'agent' && m.status === 'thinking'
      )
      if (thinkingIdx !== -1) {
        const updated = [...prev]
        const msg = updated[thinkingIdx]
        updated[thinkingIdx] = {
          ...msg,
          content: msg.content + delta,
        }
        setStreamingMessage({
          ...updated[thinkingIdx],
          type: 'agent',
          agent: msg.agent || 'Agent',
          agentType: msg.agentType || 'worker',
          status: 'thinking',
        })
        return updated
      }
      return prev
    })
  }, [setMessages, setStreamingMessage])

  const handleAgentComplete = useCallback((
    content: string,
    agent: string,
    thinking: string,
    agentType?: string
  ) => {
    const completeAgent = getDisplayName(agent || 'Claude')

    pendingMessageRef.current = {
      content,
      agent: completeAgent,
      thinking,
    }

    setMessages((prev) => {
      const thinkingIdx = prev.findIndex(
        (m) => m.type === 'agent' && m.status === 'thinking'
      )

      if (thinkingIdx !== -1) {
        const updated = [...prev]
        const finalContent = content || updated[thinkingIdx].content
        const finalThinking = thinking || updated[thinkingIdx].thinking

        updated[thinkingIdx] = {
          ...updated[thinkingIdx],
          content: finalContent,
          thinking: finalThinking,
          isThinking: false,
          status: 'complete',
          agent: getDisplayName(agent || updated[thinkingIdx].agent || 'Agent'),
          agentType: agentType || updated[thinkingIdx].agentType,
        }

        pendingMessageRef.current = {
          content: finalContent,
          agent: completeAgent,
          thinking: finalThinking,
        }
        return updated
      }

      return [
        ...prev,
        {
          id: generateMessageId(),
          type: 'agent',
          content,
          agent: completeAgent,
          agentType: agentType || 'worker',
          status: 'complete',
          timestamp: new Date(),
          thinking,
        },
      ]
    })

    setIsLoading(false)
    setIsStreaming(false)
    setStreamingMessage(null)
  }, [setMessages, setIsLoading, setIsStreaming, setStreamingMessage])

  // ===== THINKING EVENT HANDLERS =====
  const handleThinkingStart = useCallback(() => {
    setMessages((prev) => {
      const lastMessage = prev[prev.length - 1]
      if (lastMessage && lastMessage.type === 'agent' && lastMessage.status === 'thinking') {
        const updated = { ...lastMessage, isThinking: true, thinking: '' }
        setStreamingMessage({
          ...updated,
          type: 'agent',
          agent: updated.agent || 'Agent',
          agentType: updated.agentType || 'worker',
          status: 'thinking',
        })
        return [...prev.slice(0, -1), updated]
      }
      return prev
    })
  }, [setMessages, setStreamingMessage])

  const handleThinkingDelta = useCallback((delta: string) => {
    setMessages((prev) => {
      const lastMessage = prev[prev.length - 1]
      if (lastMessage && lastMessage.type === 'agent' && lastMessage.isThinking) {
        const updated = { ...lastMessage, thinking: (lastMessage.thinking || '') + delta }
        setStreamingMessage({
          ...updated,
          type: 'agent',
          agent: updated.agent || 'Agent',
          agentType: updated.agentType || 'worker',
          status: 'thinking',
        })
        return [...prev.slice(0, -1), updated]
      }
      return prev
    })
  }, [setMessages, setStreamingMessage])

  const handleThinkingComplete = useCallback((thinking: string) => {
    setMessages((prev) => {
      const lastMessage = prev[prev.length - 1]
      if (lastMessage && lastMessage.type === 'agent' && lastMessage.isThinking) {
        const updated = { ...lastMessage, isThinking: false, thinking: thinking || lastMessage.thinking }
        setStreamingMessage({
          ...updated,
          type: 'agent',
          agent: updated.agent || 'Agent',
          agentType: updated.agentType || 'worker',
          status: 'thinking',
        })
        return [...prev.slice(0, -1), updated]
      }
      return prev
    })
  }, [setMessages, setStreamingMessage])

  // ===== MAIN EVENT DISPATCHER =====
  useEffect(() => {
    const ws = wsRef.current

    const handleEvent = (event: WebSocketEvent) => {
      switch (event.type) {
        case 'chat_start':
          handleChatStart()
          break
        case 'chat_complete':
          handleChatComplete()
          break
        case 'error':
          handleError(event.message || 'Unknown error')
          break
        case 'tool_start':
          handleToolStart(event.tool || 'Unknown', event.description || '', event.agentName)
          break
        case 'tool_complete':
          handleToolComplete(event.tool || 'Unknown', event.success ?? true, event.summary, event.agentName)
          break
        case 'agent_spawn':
          handleAgentSpawn(event.agent || 'Agent', (event as any).parentAgent || 'COO', (event as any).description)
          break
        case 'agent_complete_subagent':
          handleAgentCompleteSubagent(event.agent || '')
          break
        case 'agent_start':
          handleAgentStart(event.agent || 'Agent', event.agent_type || 'worker')
          break
        case 'agent_delta':
          handleAgentDelta(event.delta || '')
          break
        case 'agent_complete':
          handleAgentComplete(event.content || '', event.agent || 'Claude', event.thinking || '', event.agent_type)
          break
        case 'thinking_start':
          handleThinkingStart()
          break
        case 'thinking_delta':
          handleThinkingDelta(event.delta || '')
          break
        case 'thinking_complete':
          handleThinkingComplete(event.thinking || '')
          break
      }
    }

    const handleDisconnect = () => {
      setIsConnected(false)
      setIsLoading(false)
      setIsStreaming(false)
      setStreamingMessage(null)
      pendingMessageRef.current = null
    }

    ws.on('*', handleEvent)
    ws.on('disconnected', handleDisconnect)

    ws.connect()
      .then(() => setIsConnected(true))
      .catch((e) => {
        console.error('WebSocket connection failed:', e)
        setIsConnected(false)
      })

    return () => {
      ws.off('*', handleEvent)
      ws.off('disconnected', handleDisconnect)
    }
  }, [
    handleChatStart, handleChatComplete, handleError,
    handleToolStart, handleToolComplete,
    handleAgentSpawn, handleAgentCompleteSubagent, handleAgentStart, handleAgentDelta, handleAgentComplete,
    handleThinkingStart, handleThinkingDelta, handleThinkingComplete,
    setIsLoading, setIsStreaming, setStreamingMessage,
  ])

  const send = useCallback((message: string, options?: { session_id?: string; attachments?: Array<{type: string; name: string; content: string; mimeType?: string}> }) => {
    try {
      wsRef.current.send(message, options)
    } catch (e) {
      console.error('Failed to send message:', e)
      setIsLoading(false)
    }
  }, [setIsLoading])

  return { isConnected, send }
}
```

### Changes to page.tsx
1. Import `useChatWebSocket`
2. Remove the massive `handleEvent` function and WebSocket setup useEffect
3. Replace with hook usage
4. Remove lines 292-723

### Testing Phase 4
```bash
# Test each event type:
# 1. Send a message - verify chat_start shows loading indicator
# 2. Wait for response - verify agent_delta streams content correctly
# 3. Check thinking display works (if enabled in backend)
# 4. Verify tool activities show in panel when COO uses tools
# 5. Verify agent spawn shows new agents in activity panel
# 6. Verify chat_complete clears loading state
# 7. Disconnect backend, verify error handling works
```

---

## Phase 5: Extract Message State Hook

**Goal**: Extract message state management including streaming restoration.

### Files to Create

#### `frontend/hooks/useChatMessages.ts`
```typescript
import { useState, useEffect, useRef, useCallback } from 'react'
import { type Message } from '@/types/chat'
import { type StreamingMessage } from '@/lib/AgentActivityContext'

interface UseChatMessagesOptions {
  isStreaming: boolean
  streamingMessage: StreamingMessage | null
  setIsLoading: (loading: boolean) => void
}

interface UseChatMessagesReturn {
  messages: Message[]
  setMessages: React.Dispatch<React.SetStateAction<Message[]>>
  addUserMessage: (content: string, attachments?: Message['attachments']) => void
  clearMessages: () => void
  messagesEndRef: React.RefObject<HTMLDivElement>
  messagesContainerRef: React.RefObject<HTMLDivElement>
}

export function useChatMessages(options: UseChatMessagesOptions): UseChatMessagesReturn {
  const { isStreaming, streamingMessage, setIsLoading } = options

  const [messages, setMessages] = useState<Message[]>([])
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const messagesContainerRef = useRef<HTMLDivElement>(null)
  const isInitialLoadRef = useRef(true)
  const hasRestoredRef = useRef(false)

  // Restore streaming message from context when navigating back
  useEffect(() => {
    if (hasRestoredRef.current) return
    if (isStreaming && streamingMessage) {
      hasRestoredRef.current = true
      setIsLoading(true)
      setMessages((prev) => {
        const exists = prev.some(m => m.id === streamingMessage.id)
        if (exists) return prev
        return [...prev, streamingMessage]
      })
    }
  }, [isStreaming, streamingMessage, setIsLoading])

  // Keep messages in sync with streaming message updates from context
  useEffect(() => {
    if (!streamingMessage || !hasRestoredRef.current) return
    setMessages((prev) => {
      const idx = prev.findIndex(m => m.id === streamingMessage.id)
      if (idx === -1) return prev
      if (prev[idx].content === streamingMessage.content &&
          prev[idx].thinking === streamingMessage.thinking) {
        return prev
      }
      const updated = [...prev]
      updated[idx] = streamingMessage
      return updated
    })
  }, [streamingMessage])

  // Auto-scroll to bottom
  const scrollToBottom = useCallback((instant = false) => {
    if (instant && messagesContainerRef.current) {
      messagesContainerRef.current.scrollTop = messagesContainerRef.current.scrollHeight
    } else {
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
    }
  }, [])

  useEffect(() => {
    if (messages.length > 0) {
      scrollToBottom(isInitialLoadRef.current)
      isInitialLoadRef.current = false
    }
  }, [messages, scrollToBottom])

  const addUserMessage = useCallback((content: string, attachments?: Message['attachments']) => {
    setMessages((prev) => [
      ...prev,
      {
        id: crypto.randomUUID(),
        type: 'user',
        content,
        timestamp: new Date(),
        attachments,
      },
    ])
  }, [])

  const clearMessages = useCallback(() => {
    setMessages([])
    isInitialLoadRef.current = true
  }, [])

  return {
    messages,
    setMessages,
    addUserMessage,
    clearMessages,
    messagesEndRef,
    messagesContainerRef,
  }
}
```

### Changes to page.tsx
1. Import `useChatMessages`
2. Replace message state and related effects with hook
3. Remove lines 35, 56-57, 62, 70-126

### Testing Phase 5
```bash
# Test:
# 1. Send a message - should appear immediately
# 2. Navigate away during streaming, navigate back - message should restore
# 3. Messages should auto-scroll smoothly for new messages
# 4. Initial load should instant-scroll (no animation)
```

---

## Phase 6: Extract UI Components

**Goal**: Break the render function into smaller, focused components.

### Files to Create

#### `frontend/components/chat/ChatHeader.tsx`
```typescript
'use client'

import { Terminal, WifiOff, MessageSquare, ChevronLeft, ChevronRight, Activity } from 'lucide-react'

interface ChatHeaderProps {
  isConnected: boolean
  showSidebar: boolean
  onToggleSidebar: () => void
  onShowMobileHistory: () => void
  onShowMobileActivity: () => void
  isMobile: boolean
  hasActivity: boolean
  isActivityProcessing: boolean
}

export function ChatHeader({
  isConnected,
  showSidebar,
  onToggleSidebar,
  onShowMobileHistory,
  onShowMobileActivity,
  isMobile,
  hasActivity,
  isActivityProcessing,
}: ChatHeaderProps) {
  return (
    <div className="flex items-center justify-between px-3 md:px-6 py-3 md:py-4 border-b border-zinc-800/50">
      <div className="flex items-center gap-2 md:gap-3">
        {/* Desktop sidebar toggle */}
        <button
          onClick={onToggleSidebar}
          className="hidden md:block p-1 hover:bg-zinc-800/50 rounded transition-colors"
          title={showSidebar ? 'Hide history' : 'Show history'}
        >
          {showSidebar ? (
            <ChevronLeft className="w-5 h-5 text-zinc-500" />
          ) : (
            <ChevronRight className="w-5 h-5 text-zinc-500" />
          )}
        </button>
        {/* Mobile history toggle */}
        <button
          onClick={onShowMobileHistory}
          className="md:hidden p-2 -ml-1 hover:bg-zinc-800/50 active:bg-zinc-800 rounded-lg transition-colors touch-manipulation"
          title="Chat history"
        >
          <MessageSquare className="w-5 h-5 text-zinc-500" />
        </button>
        <Terminal className="w-5 h-5 md:w-6 md:h-6 text-emerald-400" />
        <div>
          <h1 className="font-medium text-white text-sm md:text-base">Chat with COO</h1>
          <p className="text-xs text-zinc-600 hidden md:block">Axel - Chief Operating Officer</p>
        </div>
      </div>

      <div className="flex items-center gap-2">
        {isMobile && (isActivityProcessing || hasActivity) && (
          <button
            onClick={onShowMobileActivity}
            className={`p-2 rounded-lg transition-all touch-manipulation ${
              isActivityProcessing ? 'bg-emerald-400/20 animate-pulse' : 'hover:bg-zinc-800/50'
            }`}
            title="View activity"
          >
            <Activity className={`w-5 h-5 ${isActivityProcessing ? 'text-emerald-400' : 'text-zinc-500'}`} />
          </button>
        )}
        {isConnected ? (
          <span className="flex items-center gap-1.5 text-xs text-green-500">
            <span className="w-1.5 h-1.5 rounded-full bg-green-500 shadow-[0_0_6px_rgba(34,197,94,0.5)]" />
            <span className="hidden sm:inline">Connected</span>
          </span>
        ) : (
          <span className="flex items-center gap-1.5 text-xs text-red-500">
            <WifiOff className="w-3 h-3" />
            <span className="hidden sm:inline">Disconnected</span>
          </span>
        )}
      </div>
    </div>
  )
}
```

#### `frontend/components/chat/SessionSidebar.tsx`
```typescript
'use client'

import { Plus, MessageSquare, Trash2 } from 'lucide-react'
import { type ChatSessionSummary } from '@/lib/api'

interface SessionSidebarProps {
  sessions: ChatSessionSummary[]
  currentSessionId: string | null
  showSidebar: boolean
  onCreateSession: () => void
  onSelectSession: (id: string) => void
  onDeleteSession: (id: string, e: React.MouseEvent) => void
}

export function SessionSidebar({
  sessions,
  currentSessionId,
  showSidebar,
  onCreateSession,
  onSelectSession,
  onDeleteSession,
}: SessionSidebarProps) {
  return (
    <div className={`hidden md:flex ${showSidebar ? 'w-64' : 'w-0'} transition-all duration-200 overflow-hidden border-r border-zinc-800/50 flex-col bg-[#0a0a0a]`}>
      <div className="p-3 border-b border-zinc-800/50 flex items-center justify-between">
        <span className="text-sm font-medium text-zinc-500">History</span>
        <button
          onClick={onCreateSession}
          className="p-1.5 hover:bg-zinc-800/50 rounded transition-colors"
          title="New chat"
        >
          <Plus className="w-4 h-4 text-emerald-400" />
        </button>
      </div>
      <div className="flex-1 overflow-auto">
        {sessions.map((session) => (
          <div
            key={session.id}
            onClick={() => onSelectSession(session.id)}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => e.key === 'Enter' && onSelectSession(session.id)}
            className={`w-full px-3 py-2 text-left hover:bg-zinc-800/30 transition-colors group flex items-center gap-2 cursor-pointer ${
              currentSessionId === session.id ? 'bg-zinc-800/50 border-l-2 border-l-emerald-400' : 'border-l-2 border-l-transparent'
            }`}
          >
            <MessageSquare className="w-4 h-4 text-zinc-600 flex-shrink-0" />
            <div className="flex-1 min-w-0">
              <p className="text-sm text-zinc-400 truncate">{session.title}</p>
              <p className="text-xs text-violet-500/60">{session.message_count} messages</p>
            </div>
            <button
              onClick={(e) => onDeleteSession(session.id, e)}
              className="opacity-0 group-hover:opacity-100 p-1 hover:bg-zinc-700 rounded transition-all"
              title="Delete"
            >
              <Trash2 className="w-3 h-3 text-zinc-500" />
            </button>
          </div>
        ))}
        {sessions.length === 0 && (
          <p className="text-xs text-zinc-600 p-3">No chat history yet</p>
        )}
      </div>
    </div>
  )
}
```

#### `frontend/components/chat/MobileHistorySheet.tsx`
```typescript
'use client'

import { Plus, MessageSquare, Trash2, X } from 'lucide-react'
import { type ChatSessionSummary } from '@/lib/api'

interface MobileHistorySheetProps {
  isOpen: boolean
  sessions: ChatSessionSummary[]
  currentSessionId: string | null
  onClose: () => void
  onCreateSession: () => void
  onSelectSession: (id: string) => void
  onDeleteSession: (id: string, e: React.MouseEvent) => void
}

export function MobileHistorySheet({
  isOpen,
  sessions,
  currentSessionId,
  onClose,
  onCreateSession,
  onSelectSession,
  onDeleteSession,
}: MobileHistorySheetProps) {
  if (!isOpen) return null

  return (
    <>
      {/* Overlay */}
      <div
        className="fixed inset-0 z-40 bg-black/80"
        onClick={onClose}
      />
      {/* Bottom Sheet */}
      <div className="fixed bottom-0 left-0 right-0 z-50 bg-[#0d0d0d] rounded-t-2xl border-t border-zinc-800/50 max-h-[70vh] flex flex-col animate-slide-up">
        <div className="flex items-center justify-between p-4 border-b border-zinc-800/50">
          <span className="font-medium text-zinc-300">Chat History</span>
          <div className="flex items-center gap-2">
            <button
              onClick={onCreateSession}
              className="p-2 hover:bg-zinc-800/50 rounded-lg transition-colors touch-manipulation"
              title="New chat"
            >
              <Plus className="w-5 h-5 text-emerald-400" />
            </button>
            <button
              onClick={onClose}
              className="p-2 hover:bg-zinc-800/50 rounded-lg transition-colors touch-manipulation"
            >
              <X className="w-5 h-5 text-zinc-500" />
            </button>
          </div>
        </div>
        <div className="flex-1 overflow-auto p-2">
          {sessions.map((session) => (
            <div
              key={session.id}
              onClick={() => {
                onSelectSession(session.id)
                onClose()
              }}
              role="button"
              tabIndex={0}
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  onSelectSession(session.id)
                  onClose()
                }
              }}
              className={`w-full px-4 py-3 text-left hover:bg-zinc-800/30 active:bg-zinc-800/50 transition-colors flex items-center gap-3 rounded-lg touch-manipulation min-h-[56px] cursor-pointer ${
                currentSessionId === session.id ? 'bg-zinc-800/50 border-l-2 border-l-emerald-400' : 'border-l-2 border-l-transparent'
              }`}
            >
              <MessageSquare className="w-5 h-5 text-zinc-600 flex-shrink-0" />
              <div className="flex-1 min-w-0">
                <p className="text-sm text-zinc-400 truncate">{session.title}</p>
                <p className="text-xs text-violet-500/60">{session.message_count} messages</p>
              </div>
              <button
                onClick={(e) => {
                  onDeleteSession(session.id, e)
                }}
                className="p-2 hover:bg-zinc-700 rounded-lg transition-colors touch-manipulation"
                title="Delete"
              >
                <Trash2 className="w-4 h-4 text-zinc-500" />
              </button>
            </div>
          ))}
          {sessions.length === 0 && (
            <p className="text-sm text-zinc-600 p-4 text-center">No chat history yet</p>
          )}
        </div>
      </div>
    </>
  )
}
```

#### `frontend/components/chat/MobileActivitySheet.tsx`
```typescript
'use client'

import { Activity, X } from 'lucide-react'
import ActivityPanel from '@/components/ActivityPanel'
import { type PanelAgentActivity, type PanelToolActivity } from '@/lib/AgentActivityContext'

interface MobileActivitySheetProps {
  isOpen: boolean
  agents: PanelAgentActivity[]
  tools: PanelToolActivity[]
  isProcessing: boolean
  onClose: () => void
  onClear: () => void
}

export function MobileActivitySheet({
  isOpen,
  agents,
  tools,
  isProcessing,
  onClose,
  onClear,
}: MobileActivitySheetProps) {
  if (!isOpen) return null

  return (
    <>
      {/* Overlay */}
      <div
        className="fixed inset-0 z-40 bg-black/80"
        onClick={onClose}
      />
      {/* Bottom Sheet */}
      <div className="fixed bottom-0 left-0 right-0 z-50 bg-[#0d0d0d] rounded-t-2xl border-t border-zinc-800/50 max-h-[70vh] flex flex-col animate-slide-up">
        <div className="flex items-center justify-between p-4 border-b border-zinc-800/50">
          <div className="flex items-center gap-2">
            <Activity className={`w-5 h-5 ${isProcessing ? 'text-emerald-400 animate-pulse' : 'text-zinc-500'}`} />
            <span className="font-medium text-zinc-300">Activity</span>
            {isProcessing && (
              <span className="px-1.5 py-0.5 rounded-full bg-emerald-400/20 text-emerald-400 text-xs">
                Active
              </span>
            )}
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-zinc-800/50 rounded-lg transition-colors touch-manipulation"
          >
            <X className="w-5 h-5 text-zinc-500" />
          </button>
        </div>
        <div className="flex-1 overflow-auto">
          <ActivityPanel
            agents={agents}
            tools={tools}
            isProcessing={isProcessing}
            onClear={onClear}
          />
        </div>
      </div>
    </>
  )
}
```

#### `frontend/components/chat/EmptyState.tsx`
```typescript
'use client'

import { Terminal } from 'lucide-react'

interface EmptyStateProps {
  onQuickSend: (message: string) => void
}

const SUGGESTIONS = [
  'What is the current status of the ASA project?',
  'Research sparse attention implementations for ASA',
]

export function EmptyState({ onQuickSend }: EmptyStateProps) {
  return (
    <div className="flex flex-col items-center justify-center h-full text-center px-4">
      <Terminal className="w-12 h-12 md:w-16 md:h-16 text-emerald-400/50 mb-4" />
      <h2 className="text-lg md:text-xl font-medium text-zinc-400 mb-2">
        Chat with the COO
      </h2>
      <p className="text-sm text-zinc-600 max-w-md">
        You are connected to Axel, your COO. Ask about your swarms,
        request research, or coordinate tasks.
      </p>
      <div className="mt-6 space-y-2 text-left w-full max-w-md">
        <p className="text-xs text-zinc-600">Try asking:</p>
        <div className="space-y-2">
          {SUGGESTIONS.map((suggestion) => (
            <button
              key={suggestion}
              onClick={() => onQuickSend(suggestion)}
              className="block w-full text-left text-sm text-zinc-500 hover:text-violet-400 active:text-violet-300 transition-all duration-200 p-3 md:p-2 bg-zinc-900/50 md:bg-transparent rounded-lg md:rounded-none touch-manipulation border border-zinc-800/50 md:border-0 hover:bg-violet-500/5 hover:border-violet-500/30"
            >
              <span className="text-emerald-400 mr-2">&gt;</span>{suggestion}
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}
```

#### `frontend/components/chat/MessageList.tsx`
```typescript
'use client'

import { type Message } from '@/types/chat'
import ChatMessage from '@/components/ChatMessage'
import AgentResponse from '@/components/AgentResponse'
import { EmptyState } from './EmptyState'
import { forwardRef } from 'react'

interface MessageListProps {
  messages: Message[]
  onQuickSend: (message: string) => void
  messagesEndRef: React.RefObject<HTMLDivElement>
}

export const MessageList = forwardRef<HTMLDivElement, MessageListProps>(
  function MessageList({ messages, onQuickSend, messagesEndRef }, ref) {
    return (
      <div ref={ref} className="flex-1 overflow-auto p-3 md:p-6 space-y-4 md:space-y-6 relative">
        {messages.length === 0 ? (
          <EmptyState onQuickSend={onQuickSend} />
        ) : (
          messages.map((message) =>
            message.type === 'user' ? (
              <ChatMessage
                key={message.id}
                content={message.content}
                timestamp={message.timestamp}
                attachments={message.attachments}
              />
            ) : (
              <AgentResponse
                key={message.id}
                agent={message.agent || 'Agent'}
                agentType={message.agentType || 'worker'}
                content={message.content}
                status={message.status}
                thinking={message.thinking}
                isThinking={message.isThinking}
              />
            )
          )
        )}
        <div ref={messagesEndRef} />
      </div>
    )
  }
)
```

### Final page.tsx (~200 lines)
```typescript
'use client'

import { useState, useCallback } from 'react'
import { useAgentActivity } from '@/lib/AgentActivityContext'
import { useMobileLayout } from '@/components/MobileLayout'
import { useChatSessions } from '@/hooks/useChatSessions'
import { useChatMessages } from '@/hooks/useChatMessages'
import { useChatWebSocket } from '@/hooks/useChatWebSocket'
import { useLoadingTimeout } from '@/hooks/useLoadingTimeout'
import { type Attachment } from '@/components/ChatInput'
import ChatInput from '@/components/ChatInput'

// UI Components
import { ChatHeader } from '@/components/chat/ChatHeader'
import { SessionSidebar } from '@/components/chat/SessionSidebar'
import { MobileHistorySheet } from '@/components/chat/MobileHistorySheet'
import { MobileActivitySheet } from '@/components/chat/MobileActivitySheet'
import { MessageList } from '@/components/chat/MessageList'

export default function ChatPage() {
  const [showSidebar, setShowSidebar] = useState(true)
  const [showMobileHistory, setShowMobileHistory] = useState(false)
  const [showMobileActivity, setShowMobileActivity] = useState(false)

  const { isMobile } = useMobileLayout()
  const {
    panelAgentActivities: agentActivities,
    panelToolActivities: toolActivities,
    setPanelAgentActivities: setAgentActivities,
    setPanelToolActivities: setToolActivities,
    clearPanelActivities,
    streamingMessage,
    setStreamingMessage,
    isStreaming,
    setIsStreaming,
  } = useAgentActivity()

  // Session management
  const {
    sessionId,
    sessions,
    loadSession,
    createNewSession,
    deleteSession,
    getOrCreateSessionId,
  } = useChatSessions()

  // Loading state with timeout
  const { isLoading, setIsLoading } = useLoadingTimeout(setAgentActivities)

  // Message state
  const {
    messages,
    setMessages,
    addUserMessage,
    clearMessages,
    messagesEndRef,
    messagesContainerRef,
  } = useChatMessages({
    isStreaming,
    streamingMessage,
    setIsLoading,
  })

  // WebSocket connection
  const { isConnected, send } = useChatWebSocket({
    setMessages,
    setAgentActivities,
    setToolActivities,
    setStreamingMessage,
    setIsLoading,
    setIsStreaming,
  })

  // Handle session selection (loads messages)
  const handleSelectSession = useCallback(async (id: string) => {
    const loadedMessages = await loadSession(id)
    setMessages(loadedMessages)
  }, [loadSession, setMessages])

  // Handle new session creation
  const handleCreateSession = useCallback(async () => {
    await createNewSession()
    clearMessages()
  }, [createNewSession, clearMessages])

  // Send message handler
  const handleSend = useCallback(async (content: string, attachments: Attachment[]) => {
    // Clear old completed activities
    setAgentActivities((prev) => prev.filter(a => a.status !== 'complete' && a.status !== 'error'))
    setToolActivities((prev) => prev.filter(t => t.status === 'running'))

    // Get or create session (race-condition safe)
    const currentSessionId = await getOrCreateSessionId()
    if (!currentSessionId) return

    // Add user message to UI
    addUserMessage(content, attachments.length > 0 ? attachments : undefined)

    // Build message with text attachments inline
    let fullMessage = content
    const imageAttachments = attachments.filter(a => a.type === 'image').map(a => ({
      type: a.type,
      name: a.name,
      content: a.content,
      mimeType: a.mimeType,
    }))

    if (attachments.length > 0) {
      const descriptions = attachments.map((a) => {
        if (a.type === 'text') {
          return `\n\n--- Attached Text: ${a.name} ---\n${a.content}\n--- End Attachment ---`
        } else if (a.type === 'image') {
          return `\n\n[Image attached: ${a.name}]`
        }
        return `\n\n[Document: ${a.name}]`
      })
      fullMessage += descriptions.join('')
    }

    // Send via WebSocket
    send(fullMessage, {
      session_id: currentSessionId,
      attachments: imageAttachments.length > 0 ? imageAttachments : undefined,
    })
  }, [getOrCreateSessionId, addUserMessage, setAgentActivities, setToolActivities, send])

  // Quick send for suggestion buttons
  const handleQuickSend = useCallback((content: string) => {
    handleSend(content, [])
  }, [handleSend])

  // Activity state for mobile
  const hasActivity = agentActivities.length > 0 || toolActivities.length > 0
  const isActivityProcessing = agentActivities.some(
    (a) => a.status !== 'complete' && a.status !== 'error'
  )

  return (
    <div className="flex h-full relative bg-[#0d0d0d] overflow-hidden">
      <SessionSidebar
        sessions={sessions}
        currentSessionId={sessionId}
        showSidebar={showSidebar}
        onCreateSession={handleCreateSession}
        onSelectSession={handleSelectSession}
        onDeleteSession={deleteSession}
      />

      <MobileHistorySheet
        isOpen={isMobile && showMobileHistory}
        sessions={sessions}
        currentSessionId={sessionId}
        onClose={() => setShowMobileHistory(false)}
        onCreateSession={handleCreateSession}
        onSelectSession={handleSelectSession}
        onDeleteSession={deleteSession}
      />

      <MobileActivitySheet
        isOpen={isMobile && showMobileActivity}
        agents={agentActivities}
        tools={toolActivities}
        isProcessing={isActivityProcessing}
        onClose={() => setShowMobileActivity(false)}
        onClear={clearPanelActivities}
      />

      <div className="flex-1 flex flex-col min-w-0 bg-[#0d0d0d]">
        <ChatHeader
          isConnected={isConnected}
          showSidebar={showSidebar}
          onToggleSidebar={() => setShowSidebar(!showSidebar)}
          onShowMobileHistory={() => setShowMobileHistory(true)}
          onShowMobileActivity={() => setShowMobileActivity(true)}
          isMobile={isMobile}
          hasActivity={hasActivity}
          isActivityProcessing={isActivityProcessing}
        />

        <MessageList
          ref={messagesContainerRef}
          messages={messages}
          onQuickSend={handleQuickSend}
          messagesEndRef={messagesEndRef}
        />

        <div className="p-2 md:p-4 border-t border-zinc-800/50 pb-[calc(0.5rem+env(safe-area-inset-bottom))] md:pb-4 bg-[#0a0a0a]">
          <ChatInput
            onSend={handleSend}
            disabled={!isConnected}
            placeholder={
              !isConnected
                ? 'Connecting...'
                : isLoading
                ? 'Type to interject...'
                : 'Type a message...'
            }
          />
        </div>
      </div>
    </div>
  )
}
```

### Testing Phase 6
```bash
# Full regression test:
# 1. Desktop: sidebar toggle, create/switch/delete sessions
# 2. Mobile: history sheet, activity sheet animations
# 3. Send messages with attachments
# 4. Verify streaming works correctly
# 5. Verify activity panel shows tool/agent activity
# 6. Test navigation away and back during streaming
# 7. Test error scenarios (disconnect backend)
```

---

## Implementation Order Summary

| Phase | Files Created | Lines Removed from page.tsx | Risk Level |
|-------|--------------|----------------------------|------------|
| 1 | `types/chat.ts`, `lib/chatUtils.ts` | ~10 | Low |
| 2 | `hooks/useLoadingTimeout.ts` | ~30 | Low |
| 3 | `hooks/useChatSessions.ts` | ~130 | Medium |
| 4 | `hooks/useChatWebSocket.ts` | ~400 | High |
| 5 | `hooks/useChatMessages.ts` | ~60 | Medium |
| 6 | 6 UI components | ~400 | Medium |

**Total**: ~1030 lines removed, replaced with ~200 lines orchestrator

---

## Rollback Strategy

Each phase can be reverted independently by:
1. Removing the new files
2. Reverting `page.tsx` to the previous phase's state
3. Git makes this trivial: tag each phase completion

```bash
git tag chatpage-refactor-phase-1
git tag chatpage-refactor-phase-2
# etc.
```

---

## Additional Fixes Applied

### Race Condition Fix (Phase 3)
The `getOrCreateSessionId()` function uses a mutex pattern:
```typescript
const sessionCreationLock = useRef<Promise<string | null> | null>(null)

const getOrCreateSessionId = useCallback(async () => {
  if (sessionId) return sessionId
  if (sessionCreationLock.current) return sessionCreationLock.current

  sessionCreationLock.current = (async () => {
    try {
      const session = await createChatSession()
      // ...
    } finally {
      sessionCreationLock.current = null
    }
  })()

  return sessionCreationLock.current
}, [sessionId])
```

### Missing Dependency Fix (Phase 3)
The initialization effect now properly depends on `createNewSession`:
```typescript
useEffect(() => {
  // init logic
}, [createNewSession]) // No more eslint-disable!
```

---

## Testing Checklist

After each phase, verify:

- [ ] No TypeScript errors (`npm run type-check`)
- [ ] No ESLint errors (`npm run lint`)
- [ ] Dev server starts without errors
- [ ] Chat page loads
- [ ] Can send messages
- [ ] Responses stream correctly
- [ ] Session switching works
- [ ] Mobile layout works
- [ ] Activity panel updates correctly
- [ ] No console errors during normal operation
