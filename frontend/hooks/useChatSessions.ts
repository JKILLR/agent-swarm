'use client'

import { useState, useRef, useCallback, useEffect } from 'react'
import {
  getChatSessions,
  getChatSession,
  createChatSession,
  deleteChatSession,
  addChatMessage,
  type ChatSessionSummary,
} from '@/lib/api'
import { type Attachment } from '@/components/ChatInput'

// Message type matching page.tsx
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

interface UseChatSessionsOptions {
  onSessionLoad?: (messages: Message[]) => void
  onSessionCreate?: (sessionId: string) => void
  onMessagesChange?: (messages: Message[]) => void
}

interface UseChatSessionsReturn {
  sessions: ChatSessionSummary[]
  sessionId: string | null
  sessionIdRef: React.RefObject<string | null>
  isInitialized: boolean
  loadSessions: () => Promise<void>
  loadSession: (id: string) => Promise<void>
  createSession: () => Promise<void>
  deleteSession: (id: string, e: React.MouseEvent) => Promise<void>
  saveMessage: (role: 'user' | 'assistant', content: string, agent?: string, thinking?: string) => Promise<void>
  ensureSession: () => Promise<string>
  setSessionId: React.Dispatch<React.SetStateAction<string | null>>
}

// Map backend agent names to display names
const getDisplayName = (name: string) => {
  if (name === 'Supreme Orchestrator') return 'Axel'
  return name
}

// Improved key generation for message deduplication
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

// Convert API messages to local format
const convertMessages = (apiMessages: Array<{
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: string
  agent?: string
  thinking?: string
}>): Message[] => {
  return apiMessages.map((m) => ({
    id: m.id,
    type: m.role === 'user' ? 'user' as const : 'agent' as const,
    content: m.content,
    agent: getDisplayName(m.agent || 'Claude'),
    agentType: 'assistant',
    status: 'complete' as const,
    timestamp: new Date(m.timestamp),
    thinking: m.thinking,
  }))
}

export function useChatSessions(options: UseChatSessionsOptions = {}): UseChatSessionsReturn {
  const { onSessionLoad, onSessionCreate, onMessagesChange } = options

  // State
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [sessions, setSessions] = useState<ChatSessionSummary[]>([])
  const [isInitialized, setIsInitialized] = useState(false)

  // Refs for race condition protection
  const sessionIdRef = useRef<string | null>(null)
  const sessionCreationLock = useRef<Promise<string> | null>(null)
  const loadSequenceRef = useRef(0)
  const initRef = useRef(false)

  // Sync sessionIdRef with sessionId state
  useEffect(() => {
    sessionIdRef.current = sessionId
  }, [sessionId])

  // Load sessions list
  const loadSessions = useCallback(async () => {
    try {
      const sessionList = await getChatSessions()
      setSessions(sessionList)
    } catch (e) {
      console.error('Failed to load sessions:', e)
    }
  }, [])

  // Load a specific session with rapid switching protection
  const loadSession = useCallback(async (id: string) => {
    const loadSequence = ++loadSequenceRef.current

    try {
      const session = await getChatSession(id)

      // Check if this load is still current (user may have switched sessions)
      if (loadSequence !== loadSequenceRef.current) {
        return // User switched to different session
      }

      setSessionId(id)
      sessionIdRef.current = id

      // Convert and dedupe messages
      const loadedMessages = dedupeMessages(convertMessages(session.messages))

      // Call onSessionLoad callback with messages
      onSessionLoad?.(loadedMessages)
    } catch (e) {
      console.error('Failed to load session:', e)
    }
  }, [onSessionLoad])

  // Create new session
  const createSession = useCallback(async () => {
    try {
      const session = await createChatSession()
      setSessionId(session.id)
      sessionIdRef.current = session.id
      onSessionCreate?.(session.id)
      onMessagesChange?.([]) // Clear messages for new session
      await loadSessions()
    } catch (e) {
      console.error('Failed to create session:', e)
    }
  }, [loadSessions, onSessionCreate, onMessagesChange])

  // Delete a session
  const deleteSession = useCallback(async (id: string, e: React.MouseEvent) => {
    e.stopPropagation()
    try {
      await deleteChatSession(id)
      if (sessionIdRef.current === id) {
        setSessionId(null)
        sessionIdRef.current = null
        onMessagesChange?.([]) // Clear messages when deleting current session
      }
      await loadSessions()
    } catch (e) {
      console.error('Failed to delete session:', e)
    }
  }, [loadSessions, onMessagesChange])

  // Save message to backend
  const saveMessage = useCallback(async (role: 'user' | 'assistant', content: string, agent?: string, thinking?: string) => {
    const currentId = sessionIdRef.current
    if (!currentId) return
    try {
      await addChatMessage(currentId, role, content, agent, thinking)
      await loadSessions() // Refresh list to update message counts
    } catch (e) {
      console.error('Failed to save message:', e)
    }
  }, [loadSessions])

  // Race-condition protected session creation
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
        onSessionCreate?.(session.id)
        await loadSessions()
        return session.id
      } finally {
        sessionCreationLock.current = null
      }
    })()

    return sessionCreationLock.current
  }, [loadSessions, onSessionCreate])  // Note: NO sessionId dependency - we use ref

  // Initialize: load sessions and create/load initial session
  useEffect(() => {
    if (initRef.current) return // Already initialized
    initRef.current = true

    const init = async () => {
      const sessionList = await getChatSessions()
      setSessions(sessionList)
      if (sessionList.length > 0) {
        // Load most recent session
        try {
          const session = await getChatSession(sessionList[0].id)
          setSessionId(sessionList[0].id)
          sessionIdRef.current = sessionList[0].id

          // Convert and dedupe messages
          const loadedMessages = dedupeMessages(convertMessages(session.messages))
          onSessionLoad?.(loadedMessages)
        } catch (e) {
          console.error('Failed to load session:', e)
        }
      } else {
        // Create first session
        await createSession()
      }
      setIsInitialized(true)
    }
    init()
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  return {
    sessions,
    sessionId,
    sessionIdRef,
    isInitialized,
    loadSessions,
    loadSession,
    createSession,
    deleteSession,
    saveMessage,
    ensureSession,
    setSessionId,
  }
}
