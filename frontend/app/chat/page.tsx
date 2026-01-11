'use client'

import { useEffect, useState, useRef, useCallback } from 'react'
import ChatInput, { type Attachment } from '@/components/ChatInput'
import ChatMessage from '@/components/ChatMessage'
import AgentResponse from '@/components/AgentResponse'
import { useAgentActivity } from '@/lib/AgentActivityContext'
import { useMobileLayout } from '@/components/MobileLayout'
import MobileHistorySheet from '@/components/chat/MobileHistorySheet'
import MobileActivitySheet from '@/components/chat/MobileActivitySheet'
import { useChatSessions, type Message } from '@/hooks/useChatSessions'
import { useWebSocketEvents } from '@/hooks/useWebSocketEvents'
import { addChatMessage } from '@/lib/api'
import SessionSidebar from '@/components/chat/SessionSidebar'
import ChatHeader from '@/components/chat/ChatHeader'
import EmptyState from '@/components/chat/EmptyState'

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [showSidebar, setShowSidebar] = useState(true)

  // Session management via hook
  const {
    sessions,
    sessionId,
    sessionIdRef,
    loadSessions,
    loadSession,
    createSession: createNewSession,
    deleteSession: handleDeleteSession,
    ensureSession,
  } = useChatSessions({
    onSessionLoad: (loadedMessages) => {
      setMessages(loadedMessages)
    },
    onMessagesChange: (newMessages) => {
      setMessages(newMessages)
    },
  })
  const [showMobileHistory, setShowMobileHistory] = useState(false)
  // Use global context for activity state (persists across navigation)
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
  const [showMobileActivity, setShowMobileActivity] = useState(false)
  const { isMobile } = useMobileLayout()
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const messagesContainerRef = useRef<HTMLDivElement>(null)
  const pendingMessageRef = useRef<{ content: string; agent?: string; thinking?: string } | null>(null)
  const loadingTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const isInitialLoadRef = useRef(true) // Track if this is initial load

  // Loading timeout constant (5 minutes - long tasks are common with agents)
  const LOADING_TIMEOUT_MS = 5 * 60 * 1000

  // Map backend agent names to display names
  const getDisplayName = (name: string) => {
    if (name === 'Supreme Orchestrator') return 'Axel'
    return name
  }

  // WebSocket event handling via hook
  const { isConnected, send } = useWebSocketEvents({
    // Chat lifecycle
    onChatStart: () => {
      setIsLoading(true)
      setIsStreaming(true)
      // Clear activities for new chat
      setAgentActivities([])
      setToolActivities([])
      // Add COO as the first active agent
      setAgentActivities([{
        id: crypto.randomUUID(),
        name: 'Axel',
        status: 'thinking',
        startTime: new Date(),
      }])
      // Mark any stale thinking messages as complete before new response cycle
      setMessages((prev) => prev.map((m) =>
        m.type === 'agent' && m.status === 'thinking'
          ? { ...m, status: 'complete' as const }
          : m
      ))
    },

    onChatComplete: () => {
      setIsLoading(false)
      setIsStreaming(false)
      setStreamingMessage(null)
      // Mark all agents as complete
      setAgentActivities((prev) =>
        prev.map((a) => ({
          ...a,
          status: 'complete' as const,
          endTime: a.endTime || new Date(),
        }))
      )
      // Clear pending message ref
      pendingMessageRef.current = null
    },

    onError: (message) => {
      setIsLoading(false)
      // Clear pending message on error to prevent leaks
      pendingMessageRef.current = null
      // Mark agents as error
      setAgentActivities((prev) =>
        prev.map((a) =>
          a.status !== 'complete' ? { ...a, status: 'error' as const, endTime: new Date() } : a
        )
      )
      console.error('Chat error:', message)
    },

    // Agent messages
    onAgentStart: (agent, agentType) => {
      // Always create a fresh message for new agent_start
      // First, ensure any existing thinking messages are marked complete
      setMessages((prev) => {
        // Mark any existing thinking messages as complete (they're stale)
        const cleaned = prev.map((m) =>
          m.type === 'agent' && m.status === 'thinking'
            ? { ...m, status: 'complete' as const }
            : m
        )

        // Create new thinking message
        const newStreamingMsg = {
          id: crypto.randomUUID(),
          type: 'agent' as const,
          content: '',
          agent: agent,
          agentType: agentType,
          status: 'thinking' as const,
          timestamp: new Date(),
        }
        // Store in context for persistence across navigation
        setStreamingMessage(newStreamingMsg)
        return [...cleaned, newStreamingMsg]
      })
    },

    onAgentDelta: (delta) => {
      setMessages((prev) => {
        // Find the LAST message with status=thinking (most recent response)
        const thinkingIdx = prev.map((m, i) => ({ m, i }))
          .filter(({ m }) => m.type === 'agent' && m.status === 'thinking')
          .pop()?.i ?? -1

        if (thinkingIdx !== -1) {
          // Update the thinking message with new content
          const updated = [...prev]
          const msg = updated[thinkingIdx]
          updated[thinkingIdx] = {
            ...msg,
            content: msg.content + delta,
          }
          // Cast to StreamingMessage
          setStreamingMessage({
            ...updated[thinkingIdx],
            type: 'agent' as const,
            agent: msg.agent || 'Agent',
            agentType: msg.agentType || 'worker',
            status: 'thinking' as const,
          })
          return updated
        }

        // No thinking message found - create one (delta arrived before agent_start)
        const newStreamingMsg = {
          id: crypto.randomUUID(),
          type: 'agent' as const,
          content: delta,
          agent: 'Axel',
          agentType: 'orchestrator',
          status: 'thinking' as const,
          timestamp: new Date(),
        }
        setStreamingMessage(newStreamingMsg)
        return [...prev, newStreamingMsg]
      })
    },

    onAgentComplete: (content, agent, agentType, thinking) => {
      // Track message for saving
      const completeContent = content
      const completeAgent = agent
      const completeThinking = thinking || ''

      console.log('[agent_complete] Received, content length:', completeContent.length)

      // Update the existing thinking message to complete status
      // IMPORTANT: Use the streamed content if available (it's more accurate)
      setMessages((prev) => {
        // Find the LAST existing thinking message (most recent response)
        const thinkingIdx = prev.map((m, i) => ({ m, i }))
          .filter(({ m }) => m.type === 'agent' && m.status === 'thinking')
          .pop()?.i ?? -1

        if (thinkingIdx !== -1) {
          // Update existing message to complete
          const updated = [...prev]
          const existingContent = updated[thinkingIdx].content

          // Prefer existing (streamed) content if it exists, otherwise use event content
          // This prevents duplication when content was already streamed via deltas
          const finalContent = existingContent || completeContent
          const finalThinking = completeThinking || updated[thinkingIdx].thinking

          updated[thinkingIdx] = {
            ...updated[thinkingIdx],
            content: finalContent,
            thinking: finalThinking,
            isThinking: false,
            status: 'complete',
            agent: agent,
            agentType: agentType,
          }

          // Update pending message ref for chat history saving
          pendingMessageRef.current = {
            content: finalContent,
            agent: completeAgent,
            thinking: finalThinking,
          }

          console.log('[agent_complete] Updated existing message to complete')
          return updated
        }

        // No thinking message found - this shouldn't normally happen
        // Only add a new message if we have content and there's no duplicate
        if (completeContent) {
          // Check if this content already exists in a recent message to prevent duplicates
          const isDuplicate = prev.some(
            (m) => m.type === 'agent' && m.content === completeContent
          )

          if (isDuplicate) {
            console.log('[agent_complete] Skipping duplicate message')
            return prev
          }

          console.log('[agent_complete] Adding new complete message')
          pendingMessageRef.current = {
            content: completeContent,
            agent: completeAgent,
            thinking: completeThinking,
          }

          return [
            ...prev,
            {
              id: crypto.randomUUID(),
              type: 'agent',
              content: completeContent,
              agent: completeAgent,
              agentType: agentType,
              status: 'complete',
              timestamp: new Date(),
              thinking: completeThinking,
            },
          ]
        }

        return prev
      })

      // Also set isLoading=false here as a fallback
      setIsLoading(false)
      setIsStreaming(false)
      setStreamingMessage(null)
    },

    // Thinking
    onThinkingStart: () => {
      setMessages((prev) => {
        const lastMessage = prev[prev.length - 1]
        if (lastMessage && lastMessage.type === 'agent' && lastMessage.status === 'thinking') {
          const updated = { ...lastMessage, isThinking: true, thinking: '' }
          setStreamingMessage({
            ...updated,
            type: 'agent' as const,
            agent: updated.agent || 'Agent',
            agentType: updated.agentType || 'worker',
            status: 'thinking' as const,
          })
          return [...prev.slice(0, -1), updated]
        }
        return prev
      })
    },

    onThinkingDelta: (delta) => {
      setMessages((prev) => {
        const lastMessage = prev[prev.length - 1]
        if (lastMessage && lastMessage.type === 'agent' && lastMessage.isThinking) {
          const updated = { ...lastMessage, thinking: (lastMessage.thinking || '') + delta }
          setStreamingMessage({
            ...updated,
            type: 'agent' as const,
            agent: updated.agent || 'Agent',
            agentType: updated.agentType || 'worker',
            status: 'thinking' as const,
          })
          return [...prev.slice(0, -1), updated]
        }
        return prev
      })
    },

    onThinkingComplete: (thinking) => {
      setMessages((prev) => {
        const lastMessage = prev[prev.length - 1]
        if (lastMessage && lastMessage.type === 'agent' && lastMessage.isThinking) {
          const updated = { ...lastMessage, isThinking: false, thinking: thinking || lastMessage.thinking }
          setStreamingMessage({
            ...updated,
            type: 'agent' as const,
            agent: updated.agent || 'Agent',
            agentType: updated.agentType || 'worker',
            status: 'thinking' as const,
          })
          return [...prev.slice(0, -1), updated]
        }
        return prev
      })
    },

    // Tools (for panel activities)
    onToolStart: (tool, description, agentName) => {
      // Add new tool activity with agent attribution
      setToolActivities((prev) => [
        ...prev,
        {
          id: crypto.randomUUID(),
          tool: tool,
          description: description,
          status: 'running',
          timestamp: new Date(),
          agentName: agentName,
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
    },

    onToolComplete: (tool, success, summary, agentName) => {
      // Update tool activity status
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
          summary: summary,
          endTime: new Date(),
          agentName: agentName || updated[actualIdx].agentName,
        }
        return updated
      })
      // Check if COO should go back to working (from delegating) when no subagents are working
      setAgentActivities((prev) => {
        const cooIdx = prev.findIndex(a => a.name.includes('COO'))
        if (cooIdx !== -1 && prev[cooIdx].status === 'delegating') {
          // Check if there are still active subagents
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
    },

    // Agent hierarchy (for panel activities)
    onAgentSpawn: (agent, parentAgent, description) => {
      setAgentActivities((prev) => {
        // Check if this agent already exists
        const existingIdx = prev.findIndex(a => a.name === agent)

        // Mark parent agent as delegating
        let updated = prev.map(a =>
          a.name === parentAgent || (parentAgent === 'COO' && a.name.includes('COO'))
            ? { ...a, status: 'delegating' as const }
            : a
        )

        if (existingIdx !== -1) {
          // Update existing agent entry instead of adding duplicate
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

        // Add new agent only if it doesn't exist
        return [
          ...updated,
          {
            id: crypto.randomUUID(),
            name: agent,
            status: 'working' as const,
            description: description?.substring(0, 100),
            startTime: new Date(),
          },
        ]
      })
    },

    onAgentSubComplete: (agent) => {
      setAgentActivities((prev) => {
        return prev.map(a => {
          if (a.name === agent) {
            return { ...a, status: 'complete' as const, endTime: new Date() }
          }
          // If this was a delegating agent's child completing, restore working status
          if (a.status === 'delegating') {
            // Check if there are still other active children
            const otherActive = prev.some(other =>
              other.name !== agent &&
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
    },

    // Connection
    onDisconnect: () => {
      setIsLoading(false)
      setIsStreaming(false)
      setStreamingMessage(null)
      // Clear pending message on disconnect to prevent leaks
      pendingMessageRef.current = null
    },
  })

  // Restore streaming message from context when navigating back
  // This runs once on mount to restore any in-progress streaming
  const hasRestoredRef = useRef(false)
  useEffect(() => {
    if (hasRestoredRef.current) return
    if (isStreaming && streamingMessage) {
      hasRestoredRef.current = true
      setIsLoading(true)
      // Add the streaming message to messages if not already there
      setMessages((prev) => {
        // Check if this message is already in the list
        const exists = prev.some(m => m.id === streamingMessage.id)
        if (exists) return prev
        return [...prev, streamingMessage]
      })
    }
  }, [isStreaming, streamingMessage])

  // Keep messages in sync with streaming message updates from context
  useEffect(() => {
    if (!streamingMessage || !hasRestoredRef.current) return
    setMessages((prev) => {
      const idx = prev.findIndex(m => m.id === streamingMessage.id)
      if (idx === -1) return prev
      // Only update if content changed
      if (prev[idx].content === streamingMessage.content &&
          prev[idx].thinking === streamingMessage.thinking) {
        return prev
      }
      const updated = [...prev]
      updated[idx] = streamingMessage
      return updated
    })
  }, [streamingMessage])

  const scrollToBottom = useCallback((instant = false) => {
    if (instant && messagesContainerRef.current) {
      // Instant scroll for initial load
      messagesContainerRef.current.scrollTop = messagesContainerRef.current.scrollHeight
    } else {
      // Smooth scroll for new messages
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
    }
  }, [])

  useEffect(() => {
    if (messages.length > 0) {
      // Use instant scroll on initial load, smooth scroll for subsequent updates
      scrollToBottom(isInitialLoadRef.current)
      isInitialLoadRef.current = false
    }
  }, [messages, scrollToBottom])

  // Loading timeout effect - reset loading state if stuck
  useEffect(() => {
    if (isLoading) {
      // Start timeout when loading begins
      loadingTimeoutRef.current = setTimeout(() => {
        console.warn('Loading timeout reached, resetting state')
        setIsLoading(false)
        // Mark all non-complete agents as error
        setAgentActivities((prev) =>
          prev.map((a) =>
            a.status !== 'complete'
              ? { ...a, status: 'error' as const, endTime: new Date() }
              : a
          )
        )
      }, LOADING_TIMEOUT_MS)
    } else {
      // Clear timeout when loading ends
      if (loadingTimeoutRef.current) {
        clearTimeout(loadingTimeoutRef.current)
        loadingTimeoutRef.current = null
      }
    }

    // Cleanup on unmount
    return () => {
      if (loadingTimeoutRef.current) {
        clearTimeout(loadingTimeoutRef.current)
        loadingTimeoutRef.current = null
      }
    }
  }, [isLoading, setAgentActivities, LOADING_TIMEOUT_MS])

  const handleSend = useCallback(async (content: string, attachments: Attachment[]) => {
    // Clear old completed activities when starting a new message
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

    // Add user message with attachments
    setMessages((prev) => [
      ...prev,
      {
        id: crypto.randomUUID(),
        type: 'user',
        content,
        timestamp: new Date(),
        attachments: attachments.length > 0 ? attachments : undefined,
      },
    ])

    // Save user message to backend (use currentSessionId directly)
    try {
      await addChatMessage(currentSessionId, 'user', content)
      await loadSessions()
    } catch (e) {
      console.error('Failed to save message:', e)
    }

    // Build message with text attachments inline
    let fullMessage = content

    // Prepare image attachments to send to backend
    const imageAttachments = attachments.filter(a => a.type === 'image').map(a => ({
      type: a.type,
      name: a.name,
      content: a.content,
      mimeType: a.mimeType,
    }))

    // Add text attachments inline in the message
    if (attachments.length > 0) {
      const descriptions = attachments.map((a) => {
        if (a.type === 'text') {
          return `\n\n--- Attached Text: ${a.name} ---\n${a.content}\n--- End Attachment ---`
        } else if (a.type === 'image') {
          return `\n\n[Image attached: ${a.name}]`
        } else {
          return `\n\n[Document: ${a.name}]`
        }
      })
      fullMessage += descriptions.join('')
    }

    // Send via WebSocket with session context and image data
    try {
      send(fullMessage, {
        session_id: currentSessionId,
        attachments: imageAttachments.length > 0 ? imageAttachments : undefined,
      })
    } catch (e) {
      console.error('Failed to send message:', e)
      setIsLoading(false)
    }
  }, [ensureSession, loadSessions, send, setAgentActivities, setToolActivities])

  // Simplified send for suggestion buttons
  const handleQuickSend = useCallback((content: string) => {
    handleSend(content, [])
  }, [handleSend])

  // Activity state for mobile panel
  const hasActivity = agentActivities.length > 0 || toolActivities.length > 0
  const isActivityProcessing = agentActivities.some(
    (a) => a.status !== 'complete' && a.status !== 'error'
  )

  return (
    <div className="flex h-full relative bg-[#0d0d0d] overflow-hidden">
      {/* Session Sidebar - Hidden on mobile, use bottom sheet instead */}
      <SessionSidebar
        sessions={sessions}
        currentSessionId={sessionId}
        showSidebar={showSidebar}
        onSessionClick={loadSession}
        onNewSession={createNewSession}
        onDeleteSession={handleDeleteSession}
      />

      {/* Mobile History Bottom Sheet */}
      {isMobile && (
        <MobileHistorySheet
          isOpen={showMobileHistory}
          onClose={() => setShowMobileHistory(false)}
          sessions={sessions}
          currentSessionId={sessionId}
          onSessionClick={loadSession}
          onNewSession={createNewSession}
          onDeleteSession={handleDeleteSession}
        />
      )}

      {/* Mobile Activity Bottom Sheet */}
      {isMobile && (
        <MobileActivitySheet
          isOpen={showMobileActivity}
          onClose={() => setShowMobileActivity(false)}
          agents={agentActivities}
          tools={toolActivities}
          isProcessing={isActivityProcessing}
          onClear={clearPanelActivities}
        />
      )}

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col min-w-0 bg-[#0d0d0d]">
        {/* Chat area uses base #0d0d0d */}
        {/* Header */}
        <ChatHeader
          isConnected={isConnected}
          isMobile={isMobile}
          showSidebar={showSidebar}
          hasActivity={hasActivity}
          isActivityProcessing={isActivityProcessing}
          onToggleSidebar={() => setShowSidebar(!showSidebar)}
          onShowHistory={() => setShowMobileHistory(true)}
          onShowActivity={() => setShowMobileActivity(true)}
        />

        {/* Messages */}
        <div ref={messagesContainerRef} className="flex-1 overflow-auto p-3 md:p-6 space-y-4 md:space-y-6 relative">
          {messages.length === 0 ? (
            <EmptyState onQuickSend={handleQuickSend} />
          ) : (
            messages.map((message) => (
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
            ))
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Input - Full width on mobile with safe area padding */}
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
