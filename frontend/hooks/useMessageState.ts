'use client'

import { useState, useRef, useEffect, useCallback } from 'react'
import { type Message } from './useChatSessions'

// Streaming message type (extends Message with required agent info)
export interface StreamingMessage extends Message {
  type: 'agent'
  agent: string
  agentType: string
  status: 'thinking' | 'complete'
}

export interface UseMessageStateOptions {
  streamingMessage: StreamingMessage | null
  setStreamingMessage: (msg: StreamingMessage | null) => void
  isStreaming: boolean
}

export interface UseMessageStateReturn {
  messages: Message[]

  // User messages
  addUserMessage: (content: string, attachments?: Message['attachments']) => string  // returns ID

  // Agent messages
  startAgentMessage: (agent: string, agentType: string) => string  // returns ID
  updateAgentContent: (delta: string) => void
  startThinking: () => void
  updateThinking: (delta: string) => void
  completeThinking: (thinking: string) => void
  completeAgentMessage: (content: string, agent: string, agentType: string, thinking?: string) => void

  // Bulk operations
  setMessages: React.Dispatch<React.SetStateAction<Message[]>>
  markAllComplete: () => void

  // Internal - for session loading
  loadMessages: (messages: Message[]) => void
}

export function useMessageState(options: UseMessageStateOptions): UseMessageStateReturn {
  const { streamingMessage, setStreamingMessage, isStreaming } = options

  const [messages, setMessages] = useState<Message[]>([])

  // Track if we've restored streaming message from context (navigation back scenario)
  const hasRestoredRef = useRef(false)

  // Restore streaming message from context when navigating back
  // This runs once on mount to restore any in-progress streaming
  useEffect(() => {
    if (hasRestoredRef.current) return
    if (isStreaming && streamingMessage) {
      hasRestoredRef.current = true
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

  // Add a user message
  const addUserMessage = useCallback((content: string, attachments?: Message['attachments']): string => {
    const id = crypto.randomUUID()
    setMessages((prev) => [
      ...prev,
      {
        id,
        type: 'user',
        content,
        timestamp: new Date(),
        attachments: attachments && attachments.length > 0 ? attachments : undefined,
      },
    ])
    return id
  }, [])

  // Start a new agent message (thinking state)
  const startAgentMessage = useCallback((agent: string, agentType: string): string => {
    const id = crypto.randomUUID()
    const newStreamingMsg: StreamingMessage = {
      id,
      type: 'agent',
      content: '',
      agent,
      agentType,
      status: 'thinking',
      timestamp: new Date(),
    }

    // First, mark any existing thinking messages as complete to prevent stacking
    setMessages((prev) => {
      const cleaned = prev.map((m) =>
        m.type === 'agent' && m.status === 'thinking'
          ? { ...m, status: 'complete' as const }
          : m
      )
      return [...cleaned, newStreamingMsg]
    })

    // Store in context for persistence across navigation
    setStreamingMessage(newStreamingMsg)

    return id
  }, [setStreamingMessage])

  // Update content of the current thinking agent message
  const updateAgentContent = useCallback((delta: string) => {
    setMessages((prev) => {
      // Find ANY message with status=thinking, not just the last one
      const thinkingIdx = prev.findIndex(
        (m) => m.type === 'agent' && m.status === 'thinking'
      )
      if (thinkingIdx !== -1) {
        // Update the thinking message with new content
        const updated = [...prev]
        const msg = updated[thinkingIdx]
        updated[thinkingIdx] = {
          ...msg,
          content: msg.content + delta,
        }
        // Update streaming message in context
        setStreamingMessage({
          ...updated[thinkingIdx],
          type: 'agent' as const,
          agent: msg.agent || 'Agent',
          agentType: msg.agentType || 'worker',
          status: 'thinking' as const,
        })
        return updated
      }
      return prev
    })
  }, [setStreamingMessage])

  // Start thinking mode on current agent message
  const startThinking = useCallback(() => {
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
  }, [setStreamingMessage])

  // Update thinking content
  const updateThinking = useCallback((delta: string) => {
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
  }, [setStreamingMessage])

  // Complete thinking
  const completeThinking = useCallback((thinking: string) => {
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
  }, [setStreamingMessage])

  // Complete the agent message
  const completeAgentMessage = useCallback((content: string, agent: string, agentType: string, thinking?: string) => {
    const completeContent = content
    const completeAgent = agent
    const completeThinkingContent = thinking || ''

    // DEBUG: Log what we received
    console.log('[DEBUG agent_complete] Received:', {
      contentLength: completeContent.length,
      contentPreview: completeContent.substring(0, 200),
      agent: completeAgent,
      thinkingLength: completeThinkingContent.length,
    })

    // Always update the existing thinking message (regardless of agentType)
    setMessages((prev) => {
      // Find ANY existing thinking message (not filtered by agentType)
      const thinkingIdx = prev.findIndex(
        (m) => m.type === 'agent' && m.status === 'thinking'
      )

      // DEBUG: Log current message state
      console.log('[DEBUG agent_complete] Messages state:', {
        totalMessages: prev.length,
        thinkingIdx,
        thinkingMessage: thinkingIdx !== -1 ? {
          content: prev[thinkingIdx].content.substring(0, 100),
          status: prev[thinkingIdx].status,
        } : null,
      })

      if (thinkingIdx !== -1) {
        // Update existing message to complete, preserving thinking
        const updated = [...prev]
        const finalContent = completeContent || updated[thinkingIdx].content
        const finalThinking = completeThinkingContent || updated[thinkingIdx].thinking

        // DEBUG: Log what we're setting
        console.log('[DEBUG agent_complete] Setting finalContent:', {
          fromEvent: completeContent.length,
          fromExisting: updated[thinkingIdx].content.length,
          finalLength: finalContent.length,
          finalPreview: finalContent.substring(0, 200),
        })

        updated[thinkingIdx] = {
          ...updated[thinkingIdx],
          content: finalContent,
          thinking: finalThinking,
          isThinking: false,
          status: 'complete',
          agent: agent,
          agentType: agentType,
        }
        console.log('[agent_complete] Updated thinking message to complete:', finalContent.substring(0, 100))
        return updated
      }

      // No thinking message found, add new complete message
      console.log('[agent_complete] No thinking message found, adding new message:', completeContent.substring(0, 100))
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
          thinking: completeThinkingContent,
        },
      ]
    })

    // Clear streaming message
    setStreamingMessage(null)
  }, [setStreamingMessage])

  // Mark all messages as complete
  const markAllComplete = useCallback(() => {
    setMessages((prev) =>
      prev.map((m) =>
        m.type === 'agent' && m.status === 'thinking'
          ? { ...m, status: 'complete' as const, isThinking: false }
          : m
      )
    )
  }, [])

  // Load messages (for session loading)
  const loadMessages = useCallback((newMessages: Message[]) => {
    setMessages(newMessages)
    // Reset restoration flag when loading new messages
    hasRestoredRef.current = false
  }, [])

  return {
    messages,
    addUserMessage,
    startAgentMessage,
    updateAgentContent,
    startThinking,
    updateThinking,
    completeThinking,
    completeAgentMessage,
    setMessages,
    markAllComplete,
    loadMessages,
  }
}
