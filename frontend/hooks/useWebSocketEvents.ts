'use client'

import { useEffect, useState, useRef, useCallback } from 'react'
import { getChatWebSocket, type WebSocketEvent, type ChatWebSocket } from '@/lib/websocket'
import { wsDebug } from '@/lib/debug'

// Connection states for clearer lifecycle management
type ConnectionState = 'disconnected' | 'connecting' | 'connected' | 'reconnecting'

export interface UseWebSocketEventsOptions {
  // Chat lifecycle
  onChatStart: () => void
  onChatComplete: () => void
  onError: (message: string) => void

  // Agent messages
  onAgentStart: (agent: string, agentType: string) => void
  onAgentDelta: (delta: string) => void
  onAgentComplete: (content: string, agent: string, agentType: string, thinking?: string) => void

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

  // Connection
  onDisconnect?: () => void
  onReconnect?: () => void
}

export interface UseWebSocketEventsReturn {
  isConnected: boolean
  connectionState: ConnectionState
  send: (message: string, options?: SendOptions) => void
  reconnect: () => void
}

interface SendOptions {
  swarm?: string
  session_id?: string
  attachments?: Array<{
    type: string
    name: string
    content: string
    mimeType?: string
  }>
}

// Map backend agent names to display names
const getDisplayName = (name: string) => {
  if (name === 'Supreme Orchestrator') return 'Axel'
  return name
}

export function useWebSocketEvents(
  options: UseWebSocketEventsOptions,
  wsInstance?: ChatWebSocket  // Optional for testing
): UseWebSocketEventsReturn {
  const [isConnected, setIsConnected] = useState(false)
  const [connectionState, setConnectionState] = useState<ConnectionState>('disconnected')
  const wsRef = useRef(wsInstance || getChatWebSocket())

  // Store options in ref to avoid stale closures
  const optionsRef = useRef(options)
  useEffect(() => {
    optionsRef.current = options
  }, [options])

  const handleEvent = useCallback((event: WebSocketEvent) => {
    const opts = optionsRef.current

    switch (event.type) {
      case 'chat_start':
        wsDebug.log('chat_start received')
        opts.onChatStart()
        break

      case 'tool_start':
        wsDebug.log('tool_start:', event.tool, event.description)
        opts.onToolStart(
          event.tool || 'Unknown',
          event.description || '',
          event.agentName
        )
        break

      case 'tool_complete':
        wsDebug.log('tool_complete:', event.tool, event.success)
        opts.onToolComplete(
          event.tool || 'Unknown',
          event.success ?? true,
          event.summary,
          event.agentName
        )
        break

      case 'agent_spawn':
        wsDebug.log('agent_spawn:', event.agent, 'parent:', event.parentAgent)
        opts.onAgentSpawn(
          getDisplayName(event.agent || 'Agent'),
          getDisplayName(event.parentAgent || 'COO'),
          (event as { description?: string }).description
        )
        break

      case 'agent_complete_subagent':
        wsDebug.log('agent_complete_subagent:', event.agent)
        opts.onAgentSubComplete(getDisplayName(event.agent || ''))
        break

      case 'agent_start':
        wsDebug.log('agent_start:', event.agent, event.agent_type)
        opts.onAgentStart(
          getDisplayName(event.agent || 'Agent'),
          event.agent_type || 'worker'
        )
        break

      case 'thinking_start':
        wsDebug.log('thinking_start')
        opts.onThinkingStart()
        break

      case 'thinking_delta':
        wsDebug.log('thinking_delta:', event.delta?.length, 'chars')
        opts.onThinkingDelta(event.delta || '')
        break

      case 'thinking_complete':
        wsDebug.log('thinking_complete:', event.thinking?.length, 'chars')
        opts.onThinkingComplete(event.thinking || '')
        break

      case 'agent_delta':
        wsDebug.log('agent_delta:', event.delta?.length, 'chars')
        opts.onAgentDelta(event.delta || '')
        break

      case 'agent_complete':
        wsDebug.log('agent_complete:', event.agent, 'content:', event.content?.substring(0, 100))
        opts.onAgentComplete(
          event.content || '',
          getDisplayName(event.agent || 'Claude'),
          event.agent_type || 'worker',
          event.thinking
        )
        break

      case 'chat_complete':
        wsDebug.log('chat_complete')
        opts.onChatComplete()
        break

      case 'error':
        wsDebug.error('WebSocket error event:', event.message)
        opts.onError(event.message || 'Unknown error')
        break
    }
  }, [])

  const handleDisconnect = useCallback(() => {
    wsDebug.log('Disconnected')
    setIsConnected(false)
    setConnectionState('disconnected')
    optionsRef.current.onDisconnect?.()
  }, [])

  // Connect to WebSocket
  useEffect(() => {
    const ws = wsRef.current

    ws.on('*', handleEvent)
    ws.on('disconnected', handleDisconnect)

    // Connect to WebSocket
    setConnectionState('connecting')
    ws.connect()
      .then(() => {
        setIsConnected(true)
        setConnectionState('connected')
        optionsRef.current.onReconnect?.()
      })
      .catch((e) => {
        wsDebug.error('WebSocket connection failed:', e)
        setIsConnected(false)
        setConnectionState('disconnected')
      })

    return () => {
      ws.off('*', handleEvent)
      ws.off('disconnected', handleDisconnect)
    }
  }, [handleEvent, handleDisconnect])

  const send = useCallback((message: string, sendOptions?: SendOptions) => {
    try {
      wsRef.current.send(message, sendOptions)
    } catch (e) {
      wsDebug.error('Failed to send message:', e)
      throw e
    }
  }, [])

  const reconnect = useCallback(() => {
    const ws = wsRef.current
    ws.resetReconnectAttempts()
    setConnectionState('reconnecting')
    ws.connect()
      .then(() => {
        setIsConnected(true)
        setConnectionState('connected')
        optionsRef.current.onReconnect?.()
      })
      .catch((e) => {
        wsDebug.error('Reconnection failed:', e)
        setIsConnected(false)
        setConnectionState('disconnected')
      })
  }, [])

  return {
    isConnected,
    connectionState,
    send,
    reconnect,
  }
}
