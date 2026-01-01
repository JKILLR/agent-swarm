'use client'

import { useEffect, useState, useRef, useCallback } from 'react'
import { getChatWebSocket, type WebSocketEvent } from '@/lib/websocket'
import ChatInput, { type Attachment } from '@/components/ChatInput'
import ChatMessage from '@/components/ChatMessage'
import AgentResponse from '@/components/AgentResponse'
import { Bot, WifiOff } from 'lucide-react'

interface Message {
  id: string
  type: 'user' | 'agent'
  content: string
  agent?: string
  agentType?: string
  status?: 'thinking' | 'complete'
  timestamp: Date
  attachments?: Attachment[]
}

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([])
  const [isConnected, setIsConnected] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const wsRef = useRef(getChatWebSocket())

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  // Connect to WebSocket
  useEffect(() => {
    const ws = wsRef.current

    const handleEvent = (event: WebSocketEvent) => {
      switch (event.type) {
        case 'chat_start':
          setIsLoading(true)
          break

        case 'agent_start':
          // Add thinking indicator for agent
          setMessages((prev) => [
            ...prev,
            {
              id: `agent-${Date.now()}`,
              type: 'agent',
              content: '',
              agent: event.agent || 'Agent',
              agentType: event.agent_type || 'worker',
              status: 'thinking',
              timestamp: new Date(),
            },
          ])
          break

        case 'agent_complete':
          // Update or add agent response
          setMessages((prev) => {
            // Remove any thinking indicators for this agent type
            const filtered = prev.filter(
              (m) => !(m.type === 'agent' && m.status === 'thinking' && m.agentType === event.agent_type)
            )
            return [
              ...filtered,
              {
                id: `agent-${Date.now()}-${event.agent}`,
                type: 'agent',
                content: event.content || '',
                agent: event.agent || 'Agent',
                agentType: event.agent_type || 'worker',
                status: 'complete',
                timestamp: new Date(),
              },
            ]
          })
          break

        case 'chat_complete':
          setIsLoading(false)
          break

        case 'error':
          setIsLoading(false)
          console.error('Chat error:', event.message)
          break
      }
    }

    ws.on('*', handleEvent)

    ws.connect()
      .then(() => setIsConnected(true))
      .catch(() => setIsConnected(false))

    ws.on('disconnected', () => setIsConnected(false))

    return () => {
      ws.off('*', handleEvent)
    }
  }, [])

  const handleSend = useCallback((content: string, attachments: Attachment[]) => {
    // Add user message with attachments
    setMessages((prev) => [
      ...prev,
      {
        id: `user-${Date.now()}`,
        type: 'user',
        content,
        timestamp: new Date(),
        attachments: attachments.length > 0 ? attachments : undefined,
      },
    ])

    // Build message with attachment context for the AI
    let fullMessage = content

    if (attachments.length > 0) {
      const attachmentDescriptions = attachments.map((a) => {
        if (a.type === 'text') {
          return `\n\n--- Attached Text: ${a.name} ---\n${a.content}\n--- End Attachment ---`
        } else if (a.type === 'image') {
          return `\n\n[Attached Image: ${a.name}]`
        } else {
          return `\n\n[Attached Document: ${a.name}]`
        }
      })
      fullMessage += attachmentDescriptions.join('')
    }

    // Send via WebSocket
    try {
      wsRef.current.send(fullMessage)
    } catch (e) {
      console.error('Failed to send message:', e)
      setIsLoading(false)
    }
  }, [])

  // Simplified send for suggestion buttons
  const handleQuickSend = useCallback((content: string) => {
    handleSend(content, [])
  }, [handleSend])

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between px-6 py-4 border-b border-zinc-800">
        <div className="flex items-center gap-3">
          <Bot className="w-6 h-6 text-blue-500" />
          <div>
            <h1 className="font-semibold text-white">Chat</h1>
            <p className="text-xs text-zinc-500">Talk to the Supreme Orchestrator</p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {isConnected ? (
            <span className="flex items-center gap-1 text-xs text-green-400">
              <span className="w-2 h-2 rounded-full bg-green-400" />
              Connected
            </span>
          ) : (
            <span className="flex items-center gap-1 text-xs text-red-400">
              <WifiOff className="w-3 h-3" />
              Disconnected
            </span>
          )}
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-auto p-6 space-y-6">
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-center">
            <Bot className="w-16 h-16 text-zinc-700 mb-4" />
            <h2 className="text-xl font-semibold text-zinc-400 mb-2">
              Start a conversation
            </h2>
            <p className="text-sm text-zinc-500 max-w-md">
              Ask about your swarms, request research, or coordinate tasks across agents.
              The Supreme Orchestrator will route your request to the appropriate team.
            </p>
            <div className="mt-6 space-y-2 text-left">
              <p className="text-xs text-zinc-600">Try asking:</p>
              <div className="space-y-1">
                <button
                  onClick={() => handleQuickSend('What is the current status of the ASA project?')}
                  className="block text-sm text-zinc-400 hover:text-blue-400 transition-colors"
                >
                  → What is the current status of the ASA project?
                </button>
                <button
                  onClick={() => handleQuickSend('Research sparse attention implementations for ASA')}
                  className="block text-sm text-zinc-400 hover:text-blue-400 transition-colors"
                >
                  → Research sparse attention implementations for ASA
                </button>
              </div>
            </div>
          </div>
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
              />
            )
          ))
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="p-4 border-t border-zinc-800">
        <ChatInput
          onSend={handleSend}
          disabled={isLoading || !isConnected}
          placeholder={
            !isConnected
              ? 'Connecting...'
              : isLoading
              ? 'Waiting for response...'
              : 'Type a message...'
          }
        />
      </div>
    </div>
  )
}
