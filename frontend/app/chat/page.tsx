'use client'

import { useEffect, useState, useRef, useCallback } from 'react'
import { getChatWebSocket, type WebSocketEvent } from '@/lib/websocket'
import {
  getChatSessions,
  getChatSession,
  createChatSession,
  deleteChatSession,
  addChatMessage,
  type ChatSessionSummary,
} from '@/lib/api'
import ChatInput, { type Attachment } from '@/components/ChatInput'
import ChatMessage from '@/components/ChatMessage'
import AgentResponse from '@/components/AgentResponse'
import { Bot, WifiOff, Plus, MessageSquare, Trash2, ChevronLeft, ChevronRight } from 'lucide-react'

interface Message {
  id: string
  type: 'user' | 'agent'
  content: string
  agent?: string
  agentType?: string
  status?: 'thinking' | 'complete'
  timestamp: Date
  attachments?: Attachment[]
  thinking?: string
  isThinking?: boolean  // True while actively receiving thinking content
}

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([])
  const [isConnected, setIsConnected] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [sessions, setSessions] = useState<ChatSessionSummary[]>([])
  const [showSidebar, setShowSidebar] = useState(true)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const wsRef = useRef(getChatWebSocket())
  const pendingMessageRef = useRef<{ content: string; agent?: string; thinking?: string } | null>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  // Load sessions list
  const loadSessions = useCallback(async () => {
    try {
      const sessionList = await getChatSessions()
      setSessions(sessionList)
    } catch (e) {
      console.error('Failed to load sessions:', e)
    }
  }, [])

  // Load a specific session
  const loadSession = useCallback(async (id: string) => {
    try {
      const session = await getChatSession(id)
      setSessionId(id)
      // Convert API messages to local format
      const loadedMessages: Message[] = session.messages.map((m) => ({
        id: m.id,
        type: m.role === 'user' ? 'user' : 'agent',
        content: m.content,
        agent: m.agent || 'Claude',
        agentType: 'assistant',
        status: 'complete' as const,
        timestamp: new Date(m.timestamp),
        thinking: m.thinking,
      }))
      setMessages(loadedMessages)
    } catch (e) {
      console.error('Failed to load session:', e)
    }
  }, [])

  // Create new session
  const createNewSession = useCallback(async () => {
    try {
      const session = await createChatSession()
      setSessionId(session.id)
      setMessages([])
      await loadSessions()
    } catch (e) {
      console.error('Failed to create session:', e)
    }
  }, [loadSessions])

  // Delete a session
  const handleDeleteSession = useCallback(async (id: string, e: React.MouseEvent) => {
    e.stopPropagation()
    try {
      await deleteChatSession(id)
      if (sessionId === id) {
        setSessionId(null)
        setMessages([])
      }
      await loadSessions()
    } catch (e) {
      console.error('Failed to delete session:', e)
    }
  }, [sessionId, loadSessions])

  // Save message to backend
  const saveMessage = useCallback(async (role: 'user' | 'assistant', content: string, agent?: string, thinking?: string) => {
    if (!sessionId) return
    try {
      await addChatMessage(sessionId, role, content, agent, thinking)
      await loadSessions() // Refresh list to update message counts
    } catch (e) {
      console.error('Failed to save message:', e)
    }
  }, [sessionId, loadSessions])

  // Initialize: load sessions and create/load initial session
  useEffect(() => {
    const init = async () => {
      await loadSessions()
      const sessionList = await getChatSessions()
      if (sessionList.length > 0) {
        // Load most recent session
        await loadSession(sessionList[0].id)
      } else {
        // Create first session
        await createNewSession()
      }
    }
    init()
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

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

        case 'thinking_start':
          // Claude is starting to think - mark the message as actively thinking
          setMessages((prev) => {
            const lastMessage = prev[prev.length - 1]
            if (lastMessage && lastMessage.type === 'agent' && lastMessage.status === 'thinking') {
              return [
                ...prev.slice(0, -1),
                {
                  ...lastMessage,
                  isThinking: true,
                  thinking: '',
                },
              ]
            }
            return prev
          })
          break

        case 'thinking_delta':
          // Streaming thinking content
          setMessages((prev) => {
            const lastMessage = prev[prev.length - 1]
            if (lastMessage && lastMessage.type === 'agent' && lastMessage.isThinking) {
              return [
                ...prev.slice(0, -1),
                {
                  ...lastMessage,
                  thinking: (lastMessage.thinking || '') + (event.delta || ''),
                },
              ]
            }
            return prev
          })
          break

        case 'thinking_complete':
          // Thinking is done, now waiting for response
          setMessages((prev) => {
            const lastMessage = prev[prev.length - 1]
            if (lastMessage && lastMessage.type === 'agent' && lastMessage.isThinking) {
              return [
                ...prev.slice(0, -1),
                {
                  ...lastMessage,
                  isThinking: false,
                  thinking: event.thinking || lastMessage.thinking,
                },
              ]
            }
            return prev
          })
          break

        case 'agent_delta':
          // Streaming text delta - append to current agent message
          setMessages((prev) => {
            const lastMessage = prev[prev.length - 1]
            if (lastMessage && lastMessage.type === 'agent' && lastMessage.status === 'thinking') {
              // Update the thinking message with new content
              return [
                ...prev.slice(0, -1),
                {
                  ...lastMessage,
                  content: lastMessage.content + (event.delta || ''),
                },
              ]
            }
            return prev
          })
          break

        case 'agent_complete':
          // Track message for saving
          pendingMessageRef.current = {
            content: event.content || '',
            agent: event.agent || 'Claude',
            thinking: event.thinking,
          }

          // Update or add agent response
          setMessages((prev) => {
            // Find existing thinking message for this agent
            const thinkingIdx = prev.findIndex(
              (m) => m.type === 'agent' && m.status === 'thinking' && m.agentType === event.agent_type
            )

            if (thinkingIdx !== -1) {
              // Update existing message to complete, preserving thinking
              const updated = [...prev]
              const finalContent = event.content || updated[thinkingIdx].content
              const finalThinking = event.thinking || updated[thinkingIdx].thinking
              updated[thinkingIdx] = {
                ...updated[thinkingIdx],
                content: finalContent,
                thinking: finalThinking,
                isThinking: false,
                status: 'complete',
              }
              // Update pending message with actual content
              pendingMessageRef.current = {
                content: finalContent,
                agent: event.agent || 'Claude',
                thinking: finalThinking,
              }
              return updated
            }

            // No thinking message found, add new complete message
            return [
              ...prev,
              {
                id: `agent-${Date.now()}-${event.agent}`,
                type: 'agent',
                content: event.content || '',
                agent: event.agent || 'Agent',
                agentType: event.agent_type || 'worker',
                status: 'complete',
                timestamp: new Date(),
                thinking: event.thinking,
              },
            ]
          })
          break

        case 'chat_complete':
          setIsLoading(false)
          // Save the assistant message to backend
          if (pendingMessageRef.current) {
            const msg = pendingMessageRef.current
            saveMessage('assistant', msg.content, msg.agent, msg.thinking)
            pendingMessageRef.current = null
          }
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
  }, [saveMessage])

  const handleSend = useCallback(async (content: string, attachments: Attachment[]) => {
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

    // Save user message to backend
    await saveMessage('user', content)

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
  }, [saveMessage])

  // Simplified send for suggestion buttons
  const handleQuickSend = useCallback((content: string) => {
    handleSend(content, [])
  }, [handleSend])

  return (
    <div className="flex h-full">
      {/* Session Sidebar */}
      <div className={`${showSidebar ? 'w-64' : 'w-0'} transition-all duration-200 overflow-hidden border-r border-zinc-800 flex flex-col bg-zinc-900/50`}>
        <div className="p-3 border-b border-zinc-800 flex items-center justify-between">
          <span className="text-sm font-medium text-zinc-400">History</span>
          <button
            onClick={createNewSession}
            className="p-1.5 hover:bg-zinc-800 rounded transition-colors"
            title="New chat"
          >
            <Plus className="w-4 h-4 text-zinc-400" />
          </button>
        </div>
        <div className="flex-1 overflow-auto">
          {sessions.map((session) => (
            <button
              key={session.id}
              onClick={() => loadSession(session.id)}
              className={`w-full px-3 py-2 text-left hover:bg-zinc-800/50 transition-colors group flex items-center gap-2 ${
                sessionId === session.id ? 'bg-zinc-800' : ''
              }`}
            >
              <MessageSquare className="w-4 h-4 text-zinc-500 flex-shrink-0" />
              <div className="flex-1 min-w-0">
                <p className="text-sm text-zinc-300 truncate">{session.title}</p>
                <p className="text-xs text-zinc-600">{session.message_count} messages</p>
              </div>
              <button
                onClick={(e) => handleDeleteSession(session.id, e)}
                className="opacity-0 group-hover:opacity-100 p-1 hover:bg-zinc-700 rounded transition-all"
                title="Delete"
              >
                <Trash2 className="w-3 h-3 text-zinc-500" />
              </button>
            </button>
          ))}
          {sessions.length === 0 && (
            <p className="text-xs text-zinc-600 p-3">No chat history yet</p>
          )}
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-zinc-800">
          <div className="flex items-center gap-3">
            <button
              onClick={() => setShowSidebar(!showSidebar)}
              className="p-1 hover:bg-zinc-800 rounded transition-colors"
              title={showSidebar ? 'Hide history' : 'Show history'}
            >
              {showSidebar ? (
                <ChevronLeft className="w-5 h-5 text-zinc-400" />
              ) : (
                <ChevronRight className="w-5 h-5 text-zinc-400" />
              )}
            </button>
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
                  thinking={message.thinking}
                  isThinking={message.isThinking}
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
    </div>
  )
}
