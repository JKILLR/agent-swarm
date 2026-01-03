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
import { useAgentActivity } from '@/lib/AgentActivityContext'
import { useMobileLayout } from '@/components/MobileLayout'
import { Terminal, WifiOff, Plus, MessageSquare, Trash2, ChevronLeft, ChevronRight, X } from 'lucide-react'

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
  const [showMobileHistory, setShowMobileHistory] = useState(false)
  // Use global context for activity state (persists across navigation)
  const {
    panelAgentActivities: agentActivities,
    panelToolActivities: toolActivities,
    setPanelAgentActivities: setAgentActivities,
    setPanelToolActivities: setToolActivities,
  } = useAgentActivity()
  const { isMobile } = useMobileLayout()
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
          // Clear activities for new chat
          setAgentActivities([])
          setToolActivities([])
          // Add COO as the first active agent
          setAgentActivities([{
            id: `agent-coo-${Date.now()}`,
            name: 'Supreme Orchestrator (COO)',
            status: 'thinking',
            startTime: new Date(),
          }])
          break

        case 'tool_start':
          // Add new tool activity with agent attribution
          setToolActivities((prev) => [
            ...prev,
            {
              id: `tool-${Date.now()}-${event.tool}`,
              tool: event.tool || 'Unknown',
              description: event.description || '',
              status: 'running',
              timestamp: new Date(),
              agentName: event.agentName,
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
          break

        case 'tool_complete':
          // Update tool activity status
          setToolActivities((prev) => {
            const idx = [...prev].reverse().findIndex(
              (t) => t.tool === event.tool && t.status === 'running'
            )
            if (idx === -1) return prev
            const actualIdx = prev.length - 1 - idx
            const updated = [...prev]
            updated[actualIdx] = {
              ...updated[actualIdx],
              status: event.success ? 'complete' : 'error',
              summary: event.summary,
              endTime: new Date(),
              agentName: event.agentName || updated[actualIdx].agentName,
            }
            return updated
          })
          break

        case 'agent_spawn':
          // Add new agent activity when COO delegates
          setAgentActivities((prev) => {
            // Mark parent agent as delegating
            const parentAgent = (event as { parentAgent?: string }).parentAgent || 'COO'
            const updated = prev.map(a =>
              a.name === parentAgent || (parentAgent === 'COO' && a.name.includes('COO'))
                ? { ...a, status: 'delegating' as const }
                : a
            )
            // Add the new agent
            return [
              ...updated,
              {
                id: `agent-${Date.now()}-${event.agent}`,
                name: event.agent || 'Agent',
                status: 'working' as const,
                description: (event as { description?: string }).description?.substring(0, 100),
                startTime: new Date(),
              },
            ]
          })
          break

        case 'agent_complete_subagent':
          // Mark a sub-agent as complete
          setAgentActivities((prev) => {
            const agentName = event.agent || ''
            return prev.map(a => {
              if (a.name === agentName) {
                return { ...a, status: 'complete' as const, endTime: new Date() }
              }
              // If this was a delegating agent's child completing, restore working status
              if (a.status === 'delegating') {
                // Check if there are still other active children
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
          // Mark all agents as complete
          setAgentActivities((prev) =>
            prev.map((a) => ({
              ...a,
              status: 'complete' as const,
              endTime: a.endTime || new Date(),
            }))
          )
          // Save the assistant message to backend
          if (pendingMessageRef.current) {
            const msg = pendingMessageRef.current
            saveMessage('assistant', msg.content, msg.agent, msg.thinking)
            pendingMessageRef.current = null
          }
          break

        case 'error':
          setIsLoading(false)
          // Mark agents as error
          setAgentActivities((prev) =>
            prev.map((a) =>
              a.status !== 'complete' ? { ...a, status: 'error' as const, endTime: new Date() } : a
            )
          )
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
  }, [saveMessage, setAgentActivities, setToolActivities])

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
      wsRef.current.send(fullMessage, {
        session_id: sessionId || undefined,
        attachments: imageAttachments.length > 0 ? imageAttachments : undefined,
      })
    } catch (e) {
      console.error('Failed to send message:', e)
      setIsLoading(false)
    }
  }, [saveMessage, sessionId])

  // Simplified send for suggestion buttons
  const handleQuickSend = useCallback((content: string) => {
    handleSend(content, [])
  }, [handleSend])

  return (
    <div className="flex h-full relative bg-[#0d0d0d]">
      {/* Session Sidebar - Hidden on mobile, use bottom sheet instead */}
      <div className={`hidden md:flex ${showSidebar ? 'w-64' : 'w-0'} transition-all duration-200 overflow-hidden border-r border-zinc-800/50 flex-col bg-[#0d0d0d]`}>
        <div className="p-3 border-b border-zinc-800/50 flex items-center justify-between">
          <span className="text-sm font-medium text-zinc-500">History</span>
          <button
            onClick={createNewSession}
            className="p-1.5 hover:bg-zinc-800/50 rounded transition-colors"
            title="New chat"
          >
            <Plus className="w-4 h-4 text-orange-500" />
          </button>
        </div>
        <div className="flex-1 overflow-auto">
          {sessions.map((session) => (
            <button
              key={session.id}
              onClick={() => loadSession(session.id)}
              className={`w-full px-3 py-2 text-left hover:bg-zinc-800/30 transition-colors group flex items-center gap-2 ${
                sessionId === session.id ? 'bg-zinc-800/50 border-l-2 border-l-orange-500' : 'border-l-2 border-l-transparent'
              }`}
            >
              <MessageSquare className="w-4 h-4 text-zinc-600 flex-shrink-0" />
              <div className="flex-1 min-w-0">
                <p className="text-sm text-zinc-400 truncate">{session.title}</p>
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

      {/* Mobile History Bottom Sheet */}
      {isMobile && showMobileHistory && (
        <>
          {/* Overlay */}
          <div
            className="fixed inset-0 z-40 bg-black/80"
            onClick={() => setShowMobileHistory(false)}
          />
          {/* Bottom Sheet */}
          <div className="fixed bottom-0 left-0 right-0 z-50 bg-[#0d0d0d] rounded-t-2xl border-t border-zinc-800/50 max-h-[70vh] flex flex-col animate-slide-up">
            <div className="flex items-center justify-between p-4 border-b border-zinc-800/50">
              <span className="font-medium text-zinc-300">Chat History</span>
              <div className="flex items-center gap-2">
                <button
                  onClick={createNewSession}
                  className="p-2 hover:bg-zinc-800/50 rounded-lg transition-colors touch-manipulation"
                  title="New chat"
                >
                  <Plus className="w-5 h-5 text-orange-500" />
                </button>
                <button
                  onClick={() => setShowMobileHistory(false)}
                  className="p-2 hover:bg-zinc-800/50 rounded-lg transition-colors touch-manipulation"
                >
                  <X className="w-5 h-5 text-zinc-500" />
                </button>
              </div>
            </div>
            <div className="flex-1 overflow-auto p-2">
              {sessions.map((session) => (
                <button
                  key={session.id}
                  onClick={() => {
                    loadSession(session.id)
                    setShowMobileHistory(false)
                  }}
                  className={`w-full px-4 py-3 text-left hover:bg-zinc-800/30 active:bg-zinc-800/50 transition-colors flex items-center gap-3 rounded-lg touch-manipulation min-h-[56px] ${
                    sessionId === session.id ? 'bg-zinc-800/50 border-l-2 border-l-orange-500' : 'border-l-2 border-l-transparent'
                  }`}
                >
                  <MessageSquare className="w-5 h-5 text-zinc-600 flex-shrink-0" />
                  <div className="flex-1 min-w-0">
                    <p className="text-sm text-zinc-400 truncate">{session.title}</p>
                    <p className="text-xs text-zinc-600">{session.message_count} messages</p>
                  </div>
                  <button
                    onClick={(e) => {
                      handleDeleteSession(session.id, e)
                    }}
                    className="p-2 hover:bg-zinc-700 rounded-lg transition-colors touch-manipulation"
                    title="Delete"
                  >
                    <Trash2 className="w-4 h-4 text-zinc-500" />
                  </button>
                </button>
              ))}
              {sessions.length === 0 && (
                <p className="text-sm text-zinc-600 p-4 text-center">No chat history yet</p>
              )}
            </div>
          </div>
        </>
      )}

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col min-w-0 bg-[#0d0d0d]">
        {/* Header */}
        <div className="flex items-center justify-between px-3 md:px-6 py-3 md:py-4 border-b border-zinc-800/50">
          <div className="flex items-center gap-2 md:gap-3">
            {/* Desktop sidebar toggle */}
            <button
              onClick={() => setShowSidebar(!showSidebar)}
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
              onClick={() => setShowMobileHistory(true)}
              className="md:hidden p-2 -ml-1 hover:bg-zinc-800/50 active:bg-zinc-800 rounded-lg transition-colors touch-manipulation"
              title="Chat history"
            >
              <MessageSquare className="w-5 h-5 text-zinc-500" />
            </button>
            <Terminal className="w-5 h-5 md:w-6 md:h-6 text-orange-500" />
            <div>
              <h1 className="font-medium text-white text-sm md:text-base">Chat with COO</h1>
              <p className="text-xs text-zinc-600 hidden md:block">Supreme Orchestrator - Chief Operating Officer</p>
            </div>
          </div>

          <div className="flex items-center gap-2">
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

        {/* Messages */}
        <div className="flex-1 overflow-auto p-3 md:p-6 space-y-4 md:space-y-6 relative">
          {messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-center px-4">
              <Terminal className="w-12 h-12 md:w-16 md:h-16 text-orange-500/50 mb-4" />
              <h2 className="text-lg md:text-xl font-medium text-zinc-400 mb-2">
                Chat with the COO
              </h2>
              <p className="text-sm text-zinc-600 max-w-md">
                You are connected to the Supreme Orchestrator (COO). Ask about your swarms,
                request research, or coordinate tasks.
              </p>
              <div className="mt-6 space-y-2 text-left w-full max-w-md">
                <p className="text-xs text-zinc-600">Try asking:</p>
                <div className="space-y-2">
                  <button
                    onClick={() => handleQuickSend('What is the current status of the ASA project?')}
                    className="block w-full text-left text-sm text-zinc-500 hover:text-violet-400 active:text-violet-300 transition-all duration-200 p-3 md:p-2 bg-zinc-900/50 md:bg-transparent rounded-lg md:rounded-none touch-manipulation border border-zinc-800/50 md:border-0 hover:bg-violet-500/5 hover:border-violet-500/30"
                  >
                    <span className="text-orange-500 mr-2">&gt;</span>What is the current status of the ASA project?
                  </button>
                  <button
                    onClick={() => handleQuickSend('Research sparse attention implementations for ASA')}
                    className="block w-full text-left text-sm text-zinc-500 hover:text-violet-400 active:text-violet-300 transition-all duration-200 p-3 md:p-2 bg-zinc-900/50 md:bg-transparent rounded-lg md:rounded-none touch-manipulation border border-zinc-800/50 md:border-0 hover:bg-violet-500/5 hover:border-violet-500/30"
                  >
                    <span className="text-orange-500 mr-2">&gt;</span>Research sparse attention implementations for ASA
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

        {/* Input - Full width on mobile with safe area padding */}
        <div className="p-2 md:p-4 border-t border-zinc-800/50 pb-[calc(0.5rem+env(safe-area-inset-bottom))] md:pb-4 bg-[#0d0d0d]">
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
