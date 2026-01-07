// Auto-detect WebSocket URL based on current host
function getWsBase(): string {
  if (typeof window === 'undefined') {
    return process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000'
  }
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  const host = window.location.hostname
  const port = '8000'
  return process.env.NEXT_PUBLIC_WS_URL || `${protocol}//${host}:${port}`
}

// Connection states for clearer lifecycle management
type ConnectionState = 'disconnected' | 'connecting' | 'connected' | 'reconnecting'

export type WebSocketEventType =
  | 'chat_start'
  | 'agent_start'
  | 'agent_delta'
  | 'agent_chunk'
  | 'agent_complete'
  | 'agent_complete_subagent'
  | 'thinking_start'
  | 'thinking_delta'
  | 'thinking_complete'
  | 'chat_complete'
  | 'tool_start'
  | 'tool_complete'
  | 'agent_spawn'
  | 'executor_pool_status'
  | 'error'
  | 'disconnected'
  | 'max_reconnect_reached'

export interface WebSocketEvent {
  type: WebSocketEventType
  agent?: string
  agent_type?: string
  content?: string
  delta?: string
  thinking?: string
  message?: string
  success?: boolean
  // Tool event fields
  tool?: string
  description?: string
  summary?: string
  input?: Record<string, string>
  // Agent attribution for tools
  agentName?: string
  parentAgent?: string
  // File modification info
  filePath?: string
  fileOperation?: 'read' | 'write' | 'edit'
  // Executor pool status
  activeCount?: number
  availableSlots?: number
}

export type EventHandler = (event: WebSocketEvent) => void

export class ChatWebSocket {
  private ws: WebSocket | null = null
  private handlers: Map<string, EventHandler[]> = new Map()
  private reconnectAttempts = 0
  private maxReconnectAttempts = 5
  private reconnectDelay = 1000
  private connectionPromise: Promise<void> | null = null
  private isIntentionalDisconnect = false
  private reconnectTimeout: NodeJS.Timeout | null = null
  private connectionState: ConnectionState = 'disconnected'
  private heartbeatInterval: NodeJS.Timeout | null = null
  private lastPongTime: number = 0
  private connectionId = 0 // Track connection identity to handle stale callbacks

  connect(): Promise<void> {
    // Guard: if already connected, return resolved promise
    if (this.connectionState === 'connected' && this.ws?.readyState === WebSocket.OPEN) {
      return Promise.resolve()
    }

    // Guard: if currently connecting, return the existing promise
    // This is the key fix - we check connectionState BEFORE checking ws.readyState
    if (this.connectionState === 'connecting' && this.connectionPromise) {
      return this.connectionPromise
    }

    // Guard: if reconnecting, return the existing promise
    if (this.connectionState === 'reconnecting' && this.connectionPromise) {
      return this.connectionPromise
    }

    // Clear any pending reconnect
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout)
      this.reconnectTimeout = null
    }

    // Stop heartbeat from previous connection
    this.stopHeartbeat()

    // Clean up any existing dead connection (CLOSED or CLOSING state)
    if (this.ws && (this.ws.readyState === WebSocket.CLOSED || this.ws.readyState === WebSocket.CLOSING)) {
      this.ws.onclose = null
      this.ws.onerror = null
      this.ws.onmessage = null
      this.ws.onopen = null
      this.ws = null
    }

    // If we still have a live connection, don't create a new one
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.connectionState = 'connected'
      return Promise.resolve()
    }

    // If there's a connection in progress (CONNECTING state), wait for it
    if (this.ws && this.ws.readyState === WebSocket.CONNECTING) {
      // Create a promise that resolves when the existing connection opens or fails
      if (!this.connectionPromise) {
        this.connectionPromise = new Promise((resolve, reject) => {
          const checkState = () => {
            if (this.ws?.readyState === WebSocket.OPEN) {
              this.connectionState = 'connected'
              resolve()
            } else if (!this.ws || this.ws.readyState === WebSocket.CLOSED) {
              reject(new Error('Connection failed'))
            } else {
              setTimeout(checkState, 50)
            }
          }
          checkState()
        })
      }
      return this.connectionPromise
    }

    this.isIntentionalDisconnect = false
    this.connectionState = 'connecting'
    const currentConnectionId = ++this.connectionId

    this.connectionPromise = new Promise((resolve, reject) => {
      try {
        const wsUrl = `${getWsBase()}/ws/chat`
        console.log(`WebSocket connecting to ${wsUrl}`)
        this.ws = new WebSocket(wsUrl)

        this.ws.onopen = () => {
          // Check if this is still the current connection attempt
          if (this.connectionId !== currentConnectionId) {
            console.log('Stale connection opened, ignoring')
            return
          }

          console.log('WebSocket connected')
          this.connectionState = 'connected'
          this.reconnectAttempts = 0
          this.lastPongTime = Date.now()
          this.startHeartbeat()
          resolve()
        }

        this.ws.onmessage = (event) => {
          // Update last pong time on any message (acts as heartbeat response)
          this.lastPongTime = Date.now()

          try {
            const data = JSON.parse(event.data) as WebSocketEvent
            // Log ALL events for debugging (temporary)
            console.log(`[WS Event] ${data.type}:`, data.type === 'agent_complete'
              ? { agent: data.agent, contentLen: data.content?.length, contentPreview: data.content?.substring(0, 100) }
              : data.type === 'agent_delta'
              ? { agent: data.agent, deltaLen: data.delta?.length }
              : data)
            this.emit(data.type, data)
            this.emit('*', data) // Wildcard handler
          } catch (e) {
            console.error('Failed to parse WebSocket message:', e)
          }
        }

        this.ws.onclose = (event) => {
          // Check if this is still the current connection
          if (this.connectionId !== currentConnectionId) {
            console.log('Stale connection closed, ignoring')
            return
          }

          console.log(`WebSocket disconnected (code: ${event.code}, reason: ${event.reason || 'none'})`)
          this.connectionState = 'disconnected'
          this.connectionPromise = null
          this.stopHeartbeat()
          // Use 'disconnected' as type instead of 'error' to prevent triggering error handlers
          this.emit('disconnected', { type: 'disconnected', message: 'Disconnected' })

          // Only reconnect if not intentionally disconnected
          if (!this.isIntentionalDisconnect) {
            this.attemptReconnect()
          }
        }

        this.ws.onerror = (error) => {
          // Check if this is still the current connection attempt
          if (this.connectionId !== currentConnectionId) {
            console.log('Stale connection error, ignoring')
            return
          }

          console.error('WebSocket error:', error)
          // Don't set connectionPromise to null here - let onclose handle cleanup
          // onerror is always followed by onclose
          reject(error)
        }
      } catch (e) {
        console.error('Failed to create WebSocket:', e)
        this.connectionState = 'disconnected'
        this.connectionPromise = null
        reject(e)
      }
    })

    return this.connectionPromise
  }

  private startHeartbeat() {
    this.stopHeartbeat()
    this.lastPongTime = Date.now()

    // Check connection health every 30 seconds
    this.heartbeatInterval = setInterval(() => {
      const timeSinceLastMessage = Date.now() - this.lastPongTime
      // If no message received in 60 seconds, connection might be dead
      if (timeSinceLastMessage > 60000) {
        console.log('WebSocket connection appears stale, reconnecting...')
        this.reconnect()
      }
    }, 30000)
  }

  private stopHeartbeat() {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval)
      this.heartbeatInterval = null
    }
  }

  private reconnect() {
    this.stopHeartbeat()
    if (this.ws) {
      this.ws.onclose = null // Prevent double reconnect
      this.ws.close()
      this.ws = null
    }
    this.connectionState = 'disconnected'
    this.connectionPromise = null
    this.attemptReconnect()
  }

  private attemptReconnect() {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.log('Max reconnection attempts reached')
      // Use 'max_reconnect_reached' as type instead of 'error' to prevent triggering error handlers
      this.emit('max_reconnect_reached', { type: 'max_reconnect_reached', message: 'Max reconnection attempts reached' })
      return
    }

    // Don't start another reconnect if one is pending
    if (this.reconnectTimeout) {
      return
    }

    // Don't reconnect if we're already connecting
    if (this.connectionState === 'connecting' || this.connectionState === 'reconnecting') {
      return
    }

    this.reconnectAttempts++
    this.connectionState = 'reconnecting'
    const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1)
    console.log(`Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`)

    this.reconnectTimeout = setTimeout(() => {
      this.reconnectTimeout = null
      this.connectionState = 'disconnected' // Reset state before connecting
      this.connect().catch((e) => {
        console.error('Reconnection failed:', e)
      })
    }, delay)
  }

  disconnect() {
    this.isIntentionalDisconnect = true
    this.connectionState = 'disconnected'

    // Clear any pending reconnect
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout)
      this.reconnectTimeout = null
    }

    // Stop heartbeat
    this.stopHeartbeat()

    if (this.ws) {
      this.ws.onclose = null // Prevent reconnect trigger
      this.ws.onerror = null
      this.ws.onmessage = null
      this.ws.onopen = null
      this.ws.close()
      this.ws = null
    }
    this.connectionPromise = null
  }

  send(message: string, options?: { swarm?: string; session_id?: string; attachments?: Array<{type: string; name: string; content: string; mimeType?: string}> }) {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      console.error('WebSocket not connected, cannot send message')
      throw new Error('WebSocket not connected')
    }

    try {
      this.ws.send(JSON.stringify({
        message,
        swarm: options?.swarm,
        session_id: options?.session_id,
        attachments: options?.attachments,
      }))
    } catch (e) {
      console.error('Failed to send WebSocket message:', e)
      throw e
    }
  }

  on(event: string, handler: EventHandler) {
    if (!this.handlers.has(event)) {
      this.handlers.set(event, [])
    }
    this.handlers.get(event)!.push(handler)
  }

  off(event: string, handler: EventHandler) {
    const handlers = this.handlers.get(event)
    if (handlers) {
      const index = handlers.indexOf(handler)
      if (index !== -1) {
        handlers.splice(index, 1)
      }
    }
  }

  private emit(event: string, data: WebSocketEvent) {
    const handlers = this.handlers.get(event)
    if (handlers) {
      // Create a copy to avoid issues if handlers modify the array
      [...handlers].forEach(handler => {
        try {
          handler(data)
        } catch (e) {
          console.error(`Error in WebSocket event handler for ${event}:`, e)
        }
      })
    }
  }

  get isConnected(): boolean {
    return this.connectionState === 'connected' && this.ws?.readyState === WebSocket.OPEN
  }

  get state(): ConnectionState {
    return this.connectionState
  }

  // Reset reconnection counter (call when user manually triggers reconnect)
  resetReconnectAttempts() {
    this.reconnectAttempts = 0
  }
}

// Singleton instance
let wsInstance: ChatWebSocket | null = null

export function getChatWebSocket(): ChatWebSocket {
  if (!wsInstance) {
    wsInstance = new ChatWebSocket()
  }
  return wsInstance
}
