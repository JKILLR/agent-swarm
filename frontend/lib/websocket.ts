const WS_BASE = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000'

export type WebSocketEventType =
  | 'chat_start'
  | 'agent_start'
  | 'agent_delta'
  | 'agent_chunk'
  | 'agent_complete'
  | 'chat_complete'
  | 'error'

export interface WebSocketEvent {
  type: WebSocketEventType
  agent?: string
  agent_type?: string
  content?: string
  delta?: string
  message?: string
  success?: boolean
}

export type EventHandler = (event: WebSocketEvent) => void

export class ChatWebSocket {
  private ws: WebSocket | null = null
  private handlers: Map<string, EventHandler[]> = new Map()
  private reconnectAttempts = 0
  private maxReconnectAttempts = 5
  private reconnectDelay = 1000

  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        this.ws = new WebSocket(`${WS_BASE}/ws/chat`)

        this.ws.onopen = () => {
          console.log('WebSocket connected')
          this.reconnectAttempts = 0
          resolve()
        }

        this.ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data) as WebSocketEvent
            this.emit(data.type, data)
            this.emit('*', data) // Wildcard handler
          } catch (e) {
            console.error('Failed to parse WebSocket message:', e)
          }
        }

        this.ws.onclose = () => {
          console.log('WebSocket disconnected')
          this.emit('disconnected', { type: 'error', message: 'Disconnected' })
          this.attemptReconnect()
        }

        this.ws.onerror = (error) => {
          console.error('WebSocket error:', error)
          reject(error)
        }
      } catch (e) {
        reject(e)
      }
    })
  }

  private attemptReconnect() {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.log('Max reconnection attempts reached')
      return
    }

    this.reconnectAttempts++
    const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1)
    console.log(`Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`)

    setTimeout(() => {
      this.connect().catch(() => {})
    }, delay)
  }

  disconnect() {
    if (this.ws) {
      this.ws.close()
      this.ws = null
    }
  }

  send(message: string, swarm?: string) {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      throw new Error('WebSocket not connected')
    }

    this.ws.send(JSON.stringify({ message, swarm }))
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
      handlers.forEach(handler => handler(data))
    }
  }

  get isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN
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
