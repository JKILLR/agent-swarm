const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export interface Swarm {
  name: string
  description: string
  status: string
  agent_count: number
  priorities: string[]
  version?: string
}

export interface Agent {
  name: string
  type: string
  model: string
  background: boolean
  description: string
}

export interface HealthStatus {
  status: string
  swarm_count: number
  agent_count: number
}

export async function getSwarms(): Promise<Swarm[]> {
  const res = await fetch(`${API_BASE}/api/swarms`)
  if (!res.ok) throw new Error('Failed to fetch swarms')
  return res.json()
}

export async function getSwarm(name: string): Promise<Swarm> {
  const res = await fetch(`${API_BASE}/api/swarms/${name}`)
  if (!res.ok) throw new Error(`Failed to fetch swarm: ${name}`)
  return res.json()
}

export async function getSwarmAgents(name: string): Promise<Agent[]> {
  const res = await fetch(`${API_BASE}/api/swarms/${name}/agents`)
  if (!res.ok) throw new Error(`Failed to fetch agents for swarm: ${name}`)
  return res.json()
}

export async function createSwarm(name: string, description: string = '', template: string = '_template') {
  const res = await fetch(`${API_BASE}/api/swarms`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, description, template }),
  })
  if (!res.ok) {
    const error = await res.json()
    throw new Error(error.detail || 'Failed to create swarm')
  }
  return res.json()
}

export async function getStatus(): Promise<HealthStatus> {
  const res = await fetch(`${API_BASE}/api/status`)
  if (!res.ok) throw new Error('Failed to fetch status')
  return res.json()
}

// Chat History Types
export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: string
  agent?: string
  thinking?: string
}

export interface ChatSession {
  id: string
  title: string
  swarm?: string
  created_at: string
  updated_at: string
  messages: ChatMessage[]
}

export interface ChatSessionSummary {
  id: string
  title: string
  swarm?: string
  created_at: string
  updated_at: string
  message_count: number
}

// Chat History API
export async function getChatSessions(): Promise<ChatSessionSummary[]> {
  const res = await fetch(`${API_BASE}/api/chat/sessions`)
  if (!res.ok) throw new Error('Failed to fetch chat sessions')
  return res.json()
}

export async function getChatSession(sessionId: string): Promise<ChatSession> {
  const res = await fetch(`${API_BASE}/api/chat/sessions/${sessionId}`)
  if (!res.ok) throw new Error(`Failed to fetch session: ${sessionId}`)
  return res.json()
}

export async function createChatSession(swarm?: string, title?: string): Promise<ChatSession> {
  const params = new URLSearchParams()
  if (swarm) params.append('swarm', swarm)
  if (title) params.append('title', title)

  const res = await fetch(`${API_BASE}/api/chat/sessions?${params}`, {
    method: 'POST',
  })
  if (!res.ok) throw new Error('Failed to create chat session')
  return res.json()
}

export async function deleteChatSession(sessionId: string): Promise<void> {
  const res = await fetch(`${API_BASE}/api/chat/sessions/${sessionId}`, {
    method: 'DELETE',
  })
  if (!res.ok) throw new Error('Failed to delete session')
}

export async function addChatMessage(
  sessionId: string,
  role: 'user' | 'assistant',
  content: string,
  agent?: string,
  thinking?: string
): Promise<ChatMessage> {
  const params = new URLSearchParams()
  params.append('role', role)
  params.append('content', content)
  if (agent) params.append('agent', agent)
  if (thinking) params.append('thinking', thinking)

  const res = await fetch(`${API_BASE}/api/chat/sessions/${sessionId}/messages?${params}`, {
    method: 'POST',
  })
  if (!res.ok) throw new Error('Failed to add message')
  return res.json()
}

// Background Jobs API
export interface Job {
  id: string
  type: string
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled'
  created_at: string
  started_at?: string
  completed_at?: string
  prompt: string
  swarm?: string
  session_id?: string
  progress: number
  current_activity: string
  activities: Array<{ tool: string; time: string }>
  result?: string
  error?: string
}

export async function getJobs(sessionId?: string, limit: number = 20): Promise<Job[]> {
  const params = new URLSearchParams()
  if (sessionId) params.append('session_id', sessionId)
  params.append('limit', limit.toString())

  const res = await fetch(`${API_BASE}/api/jobs?${params}`)
  if (!res.ok) throw new Error('Failed to fetch jobs')
  return res.json()
}

export async function cancelJob(jobId: string): Promise<{ success: boolean }> {
  const res = await fetch(`${API_BASE}/api/jobs/${jobId}`, {
    method: 'DELETE',
  })
  if (!res.ok) throw new Error('Failed to cancel job')
  return res.json()
}
