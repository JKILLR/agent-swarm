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

export async function sendChat(message: string) {
  const res = await fetch(`${API_BASE}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message }),
  })
  if (!res.ok) throw new Error('Failed to send message')
  return res.json()
}
