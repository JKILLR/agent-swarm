import { clsx, type ClassValue } from 'clsx'
import { twMerge } from 'tailwind-merge'

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function getAgentColor(agentType: string): string {
  const colors: Record<string, string> = {
    orchestrator: 'agent-orchestrator',
    supreme: 'agent-orchestrator',
    researcher: 'agent-researcher',
    implementer: 'agent-implementer',
    critic: 'agent-critic',
    benchmarker: 'agent-benchmarker',
    monitor: 'agent-monitor',
    worker: 'agent-worker',
    summary: 'agent-summary',
  }
  return colors[agentType.toLowerCase()] || 'agent-worker'
}

export function getAgentIcon(agentType: string): string {
  const icons: Record<string, string> = {
    orchestrator: '🎯',
    supreme: '👑',
    researcher: '🔬',
    implementer: '🔧',
    critic: '🔍',
    benchmarker: '📊',
    monitor: '👁',
    worker: '⚙',
    summary: '📋',
  }
  return icons[agentType.toLowerCase()] || '•'
}

export function getStatusColor(status: string): string {
  const colors: Record<string, string> = {
    active: 'text-green-400',
    paused: 'text-yellow-400',
    archived: 'text-zinc-500',
  }
  return colors[status.toLowerCase()] || 'text-zinc-400'
}
