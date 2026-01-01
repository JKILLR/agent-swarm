'use client'

import { type Agent } from '@/lib/api'
import { cn, getAgentColor, getAgentIcon } from '@/lib/utils'

interface AgentPanelProps {
  agent: Agent
}

export default function AgentPanel({ agent }: AgentPanelProps) {
  const colorClass = getAgentColor(agent.type)
  const icon = getAgentIcon(agent.type)

  return (
    <div
      className={cn(
        'p-4 bg-zinc-900 border border-zinc-800 rounded-lg',
        `border-l-4 border-l-${colorClass}`
      )}
    >
      <div className="flex items-center gap-2 mb-2">
        <span className="text-lg">{icon}</span>
        <span className={cn('font-medium', `text-${colorClass}`)}>
          {agent.name}
        </span>
        <span className="text-xs text-zinc-500 bg-zinc-800 px-2 py-0.5 rounded">
          {agent.type}
        </span>
        {agent.background && (
          <span className="text-xs text-zinc-500 bg-zinc-800 px-2 py-0.5 rounded">
            background
          </span>
        )}
      </div>

      <p className="text-sm text-zinc-400 mb-2">
        {agent.description || 'No description'}
      </p>

      <div className="text-xs text-zinc-500">
        Model: {agent.model}
      </div>
    </div>
  )
}
