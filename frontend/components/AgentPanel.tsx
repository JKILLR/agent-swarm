'use client'

import { type Agent } from '@/lib/api'
import { cn, getAgentColor, getAgentIcon } from '@/lib/utils'
import { useAgentActivity } from '@/lib/AgentActivityContext'
import { Loader2 } from 'lucide-react'

interface AgentPanelProps {
  agent: Agent
  swarmName?: string
}

export default function AgentPanel({ agent, swarmName }: AgentPanelProps) {
  const colorClass = getAgentColor(agent.type)
  const icon = getAgentIcon(agent.type)
  const { getAgentActivity } = useAgentActivity()

  // Get activity for this agent (try common swarm name patterns)
  const activity = swarmName
    ? getAgentActivity(swarmName, agent.name)
    : undefined

  const isActive = activity?.status === 'active'

  return (
    <div
      className={cn(
        'p-4 bg-zinc-900 border border-zinc-800 rounded-lg transition-all',
        `border-l-4 border-l-${colorClass}`,
        isActive && 'ring-2 ring-amber-500/50 bg-amber-500/5'
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

        {/* Activity indicator */}
        {isActive && (
          <span className="ml-auto flex items-center gap-1 text-xs text-amber-400">
            <Loader2 className="w-3 h-3 animate-spin" />
            Working
          </span>
        )}
      </div>

      <p className="text-sm text-zinc-400 mb-2">
        {agent.description || 'No description'}
      </p>

      {/* Show current task if active */}
      {isActive && activity?.currentTask && (
        <div className="text-xs text-amber-400/80 mb-2 bg-amber-500/10 px-2 py-1 rounded">
          {activity.currentTask}
        </div>
      )}

      <div className="text-xs text-zinc-500">
        Model: {agent.model}
      </div>
    </div>
  )
}
