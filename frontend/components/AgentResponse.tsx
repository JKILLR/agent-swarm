'use client'

import { useState } from 'react'
import { ChevronDown, ChevronUp } from 'lucide-react'
import { cn, getAgentColor, getAgentIcon } from '@/lib/utils'

interface AgentResponseProps {
  agent: string
  agentType: string
  content: string
  status?: 'thinking' | 'complete'
  isCollapsible?: boolean
}

export default function AgentResponse({
  agent,
  agentType,
  content,
  status = 'complete',
  isCollapsible = true,
}: AgentResponseProps) {
  const [isExpanded, setIsExpanded] = useState(true)
  const colorClass = getAgentColor(agentType)
  const icon = getAgentIcon(agentType)
  const shouldCollapse = isCollapsible && content.length > 500

  return (
    <div
      className={cn(
        'rounded-lg border-l-4 bg-zinc-900/50 overflow-hidden animate-slide-up',
        `border-${colorClass}`
      )}
      style={{ borderLeftColor: `var(--color-${colorClass}, #6b7280)` }}
    >
      {/* Header */}
      <div
        className={cn(
          'flex items-center justify-between px-4 py-3 border-b border-zinc-800',
          shouldCollapse && 'cursor-pointer hover:bg-zinc-800/50'
        )}
        onClick={() => shouldCollapse && setIsExpanded(!isExpanded)}
      >
        <div className="flex items-center gap-2">
          <span className="text-lg">{icon}</span>
          <span className={cn('font-medium', `text-${colorClass}`)}>
            {agent}
          </span>
          {status === 'thinking' && (
            <span className="flex items-center gap-1 text-zinc-500 text-sm">
              <span className="thinking-dot">.</span>
              <span className="thinking-dot">.</span>
              <span className="thinking-dot">.</span>
            </span>
          )}
        </div>

        {shouldCollapse && (
          <button className="text-zinc-500 hover:text-zinc-300">
            {isExpanded ? (
              <ChevronUp className="w-4 h-4" />
            ) : (
              <ChevronDown className="w-4 h-4" />
            )}
          </button>
        )}
      </div>

      {/* Content */}
      {status === 'thinking' ? (
        <div className="px-4 py-6 flex items-center justify-center">
          <div className="flex items-center gap-2 text-zinc-500">
            <div className="w-2 h-2 rounded-full bg-zinc-500 animate-pulse" />
            <div className="w-2 h-2 rounded-full bg-zinc-500 animate-pulse delay-75" />
            <div className="w-2 h-2 rounded-full bg-zinc-500 animate-pulse delay-150" />
          </div>
        </div>
      ) : (
        <div
          className={cn(
            'px-4 py-3 text-sm text-zinc-300 whitespace-pre-wrap transition-all duration-300',
            !isExpanded && 'max-h-24 overflow-hidden relative'
          )}
        >
          {content}
          {!isExpanded && (
            <div className="absolute bottom-0 left-0 right-0 h-12 bg-gradient-to-t from-zinc-900/90 to-transparent" />
          )}
        </div>
      )}
    </div>
  )
}
