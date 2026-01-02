'use client'

import { useState } from 'react'
import { ChevronDown, ChevronUp, Brain } from 'lucide-react'
import { cn, getAgentColor, getAgentIcon } from '@/lib/utils'

interface AgentResponseProps {
  agent: string
  agentType: string
  content: string
  status?: 'thinking' | 'complete'
  isCollapsible?: boolean
  thinking?: string
  isThinking?: boolean
}

export default function AgentResponse({
  agent,
  agentType,
  content,
  status = 'complete',
  isCollapsible = true,
  thinking,
  isThinking = false,
}: AgentResponseProps) {
  const [isExpanded, setIsExpanded] = useState(true)
  const [isThinkingExpanded, setIsThinkingExpanded] = useState(true)
  const colorClass = getAgentColor(agentType)
  const icon = getAgentIcon(agentType)
  const shouldCollapse = isCollapsible && content.length > 500
  const hasThinking = thinking && thinking.length > 0

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
          {status === 'thinking' && !content && !isThinking && (
            <span className="flex items-center gap-1 text-zinc-500 text-sm">
              <span className="thinking-dot">.</span>
              <span className="thinking-dot">.</span>
              <span className="thinking-dot">.</span>
            </span>
          )}
          {isThinking && (
            <span className="flex items-center gap-1 text-purple-400 text-sm">
              <Brain className="w-3 h-3 animate-pulse" />
              <span>Thinking</span>
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

      {/* Thinking Section */}
      {(hasThinking || isThinking) && (
        <div className="border-b border-zinc-800">
          <button
            onClick={() => setIsThinkingExpanded(!isThinkingExpanded)}
            className="w-full px-4 py-2 flex items-center justify-between text-xs text-purple-400 hover:bg-zinc-800/30 transition-colors"
          >
            <div className="flex items-center gap-2">
              <Brain className="w-3 h-3" />
              <span>
                {isThinking ? 'Thinking...' : `Thinking (${thinking?.length || 0} chars)`}
              </span>
              {isThinking && (
                <span className="flex items-center gap-0.5">
                  <span className="w-1 h-1 rounded-full bg-purple-400 animate-pulse" />
                  <span className="w-1 h-1 rounded-full bg-purple-400 animate-pulse" style={{ animationDelay: '150ms' }} />
                  <span className="w-1 h-1 rounded-full bg-purple-400 animate-pulse" style={{ animationDelay: '300ms' }} />
                </span>
              )}
            </div>
            {isThinkingExpanded ? (
              <ChevronUp className="w-3 h-3" />
            ) : (
              <ChevronDown className="w-3 h-3" />
            )}
          </button>
          {isThinkingExpanded && (
            <div className="px-4 py-3 bg-purple-950/20 text-xs text-purple-300/70 whitespace-pre-wrap max-h-64 overflow-y-auto font-mono">
              {thinking || '...'}
            </div>
          )}
        </div>
      )}

      {/* Content */}
      {status === 'thinking' && !content && !hasThinking && !isThinking ? (
        <div className="px-4 py-6 flex items-center justify-center">
          <div className="flex items-center gap-2 text-zinc-500">
            <div className="w-2 h-2 rounded-full bg-zinc-500 animate-pulse" />
            <div className="w-2 h-2 rounded-full bg-zinc-500 animate-pulse" style={{ animationDelay: '75ms' }} />
            <div className="w-2 h-2 rounded-full bg-zinc-500 animate-pulse" style={{ animationDelay: '150ms' }} />
          </div>
        </div>
      ) : content ? (
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
      ) : isThinking ? (
        <div className="px-4 py-3 text-sm text-zinc-500 italic">
          Waiting for response...
        </div>
      ) : null}
    </div>
  )
}
