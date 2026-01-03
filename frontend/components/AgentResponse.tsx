'use client'

import { useState, useMemo } from 'react'
import { ChevronDown, ChevronUp, Brain } from 'lucide-react'
import { cn, getAgentColor, getAgentIcon } from '@/lib/utils'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

interface AgentResponseProps {
  agent: string
  agentType: string
  content: string
  status?: 'thinking' | 'complete'
  isCollapsible?: boolean
  thinking?: string
  isThinking?: boolean
}

// Highlight CEO decision blocks
function highlightDecisions(content: string): React.ReactNode[] {
  const parts = content.split(/([\s\S]*?CEO DECISION[\s\S]*?)(?=\n\n|$)/gi)

  return parts.map((part, index) => {
    if (part.toLowerCase().includes('ceo decision')) {
      return (
        <div
          key={index}
          className="my-4 p-4 rounded-lg border border-orange-500/50 bg-orange-500/5 text-orange-200"
        >
          <ReactMarkdown remarkPlugins={[remarkGfm]} className="prose prose-invert prose-sm max-w-none prose-orange">
            {part}
          </ReactMarkdown>
        </div>
      )
    }
    return (
      <ReactMarkdown
        key={index}
        remarkPlugins={[remarkGfm]}
        className="prose prose-invert prose-sm max-w-none prose-zinc"
      >
        {part}
      </ReactMarkdown>
    )
  })
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

  // Use orange for orchestrator/supreme types, otherwise use agent color
  const isOrchestratorType = agentType.toLowerCase() === 'orchestrator' || agentType.toLowerCase() === 'supreme'
  const borderColor = isOrchestratorType ? '#ea580c' : `var(--color-${colorClass}, #6b7280)`

  return (
    <div
      className={cn(
        'rounded-lg border-l-2 bg-zinc-900/30 overflow-hidden animate-slide-up'
      )}
      style={{ borderLeftColor: borderColor }}
    >
      {/* Header */}
      <div
        className={cn(
          'flex items-center justify-between px-4 py-3 border-b border-zinc-800/50',
          shouldCollapse && 'cursor-pointer hover:bg-zinc-800/30'
        )}
        onClick={() => shouldCollapse && setIsExpanded(!isExpanded)}
      >
        <div className="flex items-center gap-2">
          <span className="text-lg">{icon}</span>
          <span className={cn(
            'font-medium',
            isOrchestratorType ? 'text-orange-500' : `text-${colorClass}`
          )} style={{ color: isOrchestratorType ? '#ea580c' : undefined }}>
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
            <span className="flex items-center gap-1 text-violet-400 text-sm">
              <Brain className="w-3 h-3 animate-pulse" />
              <span>Thinking</span>
            </span>
          )}
        </div>

        {shouldCollapse && (
          <button className="text-zinc-600 hover:text-zinc-400">
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
        <div className="border-b border-zinc-800/50">
          <button
            onClick={() => setIsThinkingExpanded(!isThinkingExpanded)}
            className="w-full px-4 py-2 flex items-center justify-between text-xs text-violet-400/80 hover:bg-violet-500/5 transition-all duration-200"
          >
            <div className="flex items-center gap-2">
              <Brain className="w-3 h-3" />
              <span>
                {isThinking ? 'Thinking...' : `Thinking (${thinking?.length || 0} chars)`}
              </span>
              {isThinking && (
                <span className="flex items-center gap-0.5">
                  <span className="w-1 h-1 rounded-full bg-violet-400 animate-pulse" />
                  <span className="w-1 h-1 rounded-full bg-violet-400 animate-pulse" style={{ animationDelay: '150ms' }} />
                  <span className="w-1 h-1 rounded-full bg-violet-400 animate-pulse" style={{ animationDelay: '300ms' }} />
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
            <div className="px-4 py-3 bg-violet-950/10 text-xs text-violet-300/60 whitespace-pre-wrap max-h-64 overflow-y-auto font-mono border-l-2 border-violet-500/30">
              {thinking || '...'}
            </div>
          )}
        </div>
      )}

      {/* Content */}
      {status === 'thinking' && !content && !hasThinking && !isThinking ? (
        <div className="px-4 py-6 flex items-center justify-center">
          <div className="flex items-center gap-2 text-zinc-600">
            <div className="w-1.5 h-1.5 rounded-full bg-zinc-600 animate-pulse" />
            <div className="w-1.5 h-1.5 rounded-full bg-zinc-600 animate-pulse" style={{ animationDelay: '75ms' }} />
            <div className="w-1.5 h-1.5 rounded-full bg-zinc-600 animate-pulse" style={{ animationDelay: '150ms' }} />
          </div>
        </div>
      ) : content ? (
        <div
          className={cn(
            'px-4 py-3 text-sm text-zinc-300 transition-all duration-300',
            !isExpanded && 'max-h-24 overflow-hidden relative'
          )}
        >
          {highlightDecisions(content)}
          {!isExpanded && (
            <div className="absolute bottom-0 left-0 right-0 h-12 bg-gradient-to-t from-zinc-900/90 to-transparent" />
          )}
        </div>
      ) : isThinking ? (
        <div className="px-4 py-3 text-sm text-zinc-600 italic">
          Waiting for response...
        </div>
      ) : null}
    </div>
  )
}
