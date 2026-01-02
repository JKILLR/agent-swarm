'use client'

import { useState, useEffect, useRef } from 'react'
import {
  Loader2,
  Check,
  X,
  Zap,
  Search,
  FileText,
  Terminal,
  Globe,
  GitBranch,
  Users,
  Bot,
  ChevronRight,
  ChevronDown,
  Activity,
  Clock,
  Brain,
} from 'lucide-react'

export interface AgentActivity {
  id: string
  name: string
  status: 'thinking' | 'working' | 'delegating' | 'complete' | 'error'
  description?: string
  startTime: Date
  endTime?: Date
}

export interface ToolActivity {
  id: string
  tool: string
  description: string
  status: 'running' | 'complete' | 'error'
  timestamp: Date
  endTime?: Date
  summary?: string
  agentName?: string
}

interface ActivityPanelProps {
  agents: AgentActivity[]
  tools: ToolActivity[]
  isProcessing: boolean
  onClear?: () => void
}

const toolIcons: Record<string, React.ReactNode> = {
  Task: <Users className="w-3.5 h-3.5" />,
  Agent: <Bot className="w-3.5 h-3.5 text-purple-400" />,
  Read: <FileText className="w-3.5 h-3.5" />,
  Write: <FileText className="w-3.5 h-3.5" />,
  Edit: <FileText className="w-3.5 h-3.5" />,
  Bash: <Terminal className="w-3.5 h-3.5" />,
  Glob: <Search className="w-3.5 h-3.5" />,
  Grep: <Search className="w-3.5 h-3.5" />,
  WebSearch: <Globe className="w-3.5 h-3.5" />,
  WebFetch: <Globe className="w-3.5 h-3.5" />,
  GitCommit: <GitBranch className="w-3.5 h-3.5" />,
  GitSync: <GitBranch className="w-3.5 h-3.5" />,
  GitStatus: <GitBranch className="w-3.5 h-3.5" />,
}

function formatDuration(start: Date, end?: Date): string {
  const endTime = end || new Date()
  const seconds = Math.floor((endTime.getTime() - start.getTime()) / 1000)
  if (seconds < 60) return `${seconds}s`
  const minutes = Math.floor(seconds / 60)
  const remainingSeconds = seconds % 60
  return `${minutes}m ${remainingSeconds}s`
}

function ElapsedTime({ start, end }: { start: Date; end?: Date }) {
  const [elapsed, setElapsed] = useState(formatDuration(start, end))

  useEffect(() => {
    if (end) {
      setElapsed(formatDuration(start, end))
      return
    }

    const interval = setInterval(() => {
      setElapsed(formatDuration(start))
    }, 1000)

    return () => clearInterval(interval)
  }, [start, end])

  return (
    <span className="flex items-center gap-1 text-zinc-500">
      <Clock className="w-3 h-3" />
      {elapsed}
    </span>
  )
}

function StatusIndicator({
  status,
}: {
  status: 'thinking' | 'working' | 'delegating' | 'complete' | 'error' | 'running'
}) {
  switch (status) {
    case 'thinking':
      return (
        <div className="flex items-center gap-1">
          <Brain className="w-3.5 h-3.5 text-blue-400 animate-pulse" />
          <span className="text-blue-400 text-xs">Thinking</span>
        </div>
      )
    case 'working':
    case 'running':
      return (
        <div className="flex items-center gap-1">
          <Loader2 className="w-3.5 h-3.5 text-amber-500 animate-spin" />
          <span className="text-amber-500 text-xs">Working</span>
        </div>
      )
    case 'delegating':
      return (
        <div className="flex items-center gap-1">
          <Users className="w-3.5 h-3.5 text-purple-400 animate-pulse" />
          <span className="text-purple-400 text-xs">Delegating</span>
        </div>
      )
    case 'complete':
      return (
        <div className="flex items-center gap-1">
          <Check className="w-3.5 h-3.5 text-green-500" />
          <span className="text-green-500 text-xs">Done</span>
        </div>
      )
    case 'error':
      return (
        <div className="flex items-center gap-1">
          <X className="w-3.5 h-3.5 text-red-500" />
          <span className="text-red-500 text-xs">Error</span>
        </div>
      )
    default:
      return null
  }
}

export default function ActivityPanel({
  agents,
  tools,
  isProcessing,
  onClear,
}: ActivityPanelProps) {
  const [isExpanded, setIsExpanded] = useState(true)
  const [showAgents, setShowAgents] = useState(true)
  const [showTools, setShowTools] = useState(true)
  const scrollRef = useRef<HTMLDivElement>(null)

  // Auto-scroll to bottom when new items added
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [tools.length])

  const activeAgents = agents.filter((a) => a.status !== 'complete' && a.status !== 'error')
  const activeTools = tools.filter((t) => t.status === 'running')
  const totalActive = activeAgents.length + activeTools.length

  // Don't show if nothing happening and no history
  if (!isProcessing && agents.length === 0 && tools.length === 0) {
    return null
  }

  return (
    <div className="bg-zinc-900/95 border border-zinc-700 rounded-lg backdrop-blur shadow-xl overflow-hidden">
      {/* Header */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full px-3 py-2 flex items-center justify-between bg-zinc-800/50 hover:bg-zinc-800 transition-colors"
      >
        <div className="flex items-center gap-2">
          <Activity className="w-4 h-4 text-amber-500" />
          <span className="text-sm font-medium text-zinc-300">Activity</span>
          {totalActive > 0 && (
            <span className="px-1.5 py-0.5 rounded-full bg-amber-500/20 text-amber-400 text-xs font-medium">
              {totalActive} active
            </span>
          )}
          {isProcessing && totalActive === 0 && (
            <span className="flex items-center gap-1 text-xs text-blue-400">
              <Loader2 className="w-3 h-3 animate-spin" />
              Processing
            </span>
          )}
        </div>
        {isExpanded ? (
          <ChevronDown className="w-4 h-4 text-zinc-500" />
        ) : (
          <ChevronRight className="w-4 h-4 text-zinc-500" />
        )}
      </button>

      {isExpanded && (
        <div ref={scrollRef} className="max-h-64 overflow-y-auto">
          {/* Active Agents Section */}
          {agents.length > 0 && (
            <div className="border-b border-zinc-800">
              <button
                onClick={() => setShowAgents(!showAgents)}
                className="w-full px-3 py-1.5 flex items-center gap-2 text-xs text-zinc-400 hover:bg-zinc-800/30"
              >
                {showAgents ? (
                  <ChevronDown className="w-3 h-3" />
                ) : (
                  <ChevronRight className="w-3 h-3" />
                )}
                <Bot className="w-3 h-3 text-purple-400" />
                <span>Agents ({agents.length})</span>
              </button>

              {showAgents && (
                <div className="divide-y divide-zinc-800/50">
                  {agents.map((agent) => (
                    <div
                      key={agent.id}
                      className={`px-3 py-2 flex items-start justify-between gap-2 ${
                        agent.status !== 'complete' && agent.status !== 'error'
                          ? 'bg-purple-500/5 border-l-2 border-l-purple-500'
                          : 'border-l-2 border-l-transparent'
                      }`}
                    >
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2">
                          <Bot className="w-3.5 h-3.5 text-purple-400 flex-shrink-0" />
                          <span className="text-sm font-medium text-zinc-200 truncate">
                            {agent.name}
                          </span>
                        </div>
                        {agent.description && (
                          <p className="mt-0.5 text-xs text-zinc-500 truncate pl-5">
                            {agent.description}
                          </p>
                        )}
                      </div>
                      <div className="flex flex-col items-end gap-1 flex-shrink-0">
                        <StatusIndicator status={agent.status} />
                        <ElapsedTime start={agent.startTime} end={agent.endTime} />
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Tools Section */}
          {tools.length > 0 && (
            <div>
              <button
                onClick={() => setShowTools(!showTools)}
                className="w-full px-3 py-1.5 flex items-center gap-2 text-xs text-zinc-400 hover:bg-zinc-800/30"
              >
                {showTools ? (
                  <ChevronDown className="w-3 h-3" />
                ) : (
                  <ChevronRight className="w-3 h-3" />
                )}
                <Zap className="w-3 h-3 text-amber-400" />
                <span>Tools ({tools.length})</span>
              </button>

              {showTools && (
                <div className="divide-y divide-zinc-800/50">
                  {tools.slice(-20).map((tool) => (
                    <div
                      key={tool.id}
                      className={`px-3 py-2 flex items-start gap-2 text-xs ${
                        tool.status === 'running'
                          ? 'bg-amber-500/5 border-l-2 border-l-amber-500'
                          : tool.status === 'error'
                            ? 'bg-red-500/5 border-l-2 border-l-red-500'
                            : 'border-l-2 border-l-green-500/30'
                      }`}
                    >
                      {/* Status indicator */}
                      <div className="flex-shrink-0 mt-0.5">
                        {tool.status === 'running' ? (
                          <Loader2 className="w-3 h-3 text-amber-500 animate-spin" />
                        ) : tool.status === 'error' ? (
                          <X className="w-3 h-3 text-red-500" />
                        ) : (
                          <Check className="w-3 h-3 text-green-500" />
                        )}
                      </div>

                      {/* Tool icon */}
                      <div className="flex-shrink-0 mt-0.5 text-zinc-500">
                        {toolIcons[tool.tool] || <Zap className="w-3 h-3" />}
                      </div>

                      {/* Content */}
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2">
                          <span className="font-medium text-zinc-300">{tool.tool}</span>
                          {tool.agentName && (
                            <span className="text-zinc-600">via {tool.agentName}</span>
                          )}
                        </div>
                        <p className="text-zinc-500 truncate">{tool.description}</p>
                        {tool.summary && tool.status !== 'running' && (
                          <p className="mt-0.5 text-zinc-600 truncate">{tool.summary}</p>
                        )}
                      </div>

                      {/* Time */}
                      <div className="flex-shrink-0 text-zinc-600">
                        <ElapsedTime start={tool.timestamp} end={tool.endTime} />
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Empty state while processing */}
          {isProcessing && agents.length === 0 && tools.length === 0 && (
            <div className="px-3 py-4 flex items-center justify-center gap-2 text-sm text-zinc-500">
              <Loader2 className="w-4 h-4 animate-spin text-blue-400" />
              <span>Initializing...</span>
            </div>
          )}
        </div>
      )}

      {/* Footer with clear button */}
      {isExpanded && !isProcessing && (agents.length > 0 || tools.length > 0) && onClear && (
        <div className="px-3 py-1.5 border-t border-zinc-800 flex justify-end">
          <button
            onClick={onClear}
            className="text-xs text-zinc-500 hover:text-zinc-300 transition-colors"
          >
            Clear history
          </button>
        </div>
      )}
    </div>
  )
}
