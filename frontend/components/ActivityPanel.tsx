'use client'

import { useState, useEffect, useRef } from 'react'
import { createPortal } from 'react-dom'
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
  Maximize2,
  Minimize2,
  Bell,
  FileEdit,
  Eye,
  Pencil,
} from 'lucide-react'
import { type PanelAgentActivity, type PanelToolActivity } from '@/lib/AgentActivityContext'

// Re-export types for backwards compatibility
export type AgentActivity = PanelAgentActivity
export type ToolActivity = PanelToolActivity

interface ActivityPanelProps {
  agents: AgentActivity[]
  tools: ToolActivity[]
  isProcessing: boolean
  onClear?: () => void
  onComplete?: () => void // Callback when work completes
}

const toolIcons: Record<string, React.ReactNode> = {
  Task: <Users className="w-3.5 h-3.5" />,
  Agent: <Bot className="w-3.5 h-3.5 text-emerald-400" />,
  Read: <Eye className="w-3.5 h-3.5 text-blue-400" />,
  Write: <Pencil className="w-3.5 h-3.5 text-violet-400" />,
  Edit: <FileEdit className="w-3.5 h-3.5 text-yellow-400" />,
  Bash: <Terminal className="w-3.5 h-3.5 text-purple-400" />,
  Glob: <Search className="w-3.5 h-3.5" />,
  Grep: <Search className="w-3.5 h-3.5" />,
  WebSearch: <Globe className="w-3.5 h-3.5 text-cyan-400" />,
  WebFetch: <Globe className="w-3.5 h-3.5 text-cyan-400" />,
  GitCommit: <GitBranch className="w-3.5 h-3.5 text-emerald-400" />,
  GitSync: <GitBranch className="w-3.5 h-3.5 text-emerald-400" />,
  GitStatus: <GitBranch className="w-3.5 h-3.5 text-emerald-400" />,
}

// Get file path from tool description
function extractFilePath(description: string): string | null {
  // Match common patterns like "Reading /path/to/file" or "Writing to /path/to/file"
  const patterns = [
    /(?:Reading|Writing to|Editing)\s+([^\s]+)/i,
    /file_path['":\s]+([^\s'"]+)/i,
    /([/\\][^\s]+\.[a-z]+)/i,
  ]

  for (const pattern of patterns) {
    const match = description.match(pattern)
    if (match) {
      // Return just the filename for display
      const fullPath = match[1]
      const parts = fullPath.split(/[/\\]/)
      return parts[parts.length - 1]
    }
  }
  return null
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
    <span className="flex items-center gap-1 text-zinc-600">
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
          <Brain className="w-3.5 h-3.5 text-violet-400 animate-pulse" />
          <span className="text-violet-400 text-xs">Thinking</span>
        </div>
      )
    case 'working':
    case 'running':
      return (
        <div className="flex items-center gap-1">
          <Loader2 className="w-3.5 h-3.5 text-emerald-400 animate-spin" />
          <span className="text-emerald-400 text-xs">Working</span>
        </div>
      )
    case 'delegating':
      return (
        <div className="flex items-center gap-1">
          <Users className="w-3.5 h-3.5 text-violet-400 animate-pulse" />
          <span className="text-violet-400 text-xs">Delegating</span>
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

// Animated progress bar for running tools
function ToolProgressBar() {
  return (
    <div className="h-0.5 bg-zinc-800 rounded-full overflow-hidden mt-1">
      <div className="h-full bg-gradient-to-r from-emerald-400 via-violet-500 to-emerald-400 animate-progress-bar" />
    </div>
  )
}

// File activity indicator
function FileActivity({ tool, description }: { tool: string; description: string }) {
  const fileName = extractFilePath(description)
  if (!fileName) return null

  const isRead = tool === 'Read' || tool === 'Glob' || tool === 'Grep'
  const isWrite = tool === 'Write'
  const isEdit = tool === 'Edit'

  return (
    <span className={`text-xs px-1.5 py-0.5 rounded ${
      isRead ? 'bg-blue-500/10 text-blue-400' :
      isWrite ? 'bg-violet-500/10 text-violet-400' :
      isEdit ? 'bg-yellow-500/10 text-yellow-400' :
      'bg-zinc-800 text-zinc-400'
    }`}>
      {fileName}
    </span>
  )
}

export default function ActivityPanel({
  agents,
  tools,
  isProcessing,
  onClear,
  onComplete,
}: ActivityPanelProps) {
  const [isExpanded, setIsExpanded] = useState(true)
  const [isFullscreen, setIsFullscreen] = useState(false)
  const [showAgents, setShowAgents] = useState(true)
  const [showTools, setShowTools] = useState(true)
  const [showNotification, setShowNotification] = useState(false)
  const [portalContainer, setPortalContainer] = useState<HTMLElement | null>(null)
  const scrollRef = useRef<HTMLDivElement>(null)
  const wasProcessingRef = useRef(isProcessing)
  const notificationRef = useRef<Notification | null>(null)

  // Set up portal container for fullscreen mode
  useEffect(() => {
    setPortalContainer(document.body)
  }, [])

  // Check if we're on mobile
  const [isMobile, setIsMobile] = useState(false)
  useEffect(() => {
    const checkMobile = () => setIsMobile(window.innerWidth < 768)
    checkMobile()
    window.addEventListener('resize', checkMobile)
    return () => window.removeEventListener('resize', checkMobile)
  }, [])

  // No auto-scroll - let users control scroll position manually

  // Detect completion and show notification
  useEffect(() => {
    let timeoutId: ReturnType<typeof setTimeout> | null = null

    if (wasProcessingRef.current && !isProcessing && (agents.length > 0 || tools.length > 0)) {
      // Work just completed
      setShowNotification(true)
      onComplete?.()

      // Browser notification - wrapped in try-catch for safety
      if ('Notification' in window && Notification.permission === 'granted') {
        try {
          // Close any existing notification before creating a new one
          if (notificationRef.current) {
            notificationRef.current.close()
            notificationRef.current = null
          }

          notificationRef.current = new Notification('Agent Work Complete', {
            body: `Completed ${tools.length} tool operations`,
            icon: '/favicon.ico',
          })
        } catch (e) {
          // Notification creation can fail (e.g., missing favicon, browser restrictions)
          console.debug('Failed to create notification:', e)
        }
      }

      // Hide notification after 5 seconds
      timeoutId = setTimeout(() => setShowNotification(false), 5000)
    }
    wasProcessingRef.current = isProcessing

    // Cleanup timeout and notification on unmount or re-run
    return () => {
      if (timeoutId) {
        clearTimeout(timeoutId)
      }
      if (notificationRef.current) {
        notificationRef.current.close()
        notificationRef.current = null
      }
    }
  }, [isProcessing, agents.length, tools.length, onComplete])

  // Request notification permission on mount
  useEffect(() => {
    if ('Notification' in window && Notification.permission === 'default') {
      Notification.requestPermission()
    }
  }, [])

  // Keyboard support for fullscreen mode (Escape to close)
  useEffect(() => {
    if (!isFullscreen) return

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setIsFullscreen(false)
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [isFullscreen])

  const activeAgents = agents.filter((a) => a.status !== 'complete' && a.status !== 'error')
  const activeTools = tools.filter((t) => t.status === 'running')
  const totalActive = activeAgents.length + activeTools.length

  // Group tools by agent for better visualization
  const toolsByAgent = tools.reduce((acc, tool) => {
    const agent = tool.agentName || 'COO'
    if (!acc[agent]) acc[agent] = []
    acc[agent].push(tool)
    return acc
  }, {} as Record<string, typeof tools>)

  // Get modified files
  const modifiedFiles = tools
    .filter(t => ['Write', 'Edit'].includes(t.tool))
    .map(t => extractFilePath(t.description))
    .filter(Boolean)

  // Don't show if nothing happening and no history
  if (!isProcessing && agents.length === 0 && tools.length === 0) {
    return null
  }

  const panelContent = (
    <>
      {/* Header */}
      <div
        className={`flex items-center justify-between ${isFullscreen ? 'px-4 py-3 border-b border-zinc-800/50' : 'px-3 py-2.5 md:py-2 bg-zinc-900/30'} transition-all duration-200`}
      >
        <button
          onClick={() => !isFullscreen && setIsExpanded(!isExpanded)}
          className="flex items-center gap-2 flex-1"
        >
          <Activity className="w-4 h-4 text-emerald-400" />
          <span className="text-sm font-medium text-zinc-400">Activity</span>
          {totalActive > 0 && (
            <span className="px-1.5 py-0.5 rounded-full bg-violet-500/20 text-violet-400 text-xs font-medium animate-pulse">
              {totalActive} active
            </span>
          )}
          {isProcessing && totalActive === 0 && (
            <span className="flex items-center gap-1 text-xs text-emerald-400">
              <Loader2 className="w-3 h-3 animate-spin" />
              Processing
            </span>
          )}
          {showNotification && !isProcessing && (
            <span className="flex items-center gap-1 text-xs text-green-400 animate-pulse">
              <Bell className="w-3 h-3" />
              Complete!
            </span>
          )}
        </button>

        <div className="flex items-center gap-2 flex-shrink-0">
          {/* Modified files indicator - subtle dot with count */}
          {modifiedFiles.length > 0 && (
            <span className="flex items-center gap-1 text-yellow-500/70" title={`${modifiedFiles.length} files modified`}>
              <FileEdit className="w-3 h-3" />
              <span className="text-[10px]">{modifiedFiles.length}</span>
            </span>
          )}

          {/* Fullscreen toggle - prominent when processing */}
          <button
            onClick={() => setIsFullscreen(!isFullscreen)}
            className={`p-1.5 rounded transition-all ${
              isProcessing
                ? 'bg-emerald-400/20 hover:bg-emerald-400/30 animate-pulse'
                : 'hover:bg-zinc-800/50'
            }`}
            title={isFullscreen ? 'Minimize' : 'Watch activity live'}
          >
            {isFullscreen ? (
              <Minimize2 className="w-4 h-4 text-zinc-400" />
            ) : (
              <Maximize2 className={`w-4 h-4 ${isProcessing ? 'text-emerald-400' : 'text-zinc-500'}`} />
            )}
          </button>

          {!isFullscreen && (
            isExpanded ? (
              <ChevronDown className="w-4 h-4 text-zinc-600" />
            ) : (
              <ChevronRight className="w-4 h-4 text-zinc-600" />
            )
          )}
        </div>
      </div>

      {(isExpanded || isFullscreen) && (
        <div
          ref={scrollRef}
          className={`overflow-y-auto ${isFullscreen ? 'flex-1' : 'max-h-48 md:max-h-64'}`}
        >
          {/* Active Agents Section - Show all agents with their current status */}
          {agents.length > 0 && (
            <div className="border-b border-zinc-800/50">
              <button
                onClick={() => setShowAgents(!showAgents)}
                className="w-full px-3 py-2 md:py-1.5 flex items-center gap-2 text-xs text-zinc-500 hover:bg-zinc-800/20 active:bg-zinc-800/30 touch-manipulation min-h-[40px] md:min-h-0"
              >
                {showAgents ? (
                  <ChevronDown className="w-3 h-3" />
                ) : (
                  <ChevronRight className="w-3 h-3" />
                )}
                <Bot className="w-3 h-3 text-emerald-400" />
                <span>Agents ({agents.length})</span>
                {activeAgents.length > 0 && (
                  <span className="ml-auto text-xs text-emerald-400">
                    {activeAgents.length} working
                  </span>
                )}
              </button>

              {showAgents && (
                <div className={`divide-y divide-zinc-800/30 ${isFullscreen ? 'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2 p-2 divide-y-0' : ''}`}>
                  {agents.map((agent) => (
                    <div
                      key={agent.id}
                      className={`px-3 py-2 flex items-start justify-between gap-2 ${
                        isFullscreen ? 'bg-zinc-900/50 rounded-lg border border-zinc-800/50' : ''
                      } ${
                        agent.status !== 'complete' && agent.status !== 'error'
                          ? 'bg-emerald-400/5 border-l-2 border-l-emerald-400'
                          : 'border-l-2 border-l-transparent'
                      }`}
                    >
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2">
                          <Bot className="w-3.5 h-3.5 text-emerald-400 flex-shrink-0" />
                          <span className="text-sm font-medium text-zinc-300 truncate">
                            {agent.name}
                          </span>
                        </div>
                        {agent.description && (
                          <p className="mt-0.5 text-xs text-zinc-600 truncate pl-5">
                            {agent.description}
                          </p>
                        )}
                        {agent.status === 'working' && (
                          <ToolProgressBar />
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

          {/* Tools Section - Grouped by agent in fullscreen mode */}
          {tools.length > 0 && (
            <div>
              <button
                onClick={() => setShowTools(!showTools)}
                className="w-full px-3 py-2 md:py-1.5 flex items-center gap-2 text-xs text-zinc-500 hover:bg-zinc-800/20 active:bg-zinc-800/30 touch-manipulation min-h-[40px] md:min-h-0"
              >
                {showTools ? (
                  <ChevronDown className="w-3 h-3" />
                ) : (
                  <ChevronRight className="w-3 h-3" />
                )}
                <Zap className="w-3 h-3 text-emerald-400" />
                <span>Tools ({tools.length})</span>
                {activeTools.length > 0 && (
                  <span className="ml-auto text-xs text-emerald-400">
                    {activeTools.length} running
                  </span>
                )}
              </button>

              {showTools && (
                isFullscreen ? (
                  // Fullscreen: Group by agent
                  <div className="p-2 space-y-4">
                    {Object.entries(toolsByAgent).map(([agentName, agentTools]) => (
                      <div key={agentName} className="bg-zinc-900/50 rounded-lg border border-zinc-800/50 overflow-hidden">
                        <div className="px-3 py-2 bg-zinc-900/80 border-b border-zinc-800/50 flex items-center gap-2">
                          <Bot className="w-3 h-3 text-emerald-400" />
                          <span className="text-xs font-medium text-zinc-400">{agentName}</span>
                          <span className="text-xs text-zinc-600">({agentTools.length} tools)</span>
                        </div>
                        <div className="divide-y divide-zinc-800/30 max-h-64 overflow-y-auto">
                          {agentTools.map((tool) => (
                            <div
                              key={tool.id}
                              className={`px-3 py-2 flex items-start gap-2 text-xs ${
                                tool.status === 'running'
                                  ? 'bg-emerald-400/5 border-l-2 border-l-emerald-400'
                                  : tool.status === 'error'
                                    ? 'bg-red-500/5 border-l-2 border-l-red-500'
                                    : 'border-l-2 border-l-green-500/30'
                              }`}
                            >
                              <div className="flex-shrink-0 mt-0.5">
                                {tool.status === 'running' ? (
                                  <Loader2 className="w-3 h-3 text-emerald-400 animate-spin" />
                                ) : tool.status === 'error' ? (
                                  <X className="w-3 h-3 text-red-500" />
                                ) : (
                                  <Check className="w-3 h-3 text-green-500" />
                                )}
                              </div>
                              <div className="flex-shrink-0 mt-0.5 text-zinc-600">
                                {toolIcons[tool.tool] || <Zap className="w-3 h-3" />}
                              </div>
                              <div className="flex-1 min-w-0">
                                <div className="flex items-center gap-2 flex-wrap">
                                  <span className="font-medium text-zinc-400">{tool.tool}</span>
                                  <FileActivity tool={tool.tool} description={tool.description} />
                                </div>
                                <p className="text-zinc-600 truncate">{tool.description}</p>
                                {tool.status === 'running' && <ToolProgressBar />}
                              </div>
                              <div className="flex-shrink-0 text-zinc-600">
                                <ElapsedTime start={tool.timestamp} end={tool.endTime} />
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  // Compact: Simple list
                  <div className="divide-y divide-zinc-800/30">
                    {tools.slice(-20).map((tool) => (
                      <div
                        key={tool.id}
                        className={`px-3 py-2 flex items-start gap-2 text-xs ${
                          tool.status === 'running'
                            ? 'bg-emerald-400/5 border-l-2 border-l-emerald-400'
                            : tool.status === 'error'
                              ? 'bg-red-500/5 border-l-2 border-l-red-500'
                              : 'border-l-2 border-l-green-500/30'
                        }`}
                      >
                        {/* Status indicator */}
                        <div className="flex-shrink-0 mt-0.5">
                          {tool.status === 'running' ? (
                            <Loader2 className="w-3 h-3 text-emerald-400 animate-spin" />
                          ) : tool.status === 'error' ? (
                            <X className="w-3 h-3 text-red-500" />
                          ) : (
                            <Check className="w-3 h-3 text-green-500" />
                          )}
                        </div>

                        {/* Tool icon */}
                        <div className="flex-shrink-0 mt-0.5 text-zinc-600">
                          {toolIcons[tool.tool] || <Zap className="w-3 h-3" />}
                        </div>

                        {/* Content */}
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2 flex-wrap">
                            <span className="font-medium text-zinc-400">{tool.tool}</span>
                            {tool.agentName && (
                              <span className="text-zinc-600">via {tool.agentName}</span>
                            )}
                            <FileActivity tool={tool.tool} description={tool.description} />
                          </div>
                          <p className="text-zinc-600 truncate">{tool.description}</p>
                          {tool.summary && tool.status !== 'running' && (
                            <p className="mt-0.5 text-zinc-600 truncate">{tool.summary}</p>
                          )}
                          {tool.status === 'running' && <ToolProgressBar />}
                        </div>

                        {/* Time */}
                        <div className="flex-shrink-0 text-zinc-600">
                          <ElapsedTime start={tool.timestamp} end={tool.endTime} />
                        </div>
                      </div>
                    ))}
                  </div>
                )
              )}
            </div>
          )}

          {/* Empty state while processing */}
          {isProcessing && agents.length === 0 && tools.length === 0 && (
            <div className="px-3 py-4 flex items-center justify-center gap-2 text-sm text-zinc-600">
              <Loader2 className="w-4 h-4 animate-spin text-emerald-400" />
              <span>Initializing...</span>
            </div>
          )}
        </div>
      )}

      {/* Footer with clear button */}
      {(isExpanded || isFullscreen) && !isProcessing && (agents.length > 0 || tools.length > 0) && onClear && (
        <div className={`px-3 py-2 md:py-1.5 border-t border-zinc-800/50 flex ${isFullscreen ? 'justify-between' : 'justify-end'}`}>
          {isFullscreen && (
            <div className="text-xs text-zinc-600">
              {agents.length} agents • {tools.length} tools • {modifiedFiles.length} files modified
            </div>
          )}
          <button
            onClick={onClear}
            className="text-xs text-zinc-600 hover:text-zinc-400 active:text-zinc-300 transition-colors px-2 py-1 touch-manipulation min-h-[32px] md:min-h-0"
          >
            Clear history
          </button>
        </div>
      )}
    </>
  )

  // Fullscreen modal content
  const fullscreenContent = (
    <>
      {/* Overlay backdrop - covers entire screen including chat area */}
      <div
        className="fixed inset-0 z-[100] bg-black/80 backdrop-blur-sm"
        onClick={() => setIsFullscreen(false)}
      />
      {/* Fullscreen panel - centered modal that overlays the entire screen */}
      <div className="fixed inset-4 md:inset-12 lg:inset-16 z-[101] bg-[#0a0a0a] overflow-hidden flex flex-col rounded-xl border border-zinc-800/50 shadow-2xl shadow-black/50">
        {/* Panel Header */}
        <div className="px-4 py-3 border-b border-zinc-800/50 flex items-center justify-between bg-[#0d0d0d]">
          <div className="flex items-center gap-3">
            <Activity className={`w-5 h-5 ${isProcessing ? 'text-emerald-400 animate-pulse' : 'text-emerald-400/70'}`} />
            <span className="text-lg font-medium text-zinc-300">Live Activity Monitor</span>
            {isProcessing && totalActive > 0 && (
              <span className="px-2 py-1 rounded-full bg-violet-500/20 text-violet-400 text-sm font-medium animate-pulse">
                {totalActive} active
              </span>
            )}
            {!isProcessing && agents.length > 0 && (
              <span className="px-2 py-1 rounded-full bg-green-500/20 text-green-400 text-sm font-medium">
                Complete
              </span>
            )}
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xs text-zinc-600">Press ESC to close</span>
            <button
              onClick={() => setIsFullscreen(false)}
              className="p-2 hover:bg-zinc-800/50 rounded-lg transition-colors"
              title="Close (Escape)"
            >
              <X className="w-5 h-5 text-zinc-400" />
            </button>
          </div>
        </div>

        {/* Content Area */}
        <div ref={scrollRef} className="flex-1 overflow-auto p-4">
          {/* Active Agents Section */}
          {agents.length > 0 && (
            <div className="mb-6">
              <div className="flex items-center gap-2 mb-3">
                <Bot className="w-4 h-4 text-emerald-400" />
                <h3 className="text-sm font-medium text-zinc-400">Agents ({agents.length})</h3>
                {activeAgents.length > 0 && (
                  <span className="text-xs text-emerald-400 ml-auto">{activeAgents.length} working</span>
                )}
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                {agents.map((agent) => (
                  <div
                    key={agent.id}
                    className={`p-3 rounded-lg border ${
                      agent.status !== 'complete' && agent.status !== 'error'
                        ? 'bg-emerald-400/5 border-emerald-400/30'
                        : agent.status === 'error'
                          ? 'bg-red-500/5 border-red-500/30'
                          : 'bg-zinc-900/50 border-zinc-800/50'
                    }`}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-medium text-zinc-300 flex items-center gap-2">
                        <Bot className="w-3.5 h-3.5 text-emerald-400" />
                        {agent.name}
                      </span>
                      <StatusIndicator status={agent.status} />
                    </div>
                    {agent.description && (
                      <p className="text-xs text-zinc-600 mb-2 line-clamp-2">{agent.description}</p>
                    )}
                    <div className="flex items-center justify-between">
                      <ElapsedTime start={agent.startTime} end={agent.endTime} />
                      {agent.status === 'working' && <ToolProgressBar />}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Tools Section - Grouped by agent */}
          {tools.length > 0 && (
            <div>
              <div className="flex items-center gap-2 mb-3">
                <Zap className="w-4 h-4 text-emerald-400" />
                <h3 className="text-sm font-medium text-zinc-400">Tool Activity ({tools.length})</h3>
                {activeTools.length > 0 && (
                  <span className="text-xs text-emerald-400 ml-auto">{activeTools.length} running</span>
                )}
              </div>
              <div className="space-y-4">
                {Object.entries(toolsByAgent).map(([agentName, agentTools]) => (
                  <div key={agentName} className="bg-zinc-900/50 rounded-lg border border-zinc-800/50 overflow-hidden">
                    <div className="px-3 py-2 bg-zinc-900/80 border-b border-zinc-800/50 flex items-center gap-2">
                      <Bot className="w-3 h-3 text-emerald-400" />
                      <span className="text-sm font-medium text-zinc-400">{agentName}</span>
                      <span className="text-xs text-zinc-600">({agentTools.length} tools)</span>
                    </div>
                    <div className="divide-y divide-zinc-800/30 max-h-80 overflow-y-auto">
                      {agentTools.map((tool) => (
                        <div
                          key={tool.id}
                          className={`px-3 py-2 flex items-start gap-2 text-xs ${
                            tool.status === 'running'
                              ? 'bg-emerald-400/5 border-l-2 border-l-emerald-400'
                              : tool.status === 'error'
                                ? 'bg-red-500/5 border-l-2 border-l-red-500'
                                : 'border-l-2 border-l-green-500/30'
                          }`}
                        >
                          <div className="flex-shrink-0 mt-0.5">
                            {tool.status === 'running' ? (
                              <Loader2 className="w-3 h-3 text-emerald-400 animate-spin" />
                            ) : tool.status === 'error' ? (
                              <X className="w-3 h-3 text-red-500" />
                            ) : (
                              <Check className="w-3 h-3 text-green-500" />
                            )}
                          </div>
                          <div className="flex-shrink-0 mt-0.5 text-zinc-600">
                            {toolIcons[tool.tool] || <Zap className="w-3 h-3" />}
                          </div>
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-2 flex-wrap">
                              <span className="font-medium text-zinc-400">{tool.tool}</span>
                              <FileActivity tool={tool.tool} description={tool.description} />
                            </div>
                            <p className="text-zinc-600 truncate">{tool.description}</p>
                            {tool.status === 'running' && <ToolProgressBar />}
                          </div>
                          <div className="flex-shrink-0 text-zinc-600">
                            <ElapsedTime start={tool.timestamp} end={tool.endTime} />
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Empty state */}
          {agents.length === 0 && tools.length === 0 && (
            <div className="flex flex-col items-center justify-center h-64 text-zinc-600">
              <Activity className="w-12 h-12 mb-4 opacity-50" />
              <p>No activity to display</p>
            </div>
          )}
        </div>

        {/* Footer */}
        {!isProcessing && (agents.length > 0 || tools.length > 0) && onClear && (
          <div className="px-4 py-3 border-t border-zinc-800/50 flex justify-between items-center bg-[#0d0d0d]">
            <div className="text-xs text-zinc-600">
              {agents.length} agents • {tools.length} tools • {modifiedFiles.length} files modified
            </div>
            <button
              onClick={onClear}
              className="text-xs text-zinc-600 hover:text-zinc-400 transition-colors px-3 py-1.5 bg-zinc-800/50 rounded"
            >
              Clear history
            </button>
          </div>
        )}
      </div>
    </>
  )

  return (
    <>
      {/* Normal panel */}
      <div className="bg-[#0d0d0d] border border-zinc-800/50 rounded-lg overflow-hidden">
        {panelContent}
      </div>
      {/* Fullscreen modal - rendered via portal to escape parent container constraints */}
      {isFullscreen && portalContainer && createPortal(fullscreenContent, portalContainer)}
    </>
  )
}
