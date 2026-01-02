'use client'

import { useEffect, useState } from 'react'
import { Loader2, Check, X, Zap, Search, FileText, Terminal, Globe, GitBranch, Users } from 'lucide-react'

export interface ActivityItem {
  id: string
  tool: string
  description: string
  status: 'running' | 'complete' | 'error'
  timestamp: Date
  summary?: string
}

interface ActivityFeedProps {
  activities: ActivityItem[]
  maxItems?: number
}

const toolIcons: Record<string, React.ReactNode> = {
  Task: <Users className="w-3 h-3" />,
  Read: <FileText className="w-3 h-3" />,
  Write: <FileText className="w-3 h-3" />,
  Bash: <Terminal className="w-3 h-3" />,
  Glob: <Search className="w-3 h-3" />,
  Grep: <Search className="w-3 h-3" />,
  WebSearch: <Globe className="w-3 h-3" />,
  WebFetch: <Globe className="w-3 h-3" />,
  GitCommit: <GitBranch className="w-3 h-3" />,
  GitSync: <GitBranch className="w-3 h-3" />,
  GitStatus: <GitBranch className="w-3 h-3" />,
  ParallelTasks: <Zap className="w-3 h-3" />,
}

export default function ActivityFeed({ activities, maxItems = 10 }: ActivityFeedProps) {
  const displayActivities = activities.slice(-maxItems)

  if (displayActivities.length === 0) {
    return null
  }

  return (
    <div className="bg-zinc-900/80 border border-zinc-800 rounded-lg overflow-hidden">
      <div className="px-3 py-2 border-b border-zinc-800 flex items-center gap-2">
        <Zap className="w-4 h-4 text-amber-500" />
        <span className="text-xs font-medium text-zinc-400">Live Activity</span>
        <span className="text-xs text-zinc-600">
          ({displayActivities.filter(a => a.status === 'running').length} active)
        </span>
      </div>
      <div className="divide-y divide-zinc-800/50 max-h-48 overflow-y-auto">
        {displayActivities.map((activity) => (
          <div
            key={activity.id}
            className={`px-3 py-2 flex items-start gap-2 text-xs transition-colors ${
              activity.status === 'running'
                ? 'bg-amber-500/5 border-l-2 border-l-amber-500'
                : activity.status === 'error'
                ? 'bg-red-500/5 border-l-2 border-l-red-500'
                : 'border-l-2 border-l-green-500/30'
            }`}
          >
            {/* Status indicator */}
            <div className="flex-shrink-0 mt-0.5">
              {activity.status === 'running' ? (
                <Loader2 className="w-3 h-3 text-amber-500 animate-spin" />
              ) : activity.status === 'error' ? (
                <X className="w-3 h-3 text-red-500" />
              ) : (
                <Check className="w-3 h-3 text-green-500" />
              )}
            </div>

            {/* Tool icon */}
            <div className="flex-shrink-0 mt-0.5 text-zinc-500">
              {toolIcons[activity.tool] || <Zap className="w-3 h-3" />}
            </div>

            {/* Content */}
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2">
                <span className="font-medium text-zinc-300">{activity.tool}</span>
                <span className="text-zinc-500">{activity.description}</span>
              </div>
              {activity.summary && activity.status !== 'running' && (
                <div className="mt-1 text-zinc-600 truncate">
                  {activity.summary.substring(0, 100)}
                </div>
              )}
            </div>

            {/* Time */}
            <div className="flex-shrink-0 text-zinc-600">
              {activity.timestamp.toLocaleTimeString('en-US', {
                hour: '2-digit',
                minute: '2-digit',
                second: '2-digit'
              })}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
