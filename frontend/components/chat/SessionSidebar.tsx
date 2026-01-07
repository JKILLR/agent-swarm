'use client'

import { MessageSquare, Plus, Trash2 } from 'lucide-react'

export interface ChatSessionSummary {
  id: string
  title: string
  message_count: number
  created_at: string
  updated_at: string
}

interface SessionSidebarProps {
  sessions: ChatSessionSummary[]
  currentSessionId: string | null
  showSidebar: boolean
  onSessionClick: (id: string) => void
  onNewSession: () => void
  onDeleteSession: (id: string, e: React.MouseEvent) => void
}

export default function SessionSidebar({
  sessions,
  currentSessionId,
  showSidebar,
  onSessionClick,
  onNewSession,
  onDeleteSession,
}: SessionSidebarProps) {
  return (
    <div className={`hidden md:flex ${showSidebar ? 'w-64' : 'w-0'} transition-all duration-200 overflow-hidden border-r border-zinc-800/50 flex-col bg-[#0a0a0a]`}>
      <div className="p-3 border-b border-zinc-800/50 flex items-center justify-between">
        <span className="text-sm font-medium text-zinc-500">History</span>
        <button
          onClick={onNewSession}
          className="p-1.5 hover:bg-zinc-800/50 rounded transition-colors"
          title="New chat"
        >
          <Plus className="w-4 h-4 text-emerald-400" />
        </button>
      </div>
      <div className="flex-1 overflow-auto">
        {sessions.map((session) => (
          <div
            key={session.id}
            onClick={() => onSessionClick(session.id)}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => e.key === 'Enter' && onSessionClick(session.id)}
            className={`w-full px-3 py-2 text-left hover:bg-zinc-800/30 transition-colors group flex items-center gap-2 cursor-pointer ${
              currentSessionId === session.id ? 'bg-zinc-800/50 border-l-2 border-l-emerald-400' : 'border-l-2 border-l-transparent'
            }`}
          >
            <MessageSquare className="w-4 h-4 text-zinc-600 flex-shrink-0" />
            <div className="flex-1 min-w-0">
              <p className="text-sm text-zinc-400 truncate">{session.title}</p>
              <p className="text-xs text-violet-500/60">{session.message_count} messages</p>
            </div>
            <button
              onClick={(e) => onDeleteSession(session.id, e)}
              className="opacity-0 group-hover:opacity-100 p-1 hover:bg-zinc-700 rounded transition-all"
              title="Delete"
            >
              <Trash2 className="w-3 h-3 text-zinc-500" />
            </button>
          </div>
        ))}
        {sessions.length === 0 && (
          <p className="text-xs text-zinc-600 p-3">No chat history yet</p>
        )}
      </div>
    </div>
  )
}
