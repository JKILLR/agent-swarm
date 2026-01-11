'use client'

import { MessageSquare, Plus, Trash2, X } from 'lucide-react'
import type { ChatSessionSummary } from '@/lib/api'

interface MobileHistorySheetProps {
  isOpen: boolean
  onClose: () => void
  sessions: ChatSessionSummary[]
  currentSessionId: string | null
  onSessionClick: (id: string) => void
  onNewSession: () => void
  onDeleteSession: (id: string, e: React.MouseEvent) => void
}

export default function MobileHistorySheet({
  isOpen,
  onClose,
  sessions,
  currentSessionId,
  onSessionClick,
  onNewSession,
  onDeleteSession,
}: MobileHistorySheetProps) {
  if (!isOpen) return null

  return (
    <>
      {/* Overlay */}
      <div
        className="fixed inset-0 z-40 bg-black/80"
        onClick={onClose}
      />
      {/* Bottom Sheet */}
      <div className="fixed bottom-0 left-0 right-0 z-50 bg-[#0d0d0d] rounded-t-2xl border-t border-zinc-800/50 max-h-[70vh] flex flex-col animate-slide-up">
        <div className="flex items-center justify-between p-4 border-b border-zinc-800/50">
          <span className="font-medium text-zinc-300">Chat History</span>
          <div className="flex items-center gap-2">
            <button
              onClick={onNewSession}
              className="p-2 hover:bg-zinc-800/50 rounded-lg transition-colors touch-manipulation"
              title="New chat"
            >
              <Plus className="w-5 h-5 text-emerald-400" />
            </button>
            <button
              onClick={onClose}
              className="p-2 hover:bg-zinc-800/50 rounded-lg transition-colors touch-manipulation"
            >
              <X className="w-5 h-5 text-zinc-500" />
            </button>
          </div>
        </div>
        <div className="flex-1 overflow-auto p-2">
          {sessions.map((session) => (
            <div
              key={session.id}
              onClick={() => {
                onSessionClick(session.id)
                onClose()
              }}
              role="button"
              tabIndex={0}
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  onSessionClick(session.id)
                  onClose()
                }
              }}
              className={`w-full px-4 py-3 text-left hover:bg-zinc-800/30 active:bg-zinc-800/50 transition-colors flex items-center gap-3 rounded-lg touch-manipulation min-h-[56px] cursor-pointer ${
                currentSessionId === session.id ? 'bg-zinc-800/50 border-l-2 border-l-emerald-400' : 'border-l-2 border-l-transparent'
              }`}
            >
              <MessageSquare className="w-5 h-5 text-zinc-600 flex-shrink-0" />
              <div className="flex-1 min-w-0">
                <p className="text-sm text-zinc-400 truncate">{session.title}</p>
                <p className="text-xs text-violet-500/60">{session.message_count} messages</p>
              </div>
              <button
                onClick={(e) => {
                  onDeleteSession(session.id, e)
                }}
                className="p-2 hover:bg-zinc-700 rounded-lg transition-colors touch-manipulation"
                title="Delete"
              >
                <Trash2 className="w-4 h-4 text-zinc-500" />
              </button>
            </div>
          ))}
          {sessions.length === 0 && (
            <p className="text-sm text-zinc-600 p-4 text-center">No chat history yet</p>
          )}
        </div>
      </div>
    </>
  )
}
