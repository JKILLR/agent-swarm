'use client'

import { Terminal, WifiOff, ChevronLeft, ChevronRight, MessageSquare, Activity } from 'lucide-react'

interface ChatHeaderProps {
  isConnected: boolean
  isMobile: boolean
  showSidebar: boolean
  hasActivity: boolean
  isActivityProcessing: boolean
  onToggleSidebar: () => void
  onShowHistory: () => void
  onShowActivity: () => void
}

export default function ChatHeader({
  isConnected,
  isMobile,
  showSidebar,
  hasActivity,
  isActivityProcessing,
  onToggleSidebar,
  onShowHistory,
  onShowActivity,
}: ChatHeaderProps) {
  return (
    <div className="flex items-center justify-between px-3 md:px-6 py-3 md:py-4 border-b border-zinc-800/50">
      <div className="flex items-center gap-2 md:gap-3">
        {/* Desktop sidebar toggle */}
        <button
          onClick={onToggleSidebar}
          className="hidden md:block p-1 hover:bg-zinc-800/50 rounded transition-colors"
          title={showSidebar ? 'Hide history' : 'Show history'}
        >
          {showSidebar ? (
            <ChevronLeft className="w-5 h-5 text-zinc-500" />
          ) : (
            <ChevronRight className="w-5 h-5 text-zinc-500" />
          )}
        </button>
        {/* Mobile history toggle */}
        <button
          onClick={onShowHistory}
          className="md:hidden p-2 -ml-1 hover:bg-zinc-800/50 active:bg-zinc-800 rounded-lg transition-colors touch-manipulation"
          title="Chat history"
        >
          <MessageSquare className="w-5 h-5 text-zinc-500" />
        </button>
        <Terminal className="w-5 h-5 md:w-6 md:h-6 text-emerald-400" />
        <div>
          <h1 className="font-medium text-white text-sm md:text-base">Chat with COO</h1>
          <p className="text-xs text-zinc-600 hidden md:block">Axel - Chief Operating Officer</p>
        </div>
      </div>

      <div className="flex items-center gap-2">
        {/* Mobile activity button - only show when there's activity */}
        {isMobile && (isActivityProcessing || hasActivity) && (
          <button
            onClick={onShowActivity}
            className={`p-2 rounded-lg transition-all touch-manipulation ${
              isActivityProcessing
                ? 'bg-emerald-400/20 animate-pulse'
                : 'hover:bg-zinc-800/50'
            }`}
            title="View activity"
          >
            <Activity className={`w-5 h-5 ${isActivityProcessing ? 'text-emerald-400' : 'text-zinc-500'}`} />
          </button>
        )}
        {isConnected ? (
          <span className="flex items-center gap-1.5 text-xs text-green-500">
            <span className="w-1.5 h-1.5 rounded-full bg-green-500 shadow-[0_0_6px_rgba(34,197,94,0.5)]" />
            <span className="hidden sm:inline">Connected</span>
          </span>
        ) : (
          <span className="flex items-center gap-1.5 text-xs text-red-500">
            <WifiOff className="w-3 h-3" />
            <span className="hidden sm:inline">Disconnected</span>
          </span>
        )}
      </div>
    </div>
  )
}
