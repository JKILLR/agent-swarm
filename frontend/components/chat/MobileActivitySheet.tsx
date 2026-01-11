'use client'

import { Activity, X } from 'lucide-react'
import ActivityPanel from '@/components/ActivityPanel'
import type { PanelAgentActivity, PanelToolActivity } from '@/lib/AgentActivityContext'

interface MobileActivitySheetProps {
  isOpen: boolean
  onClose: () => void
  agents: PanelAgentActivity[]
  tools: PanelToolActivity[]
  isProcessing: boolean
  onClear: () => void
}

export default function MobileActivitySheet({
  isOpen,
  onClose,
  agents,
  tools,
  isProcessing,
  onClear,
}: MobileActivitySheetProps) {
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
          <div className="flex items-center gap-2">
            <Activity className={`w-5 h-5 ${isProcessing ? 'text-emerald-400 animate-pulse' : 'text-zinc-500'}`} />
            <span className="font-medium text-zinc-300">Activity</span>
            {isProcessing && (
              <span className="px-1.5 py-0.5 rounded-full bg-emerald-400/20 text-emerald-400 text-xs">
                Active
              </span>
            )}
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-zinc-800/50 rounded-lg transition-colors touch-manipulation"
          >
            <X className="w-5 h-5 text-zinc-500" />
          </button>
        </div>
        <div className="flex-1 overflow-auto">
          <ActivityPanel
            agents={agents}
            tools={tools}
            isProcessing={isProcessing}
            onClear={onClear}
          />
        </div>
      </div>
    </>
  )
}
