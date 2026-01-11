'use client'

import { Terminal } from 'lucide-react'

interface EmptyStateProps {
  onQuickSend: (message: string) => void
}

export default function EmptyState({ onQuickSend }: EmptyStateProps) {
  return (
    <div className="flex flex-col items-center justify-center h-full text-center px-4">
      <Terminal className="w-12 h-12 md:w-16 md:h-16 text-emerald-400/50 mb-4" />
      <h2 className="text-lg md:text-xl font-medium text-zinc-400 mb-2">
        Chat with the COO
      </h2>
      <p className="text-sm text-zinc-600 max-w-md">
        You are connected to Axel, your COO. Ask about your swarms,
        request research, or coordinate tasks.
      </p>
      <div className="mt-6 space-y-2 text-left w-full max-w-md">
        <p className="text-xs text-zinc-600">Try asking:</p>
        <div className="space-y-2">
          <button
            onClick={() => onQuickSend('What is the current status of the ASA project?')}
            className="block w-full text-left text-sm text-zinc-500 hover:text-violet-400 active:text-violet-300 transition-all duration-200 p-3 md:p-2 bg-zinc-900/50 md:bg-transparent rounded-lg md:rounded-none touch-manipulation border border-zinc-800/50 md:border-0 hover:bg-violet-500/5 hover:border-violet-500/30"
          >
            <span className="text-emerald-400 mr-2">&gt;</span>What is the current status of the ASA project?
          </button>
          <button
            onClick={() => onQuickSend('Research sparse attention implementations for ASA')}
            className="block w-full text-left text-sm text-zinc-500 hover:text-violet-400 active:text-violet-300 transition-all duration-200 p-3 md:p-2 bg-zinc-900/50 md:bg-transparent rounded-lg md:rounded-none touch-manipulation border border-zinc-800/50 md:border-0 hover:bg-violet-500/5 hover:border-violet-500/30"
          >
            <span className="text-emerald-400 mr-2">&gt;</span>Research sparse attention implementations for ASA
          </button>
        </div>
      </div>
    </div>
  )
}
