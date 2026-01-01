'use client'

import { User } from 'lucide-react'

interface ChatMessageProps {
  content: string
  timestamp?: Date
}

export default function ChatMessage({ content, timestamp }: ChatMessageProps) {
  return (
    <div className="flex items-start gap-3 animate-fade-in">
      <div className="w-8 h-8 rounded-full bg-blue-600 flex items-center justify-center flex-shrink-0">
        <User className="w-4 h-4 text-white" />
      </div>
      <div className="flex-1">
        <div className="flex items-center gap-2 mb-1">
          <span className="font-medium text-sm text-white">You</span>
          {timestamp && (
            <span className="text-xs text-zinc-500">
              {timestamp.toLocaleTimeString()}
            </span>
          )}
        </div>
        <div className="text-zinc-300 text-sm whitespace-pre-wrap">
          {content}
        </div>
      </div>
    </div>
  )
}
