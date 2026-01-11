'use client'

import { useState } from 'react'
import { User, FileText, FileImage, File, ChevronDown, ChevronUp, X } from 'lucide-react'
import type { Attachment } from './ChatInput'

interface ChatMessageProps {
  content: string
  timestamp?: Date
  attachments?: Attachment[]
}

function AttachmentPreview({ attachment }: { attachment: Attachment }) {
  const [expanded, setExpanded] = useState(false)
  const [lightbox, setLightbox] = useState(false)

  if (attachment.type === 'image' && attachment.mimeType) {
    return (
      <>
        <div
          className="relative cursor-pointer group"
          onClick={() => setLightbox(true)}
        >
          <img
            src={`data:${attachment.mimeType};base64,${attachment.content}`}
            alt={attachment.name}
            className="max-w-xs max-h-48 rounded-lg border border-zinc-800/50 object-contain"
          />
          <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity rounded-lg flex items-center justify-center">
            <span className="text-white text-sm">Click to expand</span>
          </div>
        </div>

        {/* Lightbox */}
        {lightbox && (
          <div
            className="fixed inset-0 bg-black/95 z-50 flex items-center justify-center p-4"
            onClick={() => setLightbox(false)}
          >
            <button
              className="absolute top-4 right-4 p-2 text-white hover:bg-white/10 rounded-lg"
              onClick={() => setLightbox(false)}
            >
              <X className="w-6 h-6" />
            </button>
            <img
              src={`data:${attachment.mimeType};base64,${attachment.content}`}
              alt={attachment.name}
              className="max-w-full max-h-full object-contain"
            />
          </div>
        )}
      </>
    )
  }

  if (attachment.type === 'text') {
    const preview = attachment.content.slice(0, 200)
    const hasMore = attachment.content.length > 200

    return (
      <div className="bg-zinc-900/50 rounded-lg border border-zinc-800/50 overflow-hidden max-w-md">
        <div
          className="flex items-center justify-between px-3 py-2 bg-[#0d0d0d] cursor-pointer"
          onClick={() => setExpanded(!expanded)}
        >
          <div className="flex items-center gap-2">
            <FileText className="w-4 h-4 text-emerald-400" />
            <span className="text-sm text-zinc-300">{attachment.name}</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xs text-zinc-600">
              {attachment.size.toLocaleString()} chars
            </span>
            {hasMore && (
              expanded ? (
                <ChevronUp className="w-4 h-4 text-zinc-600" />
              ) : (
                <ChevronDown className="w-4 h-4 text-zinc-600" />
              )
            )}
          </div>
        </div>
        <div className={`p-3 ${expanded ? 'max-h-96 overflow-auto' : ''}`}>
          <pre className="text-xs text-zinc-400 whitespace-pre-wrap font-mono">
            {expanded ? attachment.content : preview}
            {!expanded && hasMore && '...'}
          </pre>
        </div>
      </div>
    )
  }

  // Document/binary
  return (
    <div className="flex items-center gap-2 px-3 py-2 bg-zinc-900/50 rounded-lg border border-zinc-800/50">
      <File className="w-4 h-4 text-zinc-500" />
      <span className="text-sm text-zinc-300">{attachment.name}</span>
      <span className="text-xs text-zinc-600">
        {(attachment.size / 1024).toFixed(1)} KB
      </span>
    </div>
  )
}

export default function ChatMessage({ content, timestamp, attachments }: ChatMessageProps) {
  return (
    <div className="flex items-start gap-3 animate-fade-in">
      <div className="w-8 h-8 rounded-full bg-emerald-500 flex items-center justify-center flex-shrink-0">
        <User className="w-4 h-4 text-white" />
      </div>
      <div className="flex-1">
        <div className="flex items-center gap-2 mb-1">
          <span className="font-medium text-sm text-emerald-400">You</span>
          {timestamp && (
            <span className="text-xs text-zinc-600">
              {timestamp.toLocaleTimeString()}
            </span>
          )}
        </div>

        {/* Attachments */}
        {attachments && attachments.length > 0 && (
          <div className="flex flex-wrap gap-2 mb-2">
            {attachments.map((attachment) => (
              <AttachmentPreview key={attachment.id} attachment={attachment} />
            ))}
          </div>
        )}

        {/* Message content */}
        {content && (
          <div className="text-zinc-300 text-sm whitespace-pre-wrap">
            {content}
          </div>
        )}
      </div>
    </div>
  )
}
