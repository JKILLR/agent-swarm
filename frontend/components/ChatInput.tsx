'use client'

import { useState, useRef, useEffect, useCallback } from 'react'
import { Send, Loader2, Paperclip, X, FileText, FileImage, File } from 'lucide-react'
import { cn } from '@/lib/utils'

const LARGE_TEXT_THRESHOLD = 500 // Characters before text becomes an attachment

export interface Attachment {
  id: string
  type: 'image' | 'document' | 'text'
  name: string
  content: string // base64 for files, raw text for text attachments
  mimeType?: string
  size: number
}

interface ChatInputProps {
  onSend: (message: string, attachments: Attachment[]) => void
  disabled?: boolean
  placeholder?: string
}

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

function getAttachmentIcon(type: string) {
  switch (type) {
    case 'image':
      return <FileImage className="w-4 h-4" />
    case 'document':
      return <FileText className="w-4 h-4" />
    default:
      return <File className="w-4 h-4" />
  }
}

export default function ChatInput({
  onSend,
  disabled = false,
  placeholder = 'Type a message...',
}: ChatInputProps) {
  const [message, setMessage] = useState('')
  const [attachments, setAttachments] = useState<Attachment[]>([])
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if ((message.trim() || attachments.length > 0) && !disabled) {
      onSend(message.trim(), attachments)
      setMessage('')
      setAttachments([])
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e)
    }
  }

  const addAttachment = useCallback((attachment: Attachment) => {
    setAttachments((prev) => [...prev, attachment])
  }, [])

  const removeAttachment = useCallback((id: string) => {
    setAttachments((prev) => prev.filter((a) => a.id !== id))
  }, [])

  // Handle paste events
  const handlePaste = useCallback(async (e: React.ClipboardEvent) => {
    const items = e.clipboardData.items

    // Check for images first
    for (const item of Array.from(items)) {
      if (item.type.startsWith('image/')) {
        e.preventDefault()
        const file = item.getAsFile()
        if (file) {
          const reader = new FileReader()
          reader.onload = () => {
            const base64 = (reader.result as string).split(',')[1]
            addAttachment({
              id: `img-${Date.now()}`,
              type: 'image',
              name: `Pasted Image ${new Date().toLocaleTimeString()}`,
              content: base64,
              mimeType: file.type,
              size: file.size,
            })
          }
          reader.readAsDataURL(file)
        }
        return
      }
    }

    // Check for large text paste
    const text = e.clipboardData.getData('text')
    if (text && text.length > LARGE_TEXT_THRESHOLD) {
      e.preventDefault()

      // Create text attachment
      addAttachment({
        id: `text-${Date.now()}`,
        type: 'text',
        name: `Pasted Text (${text.length} chars)`,
        content: text,
        size: text.length,
      })

      // Add a short reference in the input
      if (!message.trim()) {
        setMessage('See attached text')
      }
    }
  }, [addAttachment, message])

  // Handle file upload
  const handleFileSelect = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files
    if (!files) return

    for (const file of Array.from(files)) {
      const reader = new FileReader()

      reader.onload = () => {
        const isImage = file.type.startsWith('image/')
        const isText = file.type.startsWith('text/') ||
          ['.md', '.txt', '.json', '.yaml', '.yml', '.py', '.js', '.ts'].some(ext =>
            file.name.toLowerCase().endsWith(ext)
          )

        if (isText) {
          // Read as text
          const textReader = new FileReader()
          textReader.onload = () => {
            addAttachment({
              id: `file-${Date.now()}-${file.name}`,
              type: 'text',
              name: file.name,
              content: textReader.result as string,
              mimeType: file.type,
              size: file.size,
            })
          }
          textReader.readAsText(file)
        } else {
          const base64 = (reader.result as string).split(',')[1]
          addAttachment({
            id: `file-${Date.now()}-${file.name}`,
            type: isImage ? 'image' : 'document',
            name: file.name,
            content: base64,
            mimeType: file.type,
            size: file.size,
          })
        }
      }

      reader.readAsDataURL(file)
    }

    // Reset input
    e.target.value = ''
  }, [addAttachment])

  // Auto-resize textarea
  useEffect(() => {
    const textarea = textareaRef.current
    if (textarea) {
      textarea.style.height = 'auto'
      textarea.style.height = `${Math.min(textarea.scrollHeight, 200)}px`
    }
  }, [message])

  return (
    <div className="space-y-2">
      {/* Attachments Preview */}
      {attachments.length > 0 && (
        <div className="flex flex-wrap gap-2 p-2 bg-zinc-900/50 rounded-lg border border-zinc-800/50">
          {attachments.map((attachment) => (
            <div
              key={attachment.id}
              className="flex items-center gap-2 px-2 md:px-3 py-2 bg-zinc-800/50 rounded-lg group"
            >
              {attachment.type === 'image' && attachment.mimeType ? (
                <img
                  src={`data:${attachment.mimeType};base64,${attachment.content}`}
                  alt={attachment.name}
                  className="w-8 h-8 md:w-10 md:h-10 object-cover rounded"
                />
              ) : (
                <div className="w-8 h-8 md:w-10 md:h-10 flex items-center justify-center bg-zinc-700/50 rounded">
                  {getAttachmentIcon(attachment.type)}
                </div>
              )}
              <div className="flex flex-col min-w-0">
                <span className="text-xs md:text-sm text-zinc-300 truncate max-w-[100px] md:max-w-[150px]">
                  {attachment.name}
                </span>
                <span className="text-xs text-zinc-600">
                  {attachment.type === 'text'
                    ? `${attachment.size} chars`
                    : formatFileSize(attachment.size)
                  }
                </span>
              </div>
              <button
                onClick={() => removeAttachment(attachment.id)}
                className="p-1.5 md:p-1 text-zinc-500 hover:text-white active:text-white hover:bg-zinc-700/50 active:bg-zinc-600/50 rounded transition-colors touch-manipulation min-w-[32px] min-h-[32px] flex items-center justify-center"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
          ))}
        </div>
      )}

      {/* Input Area - Terminal style */}
      <form onSubmit={handleSubmit} className="flex items-end gap-1.5 md:gap-2">
        <div className="flex-1 bg-[#0d0d0d] rounded-lg border border-zinc-800/50 focus-within:border-violet-500/50 transition-all duration-200 flex items-center purple-glow-focus">
          <span className="pl-3 text-emerald-400 font-medium select-none">&gt;</span>
          <textarea
            ref={textareaRef}
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyDown={handleKeyDown}
            onPaste={handlePaste}
            placeholder={placeholder}
            disabled={disabled}
            rows={1}
            className={cn(
              'flex-1 px-2 md:px-3 py-3 bg-transparent text-zinc-100 placeholder-zinc-600 resize-none focus:outline-none text-base md:text-sm font-mono',
              disabled && 'opacity-50 cursor-not-allowed'
            )}
            style={{ fontSize: '16px' }} // Prevents iOS zoom on focus
          />
        </div>

        {/* Upload Button */}
        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept="image/*,.pdf,.txt,.md,.json,.yaml,.yml,.py,.js,.ts,.tsx,.jsx,.html,.css"
          className="hidden"
          onChange={handleFileSelect}
        />
        <button
          type="button"
          onClick={() => fileInputRef.current?.click()}
          disabled={disabled}
          className={cn(
            'p-3 rounded-lg transition-all duration-200 touch-manipulation min-w-[44px] min-h-[44px] flex items-center justify-center border border-zinc-800/50',
            disabled
              ? 'bg-[#0d0d0d] text-zinc-700 cursor-not-allowed'
              : 'bg-[#0d0d0d] text-zinc-500 hover:text-violet-400 hover:border-violet-500/50 hover:shadow-[0_0_12px_rgba(139,92,246,0.15)] active:bg-zinc-900'
          )}
          title="Attach files"
        >
          <Paperclip className="w-5 h-5" />
        </button>

        {/* Send Button */}
        <button
          type="submit"
          disabled={disabled || (!message.trim() && attachments.length === 0)}
          className={cn(
            'p-3 rounded-lg transition-colors touch-manipulation min-w-[44px] min-h-[44px] flex items-center justify-center',
            disabled || (!message.trim() && attachments.length === 0)
              ? 'bg-zinc-900 text-zinc-600 cursor-not-allowed border border-zinc-800/50'
              : 'bg-emerald-500 text-white hover:bg-emerald-400 active:bg-emerald-600 border border-emerald-500'
          )}
        >
          {disabled ? (
            <Loader2 className="w-5 h-5 animate-spin" />
          ) : (
            <Send className="w-5 h-5" />
          )}
        </button>
      </form>

      {/* Hint - hidden on mobile to save space */}
      <p className="hidden md:block text-xs text-zinc-600 px-1">
        Paste images or large text directly. Press Enter to send, Shift+Enter for new line.
      </p>
    </div>
  )
}
