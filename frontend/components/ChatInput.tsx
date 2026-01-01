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
        <div className="flex flex-wrap gap-2 p-2 bg-zinc-900 rounded-lg border border-zinc-800">
          {attachments.map((attachment) => (
            <div
              key={attachment.id}
              className="flex items-center gap-2 px-3 py-2 bg-zinc-800 rounded-lg group"
            >
              {attachment.type === 'image' && attachment.mimeType ? (
                <img
                  src={`data:${attachment.mimeType};base64,${attachment.content}`}
                  alt={attachment.name}
                  className="w-10 h-10 object-cover rounded"
                />
              ) : (
                <div className="w-10 h-10 flex items-center justify-center bg-zinc-700 rounded">
                  {getAttachmentIcon(attachment.type)}
                </div>
              )}
              <div className="flex flex-col min-w-0">
                <span className="text-sm text-white truncate max-w-[150px]">
                  {attachment.name}
                </span>
                <span className="text-xs text-zinc-500">
                  {attachment.type === 'text'
                    ? `${attachment.size} chars`
                    : formatFileSize(attachment.size)
                  }
                </span>
              </div>
              <button
                onClick={() => removeAttachment(attachment.id)}
                className="p-1 text-zinc-500 hover:text-white hover:bg-zinc-700 rounded transition-colors"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
          ))}
        </div>
      )}

      {/* Input Area */}
      <form onSubmit={handleSubmit} className="flex items-end gap-2">
        <div className="flex-1 bg-zinc-800 rounded-lg border border-zinc-700 focus-within:border-zinc-600 transition-colors">
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
              'w-full px-4 py-3 bg-transparent text-white placeholder-zinc-500 resize-none focus:outline-none text-sm',
              disabled && 'opacity-50 cursor-not-allowed'
            )}
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
            'p-3 rounded-lg transition-colors',
            disabled
              ? 'bg-zinc-800 text-zinc-600 cursor-not-allowed'
              : 'bg-zinc-800 text-zinc-400 hover:text-white hover:bg-zinc-700'
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
            'p-3 rounded-lg transition-colors',
            disabled || (!message.trim() && attachments.length === 0)
              ? 'bg-zinc-800 text-zinc-500 cursor-not-allowed'
              : 'bg-blue-600 text-white hover:bg-blue-700'
          )}
        >
          {disabled ? (
            <Loader2 className="w-5 h-5 animate-spin" />
          ) : (
            <Send className="w-5 h-5" />
          )}
        </button>
      </form>

      {/* Hint */}
      <p className="text-xs text-zinc-600 px-1">
        Paste images or large text directly. Press Enter to send, Shift+Enter for new line.
      </p>
    </div>
  )
}
