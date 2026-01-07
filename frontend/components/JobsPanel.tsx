'use client'

import { useEffect, useState, useCallback, useRef } from 'react'
import { Loader2, Play, Square, CheckCircle, XCircle, Clock, Briefcase, ChevronDown, ChevronRight } from 'lucide-react'

interface Job {
  id: string
  type: string
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled'
  created_at: string
  started_at?: string
  completed_at?: string
  prompt: string
  swarm?: string
  progress: number
  current_activity: string
  activities: Array<{ tool: string; time: string }>
  result?: string
  error?: string
}

interface JobsPanelProps {
  expanded?: boolean
}

// Auto-detect API/WS URLs based on current host
function getApiBase(): string {
  if (typeof window === 'undefined') {
    return process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
  }
  const protocol = window.location.protocol
  const host = window.location.hostname
  const port = '8000'
  return process.env.NEXT_PUBLIC_API_URL || `${protocol}//${host}:${port}`
}

function getWsBase(): string {
  if (typeof window === 'undefined') {
    return process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000'
  }
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  const host = window.location.hostname
  const port = '8000'
  return process.env.NEXT_PUBLIC_WS_URL || `${protocol}//${host}:${port}`
}

const API_BASE = getApiBase()
const WS_BASE = getWsBase()

export default function JobsPanel({ expanded: defaultExpanded = false }: JobsPanelProps) {
  const [jobs, setJobs] = useState<Job[]>([])
  const [expanded, setExpanded] = useState(defaultExpanded)
  const [selectedJob, setSelectedJob] = useState<string | null>(null)
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null)
  const reconnectAttempts = useRef(0)
  const maxReconnectAttempts = 5

  // Load jobs
  const loadJobs = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/jobs?limit=10`)
      if (res.ok) {
        const data = await res.json()
        setJobs(data)
      }
    } catch (e) {
      console.error('Failed to load jobs:', e)
    }
  }, [])

  // Connect to job updates WebSocket with reconnection
  useEffect(() => {
    loadJobs()

    const connectWebSocket = () => {
      const ws = new WebSocket(`${WS_BASE}/ws/jobs`)

      ws.onopen = () => {
        console.log('Jobs WebSocket connected')
        reconnectAttempts.current = 0
        // Subscribe to all job updates
        ws.send(JSON.stringify({ action: 'subscribe_all' }))
      }

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)
          if (data.type === 'job_update') {
            // Update or add job in list
            setJobs((prev) => {
              const idx = prev.findIndex((j) => j.id === data.job.id)
              if (idx >= 0) {
                const updated = [...prev]
                updated[idx] = data.job
                return updated
              } else {
                return [data.job, ...prev].slice(0, 10)
              }
            })
          }
        } catch (e) {
          console.error('Failed to parse job update:', e)
        }
      }

      ws.onerror = (error) => {
        console.error('Jobs WebSocket error:', error)
      }

      ws.onclose = () => {
        wsRef.current = null
        // Attempt reconnection with exponential backoff
        if (reconnectAttempts.current < maxReconnectAttempts) {
          const delay = Math.min(1000 * Math.pow(2, reconnectAttempts.current), 10000)
          console.log(`Jobs WebSocket closed, reconnecting in ${delay}ms...`)
          reconnectAttempts.current++
          reconnectTimeoutRef.current = setTimeout(connectWebSocket, delay)
        } else {
          console.log('Jobs WebSocket max reconnect attempts reached')
        }
      }

      wsRef.current = ws
    }

    // Initial connection with small delay to let backend start
    const initialDelay = setTimeout(connectWebSocket, 500)

    return () => {
      clearTimeout(initialDelay)
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current)
      }
      if (wsRef.current) {
        wsRef.current.close()
      }
    }
  }, [loadJobs])

  // Cancel a job
  const cancelJob = async (jobId: string) => {
    try {
      const res = await fetch(`${API_BASE}/api/jobs/${jobId}`, {
        method: 'DELETE',
      })
      if (res.ok) {
        loadJobs()
      }
    } catch (e) {
      console.error('Failed to cancel job:', e)
    }
  }

  const runningJobs = jobs.filter((j) => j.status === 'running' || j.status === 'pending')
  const completedJobs = jobs.filter((j) => j.status === 'completed' || j.status === 'failed' || j.status === 'cancelled')

  const getStatusIcon = (status: Job['status']) => {
    switch (status) {
      case 'pending':
        return <Clock className="w-3 h-3 text-zinc-500" />
      case 'running':
        return <Loader2 className="w-3 h-3 text-violet-500 animate-spin" />
      case 'completed':
        return <CheckCircle className="w-3 h-3 text-green-500" />
      case 'failed':
        return <XCircle className="w-3 h-3 text-red-500" />
      case 'cancelled':
        return <Square className="w-3 h-3 text-zinc-500" />
    }
  }

  const getStatusColor = (status: Job['status']) => {
    switch (status) {
      case 'pending':
        return 'text-zinc-400'
      case 'running':
        return 'text-violet-400'
      case 'completed':
        return 'text-green-400'
      case 'failed':
        return 'text-red-400'
      case 'cancelled':
        return 'text-zinc-500'
    }
  }

  return (
    <div className="border-t border-zinc-800 bg-[#0c0c0c]">
      {/* Header */}
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full px-4 py-2 flex items-center justify-between hover:bg-zinc-800/50 transition-colors"
      >
        <div className="flex items-center gap-2">
          {expanded ? (
            <ChevronDown className="w-4 h-4 text-zinc-500" />
          ) : (
            <ChevronRight className="w-4 h-4 text-zinc-500" />
          )}
          <Briefcase className="w-4 h-4 text-zinc-500" />
          <span className="text-sm font-medium text-zinc-400">Background Jobs</span>
        </div>
        {runningJobs.length > 0 && (
          <span className="px-2 py-0.5 text-xs bg-violet-500/20 text-violet-400 rounded-full">
            {runningJobs.length} running
          </span>
        )}
      </button>

      {/* Expanded content */}
      {expanded && (
        <div className="px-4 pb-3 space-y-2">
          {jobs.length === 0 ? (
            <p className="text-xs text-zinc-600 py-2">No background jobs</p>
          ) : (
            <>
              {/* Running/Pending jobs */}
              {runningJobs.length > 0 && (
                <div className="space-y-1">
                  <p className="text-xs text-zinc-500 font-medium">Active</p>
                  {runningJobs.map((job) => (
                    <div
                      key={job.id}
                      className="bg-zinc-800/50 rounded p-2 space-y-1"
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          {getStatusIcon(job.status)}
                          <span className={`text-xs ${getStatusColor(job.status)}`}>
                            {job.status}
                          </span>
                        </div>
                        <button
                          onClick={() => cancelJob(job.id)}
                          className="text-xs text-red-400 hover:text-red-300"
                        >
                          Cancel
                        </button>
                      </div>
                      <p className="text-xs text-zinc-300 truncate">{job.prompt}</p>
                      {job.current_activity && (
                        <p className="text-xs text-zinc-500">{job.current_activity}</p>
                      )}
                      {job.status === 'running' && (
                        <div className="h-1 bg-zinc-700 rounded overflow-hidden">
                          <div
                            className="h-full bg-violet-500 transition-all"
                            style={{ width: `${job.progress}%` }}
                          />
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}

              {/* Completed jobs */}
              {completedJobs.length > 0 && (
                <div className="space-y-1">
                  <p className="text-xs text-zinc-500 font-medium">Recent</p>
                  {completedJobs.slice(0, 5).map((job) => (
                    <div
                      key={job.id}
                      onClick={() => setSelectedJob(selectedJob === job.id ? null : job.id)}
                      className="bg-zinc-900/50 rounded p-2 cursor-pointer hover:bg-zinc-800/30 transition-colors"
                    >
                      <div className="flex items-center gap-2">
                        {getStatusIcon(job.status)}
                        <span className="text-xs text-zinc-400 truncate flex-1">
                          {job.prompt}
                        </span>
                      </div>
                      {selectedJob === job.id && (
                        <div className="mt-2 pt-2 border-t border-zinc-800">
                          {job.result && (
                            <p className="text-xs text-zinc-500 line-clamp-3">
                              {job.result}
                            </p>
                          )}
                          {job.error && (
                            <p className="text-xs text-red-400">
                              Error: {job.error}
                            </p>
                          )}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </>
          )}
        </div>
      )}
    </div>
  )
}
