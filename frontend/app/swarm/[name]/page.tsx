'use client'

import { useEffect, useState } from 'react'
import { useParams } from 'next/navigation'
import Link from 'next/link'
import { ArrowLeft, MessageSquare, Play, Pause, Archive, FolderOpen } from 'lucide-react'
import { getSwarm, getSwarmAgents, type Agent } from '@/lib/api'
import AgentPanel from '@/components/AgentPanel'
import FileBrowser from '@/components/FileBrowser'
import { cn, getStatusColor } from '@/lib/utils'

interface SwarmDetails {
  name: string
  description: string
  status: string
  version: string
  workspace: string
  agent_count: number
  priorities: any[]
  agents: any[]
}

export default function SwarmPage() {
  const params = useParams()
  const name = params.name as string

  const [swarm, setSwarm] = useState<SwarmDetails | null>(null)
  const [agents, setAgents] = useState<Agent[]>([])
  const [loading, setLoading] = useState(true)
  const [showFiles, setShowFiles] = useState(true)

  useEffect(() => {
    if (!name) return

    Promise.all([getSwarm(name), getSwarmAgents(name)])
      .then(([swarmData, agentsData]) => {
        setSwarm(swarmData as SwarmDetails)
        setAgents(agentsData)
      })
      .catch(console.error)
      .finally(() => setLoading(false))
  }, [name])

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-zinc-500">Loading...</div>
      </div>
    )
  }

  if (!swarm) {
    return (
      <div className="flex flex-col items-center justify-center h-full">
        <div className="text-zinc-500 mb-4">Swarm not found</div>
        <Link
          href="/"
          className="text-blue-500 hover:text-blue-400 transition-colors"
        >
          ← Back to dashboard
        </Link>
      </div>
    )
  }

  // Extract priority strings
  const priorities = swarm.priorities?.map((p: any) =>
    typeof p === 'string' ? p : p.task || JSON.stringify(p)
  ) || []

  return (
    <div className="p-6">
      {/* Header */}
      <div className="mb-6">
        <Link
          href="/"
          className="inline-flex items-center gap-1 text-sm text-zinc-500 hover:text-zinc-300 transition-colors mb-4"
        >
          <ArrowLeft className="w-4 h-4" />
          Back to dashboard
        </Link>

        <div className="flex items-start justify-between">
          <div>
            <div className="flex items-center gap-3 mb-2">
              <h1 className="text-2xl font-bold text-white">{swarm.name}</h1>
              <span
                className={cn(
                  'text-xs px-2 py-1 rounded-full bg-zinc-800',
                  getStatusColor(swarm.status)
                )}
              >
                {swarm.status}
              </span>
            </div>
            <p className="text-zinc-500">{swarm.description || 'No description'}</p>
          </div>

          <div className="flex items-center gap-2">
            <Link
              href="/chat"
              className="flex items-center gap-2 px-3 py-2 text-sm bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
            >
              <MessageSquare className="w-4 h-4" />
              Chat
            </Link>
            {swarm.status === 'active' ? (
              <button className="p-2 text-yellow-500 hover:bg-zinc-800 rounded-lg transition-colors">
                <Pause className="w-4 h-4" />
              </button>
            ) : (
              <button className="p-2 text-green-500 hover:bg-zinc-800 rounded-lg transition-colors">
                <Play className="w-4 h-4" />
              </button>
            )}
            <button className="p-2 text-zinc-500 hover:bg-zinc-800 rounded-lg transition-colors">
              <Archive className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>

      {/* Info Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
        <div className="p-4 bg-zinc-900 border border-zinc-800 rounded-lg">
          <div className="text-xs text-zinc-500 mb-1">Version</div>
          <div className="text-white font-medium">{swarm.version || '0.1.0'}</div>
        </div>
        <div className="p-4 bg-zinc-900 border border-zinc-800 rounded-lg">
          <div className="text-xs text-zinc-500 mb-1">Agents</div>
          <div className="text-white font-medium">{agents.length}</div>
        </div>
        <div className="p-4 bg-zinc-900 border border-zinc-800 rounded-lg">
          <div className="text-xs text-zinc-500 mb-1">Workspace</div>
          <div className="text-white font-medium truncate">{swarm.workspace || './workspace'}</div>
        </div>
      </div>

      {/* Priorities */}
      {priorities.length > 0 && (
        <div className="mb-8">
          <h2 className="text-lg font-semibold text-white mb-4">Priorities</h2>
          <div className="space-y-2">
            {priorities.map((priority: string, index: number) => (
              <div
                key={index}
                className="flex items-start gap-3 p-3 bg-zinc-900 border border-zinc-800 rounded-lg"
              >
                <span className="flex-shrink-0 w-6 h-6 flex items-center justify-center bg-blue-500/10 text-blue-500 text-sm font-medium rounded">
                  {index + 1}
                </span>
                <span className="text-zinc-300 text-sm">{priority}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Agents */}
      <div className="mb-8">
        <h2 className="text-lg font-semibold text-white mb-4">Agents</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {agents.map((agent) => (
            <AgentPanel key={agent.name} agent={agent} />
          ))}
        </div>
      </div>

      {/* Workspace Files */}
      <div>
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-white flex items-center gap-2">
            <FolderOpen className="w-5 h-5" />
            Workspace Files
          </h2>
          <button
            onClick={() => setShowFiles(!showFiles)}
            className="text-sm text-zinc-500 hover:text-white transition-colors"
          >
            {showFiles ? 'Hide' : 'Show'}
          </button>
        </div>
        {showFiles && <FileBrowser swarmName={name} />}
      </div>
    </div>
  )
}
