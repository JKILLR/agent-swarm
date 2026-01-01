'use client'

import { useEffect, useState } from 'react'
import { Bot, Users, Activity, Plus } from 'lucide-react'
import { getSwarms, getStatus, type Swarm, type HealthStatus } from '@/lib/api'
import SwarmCard from '@/components/SwarmCard'

export default function DashboardPage() {
  const [swarms, setSwarms] = useState<Swarm[]>([])
  const [status, setStatus] = useState<HealthStatus | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    Promise.all([getSwarms(), getStatus()])
      .then(([swarmsData, statusData]) => {
        setSwarms(swarmsData)
        setStatus(statusData)
      })
      .catch(console.error)
      .finally(() => setLoading(false))
  }, [])

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-zinc-500">Loading...</div>
      </div>
    )
  }

  return (
    <div className="p-6">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-white mb-2">Dashboard</h1>
        <p className="text-zinc-500">Overview of your agent swarms</p>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
        <div className="p-4 bg-zinc-900 border border-zinc-800 rounded-lg">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-blue-500/10 rounded-lg">
              <Bot className="w-5 h-5 text-blue-500" />
            </div>
            <div>
              <div className="text-2xl font-bold text-white">
                {status?.swarm_count || 0}
              </div>
              <div className="text-sm text-zinc-500">Swarms</div>
            </div>
          </div>
        </div>

        <div className="p-4 bg-zinc-900 border border-zinc-800 rounded-lg">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-green-500/10 rounded-lg">
              <Users className="w-5 h-5 text-green-500" />
            </div>
            <div>
              <div className="text-2xl font-bold text-white">
                {status?.agent_count || 0}
              </div>
              <div className="text-sm text-zinc-500">Total Agents</div>
            </div>
          </div>
        </div>

        <div className="p-4 bg-zinc-900 border border-zinc-800 rounded-lg">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-emerald-500/10 rounded-lg">
              <Activity className="w-5 h-5 text-emerald-500" />
            </div>
            <div>
              <div className="text-2xl font-bold text-white capitalize">
                {status?.status || 'Unknown'}
              </div>
              <div className="text-sm text-zinc-500">System Status</div>
            </div>
          </div>
        </div>
      </div>

      {/* Swarms Grid */}
      <div className="mb-6 flex items-center justify-between">
        <h2 className="text-lg font-semibold text-white">Your Swarms</h2>
        <button className="flex items-center gap-2 px-3 py-2 text-sm bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors">
          <Plus className="w-4 h-4" />
          New Swarm
        </button>
      </div>

      {swarms.length === 0 ? (
        <div className="text-center py-12 bg-zinc-900 border border-zinc-800 rounded-lg">
          <Bot className="w-12 h-12 text-zinc-700 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-zinc-400 mb-2">No swarms yet</h3>
          <p className="text-sm text-zinc-500 mb-4">
            Create your first swarm to get started
          </p>
          <button className="flex items-center gap-2 px-4 py-2 mx-auto text-sm bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors">
            <Plus className="w-4 h-4" />
            Create Swarm
          </button>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {swarms.map((swarm) => (
            <SwarmCard key={swarm.name} swarm={swarm} />
          ))}
        </div>
      )}
    </div>
  )
}
