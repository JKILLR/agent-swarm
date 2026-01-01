'use client'

import Link from 'next/link'
import { Users, ArrowRight } from 'lucide-react'
import { type Swarm } from '@/lib/api'
import { cn, getStatusColor } from '@/lib/utils'

interface SwarmCardProps {
  swarm: Swarm
}

export default function SwarmCard({ swarm }: SwarmCardProps) {
  return (
    <Link
      href={`/swarm/${swarm.name}`}
      className="block p-4 bg-zinc-900 border border-zinc-800 rounded-lg hover:border-zinc-700 transition-colors group"
    >
      <div className="flex items-start justify-between mb-3">
        <div>
          <h3 className="font-semibold text-white group-hover:text-blue-400 transition-colors">
            {swarm.name}
          </h3>
          <p className="text-sm text-zinc-500 mt-1 line-clamp-2">
            {swarm.description || 'No description'}
          </p>
        </div>
        <span className={cn('text-xs px-2 py-1 rounded-full bg-zinc-800', getStatusColor(swarm.status))}>
          {swarm.status}
        </span>
      </div>

      <div className="flex items-center justify-between text-sm">
        <div className="flex items-center gap-1 text-zinc-400">
          <Users className="w-4 h-4" />
          <span>{swarm.agent_count} agents</span>
        </div>
        <ArrowRight className="w-4 h-4 text-zinc-600 group-hover:text-zinc-400 transition-colors" />
      </div>

      {swarm.priorities.length > 0 && (
        <div className="mt-3 pt-3 border-t border-zinc-800">
          <div className="text-xs text-zinc-500 mb-1">Top Priority</div>
          <div className="text-sm text-zinc-300 truncate">
            {swarm.priorities[0]}
          </div>
        </div>
      )}
    </Link>
  )
}
