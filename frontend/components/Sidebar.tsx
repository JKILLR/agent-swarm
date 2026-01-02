'use client'

import { useEffect, useState } from 'react'
import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { MessageSquare, LayoutDashboard, Bot, Settings, Loader2 } from 'lucide-react'
import { getSwarms, type Swarm } from '@/lib/api'
import { cn, getStatusColor } from '@/lib/utils'
import { useAgentActivity } from '@/lib/AgentActivityContext'
import CeoTodoPanel from './CeoTodoPanel'

export default function Sidebar() {
  const pathname = usePathname()
  const [swarms, setSwarms] = useState<Swarm[]>([])
  const [loading, setLoading] = useState(true)
  const { getSwarmActiveCount } = useAgentActivity()

  useEffect(() => {
    getSwarms()
      .then(setSwarms)
      .catch(console.error)
      .finally(() => setLoading(false))
  }, [])

  const navItems = [
    { href: '/', label: 'Dashboard', icon: LayoutDashboard },
    { href: '/chat', label: 'Chat', icon: MessageSquare },
  ]

  return (
    <aside className="w-64 bg-zinc-900 border-r border-zinc-800 flex flex-col">
      {/* Logo */}
      <div className="p-4 border-b border-zinc-800">
        <Link href="/" className="flex items-center gap-2">
          <Bot className="w-8 h-8 text-blue-500" />
          <span className="font-bold text-lg">Agent Swarm</span>
        </Link>
      </div>

      {/* Navigation */}
      <nav className="p-2">
        {navItems.map((item) => (
          <Link
            key={item.href}
            href={item.href}
            className={cn(
              'flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-colors',
              pathname === item.href
                ? 'bg-zinc-800 text-white'
                : 'text-zinc-400 hover:bg-zinc-800/50 hover:text-white'
            )}
          >
            <item.icon className="w-4 h-4" />
            {item.label}
          </Link>
        ))}
      </nav>

      {/* Swarms List */}
      <div className="flex-1 overflow-auto p-2">
        <div className="px-3 py-2 text-xs font-semibold text-zinc-500 uppercase tracking-wider">
          Swarms
        </div>

        {loading ? (
          <div className="px-3 py-2 text-sm text-zinc-500">Loading...</div>
        ) : swarms.length === 0 ? (
          <div className="px-3 py-2 text-sm text-zinc-500">No swarms found</div>
        ) : (
          <div className="space-y-1">
            {swarms.map((swarm) => {
              const encodedName = encodeURIComponent(swarm.name)
              const activeCount = getSwarmActiveCount(swarm.name)
              return (
                <Link
                  key={swarm.name}
                  href={`/swarm/${encodedName}`}
                  className={cn(
                    'flex items-center justify-between px-3 py-2 rounded-lg text-sm transition-colors',
                    pathname === `/swarm/${encodedName}`
                      ? 'bg-zinc-800 text-white'
                      : 'text-zinc-400 hover:bg-zinc-800/50 hover:text-white',
                    activeCount > 0 && 'border border-amber-500/30'
                  )}
                >
                  <span className="flex items-center gap-2 truncate">
                    {activeCount > 0 && (
                      <Loader2 className="w-3 h-3 text-amber-400 animate-spin flex-shrink-0" />
                    )}
                    {swarm.name}
                  </span>
                  <span className={cn(
                    'text-xs',
                    activeCount > 0 ? 'text-amber-400' : getStatusColor(swarm.status)
                  )}>
                    {activeCount > 0 ? `${activeCount} active` : swarm.agent_count}
                  </span>
                </Link>
              )
            })}
          </div>
        )}
      </div>

      {/* CEO Todo Panel */}
      <CeoTodoPanel />

      {/* Footer */}
      <div className="p-2 border-t border-zinc-800">
        <button className="flex items-center gap-3 px-3 py-2 w-full rounded-lg text-sm text-zinc-400 hover:bg-zinc-800/50 hover:text-white transition-colors">
          <Settings className="w-4 h-4" />
          Settings
        </button>
      </div>
    </aside>
  )
}
