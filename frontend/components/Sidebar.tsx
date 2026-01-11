'use client'

import { useEffect, useState } from 'react'
import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { MessageSquare, LayoutDashboard, Terminal, Settings, Loader2 } from 'lucide-react'
import { getSwarms, type Swarm } from '@/lib/api'
import { cn, getStatusColor } from '@/lib/utils'
import { useAgentActivity } from '@/lib/AgentActivityContext'
import CeoTodoPanel from './CeoTodoPanel'
import JobsPanel from './JobsPanel'
import ActivityPanel from './ActivityPanel'

interface SidebarProps {
  onNavigate?: () => void
}

export default function Sidebar({ onNavigate }: SidebarProps) {
  const pathname = usePathname()
  const [swarms, setSwarms] = useState<Swarm[]>([])
  const [loading, setLoading] = useState(true)
  const {
    getSwarmActiveCount,
    panelAgentActivities,
    panelToolActivities,
    clearPanelActivities,
  } = useAgentActivity()

  // Determine if there's any activity to show
  const hasActivity = panelAgentActivities.length > 0 || panelToolActivities.length > 0
  const isProcessing = panelAgentActivities.some(
    (a) => a.status !== 'complete' && a.status !== 'error'
  )

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

  // Wrapper for links to handle mobile navigation
  const handleLinkClick = () => {
    if (onNavigate) {
      onNavigate()
    }
  }

  return (
    <aside className="w-64 md:w-64 h-full bg-[#080808] border-r border-zinc-800/50 flex flex-col">
      {/* Logo - hidden on mobile since header shows title */}
      <div className="p-4 border-b border-zinc-800/50 hidden md:block">
        <Link href="/" className="flex items-center gap-2" onClick={handleLinkClick}>
          <Terminal className="w-8 h-8 text-emerald-400" />
          <span className="font-bold text-lg text-zinc-100">Agent Swarm</span>
        </Link>
      </div>

      {/* Navigation */}
      <nav className="p-2">
        {navItems.map((item) => (
          <Link
            key={item.href}
            href={item.href}
            onClick={handleLinkClick}
            className={cn(
              'flex items-center gap-3 px-3 py-2.5 md:py-2 rounded-lg text-sm transition-all duration-200 touch-manipulation min-h-[44px] md:min-h-0',
              pathname === item.href
                ? 'bg-zinc-800/50 text-emerald-400 border-l-2 border-l-emerald-400'
                : 'text-zinc-500 hover:bg-violet-500/5 hover:text-zinc-300 hover:border-l-violet-500/50 active:bg-zinc-800/50 border-l-2 border-l-transparent'
            )}
          >
            <item.icon className="w-5 h-5 md:w-4 md:h-4" />
            {item.label}
          </Link>
        ))}
      </nav>

      {/* Swarms List */}
      <div className="flex-1 overflow-auto p-2">
        <div className="px-3 py-2 text-xs font-semibold text-zinc-600 uppercase tracking-wider">
          Swarms
        </div>

        {loading ? (
          <div className="px-3 py-2 text-sm text-zinc-600">Loading...</div>
        ) : swarms.length === 0 ? (
          <div className="px-3 py-2 text-sm text-zinc-600">No swarms found</div>
        ) : (
          <div className="space-y-1">
            {swarms.map((swarm) => {
              const encodedName = encodeURIComponent(swarm.name)
              const activeCount = getSwarmActiveCount(swarm.name)
              return (
                <Link
                  key={swarm.name}
                  href={`/swarm/${encodedName}`}
                  onClick={handleLinkClick}
                  className={cn(
                    'flex items-center justify-between px-3 py-2.5 md:py-2 rounded-lg text-sm transition-all duration-200 touch-manipulation min-h-[44px] md:min-h-0',
                    pathname === `/swarm/${encodedName}`
                      ? 'bg-zinc-800/50 text-zinc-100 border-l-2 border-l-emerald-400'
                      : 'text-zinc-500 hover:bg-violet-500/5 hover:text-zinc-300 active:bg-zinc-800/50 border-l-2 border-l-transparent',
                    activeCount > 0 && 'border border-violet-500/30'
                  )}
                >
                  <span className="flex items-center gap-2 truncate">
                    {activeCount > 0 && (
                      <Loader2 className="w-3 h-3 text-emerald-400 animate-spin flex-shrink-0" />
                    )}
                    {swarm.name}
                  </span>
                  <span className={cn(
                    'text-xs',
                    activeCount > 0 ? 'text-emerald-400' : 'text-violet-400'
                  )}>
                    {activeCount > 0 ? `${activeCount} active` : swarm.agent_count}
                  </span>
                </Link>
              )
            })}
          </div>
        )}
      </div>

      {/* Activity Panel - Global agent activity */}
      {(isProcessing || hasActivity) && (
        <div className="p-2 border-t border-zinc-800/50 bg-[#0a0a0a]">
          <ActivityPanel
            agents={panelAgentActivities}
            tools={panelToolActivities}
            isProcessing={isProcessing}
            onClear={clearPanelActivities}
          />
        </div>
      )}

      {/* CEO Todo Panel */}
      <CeoTodoPanel />

      {/* Background Jobs Panel */}
      <JobsPanel />

      {/* Footer */}
      <div className="p-2 border-t border-zinc-800/50 bg-[#0c0c0c]">
        <button className="flex items-center gap-3 px-3 py-2.5 md:py-2 w-full rounded-lg text-sm text-zinc-500 hover:bg-violet-500/5 hover:text-zinc-300 active:bg-zinc-800/50 transition-all duration-200 touch-manipulation min-h-[44px] md:min-h-0">
          <Settings className="w-5 h-5 md:w-4 md:h-4" />
          Settings
        </button>
      </div>
    </aside>
  )
}
