'use client'

import { useEffect, useState, useCallback } from 'react'
import { useRouter } from 'next/navigation'
import { Crown, Users, Bot, ChevronDown, ChevronRight } from 'lucide-react'
import { getSwarms, getSwarmAgents, type Swarm, type Agent } from '@/lib/api'

interface OrgNode {
  id: string
  name: string
  type: 'ceo' | 'coo' | 'vp' | 'swarm' | 'agent'
  description?: string
  status?: string
  children?: OrgNode[]
  model?: string
  swarmName?: string
  expanded?: boolean
}

interface OrgChartProps {
  onSelectNode?: (node: OrgNode) => void
  selectedNodeId?: string
}

export default function OrgChart({ onSelectNode, selectedNodeId }: OrgChartProps) {
  const [orgTree, setOrgTree] = useState<OrgNode | null>(null)
  const [expandedNodes, setExpandedNodes] = useState<Set<string>>(new Set(['ceo', 'coo', 'operations']))
  const [loading, setLoading] = useState(true)
  const router = useRouter()

  useEffect(() => {
    async function loadOrgStructure() {
      try {
        const swarms = await getSwarms()

        // Load agents for each swarm
        const swarmsWithAgents = await Promise.all(
          swarms.map(async (swarm) => {
            try {
              const agents = await getSwarmAgents(swarm.name)
              return { ...swarm, agents }
            } catch {
              return { ...swarm, agents: [] }
            }
          })
        )

        // Build org tree
        const tree = buildOrgTree(swarmsWithAgents)
        setOrgTree(tree)

        // Auto-expand first few levels
        setExpandedNodes(new Set(['ceo', 'coo', 'operations', 'swarm_dev']))
      } catch (error) {
        console.error('Failed to load org structure:', error)
      } finally {
        setLoading(false)
      }
    }

    loadOrgStructure()
  }, [])

  const buildOrgTree = (swarms: (Swarm & { agents: Agent[] })[]): OrgNode => {
    // Find Operations swarm (VP level)
    const operationsSwarm = swarms.find(s => s.name === 'operations')
    const otherSwarms = swarms.filter(s => s.name !== 'operations')

    // Build swarm nodes
    const swarmNodes: OrgNode[] = otherSwarms.map(swarm => ({
      id: swarm.name,
      name: swarm.name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
      type: 'swarm',
      description: swarm.description,
      status: swarm.status,
      swarmName: swarm.name,
      children: swarm.agents.map(agent => ({
        id: `${swarm.name}-${agent.name}`,
        name: agent.name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
        type: 'agent' as const,
        description: agent.description,
        model: agent.model,
        swarmName: swarm.name,
      })),
    }))

    // Build Operations VP node
    const operationsNode: OrgNode = {
      id: 'operations',
      name: 'VP Operations',
      type: 'vp',
      description: operationsSwarm?.description || 'Cross-swarm coordination',
      status: operationsSwarm?.status,
      swarmName: 'operations',
      children: [
        // Operations agents
        ...(operationsSwarm?.agents || []).map(agent => ({
          id: `operations-${agent.name}`,
          name: agent.name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
          type: 'agent' as const,
          description: agent.description,
          model: agent.model,
          swarmName: 'operations',
        })),
        // Managed swarms
        ...swarmNodes,
      ],
    }

    // Build COO node
    const cooNode: OrgNode = {
      id: 'coo',
      name: 'Supreme Orchestrator',
      type: 'coo',
      description: 'COO - Manages all swarms',
      children: [operationsNode],
    }

    // Build CEO node
    const ceoNode: OrgNode = {
      id: 'ceo',
      name: 'CEO',
      type: 'ceo',
      description: 'You',
      children: [cooNode],
    }

    return ceoNode
  }

  const toggleExpand = useCallback((nodeId: string) => {
    setExpandedNodes(prev => {
      const next = new Set(prev)
      if (next.has(nodeId)) {
        next.delete(nodeId)
      } else {
        next.add(nodeId)
      }
      return next
    })
  }, [])

  const handleNodeClick = useCallback((node: OrgNode, e: React.MouseEvent) => {
    e.stopPropagation()

    if (node.children && node.children.length > 0) {
      toggleExpand(node.id)
    }

    if (onSelectNode) {
      onSelectNode(node)
    }

    // Navigate to swarm page if it's a swarm or agent
    if (node.swarmName) {
      router.push(`/swarm/${node.swarmName}`)
    }
  }, [onSelectNode, toggleExpand, router])

  const getNodeIcon = (type: OrgNode['type']) => {
    switch (type) {
      case 'ceo':
        return <Crown className="w-4 h-4" />
      case 'coo':
        return <Crown className="w-4 h-4" />
      case 'vp':
        return <Users className="w-4 h-4" />
      case 'swarm':
        return <Users className="w-4 h-4" />
      case 'agent':
        return <Bot className="w-4 h-4" />
    }
  }

  const getNodeColor = (type: OrgNode['type'], status?: string) => {
    if (status === 'paused') return 'border-zinc-600 bg-zinc-800/50 text-zinc-400'

    switch (type) {
      case 'ceo':
        return 'border-amber-500 bg-amber-500/10 text-amber-400'
      case 'coo':
        return 'border-purple-500 bg-purple-500/10 text-purple-400'
      case 'vp':
        return 'border-blue-500 bg-blue-500/10 text-blue-400'
      case 'swarm':
        return 'border-green-500 bg-green-500/10 text-green-400'
      case 'agent':
        return 'border-zinc-600 bg-zinc-800 text-zinc-300'
    }
  }

  const renderNode = (node: OrgNode, level: number = 0) => {
    const isExpanded = expandedNodes.has(node.id)
    const hasChildren = node.children && node.children.length > 0
    const isSelected = selectedNodeId === node.id

    return (
      <div key={node.id} className="flex flex-col">
        <div
          className={`
            flex items-center gap-2 px-3 py-2 rounded-lg border cursor-pointer
            transition-all duration-200 hover:scale-[1.02]
            ${getNodeColor(node.type, node.status)}
            ${isSelected ? 'ring-2 ring-white/50' : ''}
          `}
          style={{ marginLeft: `${level * 24}px` }}
          onClick={(e) => handleNodeClick(node, e)}
        >
          {hasChildren && (
            <span className="text-zinc-500">
              {isExpanded ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
            </span>
          )}
          <span className="flex-shrink-0">{getNodeIcon(node.type)}</span>
          <div className="flex-1 min-w-0">
            <div className="font-medium text-sm truncate">{node.name}</div>
            {node.description && (
              <div className="text-xs opacity-70 truncate">{node.description}</div>
            )}
          </div>
          {node.status === 'paused' && (
            <span className="text-xs px-1.5 py-0.5 bg-zinc-700 rounded">Paused</span>
          )}
          {node.type === 'agent' && node.model && (
            <span className="text-xs px-1.5 py-0.5 bg-zinc-700 rounded capitalize">
              {node.model}
            </span>
          )}
        </div>

        {hasChildren && isExpanded && (
          <div className="mt-1 space-y-1 border-l border-zinc-700 ml-4">
            {node.children!.map(child => renderNode(child, level + 1))}
          </div>
        )}
      </div>
    )
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-zinc-500">Loading organization...</div>
      </div>
    )
  }

  if (!orgTree) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-zinc-500">Failed to load organization</div>
      </div>
    )
  }

  return (
    <div className="p-4 bg-zinc-900 border border-zinc-800 rounded-lg">
      <div className="mb-4 flex items-center justify-between">
        <h3 className="text-lg font-semibold text-white">Organization Structure</h3>
        <div className="flex items-center gap-4 text-xs text-zinc-500">
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 rounded bg-amber-500/30 border border-amber-500"></div>
            <span>Executive</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 rounded bg-blue-500/30 border border-blue-500"></div>
            <span>Management</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 rounded bg-green-500/30 border border-green-500"></div>
            <span>Swarm</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 rounded bg-zinc-700 border border-zinc-600"></div>
            <span>Agent</span>
          </div>
        </div>
      </div>

      <div className="space-y-1 max-h-[500px] overflow-y-auto">
        {renderNode(orgTree)}
      </div>
    </div>
  )
}
