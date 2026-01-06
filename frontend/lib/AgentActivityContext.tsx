'use client'

import React, { createContext, useContext, useState, useCallback, useEffect, useRef } from 'react'
import { getChatWebSocket, type WebSocketEvent } from './websocket'

export interface AgentActivity {
  agent: string
  swarm: string
  status: 'active' | 'idle'
  currentTask?: string
  lastActive?: Date
}

// Activity panel specific types (for persistent display in chat)
export interface PanelAgentActivity {
  id: string
  name: string
  status: 'thinking' | 'working' | 'delegating' | 'complete' | 'error'
  description?: string
  startTime: Date
  endTime?: Date
}

export interface PanelToolActivity {
  id: string
  tool: string
  description: string
  status: 'running' | 'complete' | 'error'
  timestamp: Date
  endTime?: Date
  summary?: string
  agentName?: string
}

interface AgentActivityContextType {
  activities: Record<string, AgentActivity>
  isAgentActive: (swarm: string, agent: string) => boolean
  getAgentActivity: (swarm: string, agent: string) => AgentActivity | undefined
  getSwarmActiveCount: (swarm: string) => number
  // Panel activities (persistent across navigation)
  panelAgentActivities: PanelAgentActivity[]
  panelToolActivities: PanelToolActivity[]
  setPanelAgentActivities: React.Dispatch<React.SetStateAction<PanelAgentActivity[]>>
  setPanelToolActivities: React.Dispatch<React.SetStateAction<PanelToolActivity[]>>
  clearPanelActivities: () => void
}

const AgentActivityContext = createContext<AgentActivityContextType>({
  activities: {},
  isAgentActive: () => false,
  getAgentActivity: () => undefined,
  getSwarmActiveCount: () => 0,
  panelAgentActivities: [],
  panelToolActivities: [],
  setPanelAgentActivities: () => {},
  setPanelToolActivities: () => {},
  clearPanelActivities: () => {},
})

export function AgentActivityProvider({ children }: { children: React.ReactNode }) {
  const [activities, setActivities] = useState<Record<string, AgentActivity>>({})
  // Panel activities - persistent across navigation
  const [panelAgentActivities, setPanelAgentActivities] = useState<PanelAgentActivity[]>([])
  const [panelToolActivities, setPanelToolActivities] = useState<PanelToolActivity[]>([])
  const wsRef = useRef(getChatWebSocket())

  useEffect(() => {
    const ws = wsRef.current

    const handleEvent = (event: WebSocketEvent) => {
      if (event.type === 'agent_start') {
        const agentPath = event.agent || ''
        const [swarm, agent] = agentPath.includes('/')
          ? agentPath.split('/', 2)
          : ['unknown', agentPath]

        const key = `${swarm}/${agent}`
        setActivities((prev) => ({
          ...prev,
          [key]: {
            agent,
            swarm,
            status: 'active',
            currentTask: 'Working...',
            lastActive: new Date(),
          },
        }))
      }

      if (event.type === 'agent_complete') {
        const agentPath = event.agent || ''
        const [swarm, agent] = agentPath.includes('/')
          ? agentPath.split('/', 2)
          : ['unknown', agentPath]

        const key = `${swarm}/${agent}`
        setActivities((prev) => ({
          ...prev,
          [key]: {
            agent,
            swarm,
            status: 'idle',
            lastActive: new Date(),
          },
        }))
      }

      if (event.type === 'agent_spawn') {
        // When an agent is spawned via Task tool
        const agentName = event.agent || ''
        const description = event.description || ''

        // Agent spawn events use just the agent type name (e.g., "implementer", "researcher")
        const key = `subagent/${agentName}`
        setActivities((prev) => ({
          ...prev,
          [key]: {
            agent: agentName,
            swarm: 'subagent',
            status: 'active',
            currentTask: description,
            lastActive: new Date(),
          },
        }))
      }

      if (event.type === 'agent_complete_subagent') {
        // When a subagent completes
        const agentName = event.agent || ''
        const key = `subagent/${agentName}`
        setActivities((prev) => ({
          ...prev,
          [key]: {
            agent: agentName,
            swarm: 'subagent',
            status: 'idle',
            lastActive: new Date(),
          },
        }))
      }

      if (event.type === 'tool_start' && event.tool === 'Task') {
        // When a Task tool starts, we're spawning an agent
        const description = event.description || ''
        const match = description.match(/Spawning ([^\s]+)/)
        if (match) {
          const agentPath = match[1]
          const [swarm, agent] = agentPath.includes('/')
            ? agentPath.split('/', 2)
            : ['unknown', agentPath]

          const key = `${swarm}/${agent}`
          setActivities((prev) => ({
            ...prev,
            [key]: {
              agent,
              swarm,
              status: 'active',
              currentTask: description,
              lastActive: new Date(),
            },
          }))
        }
      }

      // Clear swarm activities when chat completes
      // NOTE: Panel activities (panelAgentActivities, panelToolActivities) are managed
      // by ChatPage to avoid double event handlers updating the same state
      if (event.type === 'chat_complete') {
        // Mark all swarm activities as idle
        setActivities((prev) => {
          const updated = { ...prev }
          for (const key in updated) {
            updated[key] = { ...updated[key], status: 'idle' }
          }
          return updated
        })
      }
    }

    // Only attach event handlers - don't manage connection lifecycle
    // The chat page or other components that need WebSocket will call connect()
    ws.on('*', handleEvent)

    return () => {
      ws.off('*', handleEvent)
    }
  }, [])

  const isAgentActive = useCallback(
    (swarm: string, agent: string): boolean => {
      const key = `${swarm}/${agent}`
      return activities[key]?.status === 'active'
    },
    [activities]
  )

  const getAgentActivity = useCallback(
    (swarm: string, agent: string): AgentActivity | undefined => {
      const key = `${swarm}/${agent}`
      return activities[key]
    },
    [activities]
  )

  const getSwarmActiveCount = useCallback(
    (swarm: string): number => {
      return Object.values(activities).filter(
        (a) => a.swarm.toLowerCase() === swarm.toLowerCase() && a.status === 'active'
      ).length
    },
    [activities]
  )

  const clearPanelActivities = useCallback(() => {
    setPanelAgentActivities([])
    setPanelToolActivities([])
  }, [])

  return (
    <AgentActivityContext.Provider
      value={{
        activities,
        isAgentActive,
        getAgentActivity,
        getSwarmActiveCount,
        panelAgentActivities,
        panelToolActivities,
        setPanelAgentActivities,
        setPanelToolActivities,
        clearPanelActivities,
      }}
    >
      {children}
    </AgentActivityContext.Provider>
  )
}

export function useAgentActivity() {
  return useContext(AgentActivityContext)
}
