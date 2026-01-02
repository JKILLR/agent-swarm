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

interface AgentActivityContextType {
  activities: Record<string, AgentActivity>
  isAgentActive: (swarm: string, agent: string) => boolean
  getAgentActivity: (swarm: string, agent: string) => AgentActivity | undefined
  getSwarmActiveCount: (swarm: string) => number
}

const AgentActivityContext = createContext<AgentActivityContextType>({
  activities: {},
  isAgentActive: () => false,
  getAgentActivity: () => undefined,
  getSwarmActiveCount: () => 0,
})

export function AgentActivityProvider({ children }: { children: React.ReactNode }) {
  const [activities, setActivities] = useState<Record<string, AgentActivity>>({})
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

      // Clear activities when chat completes
      if (event.type === 'chat_complete') {
        // Mark all as idle
        setActivities((prev) => {
          const updated = { ...prev }
          for (const key in updated) {
            updated[key] = { ...updated[key], status: 'idle' }
          }
          return updated
        })
      }
    }

    ws.on('*', handleEvent)
    ws.connect().catch(console.error)

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

  return (
    <AgentActivityContext.Provider
      value={{ activities, isAgentActive, getAgentActivity, getSwarmActiveCount }}
    >
      {children}
    </AgentActivityContext.Provider>
  )
}

export function useAgentActivity() {
  return useContext(AgentActivityContext)
}
