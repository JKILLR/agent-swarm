'use client'

import { AgentActivityProvider } from '@/lib/AgentActivityContext'

export default function Providers({ children }: { children: React.ReactNode }) {
  return (
    <AgentActivityProvider>
      {children}
    </AgentActivityProvider>
  )
}
