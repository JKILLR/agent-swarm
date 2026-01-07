'use client'

import { useState, useEffect, createContext, useContext, useCallback } from 'react'
import { Menu, X, Terminal } from 'lucide-react'
import Sidebar from './Sidebar'

interface MobileLayoutContextType {
  isMobile: boolean
  isSidebarOpen: boolean
  toggleSidebar: () => void
  closeSidebar: () => void
  openSidebar: () => void
}

const MobileLayoutContext = createContext<MobileLayoutContextType>({
  isMobile: false,
  isSidebarOpen: false,
  toggleSidebar: () => {},
  closeSidebar: () => {},
  openSidebar: () => {},
})

export function useMobileLayout() {
  return useContext(MobileLayoutContext)
}

export default function MobileLayout({ children }: { children: React.ReactNode }) {
  const [isMobile, setIsMobile] = useState(false)
  const [isSidebarOpen, setIsSidebarOpen] = useState(false)

  // Check for mobile viewport
  useEffect(() => {
    const checkMobile = () => {
      const mobile = window.innerWidth < 768 // md breakpoint
      setIsMobile(mobile)
      // Close sidebar on resize to desktop
      if (!mobile) {
        setIsSidebarOpen(false)
      }
    }

    checkMobile()
    window.addEventListener('resize', checkMobile)
    return () => window.removeEventListener('resize', checkMobile)
  }, [])

  const toggleSidebar = useCallback(() => {
    setIsSidebarOpen((prev) => !prev)
  }, [])

  const closeSidebar = useCallback(() => {
    setIsSidebarOpen(false)
  }, [])

  const openSidebar = useCallback(() => {
    setIsSidebarOpen(true)
  }, [])

  // Prevent body scroll when sidebar is open on mobile
  useEffect(() => {
    if (isMobile && isSidebarOpen) {
      document.body.style.overflow = 'hidden'
    } else {
      document.body.style.overflow = ''
    }
    return () => {
      document.body.style.overflow = ''
    }
  }, [isMobile, isSidebarOpen])

  return (
    <MobileLayoutContext.Provider
      value={{ isMobile, isSidebarOpen, toggleSidebar, closeSidebar, openSidebar }}
    >
      <div className="flex h-screen h-[100dvh] overflow-hidden bg-[#0d0d0d]">
        {/* Mobile Header with Hamburger */}
        {isMobile && (
          <header className="fixed top-0 left-0 right-0 z-40 bg-[#0d0d0d]/95 backdrop-blur border-b border-zinc-800/50 h-14 flex items-center px-4 shadow-[0_1px_0_rgba(139,92,246,0.1)]">
            <button
              onClick={toggleSidebar}
              className="p-2 -ml-2 rounded-lg hover:bg-violet-500/10 active:bg-zinc-800/50 transition-all duration-200 touch-manipulation"
              aria-label="Toggle menu"
            >
              {isSidebarOpen ? (
                <X className="w-6 h-6 text-zinc-400" />
              ) : (
                <Menu className="w-6 h-6 text-zinc-400" />
              )}
            </button>
            <Terminal className="ml-3 w-5 h-5 text-emerald-400" />
            <span className="ml-2 font-semibold text-zinc-100">Agent Swarm</span>
          </header>
        )}

        {/* Sidebar Overlay (mobile only) */}
        {isMobile && isSidebarOpen && (
          <div
            className="fixed inset-0 z-30 bg-black/80"
            onClick={closeSidebar}
            aria-hidden="true"
          />
        )}

        {/* Sidebar */}
        <div
          className={`
            ${isMobile ? 'fixed inset-y-0 left-0 z-40 pt-14' : 'relative'}
            ${isMobile ? (isSidebarOpen ? 'translate-x-0' : '-translate-x-full') : 'translate-x-0'}
            transition-transform duration-300 ease-in-out
            ${isMobile ? 'w-[280px] max-w-[85vw]' : ''}
          `}
        >
          <Sidebar onNavigate={isMobile ? closeSidebar : undefined} />
        </div>

        {/* Main Content */}
        <main
          className={`
            flex-1 overflow-auto bg-[#0d0d0d]
            ${isMobile ? 'pt-14 w-full' : ''}
          `}
        >
          {children}
        </main>
      </div>
    </MobileLayoutContext.Provider>
  )
}
