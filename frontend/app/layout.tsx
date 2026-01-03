import type { Metadata, Viewport } from 'next'
import { JetBrains_Mono } from 'next/font/google'
import './globals.css'
import Providers from '@/components/Providers'
import MobileLayout from '@/components/MobileLayout'

const jetbrainsMono = JetBrains_Mono({
  subsets: ['latin'],
  variable: '--font-mono',
})

export const metadata: Metadata = {
  title: 'Agent Swarm',
  description: 'Hierarchical AI Agent Management System',
}

export const viewport: Viewport = {
  width: 'device-width',
  initialScale: 1,
  maximumScale: 1,
  userScalable: false,
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className="dark">
      <body className={`${jetbrainsMono.className} bg-[#0d0d0d] text-zinc-100`}>
        <Providers>
          <MobileLayout>
            {children}
          </MobileLayout>
        </Providers>
      </body>
    </html>
  )
}
