/**
 * Debug utilities for WebSocket and other subsystems
 */

const DEBUG_ENABLED = process.env.NODE_ENV === 'development'

/**
 * WebSocket debug logger
 */
export function wsDebug(category: string, message: string, data?: unknown): void {
  if (!DEBUG_ENABLED) return

  const timestamp = new Date().toISOString().split('T')[1].slice(0, 12)
  const prefix = `[WS:${category}]`

  if (data !== undefined) {
    console.log(`${timestamp} ${prefix} ${message}`, data)
  } else {
    console.log(`${timestamp} ${prefix} ${message}`)
  }
}

/**
 * General debug logger
 */
export function debug(category: string, message: string, data?: unknown): void {
  if (!DEBUG_ENABLED) return

  const timestamp = new Date().toISOString().split('T')[1].slice(0, 12)
  const prefix = `[${category}]`

  if (data !== undefined) {
    console.log(`${timestamp} ${prefix} ${message}`, data)
  } else {
    console.log(`${timestamp} ${prefix} ${message}`)
  }
}
