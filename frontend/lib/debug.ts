/**
 * Debug utilities for WebSocket and other subsystems
 */

const DEBUG_ENABLED = process.env.NODE_ENV === 'development'

/**
 * WebSocket debug logger object with log and error methods
 */
export const wsDebug = {
  log: (message: string, ...args: unknown[]): void => {
    if (!DEBUG_ENABLED) return
    const timestamp = new Date().toISOString().split('T')[1].slice(0, 12)
    console.log(`${timestamp} [WS] ${message}`, ...args)
  },

  error: (message: string, ...args: unknown[]): void => {
    if (!DEBUG_ENABLED) return
    const timestamp = new Date().toISOString().split('T')[1].slice(0, 12)
    console.error(`${timestamp} [WS:ERROR] ${message}`, ...args)
  },

  warn: (message: string, ...args: unknown[]): void => {
    if (!DEBUG_ENABLED) return
    const timestamp = new Date().toISOString().split('T')[1].slice(0, 12)
    console.warn(`${timestamp} [WS:WARN] ${message}`, ...args)
  },
}

/**
 * General debug logger
 */
export const debug = {
  log: (category: string, message: string, ...args: unknown[]): void => {
    if (!DEBUG_ENABLED) return
    const timestamp = new Date().toISOString().split('T')[1].slice(0, 12)
    console.log(`${timestamp} [${category}] ${message}`, ...args)
  },

  error: (category: string, message: string, ...args: unknown[]): void => {
    if (!DEBUG_ENABLED) return
    const timestamp = new Date().toISOString().split('T')[1].slice(0, 12)
    console.error(`${timestamp} [${category}:ERROR] ${message}`, ...args)
  },
}
