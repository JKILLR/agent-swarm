'use client'

import { useState, useEffect } from 'react'
import { CheckSquare, Square, Plus, X, ChevronDown, ChevronRight, Trash2 } from 'lucide-react'

interface Todo {
  id: string
  text: string
  completed: boolean
  createdAt: string
}

const STORAGE_KEY = 'ceo-todos'

export default function CeoTodoPanel() {
  const [todos, setTodos] = useState<Todo[]>([])
  const [isOpen, setIsOpen] = useState(true)
  const [newTodo, setNewTodo] = useState('')
  const [isAdding, setIsAdding] = useState(false)

  // Load todos from localStorage on mount
  useEffect(() => {
    const saved = localStorage.getItem(STORAGE_KEY)
    if (saved) {
      try {
        setTodos(JSON.parse(saved))
      } catch (e) {
        console.error('Failed to parse saved todos:', e)
      }
    }
  }, [])

  // Save todos to localStorage whenever they change
  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(todos))
  }, [todos])

  const addTodo = () => {
    if (!newTodo.trim()) return

    const todo: Todo = {
      id: `todo-${Date.now()}`,
      text: newTodo.trim(),
      completed: false,
      createdAt: new Date().toISOString(),
    }
    setTodos([...todos, todo])
    setNewTodo('')
    setIsAdding(false)
  }

  const toggleTodo = (id: string) => {
    setTodos(
      todos.map((todo) =>
        todo.id === id ? { ...todo, completed: !todo.completed } : todo
      )
    )
  }

  const deleteTodo = (id: string) => {
    setTodos(todos.filter((todo) => todo.id !== id))
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      addTodo()
    } else if (e.key === 'Escape') {
      setNewTodo('')
      setIsAdding(false)
    }
  }

  const pendingCount = todos.filter((t) => !t.completed).length
  const completedCount = todos.filter((t) => t.completed).length

  return (
    <div className="border-t border-zinc-800 bg-[#0a0a0a]">
      {/* Header */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full px-3 py-2 flex items-center justify-between text-xs font-semibold text-zinc-500 uppercase tracking-wider hover:bg-zinc-800/30 transition-colors"
      >
        <span className="flex items-center gap-2">
          {isOpen ? (
            <ChevronDown className="w-3 h-3" />
          ) : (
            <ChevronRight className="w-3 h-3" />
          )}
          My TODOs
        </span>
        {pendingCount > 0 && (
          <span className="px-1.5 py-0.5 bg-violet-500/20 text-violet-400 rounded text-[10px] font-medium">
            {pendingCount}
          </span>
        )}
      </button>

      {isOpen && (
        <div className="px-2 pb-2">
          {/* Todo List */}
          <div className="space-y-1 max-h-48 overflow-y-auto">
            {todos.length === 0 && !isAdding && (
              <p className="px-2 py-1 text-xs text-zinc-600">No tasks yet</p>
            )}

            {/* Pending todos first */}
            {todos
              .filter((t) => !t.completed)
              .map((todo) => (
                <div
                  key={todo.id}
                  className="flex items-center gap-2 px-2 py-1 rounded hover:bg-zinc-800/50 group"
                >
                  <button
                    onClick={() => toggleTodo(todo.id)}
                    className="flex-shrink-0 text-zinc-500 hover:text-violet-400 transition-colors"
                  >
                    <Square className="w-3.5 h-3.5" />
                  </button>
                  <span className="flex-1 text-xs text-zinc-300 truncate">
                    {todo.text}
                  </span>
                  <button
                    onClick={() => deleteTodo(todo.id)}
                    className="flex-shrink-0 opacity-0 group-hover:opacity-100 text-zinc-600 hover:text-red-400 transition-all"
                  >
                    <Trash2 className="w-3 h-3" />
                  </button>
                </div>
              ))}

            {/* Completed todos */}
            {completedCount > 0 && (
              <div className="pt-1 mt-1 border-t border-zinc-800/50">
                {todos
                  .filter((t) => t.completed)
                  .map((todo) => (
                    <div
                      key={todo.id}
                      className="flex items-center gap-2 px-2 py-1 rounded hover:bg-zinc-800/50 group"
                    >
                      <button
                        onClick={() => toggleTodo(todo.id)}
                        className="flex-shrink-0 text-green-500 hover:text-green-400 transition-colors"
                      >
                        <CheckSquare className="w-3.5 h-3.5" />
                      </button>
                      <span className="flex-1 text-xs text-zinc-600 line-through truncate">
                        {todo.text}
                      </span>
                      <button
                        onClick={() => deleteTodo(todo.id)}
                        className="flex-shrink-0 opacity-0 group-hover:opacity-100 text-zinc-600 hover:text-red-400 transition-all"
                      >
                        <Trash2 className="w-3 h-3" />
                      </button>
                    </div>
                  ))}
              </div>
            )}
          </div>

          {/* Add Todo */}
          {isAdding ? (
            <div className="mt-2 flex items-center gap-1">
              <input
                type="text"
                value={newTodo}
                onChange={(e) => setNewTodo(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Enter task..."
                className="flex-1 px-2 py-1 bg-zinc-800 border border-zinc-700 rounded text-xs text-white placeholder-zinc-500 focus:outline-none focus:border-violet-500"
                autoFocus
              />
              <button
                onClick={addTodo}
                disabled={!newTodo.trim()}
                className="p-1 text-violet-400 hover:text-violet-300 disabled:text-zinc-600 transition-colors"
              >
                <Plus className="w-4 h-4" />
              </button>
              <button
                onClick={() => {
                  setNewTodo('')
                  setIsAdding(false)
                }}
                className="p-1 text-zinc-500 hover:text-zinc-300 transition-colors"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
          ) : (
            <button
              onClick={() => setIsAdding(true)}
              className="mt-2 w-full flex items-center justify-center gap-1 px-2 py-1 text-xs text-zinc-500 hover:text-zinc-300 hover:bg-zinc-800/50 rounded transition-colors"
            >
              <Plus className="w-3 h-3" />
              Add task
            </button>
          )}
        </div>
      )}
    </div>
  )
}
