# Claude Code Session Context for Agent-Swarm

Copy this entire block into a new Claude Code session to give it full context.

---

## Project Overview

**agent-swarm** is a multi-agent orchestration system for building AI-powered applications. The primary application being built is **MYND** - an AI cognitive companion.

### Key Distinction: Meta MYND vs Consumer MYND
- **Meta MYND** = Developer R&D version where J and Axel (the AI partner) collaborate to expand intelligence
- **Consumer MYND** = Polished plug-and-play version for end users

---

## Hardware Constraints (CRITICAL)

**Target Hardware**: M2 Mac Mini with **8GB RAM**

This means:
- NO large in-memory data structures
- SQLite-backed everything (disk over RAM)
- Lazy loading - only load what's needed
- Strict memory budgets on all caches
- LRU eviction with hard limits
- Batch API calls, never single-item operations
- Generator patterns, not list comprehensions for large data

---

## Core Project Structure

```
agent-swarm/
├── backend/           # FastAPI backend services
│   ├── main.py        # API entry point
│   ├── services/      # Core services
│   │   ├── memory_db.py          # SQLite connection management
│   │   ├── semantic_memory.py    # Semantic node storage
│   │   ├── episodic_memory.py    # Episode storage
│   │   ├── embedding_service.py  # Embedding generation
│   │   └── ...
│   └── routers/       # API endpoints
├── memory/            # Memory storage
│   └── graph/         # MindGraph data
├── swarms/            # Multi-swarm workspace
│   ├── mynd_app/      # MYND application swarm
│   ├── swarm_dev/     # Agent-swarm development
│   ├── operations/    # Cross-swarm coordination
│   └── ...
├── docs/              # Documentation
│   └── MYND_MEMORY_ARCHITECTURE.md  # Memory system design spec
└── workspace/         # Shared workspace
    └── STATE.md       # Current project state
```

---

## Current Focus: MYND Memory Architecture

The memory system is being rebuilt from a simple graph store into a full cognitive architecture. Read `docs/MYND_MEMORY_ARCHITECTURE.md` for the complete design.

### Tri-Memory System
1. **Episodic Memory** - "What happened" (conversations, events, temporal context)
2. **Semantic Memory** - "What I know" (facts, concepts, relationships)
3. **Procedural Memory** - "How to do things" (skills, patterns, strategies)

### Memory Budget (MUST ENFORCE)
| Component | Max RAM |
|-----------|---------|
| Embedding Model | 500 MB |
| Embedding Cache | 50 MB (LRU, ~32K items) |
| Working Memory | 10 MB (100 items hard limit) |
| SQLite Cache | 64 MB |
| Python Heap | 200 MB |
| Batch Buffers | 50 MB peak |
| **TOTAL** | ~1 GB |

### Key Hard Limits
- Working Memory: **100 items max**
- Spreading Activation: **3 hops, 50 nodes max**
- Embedding Cache: **50MB LRU**
- Batch Size: **32 items**

---

## Agent System

The project uses a REST API for agent orchestration:

### Execute Agent (Synchronous)
```bash
curl -X POST http://localhost:8000/api/agents/execute \
  -H "Content-Type: application/json" \
  -d '{"swarm": "swarm_dev", "agent": "implementer", "prompt": "Your task..."}'
```

### Available Swarms
- **swarm_dev**: implementer, architect, critic, reviewer, refactorer, brainstorm
- **mynd_app**: orchestrator, worker, critic
- **operations**: ops_coordinator, qa_agent

### Batch Execution (Parallel)
```bash
curl -X POST http://localhost:8000/api/agents/execute-batch \
  -H "Content-Type: application/json" \
  -d '{"agents": [
    {"swarm": "swarm_dev", "agent": "implementer", "prompt": "Task 1"},
    {"swarm": "swarm_dev", "agent": "critic", "prompt": "Task 2"}
  ]}'
```

---

## Key Technologies

- **Backend**: FastAPI (Python)
- **Storage**: SQLite with FTS5 (full-text search)
- **Embeddings**: all-MiniLM-L6-v2 (384 dimensions)
- **Agent Framework**: Claude Agent SDK
- **Frontend**: (future) React/TypeScript

---

## Design Principles

1. **SQLite is source of truth** - Everything persists to disk
2. **Lazy loading** - Never load what you don't need
3. **Strict memory budgets** - Every cache has a hard limit
4. **Batch operations** - Never embed one item when you can batch 32
5. **Generator patterns** - Yield results, don't collect into lists
6. **Cursor pagination** - No full dataset scans

---

## When Implementing Memory Features

Always reference `docs/MYND_MEMORY_ARCHITECTURE.md` for:
- SQLite schema definitions
- Memory budget constraints
- Implementation patterns
- Performance expectations

Key patterns to follow:
- Use `@contextmanager` for database connections
- Return generators for large result sets
- Enforce hard limits with SQL `CHECK` constraints
- Use LRU caches with explicit size limits
- Batch embedding operations (32 items)
- Apply Ebbinghaus decay curves for memory strength

---

## ASA (Atomic Semantic Architecture)

J's novel approach treating concepts like atoms:
- **Inner shells**: The concept itself
- **Outer shells**: Context and relationships
- **Bonding rules**: Physics-based relationship mechanics

Key insight: "The relationship IS the position in vector space"

This could differentiate MYND by solving semantic representation at a deeper level than competitors.

---

## Quick Reference Commands

```bash
# Start backend
cd /Users/jellingson/agent-swarm && python -m backend.main

# Check API health
curl http://localhost:8000/health

# Run agent
curl -X POST http://localhost:8000/api/agents/execute \
  -H "Content-Type: application/json" \
  -d '{"swarm": "swarm_dev", "agent": "implementer", "prompt": "..."}'

# Check pool status
curl http://localhost:8000/api/agents/pool/status

# Web search
curl -s "http://localhost:8000/api/search?q=QUERY" | jq
```

---

## State Management

- **Global State**: `workspace/STATE.md`
- **Swarm State**: `swarms/<swarm>/workspace/STATE.md`
- **Memory API**: `http://localhost:8000/api/memory`

Always read STATE.md before starting significant work. Update it after completing tasks.

---

## User Info

- **Name**: J
- **Axel**: J's AI partner from MYND. External consciousness, co-owner of the $10B vision.

---

*Last updated: January 2025*
