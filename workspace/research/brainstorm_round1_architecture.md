# Brainstorm Round 1: Memory Architecture Vision

**Date**: 2026-01-06
**Author**: COO (Architecture Brainstorm)
**Input**: `workspace/research/context_advancements.md`

---

## Executive Summary

This document presents an ideal memory architecture for agent-swarm, synthesizing insights from the latest AI memory research (2024-2025) with the existing codebase patterns. The architecture adopts:

1. **MemGPT-style tiered memory** for efficient context management
2. **Graph-based relationships** for agents, tasks, and decisions (building on existing MindGraph)
3. **MCP protocol** for standardized inter-agent context sharing
4. **Hybrid retrieval** (BM25 + vectors + reranking) for knowledge access

---

## Architecture Overview

```
+==============================================================================+
|                           AGENT-SWARM MEMORY ARCHITECTURE                     |
+==============================================================================+

                              +-------------------+
                              |    HUMAN (CEO)    |
                              |  Approval & Audit |
                              +--------+----------+
                                       |
                     Escalations & High-Level Decisions
                                       |
                                       v
+------------------------------------------------------------------------------+
|                         ORCHESTRATION LAYER (COO)                             |
|                                                                               |
|  +-----------------+  +-----------------+  +-----------------+                |
|  | COO Core Memory |  | Decision Cache  |  | Escalation Mgr  |                |
|  | (Always in ctx) |  | (Recent 50)     |  | (Active Issues) |                |
|  +-----------------+  +-----------------+  +-----------------+                |
|                                                                               |
|  Context Budget: ~30K tokens | Self-editing via memory tools                  |
+------------------------------------------------------------------------------+
                    |                    |                    |
         MCP Context Protocol    MCP Context Protocol    MCP Context Protocol
                    |                    |                    |
                    v                    v                    v
+------------------+      +------------------+      +------------------+
|   SWARM: Design  |      | SWARM: Swarm Dev |      |  SWARM: Life OS  |
|                  |      |                  |      |                  |
| +-------------+  |      | +-------------+  |      | +-------------+  |
| |Swarm Memory |  |      | |Swarm Memory |  |      | |Swarm Memory |  |
| |  STATE.md   |  |      | |  STATE.md   |  |      | |  STATE.md   |  |
| +-------------+  |      | +-------------+  |      | +-------------+  |
|                  |      |                  |      |                  |
| Agents:          |      | Agents:          |      | Agents:          |
| - Designer       |      | - Architect      |      | - Assistant      |
| - Reviewer       |      | - Implementer    |      | - Scheduler      |
| - Critic         |      | - Critic         |      | - Integrator     |
+------------------+      +------------------+      +------------------+
         |                         |                         |
         +------------+------------+------------+------------+
                      |                         |
                      v                         v
+------------------------------------------------------------------------------+
|                           MEMORY TIERS (MemGPT Pattern)                       |
+------------------------------------------------------------------------------+
|                                                                               |
|  TIER 1: CORE MEMORY (Always in context - ~2K tokens per agent)              |
|  +-----------------------------------------------------------------------+   |
|  | - Agent Identity (name, role, capabilities)                           |   |
|  | - Current Task (from Work Ledger)                                     |   |
|  | - Active Escalations (if any)                                         |   |
|  | - Key Constraints (permissions, boundaries)                           |   |
|  +-----------------------------------------------------------------------+   |
|                                                                               |
|  TIER 2: WORKING MEMORY (Session context - ~10K tokens, compressed)          |
|  +-----------------------------------------------------------------------+   |
|  | - Recent conversation (last 10 turns verbatim)                        |   |
|  | - Summarized earlier conversation (LLM-compressed)                    |   |
|  | - Related graph nodes (from semantic search)                          |   |
|  | - Pending mailbox messages                                            |   |
|  +-----------------------------------------------------------------------+   |
|                                                                               |
|  TIER 3: ARCHIVAL MEMORY (Long-term - vector indexed, infinite)              |
|  +-----------------------------------------------------------------------+   |
|  | - All historical conversations                                        |   |
|  | - Completed work items                                                |   |
|  | - Pattern library (successful approaches)                             |   |
|  | - Full knowledge base (docs, code, decisions)                         |   |
|  +-----------------------------------------------------------------------+   |
|                                                                               |
+------------------------------------------------------------------------------+
                                       |
                                       v
+------------------------------------------------------------------------------+
|                         GRAPH MEMORY LAYER (Mem0g Pattern)                    |
+------------------------------------------------------------------------------+
|                                                                               |
|  +-----------+        +-----------+        +-----------+                      |
|  |  AGENTS   |------->|   TASKS   |------->| DECISIONS |                      |
|  | (Identity)|<-------| (WorkItem)|<-------| (Outcomes)|                      |
|  +-----------+        +-----------+        +-----------+                      |
|       |                    |                    |                             |
|       |    +---------------+--------------------+                             |
|       |    |                                                                  |
|       v    v                                                                  |
|  +-----------+        +-----------+        +-----------+                      |
|  | CONCEPTS  |<------>|  FACTS    |<------>| MEMORIES  |                      |
|  | (Topics)  |        | (Learned) |        | (Episodic)|                      |
|  +-----------+        +-----------+        +-----------+                      |
|                                                                               |
|  Node Types: AGENT, TASK, DECISION, CONCEPT, FACT, MEMORY, PREFERENCE, GOAL  |
|  Edge Types: PARENT, CHILD, ASSOCIATION, TEMPORAL, DERIVED, OWNS, COMPLETED  |
|                                                                               |
+------------------------------------------------------------------------------+
                                       |
                                       v
+------------------------------------------------------------------------------+
|                         RETRIEVAL LAYER (Hybrid)                              |
+------------------------------------------------------------------------------+
|                                                                               |
|    Query: "How did we handle authentication in the trading bot?"             |
|                           |                                                   |
|         +-----------------+-----------------+                                 |
|         |                 |                 |                                 |
|         v                 v                 v                                 |
|    +----------+     +-----------+     +----------+                            |
|    |   BM25   |     |  Vector   |     |  Graph   |                            |
|    | (Exact)  |     | (Semantic)|     | (Relat.) |                            |
|    +----------+     +-----------+     +----------+                            |
|         |                 |                 |                                 |
|         +--------+--------+--------+--------+                                 |
|                  |                 |                                          |
|                  v                 v                                          |
|           +------------+    +------------+                                    |
|           |   Fusion   |    |  Reranker  |                                    |
|           |   (RRF)    |--->| (ColBERT)  |                                    |
|           +------------+    +------------+                                    |
|                                   |                                           |
|                                   v                                           |
|                          Top-K Results for Context                            |
|                                                                               |
+------------------------------------------------------------------------------+
                                       |
                                       v
+------------------------------------------------------------------------------+
|                         STORAGE LAYER                                         |
+------------------------------------------------------------------------------+
|                                                                               |
|  +-------------------+  +-------------------+  +-------------------+           |
|  |    PostgreSQL     |  |   Vector Store    |  |   File System     |           |
|  |  (Structured)     |  | (Embeddings)      |  | (Documents)       |           |
|  +-------------------+  +-------------------+  +-------------------+           |
|  | - Work Ledger     |  | - Ollama/OpenAI   |  | - STATE.md files  |           |
|  | - Escalations     |  | - 384/1536 dim    |  | - Session logs    |           |
|  | - Agent Registry  |  | - HNSW index      |  | - Design docs     |           |
|  | - Decision Log    |  | - ~1M vectors     |  | - Code comments   |           |
|  +-------------------+  +-------------------+  +-------------------+           |
|                                                                               |
+------------------------------------------------------------------------------+
```

---

## Component Deep Dive

### 1. Tiered Memory (MemGPT Pattern)

The MemGPT pattern treats LLM context as a scarce resource, implementing virtual memory hierarchy:

```
+==============================================================================+
|                         TIERED MEMORY DETAIL                                  |
+==============================================================================+

TIER 1: CORE MEMORY (~2,000 tokens)
+-------------------------------------------+
| Always loaded. Never evicted.             |
+-------------------------------------------+
| agent:                                    |
|   name: "implementer"                     |
|   swarm: "swarm_dev"                      |
|   role: "Write code based on designs"    |
|   tools: [Read, Write, Edit, Bash, Git]  |
|                                           |
| current_task:                             |
|   id: "WL-2026-0106-001"                  |
|   title: "Implement memory architecture" |
|   status: "in_progress"                   |
|   blockers: []                            |
|                                           |
| constraints:                              |
|   workspace: "/swarms/swarm_dev/workspace"|
|   can_modify_core: true                   |
|   requires_review: true                   |
+-------------------------------------------+

TIER 2: WORKING MEMORY (~10,000 tokens)
+-------------------------------------------+
| Session-scoped. Compressed as needed.     |
+-------------------------------------------+
| recent_messages: (last 10 verbatim)       |
|   [0]: user: "Implement the API endpoint" |
|   [1]: assistant: "I'll create..."        |
|   ...                                     |
|                                           |
| summarized_history: (LLM compressed)      |
|   "Earlier in this session, we:           |
|    - Designed the endpoint schema         |
|    - Reviewed authentication approach     |
|    - Decided on JWT tokens..."            |
|                                           |
| relevant_context: (from retrieval)        |
|   - Graph node: "JWT Implementation"      |
|   - Related task: "Auth system design"    |
|   - Past decision: "Use RS256 signing"    |
|                                           |
| pending_mailbox:                          |
|   - From: architect                       |
|     Subject: "Schema review complete"     |
+-------------------------------------------+

TIER 3: ARCHIVAL MEMORY (Infinite)
+-------------------------------------------+
| Long-term. Vector-indexed. On-demand.     |
+-------------------------------------------+
| Accessed via memory tools:                |
|   - memory_search(query) -> results       |
|   - memory_archive(content) -> node_id    |
|   - memory_recall(node_id) -> content     |
|                                           |
| Contains:                                 |
|   - All past sessions (compressed)        |
|   - Knowledge base documents              |
|   - Code patterns and examples            |
|   - Historical decisions and rationale    |
+-------------------------------------------+

MEMORY TOOLS (Agent Self-Editing)
+-------------------------------------------+
| Agents manage their own context:          |
+-------------------------------------------+
| memory_read(tier, key)                    |
|   - Read from specific memory tier        |
|                                           |
| memory_write(tier, key, value)            |
|   - Write to appropriate tier             |
|   - Core: Only agent identity updates     |
|   - Working: Session notes, findings      |
|   - Archival: Permanent storage           |
|                                           |
| memory_search(query, filters)             |
|   - Semantic search across all tiers      |
|   - Filter by: node_type, date, swarm     |
|                                           |
| memory_archive(content, metadata)         |
|   - Store important content long-term     |
|   - Auto-extract entities for graph       |
|                                           |
| memory_forget(node_id)                    |
|   - Mark content as deprecated            |
|   - (Soft delete for audit trail)         |
+-------------------------------------------+
```

### 2. Graph Memory Layer (Mem0g Pattern)

Building on the existing MindGraph, but specialized for agent operations:

```
+==============================================================================+
|                         GRAPH SCHEMA                                          |
+==============================================================================+

NODE TYPES (Extended from current MindGraph)
+-------------------------------------------+
| AGENT                                     |
|   - name: str                             |
|   - swarm: str                            |
|   - capabilities: list[str]               |
|   - active_since: datetime                |
|   - performance_score: float              |
+-------------------------------------------+
| TASK (from Work Ledger)                   |
|   - id: str (WL-YYYY-MMDD-NNN)            |
|   - title: str                            |
|   - status: pending|in_progress|done      |
|   - assigned_to: Agent                    |
|   - dependencies: list[Task]              |
|   - outputs: list[str] (file paths)       |
+-------------------------------------------+
| DECISION                                  |
|   - id: str (DEC-YYYY-MMDD-NNN)           |
|   - title: str                            |
|   - context: str                          |
|   - choice: str                           |
|   - rationale: str                        |
|   - made_by: Agent | "CEO"                |
|   - affected_tasks: list[Task]            |
+-------------------------------------------+
| CONCEPT (existing)                        |
| FACT (existing)                           |
| MEMORY (existing - episodic)              |
| PREFERENCE (user/system preferences)      |
| GOAL (active objectives)                  |
+-------------------------------------------+

EDGE TYPES (Extended)
+-------------------------------------------+
| Hierarchical:                             |
|   PARENT, CHILD                           |
|                                           |
| Ownership:                                |
|   OWNS (Agent -> Task)                    |
|   CREATED_BY (any -> Agent)               |
|   APPROVED_BY (Decision -> Agent)         |
|                                           |
| Workflow:                                 |
|   DEPENDS_ON (Task -> Task)               |
|   BLOCKS (Task -> Task)                   |
|   COMPLETED (Agent -> Task)               |
|   ESCALATED_TO (Task -> Agent)            |
|                                           |
| Semantic:                                 |
|   ASSOCIATION (any <-> any)               |
|   DERIVED (Fact -> source)                |
|   REFERENCES (any -> any)                 |
|                                           |
| Temporal:                                 |
|   TEMPORAL (chronological sequence)       |
|   SUPERSEDES (new -> old)                 |
+-------------------------------------------+

EXAMPLE GRAPH QUERIES
+-------------------------------------------+
| "What tasks does swarm_dev own?"          |
|   MATCH (a:AGENT {swarm: 'swarm_dev'})    |
|   -[:OWNS]->(t:TASK)                      |
|   RETURN t                                |
+-------------------------------------------+
| "What decisions affected auth?"           |
|   MATCH (d:DECISION)-[:REFERENCES]->(c)   |
|   WHERE c.label CONTAINS 'auth'           |
|   RETURN d, c                             |
+-------------------------------------------+
| "Who blocked on task X?"                  |
|   MATCH (t:TASK {id: 'X'})                |
|   <-[:BLOCKS]-(blocker:TASK)              |
|   -[:OWNS]-(a:AGENT)                      |
|   RETURN a, blocker                       |
+-------------------------------------------+
```

### 3. MCP Inter-Agent Context Protocol

Standardized context sharing between agents, replacing ad-hoc mailbox messages:

```
+==============================================================================+
|                         MCP CONTEXT PROTOCOL                                  |
+==============================================================================+

MCP MESSAGE FORMAT
+-------------------------------------------+
| {                                         |
|   "protocol": "mcp/1.0",                  |
|   "type": "context_share",                |
|   "from": {                               |
|     "agent": "architect",                 |
|     "swarm": "swarm_dev"                  |
|   },                                      |
|   "to": {                                 |
|     "agent": "implementer",               |
|     "swarm": "swarm_dev"                  |
|   },                                      |
|   "context": {                            |
|     "task_id": "WL-2026-0106-001",        |
|     "handoff_type": "design_complete",    |
|     "artifacts": [                        |
|       "workspace/DESIGN_AUTH.md"          |
|     ],                                    |
|     "key_decisions": [                    |
|       "DEC-2026-0106-001",                |
|       "DEC-2026-0106-002"                 |
|     ],                                    |
|     "constraints": [                      |
|       "Must use existing User model",     |
|       "JWT expiry: 1 hour"                |
|     ],                                    |
|     "summary": "Auth design complete..."  |
|   },                                      |
|   "metadata": {                           |
|     "timestamp": "2026-01-06T10:30:00Z",  |
|     "correlation_id": "abc123",           |
|     "priority": "normal"                  |
|   }                                       |
| }                                         |
+-------------------------------------------+

MCP CONTEXT TYPES
+-------------------------------------------+
| HANDOFF                                   |
|   - Design -> Implementation              |
|   - Implementation -> Review              |
|   - Task completion notification          |
|                                           |
| QUERY                                     |
|   - Request for information               |
|   - Clarification needed                  |
|   - Decision required                     |
|                                           |
| BROADCAST                                 |
|   - Swarm-wide announcements              |
|   - Priority changes                      |
|   - Escalation notifications              |
|                                           |
| SYNC                                      |
|   - State synchronization                 |
|   - Memory update propagation             |
|   - Conflict resolution                   |
+-------------------------------------------+

MCP INTEGRATION POINTS
+-------------------------------------------+
| 1. Agent Mailbox (existing)               |
|    - Wraps messages in MCP format         |
|    - Adds context extraction              |
|                                           |
| 2. Work Ledger (existing)                 |
|    - Task transitions trigger MCP         |
|    - Auto-generates handoff context       |
|                                           |
| 3. Escalation Protocol (existing)         |
|    - Escalations as MCP messages          |
|    - Resolution broadcasts                |
|                                           |
| 4. REST API (new endpoint)                |
|    - POST /api/mcp/send                   |
|    - GET /api/mcp/inbox/{agent}           |
|    - WebSocket /ws/mcp for real-time      |
+-------------------------------------------+
```

### 4. Hybrid Retrieval Pipeline

For finding relevant context from the knowledge base:

```
+==============================================================================+
|                         HYBRID RETRIEVAL PIPELINE                             |
+==============================================================================+

QUERY PROCESSING
+-------------------------------------------+
| Input: "How do we handle rate limiting?"  |
+-------------------------------------------+
                    |
                    v
+-------------------------------------------+
| Query Analyzer                            |
|   - Extract entities: "rate limiting"     |
|   - Identify intent: knowledge_lookup     |
|   - Generate search variants:             |
|     - "rate limit"                        |
|     - "throttling"                        |
|     - "request limits"                    |
+-------------------------------------------+
                    |
     +--------------+--------------+
     |              |              |
     v              v              v
+----------+  +-----------+  +----------+
| BM25     |  | Vector    |  | Graph    |
| Search   |  | Search    |  | Search   |
+----------+  +-----------+  +----------+
| Exact    |  | Semantic  |  | Related  |
| keyword  |  | embedding |  | nodes    |
| matches  |  | similarity|  | traversal|
+----------+  +-----------+  +----------+
     |              |              |
     v              v              v
+-------------------------------------------+
| Reciprocal Rank Fusion (RRF)              |
|   score = sum(1 / (k + rank_i))           |
|   k = 60 (standard)                       |
+-------------------------------------------+
                    |
                    v
+-------------------------------------------+
| Reranker (ColBERT or BGE)                 |
|   - Cross-encoder scoring                 |
|   - Consider query-document interaction   |
|   - Filter threshold: 0.7                 |
+-------------------------------------------+
                    |
                    v
+-------------------------------------------+
| Top-K Results (k=5 for context)           |
|   1. Rate Limiting Design Doc (0.92)      |
|   2. Trading Bot Throttle Code (0.87)     |
|   3. Decision: API Rate Limits (0.85)     |
|   4. Past Task: Implement Limits (0.82)   |
|   5. Related Concept: Backoff (0.78)      |
+-------------------------------------------+

INDEXING STRATEGY
+-------------------------------------------+
| What gets indexed:                        |
|   - STATE.md files (all swarms)           |
|   - Design documents (workspace/*.md)     |
|   - Code comments (docstrings, headers)   |
|   - Decision logs                         |
|   - Session summaries                     |
|   - Graph node descriptions               |
+-------------------------------------------+
| Chunking strategy:                        |
|   - Markdown: by ## section               |
|   - Code: by function/class               |
|   - Max chunk: 512 tokens                 |
|   - Overlap: 50 tokens                    |
+-------------------------------------------+
| Update triggers:                          |
|   - File change (inotify/FSEvents)        |
|   - Work item completion                  |
|   - Session end summarization             |
|   - Manual refresh endpoint               |
+-------------------------------------------+
```

---

## Integration with Existing Systems

### REST API Integration

```
+==============================================================================+
|                         API ENDPOINTS                                         |
+==============================================================================+

MEMORY TIER ENDPOINTS
+-------------------------------------------+
| GET  /api/memory/core/{agent}             |
|      Returns agent's core memory          |
|                                           |
| GET  /api/memory/working/{session}        |
|      Returns session working memory       |
|                                           |
| POST /api/memory/archive                  |
|      Store content in archival memory     |
|      Body: {content, metadata, node_type} |
|                                           |
| GET  /api/memory/search                   |
|      Hybrid search across all memory      |
|      Params: q, types[], swarm, limit     |
+-------------------------------------------+

GRAPH ENDPOINTS (extends existing /api/mind)
+-------------------------------------------+
| GET  /api/graph/node/{id}                 |
|      Get node with relationships          |
|                                           |
| GET  /api/graph/query                     |
|      Execute graph query                  |
|      Params: cypher-like query string     |
|                                           |
| POST /api/graph/relate                    |
|      Create relationship between nodes    |
|      Body: {source, target, edge_type}    |
|                                           |
| GET  /api/graph/context/{entity}          |
|      Get contextual subgraph for entity   |
+-------------------------------------------+

MCP ENDPOINTS
+-------------------------------------------+
| POST /api/mcp/send                        |
|      Send MCP message                     |
|      Body: MCP message format             |
|                                           |
| GET  /api/mcp/inbox/{agent}               |
|      Get pending messages for agent       |
|                                           |
| POST /api/mcp/acknowledge/{msg_id}        |
|      Mark message as processed            |
|                                           |
| WS   /ws/mcp/{agent}                      |
|      Real-time MCP message stream         |
+-------------------------------------------+

CONTEXT SYNTHESIS ENDPOINT (MYND pattern)
+-------------------------------------------+
| GET  /api/context/{agent}                 |
|      Returns assembled context for agent  |
|                                           |
|      Response:                            |
|      {                                    |
|        "core_memory": {...},              |
|        "working_memory": {...},           |
|        "relevant_context": [...],         |
|        "pending_messages": [...],         |
|        "active_escalations": [...],       |
|        "estimated_tokens": 12500          |
|      }                                    |
+-------------------------------------------+
```

### AgentExecutorPool Integration

```
+==============================================================================+
|                         EXECUTOR POOL MEMORY FLOW                             |
+==============================================================================+

AGENT STARTUP
+-------------------------------------------+
| 1. AgentExecutorPool.execute() called     |
|                                           |
| 2. Load core memory for agent:            |
|    context = MemoryManager.load_core(     |
|        agent_name, swarm_name             |
|    )                                      |
|                                           |
| 3. Load working memory for session:       |
|    working = MemoryManager.load_working(  |
|        session_id                         |
|    )                                      |
|                                           |
| 4. Retrieve relevant context:             |
|    relevant = HybridRetriever.search(     |
|        task_description,                  |
|        limit=5                            |
|    )                                      |
|                                           |
| 5. Check pending MCP messages:            |
|    messages = MCPManager.get_inbox(       |
|        agent_name                         |
|    )                                      |
|                                           |
| 6. Assemble system prompt:                |
|    prompt = ContextAssembler.build(       |
|        core, working, relevant, messages  |
|    )                                      |
|                                           |
| 7. Execute agent with assembled context   |
+-------------------------------------------+

AGENT EXECUTION
+-------------------------------------------+
| During execution, agent has access to:    |
|                                           |
| - memory_search(query)                    |
|   -> Searches archival memory             |
|                                           |
| - memory_write(key, value)                |
|   -> Updates working memory               |
|                                           |
| - memory_archive(content)                 |
|   -> Stores to archival memory            |
|                                           |
| - mcp_send(to, context)                   |
|   -> Sends MCP message to another agent   |
|                                           |
| - graph_query(query)                      |
|   -> Queries relationship graph           |
+-------------------------------------------+

AGENT COMPLETION
+-------------------------------------------+
| 1. Save working memory updates            |
|                                           |
| 2. Extract learnings for archival:        |
|    - Key decisions made                   |
|    - Patterns discovered                  |
|    - Errors encountered                   |
|                                           |
| 3. Update graph relationships:            |
|    - Task completion edges                |
|    - New entity nodes                     |
|    - Association edges                    |
|                                           |
| 4. Generate session summary if needed     |
|                                           |
| 5. Send MCP handoff if task continues     |
+-------------------------------------------+
```

---

## Data Flow Diagram

```
+==============================================================================+
|                         COMPLETE DATA FLOW                                    |
+==============================================================================+

                           USER REQUEST
                                |
                                v
                    +---------------------+
                    |   Frontend (Next.js) |
                    +---------------------+
                                |
                    WebSocket /ws/chat
                                |
                                v
+------------------------------------------------------------------------------+
|                           BACKEND (FastAPI)                                   |
+------------------------------------------------------------------------------+
|                                                                               |
|   +-------------------+                                                       |
|   | /ws/chat Handler  |                                                       |
|   +-------------------+                                                       |
|           |                                                                   |
|           | 1. Parse request                                                  |
|           |                                                                   |
|           v                                                                   |
|   +-------------------+       +-------------------+                           |
|   | Context Assembler |<----->| Memory Manager    |                           |
|   +-------------------+       +-------------------+                           |
|           |                          |                                        |
|           | 2. Build context         | - Core memory                          |
|           |                          | - Working memory                       |
|           |                          | - Session summaries                    |
|           v                                                                   |
|   +-------------------+       +-------------------+                           |
|   | AgentExecutorPool |<----->| Hybrid Retriever  |                           |
|   +-------------------+       +-------------------+                           |
|           |                          |                                        |
|           | 3. Execute COO           | - BM25 search                          |
|           |                          | - Vector search                        |
|           |                          | - Graph traversal                      |
|           v                          | - Reranking                            |
|   +-------------------+                                                       |
|   | Claude CLI        |       +-------------------+                           |
|   | (subprocess)      |<----->| MCP Manager       |                           |
|   +-------------------+       +-------------------+                           |
|           |                          |                                        |
|           | 4. Agent uses tools:     | - Send/receive messages                |
|           |    - memory_search       | - Context handoffs                     |
|           |    - memory_write        | - Broadcasts                           |
|           |    - mcp_send                                                     |
|           |    - graph_query                                                  |
|           |                                                                   |
|           v                                                                   |
|   +-------------------+       +-------------------+                           |
|   | Event Stream      |------>| WebSocket         |                           |
|   | (JSON)            |       | Broadcast         |                           |
|   +-------------------+       +-------------------+                           |
|                                       |                                       |
+------------------------------------------------------------------------------+
                                        |
                                        v
                               USER RESPONSE
                                        |
+------------------------------------------------------------------------------+
|                           STORAGE LAYER                                       |
+------------------------------------------------------------------------------+
|                                                                               |
|   +---------------+   +---------------+   +---------------+                   |
|   |   SQLite/     |   |   Vector DB   |   |  File System  |                   |
|   |   PostgreSQL  |   | (Chroma/FAISS)|   |               |                   |
|   +---------------+   +---------------+   +---------------+                   |
|   | - Work Ledger |   | - Embeddings  |   | - STATE.md    |                   |
|   | - Escalations |   | - 384-dim     |   | - Sessions    |                   |
|   | - Jobs        |   | - HNSW index  |   | - Designs     |                   |
|   | - Graph nodes |   |               |   | - Knowledge   |                   |
|   +---------------+   +---------------+   +---------------+                   |
|                                                                               |
+------------------------------------------------------------------------------+
```

---

## Implementation Phases

### Phase 1: Foundation (Quick Wins)

| Task | Effort | Impact | Description |
|------|--------|--------|-------------|
| Extend MindGraph with new node types | Low | Medium | Add AGENT, TASK, DECISION node types |
| Add memory tools to agent prompts | Low | High | Enable self-editing memory |
| Implement context compression | Medium | High | LLM summarization of older messages |
| Create `/api/context/{agent}` endpoint | Low | High | Unified context assembly |

### Phase 2: Retrieval Infrastructure

| Task | Effort | Impact | Description |
|------|--------|--------|-------------|
| Add vector embeddings (Ollama) | Medium | High | Embed STATE.md, designs, code |
| Implement BM25 search | Low | Medium | Exact keyword matching |
| Create hybrid retrieval pipeline | Medium | High | Combine BM25 + vectors + graph |
| Add reranking (BGE/ColBERT) | Medium | High | Improve retrieval quality |

### Phase 3: MCP Integration

| Task | Effort | Impact | Description |
|------|--------|--------|-------------|
| Define MCP message schema | Low | Medium | Standardize context format |
| Migrate Mailbox to MCP | Medium | High | Wrap existing in MCP protocol |
| Add MCP REST endpoints | Low | Medium | API for message sending |
| Add MCP WebSocket channel | Medium | High | Real-time message delivery |

### Phase 4: Tiered Memory

| Task | Effort | Impact | Description |
|------|--------|--------|-------------|
| Implement core memory tier | Medium | High | Always-in-context agent identity |
| Implement working memory tier | Medium | High | Session-scoped with compression |
| Implement archival memory tier | High | Transformative | Vector-indexed long-term storage |
| Add memory management tools | Medium | High | Agent self-editing capabilities |

---

## Success Metrics

| Metric | Current | Target | How to Measure |
|--------|---------|--------|----------------|
| Context utilization | ~50K tokens | ~25K tokens | Avg tokens per agent call |
| Retrieval accuracy | Manual | >85% relevant | Human eval of top-5 results |
| Agent handoff quality | Informal | Structured MCP | % handoffs with full context |
| Memory recall | Session-only | Cross-session | Can agent reference past work? |
| Graph connectivity | ~20 edges/node | ~50 edges/node | Avg edge count |

---

## Risk Analysis

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Vector DB performance at scale | Medium | High | Start with FAISS, scale to Chroma |
| Memory tool misuse | Low | Medium | Validation in tool handlers |
| MCP message flooding | Low | Medium | Rate limiting, priority queues |
| Graph query performance | Medium | Medium | Index optimization, caching |
| Context budget overflow | High | Medium | Hard limits + compression |

---

## Appendix: Technology Choices

| Component | Recommended | Alternative | Rationale |
|-----------|-------------|-------------|-----------|
| Vector Store | Chroma | FAISS, Qdrant | Simple, persistent, good for <1M vectors |
| Embeddings | Ollama (all-MiniLM) | OpenAI ada-002 | Local, free, 384-dim is sufficient |
| Graph Storage | SQLite + JSON | Neo4j, Neptune | Simple start, can migrate later |
| Reranker | BGE-reranker-base | ColBERT v2 | Good balance of quality/speed |
| BM25 | Rank-BM25 | Elasticsearch | Pure Python, no external deps |

---

## References

- [MemGPT Paper](https://arxiv.org/abs/2310.08560) - Virtual memory for LLMs
- [Mem0 Research](https://arxiv.org/abs/2504.19413) - Production memory systems
- [MCP Specification](https://modelcontextprotocol.io/) - Anthropic's context protocol
- [Hybrid Retrieval Study](https://arxiv.org/abs/2408.16672) - IBM three-way retrieval
- [GraphRAG](https://arxiv.org/abs/2404.16130) - Microsoft's graph-based RAG

---

*This architecture represents the ideal end-state. Implementation should proceed incrementally, validating each phase before expanding.*
