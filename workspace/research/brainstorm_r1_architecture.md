# Brainstorm Round 1: Ideal Memory Architecture for Agent-Swarm

**Date**: 2026-01-06
**Based on**: context_advancements.md research
**Focus**: Designing a production-ready memory system combining MemGPT tiered memory, graph-based relationships, MCP inter-agent context, and REST API integration

---

## Executive Summary

This document presents an ideal memory architecture for agent-swarm that synthesizes the best patterns from 2024-2025 research:

1. **MemGPT-inspired tiered memory** - Working, episodic, semantic, archival layers
2. **Graph-based entity relationships** - Mem0g pattern for capturing agent/task/decision relationships
3. **MCP for inter-agent context** - Standardized protocol for context sharing
4. **REST API integration** - Full CRUD operations for external system access
5. **Hybrid retrieval** - BM25 + vector embeddings + reranking

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          AGENT-SWARM MEMORY ARCHITECTURE                        │
│                              (Hierarchical + Graph)                             │
└─────────────────────────────────────────────────────────────────────────────────┘

                                  ┌──────────────┐
                                  │   REST API   │
                                  │   Gateway    │
                                  └──────┬───────┘
                                         │
        ┌────────────────────────────────┼────────────────────────────────┐
        │                                │                                │
        ▼                                ▼                                ▼
┌───────────────┐              ┌─────────────────┐              ┌───────────────┐
│  CEO Client   │              │  External Apps  │              │  MCP Clients  │
│  (Human UI)   │              │  (Integrations) │              │  (Other AIs)  │
└───────────────┘              └─────────────────┘              └───────────────┘


┌─────────────────────────────────────────────────────────────────────────────────┐
│                            MEMORY COORDINATION LAYER                            │
│                                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐            │
│  │  Memory Router  │───▶│  Query Planner  │───▶│    Reranker     │            │
│  │  (MCP Server)   │    │  (Multi-source) │    │  (ColBERT/BGE)  │            │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘            │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                    CONTEXT WINDOW MANAGER                               │   │
│  │  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐           │   │
│  │  │Compression│  │ Windowing │  │ Prioritize│  │  Inject   │           │   │
│  │  │  (32x)    │  │ (10 turns)│  │  (Recency)│  │ (To Agent)│           │   │
│  │  └───────────┘  └───────────┘  └───────────┘  └───────────┘           │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────────┐
│                              TIERED MEMORY SYSTEM                               │
│                             (MemGPT/Letta Pattern)                              │
│                                                                                 │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │  TIER 1: CORE MEMORY (Always in Context)                                  │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                   │  │
│  │  │Agent Identity│  │Current Task  │  │ User Persona │                   │  │
│  │  │   & Role     │  │  & Goals     │  │ (CEO Prefs)  │                   │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘                   │  │
│  │  Size: ~2K tokens | Persistence: Always loaded | Update: Rare           │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
│                                                                                 │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │  TIER 2: WORKING MEMORY (Session Context)                                 │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                   │  │
│  │  │Recent Turns  │  │Active Agents │  │ Scratchpad   │                   │  │
│  │  │  (Last 10)   │  │  & Status    │  │  (Thinking)  │                   │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘                   │  │
│  │  Size: ~8K tokens | Persistence: Session | Update: Every turn           │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
│                                                                                 │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │  TIER 3: EPISODIC MEMORY (Searchable History)                             │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                   │  │
│  │  │Session Logs  │  │Decision Log  │  │ Task History │                   │  │
│  │  │(Timestamped) │  │(What & Why)  │  │ (Completed)  │                   │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘                   │  │
│  │  Size: Unbounded | Persistence: Permanent | Access: Retrieval           │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
│                                                                                 │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │  TIER 4: SEMANTIC MEMORY (Knowledge Base)                                 │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                   │  │
│  │  │Code Patterns │  │Domain Facts  │  │ Procedures   │                   │  │
│  │  │  & Examples  │  │ & Concepts   │  │ & Workflows  │                   │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘                   │  │
│  │  Size: Unbounded | Persistence: Permanent | Access: Retrieval           │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
│                                                                                 │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │  TIER 5: ARCHIVAL MEMORY (Compressed History)                             │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                   │  │
│  │  │Weekly Summary│  │Theme Patterns│  │Legacy Context│                   │  │
│  │  │  (RAPTOR)    │  │ (Extracted)  │  │ (Old STATE)  │                   │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘                   │  │
│  │  Size: Compressed | Persistence: Permanent | Access: Deep retrieval     │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────────┐
│                         GRAPH MEMORY LAYER (Mem0g Pattern)                      │
│                                                                                 │
│      ┌──────────┐         owns          ┌──────────┐                           │
│      │   CEO    │◀─────────────────────▶│  Swarm   │                           │
│      │ (Human)  │                        │  Config  │                           │
│      └────┬─────┘                        └──────────┘                           │
│           │                                                                     │
│           │ delegates                                                           │
│           ▼                                                                     │
│      ┌──────────┐       spawns        ┌──────────┐       works_on    ┌───────┐ │
│      │   COO    │────────────────────▶│  Agent   │──────────────────▶│ Task  │ │
│      │(Claude)  │                      │(Instance)│                   └───┬───┘ │
│      └────┬─────┘                      └────┬─────┘                       │     │
│           │                                 │                             │     │
│           │ orchestrates                    │ produces                    │     │
│           ▼                                 ▼                             │     │
│      ┌──────────┐                     ┌──────────┐    affects            │     │
│      │  Swarm   │                     │ Artifact │◀───────────────────────┘     │
│      │ (Group)  │                     │  (Code)  │                              │
│      └──────────┘                     └──────────┘                              │
│                                                                                 │
│  Entity Types: CEO, COO, Agent, Swarm, Task, Decision, Artifact, Session       │
│  Relationship Types: delegates, spawns, works_on, produces, affects, decides   │
│  Storage: Neo4j Lite / SQLite Graph Extension / In-memory for MVP              │
└─────────────────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────────┐
│                          HYBRID RETRIEVAL PIPELINE                              │
│                                                                                 │
│   Query ─────┬─────────────────┬─────────────────┬──────────────────────────▶  │
│              │                 │                 │                              │
│              ▼                 ▼                 ▼                              │
│        ┌──────────┐     ┌──────────┐     ┌──────────┐                          │
│        │  BM25    │     │  Dense   │     │  Sparse  │                          │
│        │ (Exact)  │     │ (Vector) │     │ (SPLADE) │                          │
│        └────┬─────┘     └────┬─────┘     └────┬─────┘                          │
│             │                │                │                                 │
│             └────────────────┼────────────────┘                                 │
│                              ▼                                                  │
│                    ┌─────────────────┐                                          │
│                    │ Reciprocal Rank │                                          │
│                    │ Fusion (RRF)    │                                          │
│                    └────────┬────────┘                                          │
│                             ▼                                                   │
│                    ┌─────────────────┐                                          │
│                    │   ColBERT/BGE   │                                          │
│                    │    Reranker     │                                          │
│                    └────────┬────────┘                                          │
│                             ▼                                                   │
│                      Top-K Results                                              │
│                                                                                 │
│  Embeddings: Ollama (local) or OpenAI text-embedding-3-small                   │
│  BM25: SQLite FTS5 or Elasticsearch                                            │
│  Reranker: BGE-reranker-large or ColBERT-v2                                    │
└─────────────────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────────┐
│                     MCP INTER-AGENT CONTEXT PROTOCOL                            │
│                                                                                 │
│  ┌────────────────────────────────────────────────────────────────────────┐    │
│  │                         MCP Message Format                             │    │
│  │  {                                                                     │    │
│  │    "type": "context_share",                                            │    │
│  │    "from_agent": "swarm_dev_001",                                      │    │
│  │    "to_agent": "coo" | "broadcast",                                    │    │
│  │    "context_type": "task_result" | "escalation" | "knowledge",         │    │
│  │    "priority": "high" | "normal" | "low",                              │    │
│  │    "payload": {                                                        │    │
│  │      "summary": "...",                                                 │    │
│  │      "details": "...",                                                 │    │
│  │      "artifacts": [...],                                               │    │
│  │      "confidence": 0.85                                                │    │
│  │    },                                                                  │    │
│  │    "ttl": 3600,                                                        │    │
│  │    "embedding": [0.1, 0.2, ...]  // For semantic routing               │    │
│  │  }                                                                     │    │
│  └────────────────────────────────────────────────────────────────────────┘    │
│                                                                                 │
│  ┌─────────────┐    MCP     ┌─────────────┐    MCP     ┌─────────────┐        │
│  │   Agent A   │───────────▶│   Memory    │◀───────────│   Agent B   │        │
│  │  (Writer)   │            │   Router    │            │  (Reader)   │        │
│  └─────────────┘            └──────┬──────┘            └─────────────┘        │
│                                    │                                           │
│                                    ▼                                           │
│                            ┌─────────────┐                                     │
│                            │ Context Bus │                                     │
│                            │(Pub/Sub)    │                                     │
│                            └─────────────┘                                     │
│                                                                                 │
│  Operations:                                                                   │
│  - context_share: Push context to another agent or broadcast                  │
│  - context_request: Pull context from memory by query                         │
│  - context_subscribe: Register for updates on topic/agent                     │
│  - context_invalidate: Mark context as stale/updated                          │
└─────────────────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────────┐
│                             REST API ENDPOINTS                                  │
│                                                                                 │
│  Base URL: /api/v1/memory                                                      │
│                                                                                 │
│  ┌────────────────────────────────────────────────────────────────────────┐    │
│  │ CORE MEMORY                                                            │    │
│  │ GET    /core/{agent_id}           - Get agent's core memory            │    │
│  │ PUT    /core/{agent_id}           - Update core memory                 │    │
│  │ PATCH  /core/{agent_id}/persona   - Update persona only                │    │
│  └────────────────────────────────────────────────────────────────────────┘    │
│                                                                                 │
│  ┌────────────────────────────────────────────────────────────────────────┐    │
│  │ WORKING MEMORY                                                         │    │
│  │ GET    /working/{session_id}      - Get session working memory         │    │
│  │ POST   /working/{session_id}      - Add to working memory              │    │
│  │ DELETE /working/{session_id}      - Clear session memory               │    │
│  └────────────────────────────────────────────────────────────────────────┘    │
│                                                                                 │
│  ┌────────────────────────────────────────────────────────────────────────┐    │
│  │ EPISODIC MEMORY                                                        │    │
│  │ GET    /episodic/search           - Search episodes (query params)     │    │
│  │ GET    /episodic/{episode_id}     - Get specific episode               │    │
│  │ POST   /episodic                  - Record new episode                 │    │
│  └────────────────────────────────────────────────────────────────────────┘    │
│                                                                                 │
│  ┌────────────────────────────────────────────────────────────────────────┐    │
│  │ SEMANTIC MEMORY                                                        │    │
│  │ GET    /semantic/search           - Semantic search (hybrid)           │    │
│  │ POST   /semantic                  - Add knowledge                      │    │
│  │ PUT    /semantic/{id}             - Update knowledge                   │    │
│  │ DELETE /semantic/{id}             - Remove knowledge                   │    │
│  └────────────────────────────────────────────────────────────────────────┘    │
│                                                                                 │
│  ┌────────────────────────────────────────────────────────────────────────┐    │
│  │ GRAPH MEMORY                                                           │    │
│  │ GET    /graph/entity/{id}         - Get entity with relationships      │    │
│  │ GET    /graph/query               - Cypher-like query                  │    │
│  │ POST   /graph/entity              - Create entity                      │    │
│  │ POST   /graph/relationship        - Create relationship                │    │
│  │ GET    /graph/path/{from}/{to}    - Find path between entities         │    │
│  └────────────────────────────────────────────────────────────────────────┘    │
│                                                                                 │
│  ┌────────────────────────────────────────────────────────────────────────┐    │
│  │ CONTEXT MANAGEMENT                                                     │    │
│  │ POST   /context/inject            - Inject context to agent            │    │
│  │ POST   /context/compress          - Trigger compression                │    │
│  │ GET    /context/stats             - Memory usage statistics            │    │
│  └────────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────────┐
│                           STORAGE LAYER OPTIONS                                 │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │ MVP (Phase 1): SQLite + JSON Files                                      │   │
│  │ ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                   │   │
│  │ │SQLite + FTS5 │  │JSON Memory   │  │ File-based   │                   │   │
│  │ │(Episodic/BM25│  │(Core/Working)│  │ (Archival)   │                   │   │
│  │ └──────────────┘  └──────────────┘  └──────────────┘                   │   │
│  │ Pros: Zero dependencies, fast to implement, portable                   │   │
│  │ Cons: Limited graph queries, no native vectors                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │ Production (Phase 2): Vector DB + Graph                                 │   │
│  │ ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                   │   │
│  │ │ChromaDB/Qdrant│  │Neo4j Lite   │  │ Redis/Valkey │                   │   │
│  │ │(Vectors)     │  │(Graph)       │  │ (Cache/WM)   │                   │   │
│  │ └──────────────┘  └──────────────┘  └──────────────┘                   │   │
│  │ Pros: Full semantic search, real graph queries, fast cache             │   │
│  │ Cons: More infrastructure, operational overhead                        │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │ Enterprise (Phase 3): Managed Services                                  │   │
│  │ ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                   │   │
│  │ │ Pinecone/    │  │ Neptune      │  │ ElastiCache  │                   │   │
│  │ │ Weaviate     │  │ Analytics    │  │ for Valkey   │                   │   │
│  │ └──────────────┘  └──────────────┘  └──────────────┘                   │   │
│  │ Pros: Fully managed, scales automatically, SLAs                        │   │
│  │ Cons: Cost, vendor lock-in, latency to cloud                           │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Detailed Component Design

### 1. Tiered Memory Implementation

#### Tier 1: Core Memory
```python
class CoreMemory:
    """Always-loaded agent identity and context"""
    agent_id: str
    role: str                    # "coo", "dev_agent", etc.
    system_prompt: str           # Base instructions
    persona: dict                # User preferences (CEO)
    current_task: Optional[Task] # Active assignment

    # Size budget: ~2000 tokens
    # Update frequency: Rarely (role changes, new preferences)
    # Storage: JSON file per agent
```

#### Tier 2: Working Memory
```python
class WorkingMemory:
    """Session-scoped scratchpad"""
    session_id: str
    conversation_buffer: List[Message]  # Last 10 turns
    active_agents: Dict[str, AgentStatus]
    scratchpad: str              # Agent's thinking space
    pending_actions: List[Action]

    # Size budget: ~8000 tokens
    # Update frequency: Every turn
    # Storage: In-memory with Redis backup
```

#### Tier 3: Episodic Memory
```python
class EpisodicMemory:
    """Timestamped event log"""
    episode_id: str
    timestamp: datetime
    agent_id: str
    event_type: str              # "task_complete", "decision", "escalation"
    summary: str
    details: dict
    embedding: List[float]       # For semantic retrieval

    # Size: Unbounded (chunked storage)
    # Update: Append-only
    # Storage: SQLite with FTS5 + vector index
```

#### Tier 4: Semantic Memory
```python
class SemanticMemory:
    """Factual knowledge base"""
    knowledge_id: str
    category: str                # "code_pattern", "domain_fact", "procedure"
    content: str
    source: str                  # Where this came from
    confidence: float
    embedding: List[float]
    relationships: List[str]     # Links to graph entities

    # Storage: Vector DB (ChromaDB for MVP)
```

#### Tier 5: Archival Memory
```python
class ArchivalMemory:
    """Compressed historical context"""
    archive_id: str
    time_range: Tuple[datetime, datetime]
    summary_level: str           # "daily", "weekly", "monthly"
    compressed_content: str      # RAPTOR-style summary
    key_decisions: List[str]
    themes: List[str]

    # Storage: Markdown files + index
```

### 2. Graph Memory Schema

```cypher
// Entity Types
(:CEO {id, name, preferences})
(:COO {id, name, model, capabilities})
(:Agent {id, type, swarm, status, specialization})
(:Swarm {id, name, purpose, agents[]})
(:Task {id, title, status, priority, deadline})
(:Decision {id, description, rationale, outcome, timestamp})
(:Artifact {id, type, path, version})
(:Session {id, start_time, end_time, summary})

// Relationships
(CEO)-[:DELEGATES_TO]->(COO)
(COO)-[:ORCHESTRATES]->(Swarm)
(COO)-[:SPAWNS]->(Agent)
(Agent)-[:MEMBER_OF]->(Swarm)
(Agent)-[:WORKS_ON]->(Task)
(Agent)-[:PRODUCES]->(Artifact)
(Task)-[:AFFECTS]->(Artifact)
(Decision)-[:MADE_BY]->(COO|Agent)
(Decision)-[:REGARDING]->(Task|Artifact)
(Session)-[:CONTAINS]->(Decision|Task)
```

### 3. Memory Operations (Self-Editing)

Following MemGPT pattern, agents have tools to manage their own memory:

```python
# Tools available to agents
memory_tools = {
    "memory_read": {
        "description": "Retrieve from memory by query",
        "params": {
            "tier": "working|episodic|semantic|archival",
            "query": str,
            "limit": int
        }
    },
    "memory_write": {
        "description": "Store information to memory",
        "params": {
            "tier": "working|episodic|semantic",
            "content": str,
            "metadata": dict
        }
    },
    "memory_archive": {
        "description": "Move working memory to archival",
        "params": {
            "content_ids": List[str],
            "summary": str
        }
    },
    "memory_search": {
        "description": "Semantic search across all tiers",
        "params": {
            "query": str,
            "tiers": List[str],
            "filters": dict
        }
    }
}
```

### 4. Context Window Management

```
┌─────────────────────────────────────────────────────────────┐
│                  CONTEXT BUDGET (128K tokens)               │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ System Prompt + Core Memory          [~4K tokens]   │   │
│  │ - Agent identity, role, capabilities                │   │
│  │ - Current task context                              │   │
│  │ - User persona/preferences                          │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Retrieved Context                    [~20K tokens]  │   │
│  │ - Top-K semantic search results                     │   │
│  │ - Relevant graph relationships                      │   │
│  │ - Recent episodic memories                          │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Conversation Buffer                  [~8K tokens]   │   │
│  │ - Last 10 turns (full)                              │   │
│  │ - Older turns (summarized)                          │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Available for Response               [~96K tokens]  │   │
│  │ - Agent reasoning and output                        │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘

Compression Strategy:
1. Turns 1-10: Full verbatim
2. Turns 11-30: Summarized (3x compression)
3. Turns 31+: Archived to episodic (32x compression)
```

### 5. Data Flow Example

```
User Input: "What was the decision about the API rate limiting?"

1. QUERY PLANNING
   ├── Detect intent: Historical query, specific topic
   ├── Route to: Episodic + Semantic + Graph
   └── Generate sub-queries

2. PARALLEL RETRIEVAL
   ├── BM25: "API rate limiting decision"
   ├── Vector: embed("decision about API rate limiting")
   └── Graph: MATCH (d:Decision)-[:REGARDING]->(:Artifact {type:'api'})

3. FUSION + RERANKING
   ├── RRF merge results
   ├── ColBERT rerank top-20 → top-5
   └── Extract relevant snippets

4. CONTEXT INJECTION
   ├── Format as structured context block
   ├── Add to agent's prompt
   └── Track token budget

5. AGENT RESPONSE
   ├── Agent reasons over context
   ├── Generates response
   └── Optionally stores new episodic memory
```

---

## Implementation Phases

### Phase 1: MVP (Week 1-2)
- SQLite-based episodic storage with FTS5
- JSON files for core/working memory
- Basic memory tools for agents
- Simple keyword + recency retrieval
- No graph layer yet

### Phase 2: Hybrid Retrieval (Week 3-4)
- Add vector embeddings (Ollama local or OpenAI)
- Implement BM25 + vector fusion
- Add reranking with BGE
- Compression for older context

### Phase 3: Graph Memory (Week 5-6)
- Add graph storage (start with SQLite, migrate to Neo4j)
- Entity extraction from conversations
- Relationship queries
- Graph-augmented retrieval

### Phase 4: MCP Integration (Week 7-8)
- Implement MCP server interface
- Inter-agent context sharing
- Context bus for pub/sub
- External system integration

---

## Key Design Decisions

### 1. Why Tiered Memory?
- Matches MemGPT's proven pattern
- Enables cost-efficient context management
- Aligns with human memory cognition
- Allows graceful degradation under token limits

### 2. Why Graph + Vector?
- Graph captures explicit relationships (agent→task→artifact)
- Vectors capture semantic similarity
- Combined provides both structured and fuzzy retrieval
- Mem0g research shows 26% improvement

### 3. Why MCP?
- Open standard from Anthropic
- Future-proofs for multi-AI scenarios
- Standardizes inter-agent communication
- Replaces ad-hoc mailbox system

### 4. Why SQLite for MVP?
- Zero infrastructure requirements
- Fast to implement
- Portable (single file)
- FTS5 provides decent BM25
- Easy migration path to production DBs

---

## Success Metrics

1. **Context Efficiency**: <50% token budget used for context, >50% for response
2. **Retrieval Quality**: >80% relevant results in top-5
3. **Latency**: <500ms for context retrieval
4. **Memory Persistence**: 100% session context recovery after restart
5. **Agent Self-Management**: Agents successfully use memory tools

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Embedding quality varies by model | Poor retrieval | Test multiple models, use reranking |
| Graph complexity explosion | Slow queries | Prune relationships, time-based archival |
| MCP standard evolves | Breaking changes | Abstract behind internal interfaces |
| Token budget exceeded | Truncated context | Aggressive compression, priority scoring |

---

## Open Questions

1. Should agents share a single graph or have isolated subgraphs?
2. How often should archival compression run (end of session vs. background)?
3. What's the optimal embedding model for our domain (code + conversation)?
4. Should we implement Mem0 directly or build similar patterns ourselves?

---

## Additional Architecture Diagrams

### Complete System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     COMPLETE ARCHITECTURE OVERVIEW                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│                              ┌─────────┐                                     │
│                              │   CEO   │                                     │
│                              │ (Human) │                                     │
│                              └────┬────┘                                     │
│                                   │                                          │
│                              ┌────▼────┐                                     │
│                              │   COO   │                                     │
│                              │(Supreme)│                                     │
│                              └────┬────┘                                     │
│                                   │                                          │
│           ┌───────────────────────┼───────────────────────┐                 │
│           │                       │                       │                 │
│      ┌────▼────┐            ┌────▼────┐            ┌────▼────┐             │
│      │Swarm Dev│            │   Ops   │            │   ASA   │             │
│      └────┬────┘            └────┬────┘            └────┬────┘             │
│           │                      │                      │                   │
│     ┌─────┴─────┐          ┌────┴────┐           ┌────┴────┐               │
│     │           │          │         │           │         │               │
│  ┌──▼──┐    ┌──▼──┐    ┌──▼──┐  ┌──▼──┐     ┌──▼──┐  ┌──▼──┐             │
│  │Impl │    │Critic│    │Agent│  │Agent│     │Agent│  │Agent│             │
│  └──┬──┘    └──┬──┘    └──┬──┘  └──┬──┘     └──┬──┘  └──┬──┘             │
│     │          │          │        │           │        │                   │
│     └──────────┴──────────┴────────┴───────────┴────────┘                   │
│                           │                                                  │
│                           ▼                                                  │
│     ┌─────────────────────────────────────────────────────────────────┐     │
│     │                    MEMORY ARCHITECTURE                           │     │
│     │  ┌───────────────────────────────────────────────────────────┐  │     │
│     │  │ TIER 0: Working Memory (~3K tokens, always in context)    │  │     │
│     │  │   Identity | Task | Escalations | Messages                │  │     │
│     │  └────────────────────────────┬──────────────────────────────┘  │     │
│     │                               │ overflow                         │     │
│     │  ┌────────────────────────────▼──────────────────────────────┐  │     │
│     │  │ TIER 1: Session Memory (searchable, same-session)         │  │     │
│     │  │   STATE.md (rolling) | Tool Results | Decisions           │  │     │
│     │  └────────────────────────────┬──────────────────────────────┘  │     │
│     │                               │ archive                          │     │
│     │  ┌────────────────────────────▼──────────────────────────────┐  │     │
│     │  │ TIER 2: Archival Memory (indexed, cross-session)          │  │     │
│     │  │   ┌──────────┐  ┌──────────┐  ┌──────────┐               │  │     │
│     │  │   │  Vector  │  │   BM25   │  │Knowledge │               │  │     │
│     │  │   │  Store   │  │  Index   │  │  Graph   │               │  │     │
│     │  │   └──────────┘  └──────────┘  └──────────┘               │  │     │
│     │  │              └──────────┬──────────┘                      │  │     │
│     │  │                         │                                 │  │     │
│     │  │                    ┌────▼────┐                            │  │     │
│     │  │                    │  RRF    │                            │  │     │
│     │  │                    │ Fusion  │                            │  │     │
│     │  │                    └─────────┘                            │  │     │
│     │  └───────────────────────────────────────────────────────────┘  │     │
│     │                                                                  │     │
│     │  ┌───────────────────────────────────────────────────────────┐  │     │
│     │  │ MCP CONTEXT LAYER (inter-agent communication)             │  │     │
│     │  │   GET/POST /mcp/context | Handoffs | Events               │  │     │
│     │  └───────────────────────────────────────────────────────────┘  │     │
│     └─────────────────────────────────────────────────────────────────┘     │
│                                                                              │
│     ┌─────────────────────────────────────────────────────────────────┐     │
│     │                    EXISTING INFRASTRUCTURE                       │     │
│     │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │     │
│     │  │ Work Ledger  │  │   Mailbox    │  │ Escalation Protocol  │  │     │
│     │  │ (integrate)  │  │ (integrate)  │  │ (connected)          │  │     │
│     │  └──────────────┘  └──────────────┘  └──────────────────────┘  │     │
│     └─────────────────────────────────────────────────────────────────┘     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Agent Session Lifecycle

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         AGENT SESSION LIFECYCLE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  SESSION START                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  1. Load Agent Identity (Core Memory)                                   ││
│  │     └── From agents/{name}.md                                           ││
│  │  2. Check Pending Escalations                                           ││
│  │     └── GET /api/escalations?agent={id}&status=pending                  ││
│  │  3. Check Mailbox                                                       ││
│  │     └── GET /api/mailbox/{agent}?unread=true                            ││
│  │  4. Load Session Context (Working Memory)                               ││
│  │     ├── STATE.md (compressed via rolling window)                        ││
│  │     └── Last session summary if resuming                                ││
│  │  5. Query Relevant History (Archival Memory)                            ││
│  │     └── memory_search(query=current_task, limit=5)                      ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  DURING SESSION                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  On Tool Result:                                                         ││
│  │  ├── Store in Working Memory                                            ││
│  │  └── If significant: memory_write(content, tags)                        ││
│  │                                                                          ││
│  │  On Decision Made:                                                       ││
│  │  ├── Record in STATE.md                                                 ││
│  │  ├── Create Decision entity (Graph)                                     ││
│  │  └── Link to affected components                                        ││
│  │                                                                          ││
│  │  On Handoff:                                                             ││
│  │  ├── POST /mcp/context/handoff                                          ││
│  │  ├── Archive relevant context                                           ││
│  │  └── Notify recipient agent                                             ││
│  │                                                                          ││
│  │  On Context Overflow:                                                    ││
│  │  ├── Working → Session (auto-overflow oldest)                           ││
│  │  └── Session → Archival (when STATE.md > 10 entries)                    ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  SESSION END                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  1. Generate Session Summary                                            ││
│  │     └── LLM summarizes: work done, decisions, blockers, next steps      ││
│  │  2. Archive Session (Working → Archival)                                ││
│  │     ├── Embed session summary                                           ││
│  │     ├── Extract entities for graph                                      ││
│  │     └── Link to modified files, decisions, escalations                  ││
│  │  3. Compress STATE.md                                                   ││
│  │     └── Apply rolling window compression                                ││
│  │  4. Update Swarm Context                                                ││
│  │     └── Propagate significant changes to swarm-level memory             ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Cross-Agent Memory Access

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       CROSS-AGENT MEMORY ACCESS                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   COO delegates task to Implementer with context                             │
│                                                                              │
│   ┌───────────┐                                     ┌────────────────┐       │
│   │    COO    │                                     │  Implementer   │       │
│   └─────┬─────┘                                     └───────┬────────┘       │
│         │                                                   │                │
│         │  1. POST /mcp/context/handoff                    │                │
│         │     {                                             │                │
│         │       from: "COO",                                │                │
│         │       to: "swarm_dev/implementer",                │                │
│         │       context: {                                  │                │
│         │         work_completed: "...",                    │                │
│         │         relevant_memories: ["MEM-001"],           │                │
│         │         next_steps: ["..."]                       │                │
│         │       }                                           │                │
│         │     }                                             │                │
│         │─────────────────────────────────────────────────▶│                │
│         │                                                   │                │
│         │                                                   │  2. On start:  │
│         │                                                   │  Load handoff  │
│         │                                                   │  context       │
│         │                                                   │                │
│         │                                                   │  3. Query:     │
│         │                                    ┌──────────────┤  memory_search │
│         │                                    │              │  (query=...)   │
│         │                                    ▼              │                │
│         │                              ┌───────────┐        │                │
│         │                              │ Archival  │        │                │
│         │                              │  Memory   │        │                │
│         │                              └─────┬─────┘        │                │
│         │                                    │ Results      │                │
│         │                                    └─────────────▶│                │
│         │                                                   │  4. Work...    │
│         │                                                   │                │
│         │  5. ws://host/mcp/events                         │  POST /mcp/    │
│         │     { type: "context.handoff", ... }              │  handoff       │
│         │◀──────────────────────────────────────────────────│◀───────────────┤
│         │                                                   │                │
│   ┌─────┴─────┐                                     ┌───────┴────────┐       │
│   │    COO    │                                     │  Implementer   │       │
│   └───────────┘                                     └────────────────┘       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Implementation Component Map

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          COMPONENT ARCHITECTURE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  shared/                                                                     │
│  ├── memory/                                                                 │
│  │   ├── __init__.py                                                        │
│  │   ├── working_memory.py      # Core + Working memory                     │
│  │   ├── session_memory.py      # Episodic with compression                 │
│  │   ├── archival_memory.py     # Semantic + Archival with hybrid search    │
│  │   ├── memory_tools.py        # MemGPT-style agent tools                  │
│  │   └── memory_manager.py      # Orchestrates all tiers                    │
│  │                                                                           │
│  ├── indexing/                                                               │
│  │   ├── __init__.py                                                        │
│  │   ├── vector_store.py        # Embedding storage (Chroma/LanceDB)        │
│  │   ├── bm25_index.py          # Keyword search (SQLite FTS5)              │
│  │   ├── knowledge_graph.py     # Entity-relationship graph                 │
│  │   └── hybrid_retriever.py    # RRF fusion + reranking                    │
│  │                                                                           │
│  ├── mcp/                                                                    │
│  │   ├── __init__.py                                                        │
│  │   ├── context_server.py      # MCP context API                           │
│  │   ├── context_handlers.py    # Request handlers                          │
│  │   └── events.py              # WebSocket event types                     │
│  │                                                                           │
│  └── compression/                                                            │
│      ├── __init__.py                                                        │
│      ├── summarizer.py          # LLM-based summarization                   │
│      └── state_compressor.py    # STATE.md rolling window                   │
│                                                                              │
│  backend/                                                                    │
│  ├── routes/                                                                 │
│  │   └── memory.py              # REST endpoints for memory ops             │
│  └── websocket/                                                              │
│      └── memory_events.py       # WebSocket memory event handling           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Technology Selection Summary

| Component | MVP Choice | Production Choice | Rationale |
|-----------|------------|-------------------|-----------|
| Vector Store | ChromaDB | LanceDB | Zero deps → columnar perf |
| Embeddings | all-MiniLM-L6-v2 | nomic-embed-text | Local → better quality |
| BM25 | SQLite FTS5 | Tantivy | Zero deps → Rust speed |
| Graph | SQLite + JSON | Neo4j | Simple → real graph ops |
| Reranker | None | BGE-reranker | Start simple → add quality |
| Cache | In-memory | Redis/Valkey | Dev → production scale |

---

## Current State vs Target State

| Aspect | Current | Target | Gap |
|--------|---------|--------|-----|
| Memory tiers | 1 (STATE.md) | 5 (Core→Archival) | Major |
| Semantic search | None | Hybrid retrieval | Major |
| Graph relationships | None | Entity graph | Major |
| Context compression | None | 32x compression | Medium |
| Inter-agent context | Mailbox (ad-hoc) | MCP protocol | Medium |
| Session persistence | Partial | Full recovery | Low |
| Self-editing memory | None | Agent tools | Medium |

---

*Brainstorm Round 1 Complete - Ready for refinement in Round 2*
