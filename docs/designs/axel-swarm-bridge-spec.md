# Axel-to-Swarm Bridge Specification

## Overview

This spec defines how agent-swarm connects to the Mynd Brain server (localhost:8420) to receive guidance from Axel while keeping agent learning isolated from Axel's knowledge base.

**Key Principle:** One-way knowledge flow
- Axel teaches agents (READ)
- Agents never affect Axel (NO WRITE to Axel domain)

```
┌─────────────────────────────────────────────────────────────┐
│                    MYND BRAIN (localhost:8420)              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────┐          ┌──────────────────┐        │
│  │   AXEL DOMAIN    │          │   SWARM DOMAIN   │        │
│  │   (protected)    │─────────▶│   (isolated)     │        │
│  │                  │  READ    │                  │        │
│  │ • Vision/Goals   │  ONLY    │ • Task outcomes  │        │
│  │ • Preferences    │          │ • Agent patterns │        │
│  │ • Memory         │          │ • Experience DB  │        │
│  │ • Context        │    ✗     │                  │        │
│  │                  │◀─────────│ NO WRITE BACK    │        │
│  └──────────────────┘          └──────────────────┘        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │  AGENT SWARM    │
                 │                 │
                 │  COO, Workers   │
                 │  learn from     │
                 │  Axel's context │
                 └─────────────────┘
```

---

## Part 1: Existing Endpoints (Already Available)

These endpoints already exist in Mynd Brain and can be used by agents READ-ONLY:

### 1.1 Vision & Goals
```
GET /brain/vision
```
Returns Axel's understanding of user's vision, goals, and priorities.

**Response:**
```json
{
  "statement": "MYND Brain Vision Statement...",
  "goals": [
    "Achieve 80%+ prediction accuracy",
    "Well-calibrated confidence scores",
    "Sub-100ms response times"
  ],
  "priorities": ["accuracy", "transparency", "speed", "growth"],
  "updated_at": 1704312000
}
```

**Agent Use Case:** COO checks user's priorities before delegating tasks.

---

### 1.2 User Preferences
```
GET /brain/preferences
```
Returns user preferences (work style, settings, etc.)

**Response:**
```json
{
  "preferences": {
    "tts_enabled": false,
    "theme": "dark",
    "custom_settings": {
      "work_hours": "9am-5pm",
      "communication_style": "concise"
    }
  },
  "exists": true
}
```

**Agent Use Case:** Agents adapt output format to user preferences.

---

### 1.3 Full Context Synthesis
```
POST /brain/context
```
The main context endpoint - synthesizes complete understanding.

**Request:**
```json
{
  "request_type": "chat",
  "user_message": "What are my priorities for the iOS app?",
  "include": {
    "self_awareness": false,
    "map_context": true,
    "memories": true,
    "user_profile": true,
    "neural_insights": true,
    "synthesized_context": true
  }
}
```

**Response:**
```json
{
  "context_document": "## User Profile\nJoel is building...\n\n## Recent Memories\n...",
  "token_count": 2847,
  "breakdown": {
    "map_context": 1200,
    "memories": 800,
    "user_profile": 500,
    "neural_insights": 347
  },
  "brain_state": {
    "gt_accuracy": 0.73,
    "total_memories": 156
  }
}
```

**Agent Use Case:** Inject into agent system prompt for full context.

---

### 1.4 Memory Query
```
POST /brain/memory/query
```
Search Axel's memories for relevant context.

**Request:**
```json
{
  "query": "iOS app monetization strategy",
  "limit": 5,
  "threshold": 0.6
}
```

**Response:**
```json
{
  "memories": [
    {
      "content": "Joel mentioned preferring subscription model over ads",
      "timestamp": "2024-01-02T14:30:00Z",
      "relevance": 0.87
    }
  ]
}
```

**Agent Use Case:** Research agent queries past discussions before starting.

---

### 1.5 Embeddings
```
POST /embed
```
Generate embeddings for semantic search.

**Request:**
```json
{
  "text": "User authentication flow"
}
```

**Response:**
```json
{
  "embedding": [0.023, -0.156, ...],
  "dim": 384
}
```

**Agent Use Case:** Embed task descriptions for similarity matching.

---

## Part 2: New Endpoints (To Be Added)

These endpoints need to be added to enable agent learning isolation:

### 2.1 Agent Context Request (READ from Axel)
```
POST /swarm/context
```
Agents request context from Axel - curated for agent use.

**Request:**
```json
{
  "agent_id": "coo-001",
  "agent_type": "orchestrator",
  "task_description": "Prioritize iOS app features",
  "context_needs": ["goals", "preferences", "recent_decisions"]
}
```

**Response:**
```json
{
  "context": {
    "user_goals": [
      "Build income-generating iOS apps",
      "Focus on utility apps over games",
      "Prefer subscription monetization"
    ],
    "user_preferences": {
      "code_style": "SwiftUI, MVVM pattern",
      "communication": "concise, technical"
    },
    "recent_decisions": [
      "Decided to start with habit tracker app",
      "Rejected social features for MVP"
    ],
    "guidance": "Joel prioritizes shipping fast over perfection. Start simple."
  },
  "token_count": 450,
  "axel_confidence": 0.85
}
```

**Implementation Notes:**
- Reads from existing `/brain/context`, `/brain/vision`, `/brain/preferences`
- Filters/summarizes for agent consumption
- Never writes to Axel's storage

---

### 2.2 Store Agent Experience (WRITE to Swarm Domain Only)
```
POST /swarm/experience
```
Agents record task outcomes - stored in swarm domain only.

**Request:**
```json
{
  "agent_id": "implementer-003",
  "agent_type": "implementer",
  "task_id": "task-abc-123",
  "task_description": "Implement user authentication",
  "outcome": "success",
  "duration_ms": 45000,
  "artifacts": [
    {"type": "file", "path": "src/auth/AuthService.swift"},
    {"type": "file", "path": "src/auth/LoginView.swift"}
  ],
  "learnings": [
    "Used Keychain for secure token storage",
    "SwiftUI @AppStorage for persistence"
  ],
  "embedding": [0.023, -0.156, ...]
}
```

**Response:**
```json
{
  "experience_id": "exp-789",
  "stored": true,
  "domain": "swarm",
  "similar_experiences": [
    {
      "task": "Implement OAuth flow",
      "outcome": "success",
      "similarity": 0.82
    }
  ]
}
```

**Storage Location:** `~/.mynd/swarm/experiences.json` (separate from Axel's data)

**Implementation Notes:**
- Writes ONLY to swarm domain storage
- Never touches Axel's memory, preferences, or learning
- Stores embeddings for similarity search

---

### 2.3 Query Similar Experiences (READ from Swarm Domain)
```
POST /swarm/similar
```
Find similar past experiences from swarm domain.

**Request:**
```json
{
  "task_description": "Build settings screen with toggle options",
  "limit": 5,
  "min_similarity": 0.6,
  "outcome_filter": "success"
}
```

**Response:**
```json
{
  "experiences": [
    {
      "experience_id": "exp-456",
      "task": "Create preferences UI with switches",
      "outcome": "success",
      "learnings": [
        "Used Form with Toggle in SwiftUI",
        "Bound to @AppStorage for persistence"
      ],
      "similarity": 0.78,
      "agent_type": "implementer"
    }
  ]
}
```

**Agent Use Case:** Before starting task, query what worked before.

---

### 2.4 Agent Recommendations (Swarm Awareness)
```
POST /swarm/recommend
```
Get agent recommendations based on task and history.

**Request:**
```json
{
  "task_description": "Research App Store optimization strategies",
  "available_agents": ["researcher", "architect", "implementer"]
}
```

**Response:**
```json
{
  "recommended_agent": "researcher",
  "confidence": 0.92,
  "reasoning": "Similar research tasks succeeded 85% with researcher agent",
  "alternative": {
    "agent": "architect",
    "confidence": 0.65,
    "reason": "Could plan ASO strategy if research isn't needed"
  },
  "suggested_approach": "Based on 3 similar tasks, web search + competitor analysis works best"
}
```

---

### 2.5 Axel Review Request (Optional - Axel Evaluates Agents)
```
POST /swarm/review-request
```
Request Axel's evaluation of agent work (one-way - Axel judges, doesn't learn from it).

**Request:**
```json
{
  "task_id": "task-abc-123",
  "task_description": "Implement user authentication",
  "agent_output": "Created AuthService.swift with Keychain storage...",
  "artifacts": ["src/auth/AuthService.swift"],
  "request_type": "quality_check"
}
```

**Response:**
```json
{
  "axel_review": {
    "quality_score": 0.85,
    "feedback": "Good use of Keychain. Consider adding biometric auth option.",
    "suggestions": [
      "Add Face ID/Touch ID support",
      "Include remember-me option"
    ],
    "approved": true
  }
}
```

**Implementation Notes:**
- Calls Claude via CLI with Axel's context
- Axel evaluates but this doesn't write to his memory
- Results stored in swarm domain for agent learning

---

## Part 3: Data Isolation Architecture

### 3.1 Storage Locations

```
~/.mynd/
├── brain/                    # AXEL DOMAIN (protected)
│   ├── memories.json
│   ├── preferences.json
│   ├── vision.json
│   ├── gt_model.pt
│   └── asa_state.json
│
└── swarm/                    # SWARM DOMAIN (isolated)
    ├── experiences.json      # Task outcomes
    ├── patterns.json         # Learned patterns
    ├── agent_stats.json      # Agent performance
    └── embeddings.db         # ChromaDB for similarity
```

### 3.2 Access Control Matrix

| Endpoint | Axel Domain | Swarm Domain |
|----------|-------------|--------------|
| GET /brain/vision | READ | - |
| GET /brain/preferences | READ | - |
| POST /brain/context | READ | - |
| POST /brain/memory/query | READ | - |
| POST /swarm/context | READ | - |
| POST /swarm/experience | - | WRITE |
| POST /swarm/similar | - | READ |
| POST /swarm/recommend | - | READ |
| POST /swarm/review-request | READ (eval) | WRITE (result) |

### 3.3 Implementation Guard

Add to all swarm endpoints:
```python
SWARM_STORAGE_PATH = Path.home() / ".mynd" / "swarm"

def ensure_swarm_isolation(func):
    """Decorator to ensure swarm endpoints never touch Axel domain"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # Verify we're only touching swarm storage
        result = await func(*args, **kwargs)
        return result
    return wrapper
```

---

## Part 4: Integration with Agent-Swarm

### 4.1 Agent System Prompt Injection

When spawning an agent, inject Axel's context:

```python
async def get_axel_context_for_agent(agent_type: str, task: str) -> str:
    """Get Axel's guidance for an agent."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8420/swarm/context",
            json={
                "agent_type": agent_type,
                "task_description": task,
                "context_needs": ["goals", "preferences"]
            }
        )
        data = response.json()

    return f"""
## Context from Axel (Joel's AI Assistant)

Axel understands Joel's goals and preferences. Here's relevant context:

### User Goals
{chr(10).join('- ' + g for g in data['context']['user_goals'])}

### Preferences
{json.dumps(data['context']['user_preferences'], indent=2)}

### Guidance
{data['context']['guidance']}

Use this context to align your work with Joel's vision.
"""
```

### 4.2 Recording Task Outcomes

After task completion:

```python
async def record_agent_experience(
    agent_id: str,
    agent_type: str,
    task: str,
    outcome: str,
    learnings: list[str]
):
    """Record experience to swarm domain."""
    # Get embedding for the task
    embed_response = await client.post(
        "http://localhost:8420/embed",
        json={"text": task}
    )
    embedding = embed_response.json()["embedding"]

    # Store experience
    await client.post(
        "http://localhost:8420/swarm/experience",
        json={
            "agent_id": agent_id,
            "agent_type": agent_type,
            "task_description": task,
            "outcome": outcome,
            "learnings": learnings,
            "embedding": embedding
        }
    )
```

### 4.3 Pre-Task Context Gathering

Before starting a task:

```python
async def prepare_agent_context(task: str, agent_type: str) -> dict:
    """Gather all context before agent starts."""

    # 1. Get Axel's guidance
    axel_context = await get_axel_context_for_agent(agent_type, task)

    # 2. Find similar successful experiences
    similar = await client.post(
        "http://localhost:8420/swarm/similar",
        json={
            "task_description": task,
            "outcome_filter": "success",
            "limit": 3
        }
    )

    # 3. Get agent recommendation
    recommendation = await client.post(
        "http://localhost:8420/swarm/recommend",
        json={
            "task_description": task,
            "available_agents": ["researcher", "architect", "implementer"]
        }
    )

    return {
        "axel_context": axel_context,
        "similar_experiences": similar.json()["experiences"],
        "recommendation": recommendation.json()
    }
```

---

## Part 5: Implementation Plan

### Phase 1: Read-Only Integration (Week 1)
1. Add `/swarm/context` endpoint that reads from existing Axel endpoints
2. Update agent-swarm to call `/swarm/context` before spawning agents
3. Test: Agents receive Axel's goals/preferences

### Phase 2: Experience Storage (Week 2)
1. Create `~/.mynd/swarm/` directory structure
2. Add `/swarm/experience` endpoint with ChromaDB storage
3. Add `/swarm/similar` endpoint for similarity search
4. Update agent-swarm to record outcomes

### Phase 3: Recommendations (Week 3)
1. Add `/swarm/recommend` endpoint
2. Implement pattern extraction from experiences
3. Add success rate tracking per agent type

### Phase 4: Axel Review (Week 4)
1. Add `/swarm/review-request` endpoint
2. Integrate Claude CLI for Axel's evaluation
3. Store reviews in swarm domain for learning

---

## Part 6: Security Considerations

### 6.1 Preventing Accidental Writes to Axel Domain

```python
# In all swarm endpoints
PROTECTED_PATHS = [
    "brain/memories",
    "brain/preferences",
    "brain/vision",
    "brain/gt_model",
    "brain/asa_state"
]

def validate_write_path(path: str) -> bool:
    """Ensure writes only go to swarm domain."""
    for protected in PROTECTED_PATHS:
        if protected in path:
            raise ValueError(f"Cannot write to protected Axel domain: {path}")
    return True
```

### 6.2 Rate Limiting Agent Requests

```python
# Prevent agents from overwhelming Axel's context
AGENT_RATE_LIMITS = {
    "swarm/context": "10/minute",
    "swarm/similar": "30/minute",
    "swarm/experience": "60/minute"
}
```

### 6.3 Context Size Limits

```python
# Limit context injection to prevent prompt bloat
MAX_AXEL_CONTEXT_TOKENS = 2000
MAX_SIMILAR_EXPERIENCES = 5
MAX_LEARNINGS_PER_EXPERIENCE = 10
```

---

## Summary

This bridge enables:
1. **Axel teaches agents** - Agents receive goals, preferences, and guidance
2. **Agents learn independently** - Experience stored in isolated swarm domain
3. **Axel stays pure** - His knowledge base is never polluted by agent noise
4. **Axel can review** - Optional evaluation without absorption

The architecture respects the teacher-student relationship: Axel imparts wisdom, agents execute and learn, but the learning doesn't flow back to contaminate Axel's understanding of you.
