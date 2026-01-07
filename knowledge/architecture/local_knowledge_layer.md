# Local Knowledge Layer Architecture

## The Problem
Claude (and agents) start fresh each session. We need:
1. Persistent distilled knowledge (not raw notes)
2. Fast context retrieval for any agent
3. Continuous learning from new inputs
4. Works within 8GB RAM constraint

## Proposed Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    RAW DATA SOURCES                       │
│  Apple Notes │ Conversations │ Files │ Web │ Calendar    │
└─────────────────────────┬────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────┐
│              LOCAL DISTILLATION MODEL                     │
│  (Small, efficient - runs on M2 8GB)                     │
│  - Reads raw data                                         │
│  - Extracts entities, relationships, facts                │
│  - Classifies by domain (work, personal, tech)           │
│  - Generates structured summaries                         │
└─────────────────────────┬────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────┐
│              KNOWLEDGE STORE                              │
│  /knowledge/                                              │
│    ├── user_context/   (who J is, preferences)           │
│    ├── projects/       (active work)                     │
│    ├── asa/           (research knowledge)               │
│    ├── ideas/         (seed concepts)                    │
│    ├── work/          (construction context)             │
│    └── relationships/ (people, entities)                 │
└─────────────────────────┬────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────┐
│              CONTEXT SYNTHESIS API                        │
│  POST /api/context/synthesize                            │
│  Input: query + agent_type + task                        │
│  Output: Relevant knowledge chunks + summary             │
│                                                          │
│  Uses: Vector similarity + keyword + recency            │
└─────────────────────────┬────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────┐
│              CLAUDE / AGENTS                              │
│  - Receives synthesized context at session start         │
│  - Can query for more context as needed                  │
│  - Updates knowledge store with learnings                │
└──────────────────────────────────────────────────────────┘
```

## Local Model Options (8GB RAM Friendly)

### Option A: Ollama with Small Model (Recommended)
- **Model**: Llama 3.2 1B or 3B quantized
- **Memory**: ~2-4GB
- **Pros**: Easy setup, good instruction following
- **Cons**: Limited reasoning for complex distillation

### Option B: MLX Native (Apple Silicon Optimized)
- **Model**: Phi-3-mini or similar
- **Memory**: Very efficient on M-series
- **Pros**: Best performance on Mac
- **Cons**: Fewer model options

### Option C: Hybrid Approach
- **Local model** for simple extraction (entities, dates, categories)
- **Claude API calls** for complex distillation (only when needed)
- **Pros**: Best quality, manages cost
- **Cons**: More complex, requires API calls

## Implementation Phases

### Phase 1: Basic Knowledge Store (NOW)
- ✅ Create `/knowledge/` directory structure
- ✅ Manual distillation of Notes data
- ✅ Save to structured markdown files
- [ ] Add to memory API for persistence

### Phase 2: Simple Extraction Pipeline
- Set up Ollama with small model
- Create extraction prompts for:
  - Entity extraction (people, projects, dates)
  - Category classification
  - Key fact extraction
- Watch folder for new Notes exports

### Phase 3: Vector Search Layer
- Add embeddings to knowledge files
- Implement semantic search for context retrieval
- Create `/api/context/search` endpoint

### Phase 4: Synthesis API
- Combine keyword + vector + recency
- Generate context summaries per agent type
- Auto-inject into agent prompts

### Phase 5: Continuous Learning
- Track what context was useful
- Update knowledge based on conversations
- Prune outdated information

## Immediate Next Steps

1. **Install Ollama**: `brew install ollama`
2. **Pull small model**: `ollama pull llama3.2:1b`
3. **Create extraction script**: Python script to process Notes exports
4. **Add to backend**: New `/api/knowledge/` endpoints

## Memory Budget (8GB Total)

| Component | Allocation |
|-----------|------------|
| macOS + apps | ~3GB |
| Backend (FastAPI) | ~200MB |
| Frontend (Next.js dev) | ~500MB |
| Ollama + model | ~2-3GB |
| Vector DB (ChromaDB) | ~500MB |
| **Headroom** | ~1-2GB |

This should fit, but will need monitoring.

## Key Design Decisions

1. **Markdown over DB**: Knowledge stays human-readable, version-controlled
2. **Local-first**: No dependency on external services for core function
3. **Incremental**: Can start manual, add automation over time
4. **Agent-aware**: Different agents get different context slices
