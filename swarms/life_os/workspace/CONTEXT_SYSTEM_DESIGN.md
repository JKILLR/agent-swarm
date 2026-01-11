# Context System Architecture

## Overview

A hybrid file-based context system optimized for 8GB RAM, combining:
- **File-based core**: YAML/JSON for fast keyword lookup and git-versioned history
- **Selective embeddings**: Only for fuzzy search on emails/docs when needed

## Directory Structure

```
memory/context/
├── foundation/           # Core personal context (always loaded)
│   ├── profile.yaml      # Name, profession, current project
│   ├── communication_style.yaml  # Email tone, phrases, signatures
│   └── preferences.yaml  # Working hours, priority contacts, urgency rules
│
├── projects/             # Project-specific context (lazy loaded)
│   └── langley_5/
│       ├── project.yaml  # Project details, location, timeline
│       ├── trades.yaml   # Subcontractor list (populated from Drive)
│       ├── contacts.yaml # Key people (learned over time)
│       └── docs/         # Cached doc summaries
│           ├── po_summary.json
│           └── recent_sis.json
│
├── embeddings/           # Selective vector cache (Phase 2)
│   ├── index.json        # Metadata: what's embedded, when
│   └── emails/           # Email embeddings for style matching
│
└── working/              # Ephemeral session context
    └── current.json      # Current email thread, active docs
```

## Layer Architecture

### Layer 1: Foundation (Always in Memory)
- ~10KB YAML files
- Loaded at startup
- Updated rarely

**Use for**: Personalizing all agent responses, email drafting tone

### Layer 2: Project (Lazy Loaded)
- Loaded when project is referenced
- Cached for session duration
- Updated when Drive docs change

**Use for**: Project-specific context (trades, contacts, document references)

### Layer 3: Working (Ephemeral)
- Current task context
- Email thread being processed
- Active document references
- Cleared between sessions

**Use for**: Immediate context for current task

## Context Retrieval API

### Endpoints

```
GET  /api/context/foundation        # Get all foundation context
GET  /api/context/project/{name}    # Get project context
POST /api/context/working           # Update working context
GET  /api/context/combined          # Get merged context for current task
```

### Python Service

```python
# backend/services/context_service.py

class ContextService:
    def __init__(self):
        self.foundation = self._load_foundation()  # Always loaded
        self.project_cache = {}  # Lazy-loaded projects
        self.working = {}  # Current session context

    def get_foundation(self) -> dict:
        """Return foundation context (profile, style, preferences)"""
        return self.foundation

    def get_project(self, project_name: str) -> dict:
        """Lazy-load and return project context"""
        if project_name not in self.project_cache:
            self.project_cache[project_name] = self._load_project(project_name)
        return self.project_cache[project_name]

    def get_combined_context(self, project: str = None) -> dict:
        """Get merged context for LLM prompt injection"""
        context = {
            "user": self.foundation,
            "working": self.working
        }
        if project:
            context["project"] = self.get_project(project)
        return context

    def update_from_drive(self, folder_id: str, project: str):
        """Sync project docs from Google Drive"""
        # Fetch doc summaries, update project context
        pass
```

## Embedding Strategy (Phase 2)

**Only embed these types**:
1. **Sent emails** - For style matching when drafting
2. **Important docs** - SI summaries, key project docs
3. **Contact interactions** - Build contact context over time

**Keep lightweight**:
- Use `all-MiniLM-L6-v2` (90MB, runs on CPU)
- Lazy-load model only when searching
- Cache top 100 vectors in memory
- Store rest on disk

```python
# Only triggered for semantic search, not default retrieval
def semantic_search(query: str, collection: str = "emails") -> list:
    embeddings = load_embeddings(collection)  # Lazy load
    query_vec = embed(query)
    return cosine_similarity_search(query_vec, embeddings, top_k=5)
```

## Memory Budget (8GB Constraint)

| Component | Max RAM |
|-----------|---------|
| Foundation YAML | ~50KB |
| Active project | ~200KB |
| Working context | ~100KB |
| Embedding model (when loaded) | ~200MB |
| Cached vectors | ~10MB |
| **Total** | **<250MB** |

## Integration Points

### Email Drafting Flow
1. Load foundation context (style, tone)
2. Load project context if project-related
3. Inject into LLM prompt
4. Optional: Semantic search for similar sent emails

### Document Summarization Flow
1. Fetch doc from Drive
2. Summarize with LLM
3. Store summary in `projects/{name}/docs/`
4. Update project context with doc reference

### Learning Flow
1. Agent observes J's communication
2. Extract patterns (phrases, signature, style)
3. Update `communication_style.yaml`
4. Commit changes to git (audit trail)

## Next Steps

1. [x] Create foundation YAML structure
2. [x] Create Langley 5 project structure
3. [ ] Implement ContextService Python class
4. [ ] Add API endpoints
5. [ ] Integrate with email drafting
6. [ ] Phase 2: Add selective embeddings
