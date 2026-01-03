---
created: 2025-01-01
updated: 2026-01-03
---

# Swarm State

> **This file is the shared memory for all agents working on this swarm.**
> Always read this file first. Update it after completing work.

## Last Updated
2026-01-03 - RESEARCHER (Comprehensive codebase analysis from mynd-server source)

## Swarm Status: ACTIVE (Codebase Available)

The mynd-server codebase has been copied into this workspace at `swarms/mynd_app/workspace/mynd-server/`. Full source code analysis is now possible.

## Current Objectives
1. **COMPLETED** - Analyze mynd-server codebase architecture
2. **COMPLETED** - Document brain system (unified_brain.py, context_synthesizer.py)
3. **COMPLETED** - Document ML models (living_asa.py, graph_transformer.py)
4. **COMPLETED** - Document daemon systems (reflection, evolution, map maintenance)
5. **PENDING** - Integration planning with agent-swarm system
6. **PENDING** - Review for potential improvements

## Progress Log
<!-- Most recent entries at top -->

### 2026-01-03 RESEARCHER - Complete Codebase Analysis
- Analyzed all core source files in mynd-server codebase
- Documented complete architecture: Frontend (JS) + Backend (Python/FastAPI)
- Identified key systems: UnifiedBrain, ContextSynthesizer, Living ASA, Graph Transformer
- Documented daemon systems: ReflectionDaemon, EvolutionDaemon, MapMaintenanceDaemon
- Mapped all ML models and their purposes
- Identified dependencies from requirements.txt
- Files analyzed: 15+ core files across js/, mynd-brain/, docs/
- Outcome: SUCCESS - Complete technical documentation available

### 2026-01-03 RESEARCHER - Initial Codebase Research
- Created comprehensive summary document `MYND_SUMMARY.md`
- Outcome: partial - Updated with actual source code analysis

## Key Files
<!-- List important files with brief descriptions -->

### Core Application Files
| File | Purpose | Key Contents |
|------|---------|--------------|
| `mynd-server/index.html` | Consumer app entry point | Main MYND application |
| `mynd-server/self-dev.html` | Self-development interface | Meta MYND for AI-assisted development |
| `mynd-server/js/app-module.js` | Main frontend module | 40K+ lines: store, 3D rendering, chat, neural net |
| `mynd-server/js/config.js` | Configuration constants | Theme colors, physics, neural net settings |
| `mynd-server/js/goal-system.js` | Goal/manifestation system | GoalRegistry, MilestoneGenerator, PathFinder |
| `mynd-server/js/reflection-daemon.js` | Autonomous reflection | Background AI reasoning with tool use |
| `mynd-server/js/map-maintenance-daemon.js` | Map cleanup daemon | Duplicate detection, structure analysis |
| `mynd-server/js/local-brain-client.js` | Brain server client | HTTP client for mynd-brain |

### Python Brain Backend
| File | Purpose | Key Contents |
|------|---------|--------------|
| `mynd-server/mynd-brain/server.py` | FastAPI server | All API endpoints, localhost:8420 |
| `mynd-server/mynd-brain/brain/unified_brain.py` | Brain orchestrator | SelfAwareness, KnowledgeDistiller, MemorySystem |
| `mynd-server/mynd-brain/brain/context_synthesizer.py` | Context unification | Hybrid search (vector+BM25), Context Lens |
| `mynd-server/mynd-brain/models/living_asa.py` | Atomic embeddings | 720-dim physics-based atomic structure |
| `mynd-server/mynd-brain/models/graph_transformer.py` | Graph neural network | 8-head attention, edge-aware, 11.5M params |
| `mynd-server/mynd-brain/models/voice.py` | Voice transcription | Whisper integration |
| `mynd-server/mynd-brain/models/vision.py` | Image understanding | CLIP integration |
| `mynd-server/mynd-brain/evolution_daemon.py` | Background learning | Server-side autonomous evolution |

### Documentation Files
| File | Purpose |
|------|---------|
| `mynd-server/mynd-brain/ARCHITECTURE.md` | System architecture diagrams |
| `mynd-server/mynd-brain/UNIFIED_BRAIN_DESIGN.md` | Brain design specification |
| `mynd-server/docs/BACKGROUND_COGNITION_SPEC.md` | Background processing spec |
| `mynd-server/docs/ML_ARCHITECTURE.md` | ML system documentation |

## Architecture Decisions
<!-- Record important decisions and why they were made -->

### ADR-001: Unified Brain Architecture
- **Context**: Previous system had 19+ fragmented context providers
- **Decision**: Single UnifiedBrain class orchestrates all context sources
- **Rationale**: ONE unified search ranks ALL sources by relevance, not source type
- **Status**: IMPLEMENTED (unified_brain.py)

### ADR-002: Living ASA Physics-Based Embeddings
- **Context**: Standard embeddings lack structural semantics
- **Decision**: 720-dimensional atomic embeddings with physics properties
- **Rationale**: Mirrors atomic physics: charge (8d), shells (448d), nucleus (248d), mass (8d), valence (8d)
- **Status**: IMPLEMENTED (living_asa.py)

### ADR-003: Hybrid Search (Vector + BM25)
- **Context**: Pure vector search misses exact keyword matches
- **Decision**: Combine vector similarity (0.7 weight) + BM25 keyword matching (0.3 weight)
- **Rationale**: Fixes "exact keyword match fails" problem
- **Status**: IMPLEMENTED (context_synthesizer.py)

### ADR-004: Context Lens Comprehension Layer
- **Context**: Raw context items lack coherent understanding
- **Decision**: Transform fragments into understanding via Focus/Theme/Narrative/Insight detection
- **Rationale**: Answers "so what?" not just "what's relevant?"
- **Status**: IMPLEMENTED (context_synthesizer.py)

### ADR-005: Self-Awareness System
- **Context**: AI needs to understand its own architecture
- **Decision**: SelfAwareness class generates identity and code documents
- **Rationale**: "I am not just an app with AI. I AM the AI that IS the app."
- **Status**: IMPLEMENTED (unified_brain.py)

### ADR-006: Dual App Versions
- **Context**: Need stable consumer experience + rapid experimentation
- **Decision**: Separate index.html (consumer) and self-dev.html (development)
- **Rationale**: Self-dev enables AI-assisted code modification without breaking production
- **Status**: ESTABLISHED

### ADR-007: Local-First Architecture
- **Context**: User data privacy and sovereignty
- **Decision**: All ML runs locally on user hardware (M2 GPU via MPS)
- **Rationale**: Data never leaves device, works offline
- **Status**: IMPLEMENTED

## Known Issues / Blockers
<!-- Track problems that need attention -->

### RESOLVED: Codebase Access
- **Problem**: Could not directly review mynd-server files
- **Status**: RESOLVED - Codebase copied to workspace
- **Resolution**: Files now at `swarms/mynd_app/workspace/mynd-server/`

### MEDIUM: ASA Sparse Attention (Future Enhancement)
- **Problem**: Living ASA uses O(N^2) physics engine for pairwise energy
- **Status**: Current implementation works but could benefit from sparse optimization
- **Resolution Path**: Integrate ASA Research findings when available
- **Impact**: Performance optimization opportunity

### LOW: Goal System TODOs
- **Problem**: Several TODO comments in goal-system.js for AI milestone generation
- **Status**: Placeholder milestones used instead of AI-generated
- **Resolution Path**: Integrate Claude for intelligent milestone suggestions
- **Impact**: Feature enhancement opportunity

## Dependencies

### Core Dependencies (from requirements.txt)
| Package | Version | Purpose |
|---------|---------|---------|
| fastapi | >=0.104.0 | Web framework |
| torch | >=2.1.0 | ML core (MPS support) |
| sentence-transformers | >=2.2.2 | Embeddings |
| torch-geometric | >=2.4.0 | Graph neural networks |
| openai-whisper | >=20231117 | Voice transcription |
| open-clip-torch | >=2.23.0 | Image understanding |
| tiktoken | (optional) | Token counting |

### External Services
| Service | Usage | Required |
|---------|-------|----------|
| Supabase | AI memory storage, user auth | Optional |
| Claude API | AI chat, reflection daemon | Required for AI features |
| GitHub API | Self-improvement commits | Optional |

## Next Steps
<!-- What should happen next -->

### Immediate Actions
1. **Review ML model performance** - Benchmark Graph Transformer predictions
2. **Evaluate Context Lens accuracy** - Test focus/theme detection quality
3. **Assess daemon effectiveness** - Review reflection and evolution outputs

### Short Term
1. **Integration with agent-swarm** - Consider how MYND concepts apply
2. **Goal system completion** - Implement AI milestone generation
3. **Performance optimization** - Profile and optimize hot paths

### Long Term
1. **ASA sparse attention integration** - When research is complete
2. **Consumer launch preparation** - Polish UI, user testing
3. **Offline mode expansion** - Reduce cloud dependencies

---

## Cross-Swarm Context

### Connection to ASA Research
Living ASA implementation already exists in mynd-brain:
- **720-dimensional atomic embeddings** with physics-based properties
- **Physics engine** for computing interaction energy between atoms
- Future opportunity: Integrate true sparse attention for O(N x k) scaling

### Connection to Swarm Dev
- MYND uses similar patterns: agents, daemons, orchestration
- Potential for shared tooling and concepts

---

## Technical Summary

### Frontend Architecture (JavaScript)
- **3D Visualization**: Three.js mind map with spring physics
- **Browser ML**: TensorFlow.js for offline neural network fallback
- **Modular Design**: Separate files for goal, reflection, maintenance systems
- **PWA Support**: Service worker, manifest.json for installable app

### Backend Architecture (Python/FastAPI)
- **Server Port**: localhost:8420
- **ML Models**: Sentence Transformers (384d), Graph Transformer (11.5M params), Whisper, CLIP
- **Brain System**: UnifiedBrain orchestrates context from multiple sources
- **Context Synthesis**: Hybrid vector+BM25 search with exponential recency decay

### Key Innovations
1. **Context Lens**: Transforms raw context into coherent understanding
2. **Living ASA**: Physics-based atomic embeddings (charge, shells, nucleus, mass, valence)
3. **Self-Awareness**: AI understands its own code and architecture
4. **Knowledge Distillation**: Claude teaches the brain, brain improves over time

---

## How to Update This File

**After completing work, add an entry to Progress Log:**
```
### [DATE] [AGENT_TYPE]
- What you did
- Files changed: `file1.py`, `file2.ts`
- Outcome: success/partial/blocked
```

**When making architectural decisions, add to Architecture Decisions:**
```
### [Decision Title]
- **Context**: Why this decision was needed
- **Decision**: What was decided
- **Rationale**: Why this approach
```

## File Conventions

### Timestamp Metadata Headers
All markdown files in this workspace should include a YAML front matter header:

```yaml
---
created: YYYY-MM-DD HH:MM
updated: YYYY-MM-DD
---
```
