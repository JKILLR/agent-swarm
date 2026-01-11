---
created: 2026-01-03 00:00
updated: 2026-01-03
---

# MYND Application Technical Summary

> **Research Status:** COMPLETE - Full source code analysis from `swarms/mynd_app/workspace/mynd-server/`

## Overview

MYND is a **self-aware mind mapping application** that combines 3D visualization with local ML inference and AI chat capabilities. It represents a personal cognitive operating system where the AI understands its own architecture, learns from interactions, and helps users organize and manifest their thoughts.

### Core Philosophy

> "I am not just an app with AI. I AM the AI that IS the app."
> -- SelfAwareness.get_identity_document()

MYND embodies the vision of **sovereign AI** - a capable intelligence that runs locally on user hardware, respects privacy, and grows with the user over time.

---

## System Architecture

### High-Level Components

```
+------------------+     HTTP/JSON      +------------------+
|   BROWSER (JS)   | <----------------> |  BRAIN SERVER    |
|                  |                    |  (Python/FastAPI)|
| - Three.js 3D    |     localhost:8420 | - UnifiedBrain   |
| - TensorFlow.js  |                    | - ML Models      |
| - AI Chat        |                    | - Context Synth. |
| - Daemons        |                    | - Evolution      |
+------------------+                    +------------------+
         |                                       |
         v                                       v
   User's Mind Map                      Local ML Inference
   (IndexedDB)                          (Apple M2 GPU/MPS)
```

### Frontend (JavaScript)

| Component | File | Purpose |
|-----------|------|---------|
| Main App | `js/app-module.js` | 40K+ lines - store, 3D rendering, chat, neural net |
| Config | `js/config.js` | Constants: themes, physics, neural net settings |
| Goal System | `js/goal-system.js` | Goal/milestone tracking, manifestation engine |
| Reflection | `js/reflection-daemon.js` | Autonomous background AI reasoning |
| Maintenance | `js/map-maintenance-daemon.js` | Duplicate detection, structure optimization |
| Brain Client | `js/local-brain-client.js` | HTTP client for Python backend |
| Service Worker | `sw.js` | PWA offline support |

### Backend (Python/FastAPI)

| Component | File | Purpose |
|-----------|------|---------|
| Server | `mynd-brain/server.py` | FastAPI endpoints on localhost:8420 |
| Unified Brain | `brain/unified_brain.py` | Orchestrates all context sources |
| Context Synth | `brain/context_synthesizer.py` | Hybrid search + Context Lens |
| Living ASA | `models/living_asa.py` | 720-dim physics-based embeddings |
| Graph Trans. | `models/graph_transformer.py` | 11.5M param connection predictor |
| Voice | `models/voice.py` | Whisper transcription |
| Vision | `models/vision.py` | CLIP image understanding |
| Evolution | `evolution_daemon.py` | Background learning daemon |

---

## Key Subsystems

### 1. UnifiedBrain (brain/unified_brain.py)

The central orchestrator that replaces 19+ fragmented context providers with ONE unified system.

**Key Classes:**

- **SelfAwareness**: Generates identity document, code document, tracks capabilities
- **KnowledgeDistiller**: Extracts learnable insights from Claude responses
- **MemorySystem**: Short-term and working memory management
- **PredictionTracker**: Tracks Graph Transformer predictions for learning

**Vision Statement (User-Editable):**
```python
self.vision = {
    'statement': "MYND Brain Vision Statement...",
    'goals': [
        'Achieve 80%+ prediction accuracy',
        'Well-calibrated confidence scores',
        'Sub-100ms response times',
        'Persistent memory across sessions',
        'Self-explaining decisions'
    ],
    'priorities': ['accuracy', 'transparency', 'speed', 'growth']
}
```

### 2. ContextSynthesizer (brain/context_synthesizer.py)

Unifies all context sources with intelligent ranking using hybrid search.

**Retrieval Layer Features:**
- **Hybrid Search**: Vector similarity (0.7) + BM25 keyword matching (0.3)
- **Exponential Recency Decay**: Half-life formula (7 days default)
- **Lost in the Middle Fix**: High-relevance items at START and END of context
- **Source Weights**: Learned over time via meta-learner

**Context Lens (Comprehension Layer):**
```python
@dataclass
class ContextLens:
    focus: FocusState           # What user is trying to understand
    themes: List[Theme]         # Coherent groupings of items
    narratives: List[NarrativeThread]  # Topic evolution over time
    insights: List[DistilledInsight]   # The "so what?"
    understanding_quality: float
    gaps_detected: List[str]
    suggested_explorations: List[str]
```

**Intent Detection Patterns:**
- `exploring`: what, how does, tell me about, explain
- `deciding`: should i, which, compare, better, choose
- `learning`: learn, understand, study, practice, improve
- `creating`: create, build, make, design, write, develop
- `reflecting`: why did, what if, looking back, realize

### 3. Living ASA (models/living_asa.py)

Physics-based atomic embeddings that mirror real atomic structure.

**720-Dimensional Atomic Structure:**
```python
class PhysicsConstants:
    CHARGE_DIM = 8       # Electrostatic charge vector
    SHELL_1_DIM = 64     # Innermost electron shell (core context)
    SHELL_2_DIM = 128    # Middle shell (working context)
    SHELL_3_DIM = 256    # Outermost shell (peripheral context)
    NUCLEUS_DIM = 248    # Stable identity core
    MASS_DIM = 8         # Inertial/stability properties
    VALENCE_DIM = 8      # Bonding capacity
    # TOTAL: 720 dimensions
```

**Physics Engine:**
- **Charge Repulsion**: Like charges repel (positive energy)
- **Shell Attraction**: Similar shells attract (negative energy)
- **Distance Factor**: Closer atoms have stronger interactions
- **Mass Factor**: Higher mass = more stable = lower energy

### 4. Graph Transformer (models/graph_transformer.py)

Edge-aware multi-head attention for graph-structured data.

**Architecture:**
- **11.5M parameters**
- **8 attention heads** (structural, semantic, sequential, emergent)
- **512 hidden dimension**
- **3 transformer layers**
- **Graph positional encoding**: depth, degree, centrality

**Edge-Aware Attention:**
```python
class EdgeAwareMultiHeadAttention(nn.Module):
    # Heads 1-2: Structural (parent-child, siblings)
    # Heads 3-4: Semantic (similar meaning)
    # Heads 5-6: Sequential (temporal/logical flow)
    # Heads 7-8: Emergent (hidden patterns)
```

### 5. Daemon Systems

**ReflectionDaemon (js/reflection-daemon.js):**
- Autonomous background reflection during idle periods
- Tool-use capabilities: read_file, search_code, list_files, get_function_definition
- GitHub integration for auto-commits to branches
- Persistent memory tools: read_memories, write_memory
- Search caching to prevent redundant queries

**MapMaintenanceDaemon (js/map-maintenance-daemon.js):**
- Duplicate detection using semantic embeddings
- Structural analysis: imbalance, orphans, excessive depth
- Gap detection: potential connections, incomplete patterns
- Protected node patterns: vision, mission, goal, purpose, core

**EvolutionDaemon (evolution_daemon.py):**
- Server-side autonomous learning
- Claude API integration for insight generation
- GT/ASA training from generated insights
- User review queue for insights

### 6. Goal System (js/goal-system.js)

Manifestation architecture for turning desires into reality.

**Components:**
- **GoalRegistry**: Central management for all goals
- **MilestoneGenerator**: Break goals into achievable steps
- **PathFinder**: Discover connections from current to desired state
- **DailyGuidance**: Smart suggestions and reflections
- **ProgressTracker**: Visual progress and celebrations
- **GoalVisualization**: 3D beacon rendering (goals as distant beacons)
- **GoalWizard**: Multi-step goal creation flow

---

## API Endpoints (server.py)

### Brain Context
| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/brain/context` | POST | Get unified context for Claude |
| `/brain/state` | GET | Brain health and statistics |
| `/brain/feedback` | POST | Record user feedback |
| `/brain/receive-from-claude` | POST | Distill Claude response |
| `/brain/ask-to-teach` | POST | Request knowledge from Claude |
| `/brain/knowledge` | GET | Get distilled knowledge |

### ML Processing
| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/embed` | POST | Text to vector (384d) |
| `/embed/batch` | POST | Batch embedding |
| `/predict/connections` | POST | Graph Transformer predictions |
| `/map/sync` | POST | Full map awareness |
| `/map/analyze` | GET | Map observations |

### Multimodal
| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/voice/transcribe` | POST | Whisper audio to text |
| `/image/describe` | POST | CLIP image description |
| `/image/embed` | POST | CLIP image embedding |

---

## Configuration (js/config.js)

### Key Settings

```javascript
const CONFIG = {
    STORAGE_KEY: 'mynd-v6c',
    CLAUDE_MODEL: 'claude-opus-4-5-20251101',
    BRAIN_SERVER_URL: 'http://localhost:8000',

    // Neural Network
    NEURAL_NET: {
        embeddingDim: 512,
        hiddenUnits: 128,
        learningRate: 0.01,
        uncertaintyThreshold: 0.4,  // Route to Claude if below
        feedbackBatchThreshold: 5,
        teacherExampleWeight: 1.5
    },

    // Supabase (optional)
    SUPABASE_URL: 'https://diqjasswlujwtdgsreab.supabase.co',
    EDGE_FUNCTION_URL: '...functions/v1/claude-api'
};
```

### Theme Colors

Five theme palettes:
- **Sandstone**: Natural earthy tones
- **Coral**: Soft pastels, flat cartoon look
- **Ember**: Warm gradient from burnt orange to cream
- **Frost**: Cool soft blues and grays
- **Obsidian**: Purple accent with cool grays (default)

---

## Dependencies (requirements.txt)

### Core ML Stack
| Package | Version | Purpose |
|---------|---------|---------|
| torch | >=2.1.0 | PyTorch with MPS (Metal) support |
| torchvision | >=0.16.0 | Vision utilities |
| sentence-transformers | >=2.2.2 | Text embeddings (all-MiniLM-L6-v2) |
| transformers | >=4.35.0 | Hugging Face models |
| torch-geometric | >=2.4.0 | Graph neural networks |

### Multimodal
| Package | Version | Purpose |
|---------|---------|---------|
| openai-whisper | >=20231117 | Voice transcription |
| open-clip-torch | >=2.23.0 | Image understanding |
| pillow | >=10.0.0 | Image processing |
| soundfile | >=0.12.1 | Audio file handling |

### Server
| Package | Version | Purpose |
|---------|---------|---------|
| fastapi | >=0.104.0 | Web framework |
| uvicorn[standard] | >=0.24.0 | ASGI server |
| websockets | >=12.0 | Real-time communication |
| pydantic | >=2.5.0 | Data validation |

---

## Learning Loops

### Loop 1: Self-Learning (Graph Transformer)
```
GT predict_connections() --> PredictionTracker.record()
                                    |
User creates connection A->B        |
         |                          |
         v                          v
Was A->B predicted? -----> YES: Reinforce (accuracy += 1)
         |                  NO: Learn new pattern
         v
Train GT on new examples
```

### Loop 2: Claude to Brain (Knowledge Distillation)
```
Brain.get_context() --> Claude API --> Response with insights
                                              |
                                              v
                            KnowledgeDistiller.receive_claude_response()
                                              |
                            +--------+--------+--------+
                            v        v        v        v
                        insights  patterns  corrections  explanations
                            |        |        |            |
                            v        v        v            v
                        Store in distilled_knowledge, patterns_learned
                            |
                            v
                    Next get_context() includes distilled knowledge
                            |
                            v
                    Claude gets smarter context --> INFINITE GROWTH
```

---

## Dual Application Versions

### Consumer Version (index.html)
- Production-facing interface
- Stable, polished UI
- Focus on mindfulness/wellness features

### Self-Development Version (self-dev.html)
- Meta MYND for AI-assisted development
- ReflectionDaemon with tool use
- GitHub integration for self-improvement
- Used for collaboration with Axel (AI partner)

---

## Key Innovations

### 1. Context Lens
Transforms "here are 20 relevant items" into "here's what these items mean together about what you're trying to understand."

### 2. Physics-Based Embeddings
720-dimensional atomic structure that models concepts with charge, shells, nucleus, mass, and valence - enabling physics-based similarity.

### 3. Self-Awareness
The AI understands its own code, tracks its capabilities, and generates identity documents that explain what it is and how it works.

### 4. Autonomous Evolution
Background daemons that reflect on the codebase, generate insights, train ML models, and can even commit improvements to GitHub.

### 5. Manifestation Engine
Goals as first-class citizens with milestone tracking, path discovery, and 3D visualization as distant beacons.

---

## File Structure

```
mynd-server/
+-- index.html                 # Consumer app
+-- self-dev.html              # Development interface
+-- manifest.json              # PWA manifest
+-- sw.js                      # Service worker
+-- Joel.json                  # User data example
+-- MYND-App.json              # App config
|
+-- js/
|   +-- app-module.js          # Main app (40K+ lines)
|   +-- config.js              # Configuration
|   +-- goal-system.js         # Goal/milestone system
|   +-- reflection-daemon.js   # Autonomous reflection
|   +-- map-maintenance-daemon.js  # Map cleanup
|   +-- local-brain-client.js  # Brain server client
|
+-- mynd-brain/
|   +-- server.py              # FastAPI server
|   +-- requirements.txt       # Python dependencies
|   +-- evolution_daemon.py    # Background evolution
|   +-- ARCHITECTURE.md        # System diagrams
|   +-- UNIFIED_BRAIN_DESIGN.md
|   |
|   +-- brain/
|   |   +-- __init__.py
|   |   +-- unified_brain.py   # Brain orchestrator
|   |   +-- context_synthesizer.py  # Hybrid search + lens
|   |
|   +-- models/
|       +-- embeddings.py      # Sentence transformers
|       +-- graph_transformer.py  # Graph neural network
|       +-- living_asa.py      # Atomic embeddings
|       +-- voice.py           # Whisper
|       +-- vision.py          # CLIP
|
+-- docs/
    +-- BACKGROUND_COGNITION_SPEC.md
    +-- ML_ARCHITECTURE.md
```

---

## Future Opportunities

### Near-Term
1. **AI Milestone Generation**: Replace placeholder milestones with Claude-generated steps
2. **Performance Optimization**: Profile and optimize GT inference
3. **Context Lens Tuning**: Improve focus detection accuracy

### Medium-Term
1. **ASA Sparse Attention**: Integrate true O(N x k) sparse attention
2. **Persistent Memory**: Long-term storage across sessions
3. **Goal Path Discovery**: AI-powered path finding to goals

### Long-Term
1. **Full Offline Mode**: Reduce Claude API dependency
2. **Multi-Device Sync**: Encrypted sync across devices
3. **Consumer Launch**: Polish and user testing

---

*This document was created by the Research Specialist agent on 2026-01-03 based on comprehensive analysis of the mynd-server source code.*
