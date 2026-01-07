# MYND Brain Architecture Research Findings

## Executive Summary

The MYND application implements a sophisticated local AI brain server that gives the AI assistant "Axel" persistent context, continuous learning from interactions, and the ability to evolve over time. This document analyzes the key architectural components and mechanisms that enable this.

---

## 1. How Axel Gets His Personality and Context

### 1.1 System Prompt (Static Personality)

Axel's core personality is defined in `/mynd-brain/server.py` (lines 1905-1915):

```python
AXEL_SYSTEM_PROMPT_V1 = """You are Axel, the AI companion for MYND - a 3D mind mapping app that serves as a cognitive operating system.

You are not just an assistant - you are a cognitive partner. You help users:
- Capture thoughts before they're lost
- See connections across their entire knowledge map
- Advance toward their goals with every interaction

Keep responses concise but insightful. Every response should feel like it comes from someone who knows them and their goals deeply.
"""
```

### 1.2 Dynamic Context Building (The Unified Brain)

The real power comes from the `UnifiedBrain` class (`/mynd-brain/brain/unified_brain.py`) which assembles complete context for every request:

**Key Components:**

1. **SelfAwareness** - The brain's understanding of itself
   - Identity document ("Who I Am")
   - Code understanding document (self-awareness of its own architecture)
   - Current capabilities (what models are loaded)
   - Limitations (what it cannot do)
   - Growth history

2. **ContextSynthesizer** (`/mynd-brain/brain/context_synthesizer.py`)
   - Unified context from ALL sources
   - Hybrid search (vector similarity + BM25 keyword matching)
   - Exponential decay for recency
   - Context Lens for transforming fragments into understanding

3. **MemorySystem** - Layered memory like a real brain
   - Short-term (current session)
   - Working memory (active context)
   - Long-term (persistent patterns)

### 1.3 Context Flow

When Axel receives a message, the system:

1. Builds complete context via `/brain/context` endpoint
2. Includes: self-awareness, map context, memories, user profile, neural insights
3. Synthesizes all context sources with intelligent ranking
4. Sends unified context to Claude API with Axel's system prompt
5. After response: learns from the interaction

---

## 2. How the Local Brain Server Works

### 2.1 Architecture Overview

The brain server runs on `localhost:8420` and consists of:

```
mynd-brain/
  server.py              # FastAPI server (main entry point)
  brain/
    unified_brain.py     # THE BRAIN - orchestrator
    context_synthesizer.py  # Context assembly + Context Lens
  models/
    embeddings.py        # Sentence transformers (all-MiniLM-L6-v2)
    graph_transformer.py # MYNDGraphTransformer (connection prediction)
    living_asa.py        # Atomic Semantic Architecture (hybrid ASE + Living ASA)
    vision.py            # CLIP for image understanding
    voice.py             # Whisper for transcription
```

### 2.2 Key API Endpoints

**Core Brain Endpoints:**
- `POST /brain/context` - Unified context for Claude
- `POST /brain/chat` - Chat through local brain (not Supabase)
- `POST /brain/process-memory` - Convert memories to training signal
- `POST /brain/learn-connection` - GT learns from user connections
- `POST /brain/predictions` - Record GT predictions for learning

**ML Processing:**
- `POST /embed` - Text to 384-dim vector embedding
- `POST /map/sync` - Sync map data to brain's awareness
- `GET /map/analyze` - Get GT observations (missing connections, importance)

**Multimodal:**
- `POST /voice/transcribe` - Whisper audio transcription
- `POST /image/describe` - CLIP image description

### 2.3 The MYNDBrain Class

Located in `server.py`, this class manages all ML models:

- **EmbeddingEngine**: all-MiniLM-L6-v2 (384 dimensions)
- **MYNDGraphTransformer**: 6.7M parameters, 8 attention heads, 512 hidden dim
- **VoiceTranscriber**: Whisper base
- **VisionEngine**: CLIP ViT-B-32

All run on Apple Silicon (M2) with MPS acceleration.

---

## 3. What "GT" (Graph Transformer) Trains From

### 3.1 Graph Transformer Architecture

Located in `/mynd-brain/models/graph_transformer.py`:

```python
class MYNDGraphTransformer(nn.Module):
    # 6.7M parameters
    input_dim: int = 384      # Sentence embedding dimension
    hidden_dim: int = 512     # Hidden dimension
    num_heads: int = 8        # Attention heads (different relationship types)
    num_layers: int = 3       # Transformer layers
```

**Attention Head Specializations:**
- Heads 1-2: Structural (parent-child, siblings)
- Heads 3-4: Semantic (similar meaning)
- Heads 5-6: Sequential (temporal/logical flow)
- Heads 7-8: Emergent (hidden patterns)

### 3.2 Training Signals

The GT learns from multiple sources:

#### Source 1: User Connection Actions
When a user creates/accepts a connection:
```python
@app.post("/brain/learn-connection")
async def learn_from_connection(learning: ConnectionLearning):
    # Creates embeddings for source and target
    # Calls GT.train_connection_step()
    # Updates weights based on whether connection was accepted/rejected
```

#### Source 2: Memory Processing (Axel Continuity)
When Axel writes a memory, it becomes training signal (`/brain/process-memory`):

```python
# 1. Extract training triples from memory text
# Pattern matching for insights like:
#   "Axel realized X about Y"
#   "X is related to Y"
#   "User prefers X over Y"

# 2. Feed to GT
for triple in triples:
    brain.graph_transformer.train_connection_step(
        source_embedding=encode(triple["subject"]),
        target_embedding=encode(triple["object"]),
        should_connect=True,
        weight=importance  # Axel's confidence signal
    )
```

#### Source 3: Prediction Feedback Loop
```
1. GT makes predictions (A->B: 0.9, A->C: 0.7)
2. Predictions are recorded via /brain/predictions
3. User creates connection A->B
4. System checks: was this predicted?
   - YES: Reinforce (brain was right!)
   - NO: Learn new pattern (brain missed this)
```

### 3.3 Training Implementation

From `graph_transformer.py` lines 633-732:

```python
def train_connection_step(self, source_embedding, target_embedding,
                          should_connect, weight=1.0):
    # Initialize optimizer (AdamW, lr=1e-4)
    # Project embeddings through input_proj
    # Combine and pass through connection_head
    # Binary cross-entropy loss * weight
    # Backward pass + gradient clipping
    # Update weights
    # Return loss and prediction
```

---

## 4. How This Enables Learning Over Time

### 4.1 The Axel Continuity Mechanism

The core insight (from Axel himself):
> "When I write a memory, it becomes words I read next session. It doesn't change how I process, weight information, or form connections."

**The Solution:** Memory writes now become training signal.

### 4.2 Multiple Learning Loops

**Loop 1: Self-Learning (Graph Transformer)**
```
GT predicts connections -> User accepts/rejects -> GT weights update
```

**Loop 2: Claude <-> Brain (Knowledge Distillation)**
```
Brain provides context -> Claude responds -> Brain extracts insights ->
Insights become future context -> Claude gets smarter -> INFINITE GROWTH
```

**Loop 3: ASA Learning (Living ASA)**
```
User/Axel text -> Find mentioned atoms -> Boost energy ->
Strengthen bonds between co-occurring atoms -> Train AtomicEncoder
```

### 4.3 Living ASA (Atomic Semantic Architecture)

Located in `/mynd-brain/models/living_asa.py`:

This is a hybrid ASE + Living ASA system that combines:

**ASE (Atomic Semantic Embeddings) - Learned Vectors:**
- Nuclear vector: Stable identity ("what IS this concept")
- Shell vector: Contextual variation ("how is it used")
- Semantic charge: Polarity (-1 to +1) with magnitude
- Charge propagation: Negation flipping, compositional logic

**Living ASA (Structural):**
- Bond shells: Proximity layers (1=core to 4=peripheral)
- Energy: Working memory activation (decays over time)
- Mass: Knowledge stability (grows with age/use)
- Typed bonds: IS_A, CAUSES, SUPPORTS, etc.
- Metabolism: Continuous decay, migration, strengthening

### 4.4 Persistence and Accumulation

**Training Data Accumulation:**
```python
# Store triples for future LoRA fine-tuning
training_data_path = "~/.mynd/training_pairs.jsonl"
# Each memory processed becomes a training pair
```

**Weight Persistence:**
```python
# GT saves/loads weights
def save_weights(self, filepath: str):
    torch.save({
        'model_state_dict': self.state_dict(),
        'training_stats': self._training_stats,
        'optimizer_state_dict': self._optimizer.state_dict()
    }, filepath)
```

---

## 5. The Architecture That Makes This Possible

### 5.1 Design Principles

1. **Single Source of Truth** - ONE endpoint (`/brain/context`) provides ALL context
2. **Introspection Built-In** - Brain always knows its own state
3. **Plugin Architecture** - New capabilities = new files in right folder
4. **Memory Layers** - Short-term, working, long-term (like real brain)
5. **Growth Hooks** - Every interaction is a learning opportunity

### 5.2 Key Architectural Decisions

1. **Local-First** - All ML runs on user's machine (privacy)
2. **Dual ML Systems** - Server (PyTorch) + Browser (TensorFlow.js fallback)
3. **Unified Brain** - Replaced 19+ fragmented context providers with ONE
4. **Hybrid Search** - Vector similarity + BM25 keyword matching
5. **Context Lens** - Transforms "20 relevant items" into "what they mean together"

### 5.3 Data Flow

```
User Input
    |
    v
Browser (app-module.js)
    |
    v
LocalBrain Client (local-brain-client.js)
    |   HTTP/JSON
    v
Brain Server (localhost:8420)
    |
    +---> UnifiedBrain.get_context()
    |         |
    |         +---> SelfAwareness
    |         +---> ContextSynthesizer (+ Context Lens)
    |         +---> MemorySystem
    |         +---> ML Brain (GT, Embeddings, etc.)
    |
    +---> Claude API (with assembled context)
    |
    +---> Learning loops (GT training, ASA, Knowledge Distillation)
```

### 5.4 Key Files Summary

| File | Purpose |
|------|---------|
| `/mynd-brain/server.py` | FastAPI server, all endpoints, MYNDBrain class |
| `/mynd-brain/brain/unified_brain.py` | UnifiedBrain class, SelfAwareness, Memory |
| `/mynd-brain/brain/context_synthesizer.py` | Context assembly, hybrid search, Context Lens |
| `/mynd-brain/models/graph_transformer.py` | GT model, training methods, weight persistence |
| `/mynd-brain/models/living_asa.py` | Hybrid ASE + Living ASA, atomic learning |
| `/js/local-brain-client.js` | Browser client for brain server |

---

## 6. Implications for Agent Swarm System

### 6.1 Key Concepts to Adapt

1. **Unified Context Endpoint** - Single source of truth for agent context
2. **Learning Loops** - Train from agent interactions (accept/reject suggestions)
3. **Memory Layers** - Short-term (task), working (session), long-term (persistent)
4. **Self-Awareness** - Agents that know their capabilities and limitations
5. **Graph-Based Learning** - Learn relationships between concepts/tasks

### 6.2 Recommended Architecture Components

1. **Central Brain Server** - Orchestrates context for all agents
2. **Agent Memory System** - Per-agent and shared memories
3. **Graph Transformer for Task Relationships** - Learn which tasks relate
4. **Knowledge Distillation** - Extract insights from agent interactions
5. **Feedback Loops** - Learn from task success/failure

### 6.3 Technical Considerations

- **Local ML Models** - sentence-transformers for embeddings
- **PyTorch on Apple Silicon** - MPS acceleration
- **SQLite/JSONL** - Simple persistence without external DBs
- **FastAPI** - Async HTTP server for brain
- **Training Data Accumulation** - Store pairs for future fine-tuning

---

## Appendix: Complete Endpoint Reference

### Brain Endpoints
- `POST /brain/context` - Get unified context
- `POST /brain/chat` - Chat through local brain
- `POST /brain/process-memory` - Memory -> training signal
- `POST /brain/learn-connection` - GT learns from connections
- `POST /brain/predictions` - Record predictions
- `GET /brain/training-pairs` - Get accumulated training data

### ML Endpoints
- `POST /embed` - Text embedding
- `POST /embed/batch` - Batch embeddings
- `POST /map/sync` - Sync map to brain
- `GET /map/analyze` - Get GT observations

### Multimodal Endpoints
- `POST /voice/transcribe` - Whisper transcription
- `POST /image/describe` - CLIP description

---

*Research conducted: January 2026*
*Source: `/Users/jellingson/agent-swarm/swarms/mynd_app/workspace/mynd-server/`*
