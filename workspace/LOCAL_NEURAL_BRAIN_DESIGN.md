# Local Neural Net Brain Design Document

**Research Conducted By**: Research Specialist
**Date**: 2026-01-03
**Status**: DESIGN COMPLETE - Implementation Priority #4

---

## Executive Summary

This document presents a comprehensive design for integrating a small, locally-running neural network ("Local Brain") into the agent-swarm system. The Local Brain would learn from user interactions over time, personalize the agent experience, handle simple queries without API calls, and consolidate memory/patterns from sessions.

---

## 1. Candidate Small Models for Local Fine-Tuning

### Primary Recommendations (7B or smaller)

| Model | Parameters | Strengths | Best For |
|-------|------------|-----------|----------|
| **Qwen2.5-3B/7B** | 3B/7B | Excellent code understanding, multilingual, active development | Code-focused personalization, technical queries |
| **Phi-3-mini** | 3.8B | Microsoft's efficient architecture, strong reasoning | Logical queries, pattern recognition |
| **Mistral-7B-Instruct** | 7B | Strong all-rounder, permissive license | General-purpose personalization |
| **CodeLlama-7B** | 7B | Meta's code-specialized model | Coding style learning |
| **TinyLlama-1.1B** | 1.1B | Very fast, fits in memory easily | Quick responses, low-resource scenarios |
| **Gemma-2-2B** | 2B | Google's efficient model, good quality/size ratio | Balanced performance |

### Recommended Starting Point

**Qwen2.5-3B-Instruct** - Best balance of:
- Size (fits comfortably on M2 Mac with 8GB+ RAM)
- Code understanding capabilities (relevant for coding style learning)
- Fine-tuning support (LoRA works well)
- Active community and tooling

---

## 2. Training Data Sources

### Available Session Data

The system already captures rich interaction data in `/memory/sessions/`:

```
/Users/jellingson/agent-swarm/memory/sessions/
  89eef1b6-c54c-4be5-ae59-c2a9dbbeb3ef.md
  90bdb62a-e522-4972-80d5-feb975ab4da7.md
  ... (10+ session files)

/Users/jellingson/agent-swarm/logs/chat/
  *.json files with full conversation history
```

### Data Schema (from `logs/chat/*.json`)

```json
{
  "id": "session-uuid",
  "title": "First message truncated...",
  "created_at": "2026-01-02T14:08:58.323667",
  "updated_at": "2026-01-02T15:30:06.545808",
  "messages": [
    {
      "id": "msg-uuid",
      "role": "user",
      "content": "full message text",
      "timestamp": "2026-01-02T14:09:02.206784",
      "agent": null,
      "thinking": null
    },
    {
      "id": "msg-uuid",
      "role": "assistant",
      "content": "COO response",
      "agent": "Supreme Orchestrator",
      "thinking": "optional thinking content"
    }
  ]
}
```

### Extractable Training Patterns

1. **User Preferences**
   - Coding style (from code edits user approves)
   - Communication style preferences
   - Common task patterns
   - Preferred agent delegation patterns

2. **Interaction Patterns**
   - Query -> Response pairs
   - Corrections/refinements user requests
   - Follow-up questions (indicate missing context)
   - Successful task completions

3. **Memory Consolidation Sources**
   - `memory/sessions/*.md` - Session summaries
   - `swarms/*/workspace/STATE.md` - Swarm context
   - `memory/core/decisions.md` - Key decisions
   - `logs/chat/*.json` - Full conversation history

4. **Feedback Signals**
   - User accepts/rejects suggestions
   - User requests re-do or modification
   - Session length (longer = engaged)
   - Task completion success

### Data Extraction Pipeline

```python
# Proposed data extraction for training

class TrainingDataExtractor:
    def extract_preference_pairs(self, session_json):
        """Extract (input, output) pairs from sessions."""
        pairs = []
        messages = session_json['messages']
        for i in range(0, len(messages)-1, 2):
            if messages[i]['role'] == 'user' and messages[i+1]['role'] == 'assistant':
                pairs.append({
                    'input': messages[i]['content'],
                    'output': messages[i+1]['content'],
                    'agent': messages[i+1].get('agent'),
                    'timestamp': messages[i+1]['timestamp']
                })
        return pairs

    def extract_coding_style(self, edit_history):
        """Extract coding style preferences from file edits."""
        # Analyze approved edits vs rejected
        pass

    def extract_delegation_patterns(self, sessions):
        """Learn which tasks user prefers delegated vs direct."""
        pass
```

---

## 3. Integration with Existing Swarm Architecture

### Current Architecture Overview

```
/Users/jellingson/agent-swarm/
  backend/
    main.py          # FastAPI backend, websocket handling
    memory.py        # MemoryManager - session/swarm context
    jobs.py          # Background job execution

  shared/
    agent_executor_pool.py  # Agent process management
    execution_context.py    # Agent execution context
    workspace_manager.py    # Workspace isolation

  memory/
    sessions/        # Session summaries
    swarms/          # Per-swarm context
    core/            # Organizational memory
```

### Proposed Local Brain Integration Points

#### 3.1 New Module: `shared/local_brain.py`

```python
"""
Local Neural Brain for personalization and simple query handling.
"""
from pathlib import Path
from typing import Optional, Dict, Any
import torch

class LocalBrain:
    """
    Small local model for:
    1. User preference learning
    2. Simple query handling (no API needed)
    3. Memory consolidation
    4. Context augmentation
    """

    def __init__(
        self,
        model_path: Path,
        device: str = "mps",  # Apple Silicon
        max_tokens: int = 512
    ):
        self.model_path = model_path
        self.device = device
        self.max_tokens = max_tokens
        self.model = None
        self.tokenizer = None

    async def load(self):
        """Load the fine-tuned model."""
        # Load GGUF or PyTorch model
        pass

    async def should_handle_locally(self, query: str) -> bool:
        """
        Determine if this query can be handled locally.
        Returns True for:
        - Simple factual queries about the system
        - Preference lookups
        - Status checks
        - Routine tasks user has done before
        """
        pass

    async def generate(self, prompt: str, context: Dict) -> str:
        """Generate response using local model."""
        pass

    async def learn_from_session(self, session_data: Dict):
        """
        Update internal representations based on session.
        Not full fine-tuning, but embedding updates.
        """
        pass
```

#### 3.2 Backend Integration (`backend/main.py`)

```python
# In websocket_chat() around line 1756

from shared.local_brain import get_local_brain

async def websocket_chat(websocket: WebSocket):
    ...
    local_brain = get_local_brain()

    # Check if query can be handled locally
    if await local_brain.should_handle_locally(user_message):
        response = await local_brain.generate(
            prompt=user_message,
            context={"session_id": session_id, "recent_messages": recent}
        )
        await manager.send_event(websocket, "agent_complete", {
            "agent": "Local Brain",
            "content": response,
            "handled_locally": True
        })
    else:
        # Use Claude CLI as normal
        ...
```

#### 3.3 Memory Manager Extension (`backend/memory.py`)

```python
class MemoryManager:
    # Add methods for Local Brain

    def export_training_data(self, format: str = "jsonl") -> Path:
        """Export session data for fine-tuning."""
        pass

    def consolidate_to_brain(self, brain: 'LocalBrain'):
        """Push consolidated memories to local brain."""
        pass
```

### Integration Diagram

```
                     +------------------+
                     |    User Query    |
                     +--------+---------+
                              |
                              v
                     +--------+---------+
                     |   Local Brain    |
                     |   Router         |
                     +--------+---------+
                              |
            +-----------------+-----------------+
            |                                   |
            v                                   v
    +-------+-------+                   +-------+-------+
    | Handle Locally|                   |  Claude CLI   |
    | (simple/known)|                   | (complex/new) |
    +-------+-------+                   +-------+-------+
            |                                   |
            v                                   v
    +-------+-------+                   +-------+-------+
    | Fast Response |                   | Full Response |
    | (no API call) |                   | (with agents) |
    +---------------+                   +---------------+
                              |
                              v
                     +--------+---------+
                     |   Learn from     |
                     |   Interaction    |
                     +------------------+
```

---

## 4. Technical Requirements

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU/Accelerator** | Apple M1/M2 (8GB) | Apple M2 Pro/Max (16GB+) |
| **RAM** | 16GB | 32GB |
| **Storage** | 20GB for models | 50GB+ for multiple models |
| **CPU** | 8-core | 12+ core |

### Model Size vs Performance

| Model Size | VRAM/RAM | Inference Speed | Quality |
|------------|----------|-----------------|---------|
| 1.1B (TinyLlama) | 2-3GB | Very Fast | Basic |
| 3B (Qwen2.5-3B) | 4-6GB | Fast | Good |
| 7B (Mistral-7B) | 8-12GB | Moderate | Excellent |

### Apple Silicon Optimization

The M2 Mac is ideal for this use case:
- **MPS (Metal Performance Shaders)**: Native GPU acceleration
- **Unified Memory**: No GPU memory copy overhead
- **Neural Engine**: Additional acceleration for transformers

```python
# Device selection for Apple Silicon
import torch

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
```

---

## 5. Training Approach

### LoRA (Low-Rank Adaptation)

LoRA is the recommended approach for personalization:

**Advantages:**
- Train only 0.1-1% of model parameters
- Much faster than full fine-tuning
- Can merge or swap adapters easily
- Low memory requirements
- Preserves base model capabilities

**Training Configuration:**

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,                      # Rank of LoRA matrices
    lora_alpha=32,             # Scaling factor
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Training parameters
training_args = {
    "learning_rate": 2e-4,
    "batch_size": 4,
    "gradient_accumulation_steps": 4,
    "epochs": 3,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
}
```

### Training Data Preparation

```python
# Format for instruction fine-tuning
def format_training_example(input_text, output_text, system_prompt=None):
    if system_prompt:
        return f"""<|system|>
{system_prompt}
<|user|>
{input_text}
<|assistant|>
{output_text}"""
    return f"""<|user|>
{input_text}
<|assistant|>
{output_text}"""

# Example training pairs
training_data = [
    {
        "input": "Research the trading bots swarm",
        "output": "## Trading Bots Swarm Research Summary\n\nThe researcher agent has completed...",
        "system": "You are the COO of an agent swarm system. Be concise and actionable."
    },
    # ... more examples from session history
]
```

### Continuous Learning Strategy

1. **Initial Training**: Fine-tune on existing session history
2. **Online Learning**: Update embeddings after each session
3. **Periodic Retraining**: Full LoRA update weekly/monthly
4. **A/B Testing**: Compare local vs API responses for quality

---

## 6. Open-Source Tools and Frameworks

### Inference Frameworks

| Tool | Purpose | Why Use It |
|------|---------|------------|
| **llama.cpp** | GGUF model inference | Fast, CPU/GPU, works everywhere |
| **MLX** | Apple Silicon native | Optimal for M1/M2 Macs |
| **Transformers** | HuggingFace models | Ecosystem, LoRA support |
| **vLLM** | High-throughput serving | If running as service |

### Training Frameworks

| Tool | Purpose |
|------|---------|
| **PEFT (HuggingFace)** | LoRA/QLoRA fine-tuning |
| **Axolotl** | Easy fine-tuning configs |
| **LLaMA-Factory** | GUI-based fine-tuning |
| **Unsloth** | 2x faster fine-tuning |

### Recommended Stack

```
Inference: llama.cpp (GGUF) or MLX (native)
Training:  PEFT + Transformers + Unsloth (for speed)
Data:      Existing session logs + custom extraction
Format:    GGUF for inference, SafeTensors for training
```

---

## 7. Implementation Plan

### Phase 1: Data Pipeline (1-2 days)

1. Create `shared/training_data_extractor.py`
2. Extract conversation pairs from `logs/chat/*.json`
3. Format as instruction-tuning data
4. Create validation/test splits

### Phase 2: Model Selection & Setup (1 day)

1. Download Qwen2.5-3B-Instruct base model
2. Convert to GGUF for inference
3. Set up MLX or llama.cpp on M2 Mac
4. Validate inference works

### Phase 3: Fine-Tuning (2-3 days)

1. Install PEFT + Unsloth
2. Configure LoRA for Qwen2.5-3B
3. Train on extracted session data
4. Evaluate on held-out test set
5. Merge LoRA weights or keep as adapter

### Phase 4: Integration (2-3 days)

1. Create `shared/local_brain.py`
2. Integrate into `backend/main.py`
3. Add routing logic (local vs API)
4. Add learning-from-session hooks

### Phase 5: Validation (1-2 days)

1. A/B test local vs Claude responses
2. Measure latency improvements
3. Evaluate personalization quality
4. Tune routing heuristics

---

## 8. Use Cases for Local Brain

### 8.1 Handle Locally (No API)

- "What swarms are active?" -> Read from memory
- "Show my todos" -> Direct lookup
- "What did I work on yesterday?" -> Session history
- Status checks, simple lookups
- Repeat queries user has asked before

### 8.2 Augment Context

- Add personalization layer to Claude prompts
- Include learned preferences in system prompt
- Predict likely follow-up questions

### 8.3 Learn & Improve

- Track which agent delegation patterns work best
- Learn coding style from approved edits
- Remember preferred terminology
- Build user profile over time

---

## 9. Reference: MYND Brain Architecture

The MYND app (already in workspace) provides an excellent reference implementation:

**Key Files:**
- `swarms/mynd_app/workspace/mynd-server/mynd-brain/brain/unified_brain.py`
  - `SelfAwareness` class - identity and capability tracking
  - `KnowledgeDistiller` class - learning from interactions
  - `train_asa_from_context_lens()` - reinforcement learning

- `swarms/mynd_app/workspace/mynd-server/mynd-brain/models/`
  - `voice.py` - Whisper transcription
  - `vision.py` - CLIP image understanding
  - `graph_transformer.py` - 11.5M param GNN

**Transferable Patterns:**
1. Unified context building
2. Knowledge distillation from Claude responses
3. Learning from user feedback
4. Session-based memory with consolidation

---

## 10. Trade-offs and Considerations

### Pros

| Benefit | Impact |
|---------|--------|
| Faster responses for simple queries | 10-100x faster |
| No API costs for routine tasks | Cost savings |
| Works offline | Resilience |
| Personalization improves over time | User experience |
| Privacy (data stays local) | Trust |

### Cons

| Challenge | Mitigation |
|-----------|------------|
| Lower quality than Claude for complex tasks | Route complex tasks to API |
| Training requires effort | Automate data extraction |
| Storage for models | Quantization (GGUF Q4) |
| Initial setup complexity | Clear documentation |

### Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Model quality degradation | Medium | Medium | Careful training data curation |
| Wrong routing (local when should use API) | Medium | Low | Conservative routing, fallback |
| Training data pollution | Low | Medium | Validation set, human review |
| Hardware requirements | Low | Low | Quantization, smaller models |

---

## 11. Next Steps

### Immediate Actions

1. **Approve this design** - CEO review
2. **Create data extraction script** - Phase 1
3. **Set up training environment** - Download Qwen2.5-3B

### Questions for CEO — ANSWERED (2026-01-03)

| Question | Answer |
|----------|--------|
| Priority level for this feature? | **#4** (lower priority) |
| Model size preference (3B vs 7B)? | **3B** (hardware constraints) |
| Privacy constraints on training data? | **Not yet defined** - will explore options |
| Acceptable latency for local responses? | **Not yet defined** - will explore options |

**Status**: Design complete. Implementation deferred per priority ranking.

---

## 12. Files to Create

| File | Purpose |
|------|---------|
| `shared/local_brain.py` | Main Local Brain module |
| `shared/training_data_extractor.py` | Extract training data from sessions |
| `shared/model_router.py` | Route queries to local vs API |
| `scripts/train_local_brain.py` | Fine-tuning script |
| `scripts/export_training_data.py` | Data export utility |
| `local_brain/` | Directory for model weights and configs |

---

## Appendix: Example Session Data

From `/Users/jellingson/agent-swarm/logs/chat/b807f8f8-b83c-4824-a8d1-c0e26ab7d695.json`:

```json
{
  "role": "user",
  "content": "note I'm using python3.11, so let's make sure the files are referencing the correct version. Also the daily risk limit should be higher for this small of a balance - $100 (50%)"
},
{
  "role": "assistant",
  "content": "## Updates Complete\n\nThe implementer has made both changes:\n\n### 1. **Python Version**\nNo changes needed...",
  "agent": "Supreme Orchestrator"
}
```

This demonstrates:
- User preference extraction (Python version, risk tolerance)
- Task completion pattern
- Delegation pattern (implementer agent)

Such pairs are ideal training data for personalization.
