# Memory Approaches for LLMs: Research Summary

*Research conducted: January 2026*
*Last updated: January 6, 2026*

## 1. Context Window Limitations and Solutions

### The Problem

Large language models face fundamental limitations with context windows:

- **Fixed token limits**: Even models with 128K-1M token windows experience performance degradation
- **Context Rot**: Research measuring 18 LLMs found models "do not use their context uniformly; instead, their performance grows increasingly unreliable as input length grows"
- **Effective capacity < stated limits**: LLMs experience reasoning decline when processing inputs exceeding ~50% of maximum context length
- **Working memory overload**: Complex reasoning tasks can overwhelm LLM working memory far before hitting token limits

### Current Solutions

#### 1.1 Retrieval-Augmented Generation (RAG)
- Grounds responses in actual repositories/documents
- Avoids embedding outdated training data
- Enables dynamic knowledge access beyond training cutoff

#### 1.2 Vector Embeddings and Semantic Search
- Break text into manageable chunks
- Embed chunks and use cosine similarity for retrieval
- Acts as semantic cache for the LLM
- Retrieves most relevant pieces for specific queries

#### 1.3 Iterative Prompt Stuffing
- Process documents exceeding context window in segments
- Return structured JSON summaries per segment
- Carry forward JSON to subsequent iterations
- Enables "memory" of previously processed fragments

#### 1.4 Sliding Window Technique
- Process text in overlapping segments
- Example: 1000-token window processes tokens 1-1000, then 501-1500
- Maintains continuity across segment boundaries

#### 1.5 Hierarchical Memory
- Span individual and organizational memory layers
- Provide continuity and context across sessions
- Address lack of persistent user/organization memory

#### 1.6 Context Compression and Summarization
- Generate concise summaries before feeding to LLM
- Compress 1000-token sections to ~200-token abstracts
- Discard less relevant tokens mid-process when supported

#### 1.7 Map-Reduce Pipelines
- Summarize each data chunk into smaller chunks
- Process summaries instead of full pages
- Enables processing of arbitrarily large documents

#### 1.8 Strategic Context Placement
- Place critical information at beginning or end of context
- Models handle these positions better ("lost in the middle" problem)
- Prioritize query and top results in prominent positions

### Top Long-Context Models (2025-2026)
- **Qwen3-Coder-480B-A35B-Instruct**
- **Qwen3-30B-A3B-Thinking-2507**
- **DeepSeek-R1**
- **GPT-4 Turbo** (128K tokens)
- **Claude 2.1+** (200K tokens)
- **Gemini 1.5** (1M tokens)

Evaluated on: 164K-262K+ token windows, long-document understanding benchmarks, MoE architectures, extended context reasoning

### Key Insight: Enterprise Scale Challenge
A typical enterprise monorepo can span thousands of files and several million tokens—far exceeding even the largest context windows. This necessitates intelligent retrieval and memory systems rather than relying on raw context capacity.

---

## 2. Memory-Augmented Transformers

### Core Concept

Memory-Augmented Transformers bridge neuroscience principles with engineering advances through three taxonomic dimensions:
1. **Functional objectives**: context extension, reasoning, knowledge integration, adaptation
2. **Memory representations**: parameter-encoded, state-based, explicit, hybrid
3. **Integration mechanisms**: attention fusion, gated control, associative retrieval

### Key Architectures

#### 2.1 Large Memory Model (LM2)
*By Convergence Labs*

- Decoder-only Transformer with auxiliary memory module
- **Dedicated memory bank**: Explicit long-term storage via cross-attention
- **Hybrid memory pathway**: Original + auxiliary memory pathways
- **Dynamic updates**: Learnable input, forget, and output gates
- Selective updates prevent irrelevant data accumulation

#### 2.2 Recurrent Memory-Augmented Transformers
- Combines global attention with chunked local attention
- **Gated FIFO memory mechanism**: Persistent storage of past token representations
- Handles short-range and long-range dependencies efficiently
- Rotary positional encoding per attention head
- Avoids quadratic attention cost increase

#### 2.3 Transformer-Squared (Sun et al., 2025)
- Encodes procedural expertise into parameter space using SVD
- Dynamically blends expert vectors during inference
- Achieves 90% accuracy on unseen tasks
- Trade-off: 15% latency overhead

#### 2.4 Memory-R+ (Le et al., 2025)
- Targets resource-constrained settings (LLMs ≤1B parameters)
- **Dual episodic memory modules**: Intrinsic rewards for exploration/exploitation
- 2-14% performance improvements on reasoning tasks

#### 2.5 M+ (Wang et al., 2025)
- Splits cache into on-GPU working store + CPU-resident long-term bank
- Co-trained retriever and read-write scheduler
- Sustains coherent generation across >160K tokens
- <3% throughput overhead

#### 2.6 HRM (Wang et al., 2025a)
- Addresses reasoning depth through hierarchical convergence
- Uses coupled recurrent modules
- Enables deeper multi-step inference

#### 2.7 Phy-FusionNet
- Memory-augmented transformer for multimodal emotion recognition
- **Memory Stream Module**: FIFO-queue + decay-based updates
- Preserves long-term contextual information
- Combines periodicity and contextual attention

### Research Trends

- Shift from static caches toward **adaptive, test-time learning systems**
- Hierarchical buffering for scalability
- Surprise-gated updates to address interference
- Multi-hop inference through extended context integration
- **Lifelong Learning**: Prototypes emerging of LLMs that continuously fine-tune on user data
- Transformers with memory becoming "systems that grow with experience"

---

## 3. Episodic Memory in LLMs

### Why Episodic Memory Matters

Position paper (Pink et al., Feb 2025): "Episodic Memory is the Missing Piece for Long-Term LLM Agents"

- LLMs evolving from text-completion to agentic systems
- Required for: autonomous research, personalized support, interactive tutoring
- Five key properties underlie adaptive, context-sensitive behavior

### EM-LLM: Human-Inspired Episodic Memory
*Published at ICLR 2025*

**Key Features:**
- Integrates human episodic memory and event cognition into LLMs
- **No fine-tuning required** - works out of the box with any Transformer
- Handles practically infinite context lengths
- Maintains computational efficiency

**Architecture:**
1. **Event Segmentation**: Organizes tokens into coherent episodic events using:
   - Bayesian surprise detection
   - Graph-theoretic boundary refinement

2. **Two-Stage Memory Retrieval**:
   - Similarity-based retrieval
   - Temporally contiguous retrieval

### Alignment with Human Memory

*Trends in Cognitive Sciences, July 2025*

Current challenges:
- Popular memory-augmented approaches misaligned with human memory
- Need computational modeling framework for high-dimensional naturalistic stimuli
- Criteria needed for benchmark tasks to promote alignment

### Benchmarks and Evaluation

#### Episodic Memory Benchmark
- Targets events rich in contextual information
- Specific entities at specific times and places
- Ensures coherence and control over narratives

#### SORT (Sequence Order Recall Tasks)
- Evaluates episodic memory capabilities
- Tests correct ordering of text segments
- Extensible framework for episodic tasks
- Addresses gap in current benchmarks (which focus on facts/semantic relations)

### Survey: "Memory in the Age of AI Agents"

Unified taxonomy covering:
- **Forms**: Different memory types and structures
- **Functions**: What memory enables (reasoning, adaptation, interaction)
- **Dynamics**: How memory changes over time

Memory categorized by storage medium:
- **Token-level**: Explicit & discrete
- **Parametric**: Implicit weights
- **Latent**: Hidden states

Memory described as "cornerstone" for:
- Long-horizon reasoning
- Continual adaptation
- Complex environment interaction

---

## 4. Implementation Considerations for Agent Systems

### Recommended Architecture Patterns

1. **Hybrid Memory Systems**
   - Short-term: In-context working memory
   - Medium-term: Session-based episodic storage
   - Long-term: Persistent knowledge graphs or vector stores

2. **Event-Based Segmentation**
   - Detect natural boundaries in agent activity
   - Store coherent episodes rather than raw token streams
   - Enable efficient retrieval by episode similarity

3. **Hierarchical Retrieval**
   - First retrieve relevant episodes/memories
   - Then provide detailed context from selected memories
   - Reduces context pollution from irrelevant information

4. **Gated Memory Updates**
   - Don't store everything
   - Use surprise/importance signals for selective storage
   - Implement forgetting mechanisms for outdated information

### Open Challenges

- Balancing memory capacity vs retrieval accuracy
- Handling conflicting memories/information
- Efficient cross-session memory consolidation
- Privacy and security of stored memories
- Computational overhead of memory operations

---

## Sources

### Context Window Solutions
- [Top Techniques to Manage Context Lengths in LLMs](https://agenta.ai/blog/top-6-techniques-to-manage-context-length-in-llms)
- [The Context Window Problem: Scaling Agents Beyond Token Limits](https://factory.ai/news/context-window-problem)
- [Top LLMs for Long Context Windows in 2025](https://www.siliconflow.com/articles/en/top-LLMs-for-long-context-windows)
- [Understanding Context Window for AI Performance](https://www.qodo.ai/blog/context-windows/)
- [How To Bypass LLMs Context Limits](https://relevanceai.com/blog/how-to-overcome-context-limits-in-large-language-models)
- [RAG vs. Prompt Stuffing](https://www.spyglassmtg.com/blog/rag-vs.-prompt-stuffing-overcoming-context-window-limits-for-large-information-dense-documents)
- [Your 1M+ Context Window LLM Is Less Powerful Than You Think](https://towardsdatascience.com/your-1m-context-window-llm-is-less-powerful-than-you-think/)
- [LLM Context Windows: Why They Matter and 5 Solutions](https://www.kolena.com/guides/llm-context-windows-why-they-matter-and-5-solutions-for-context-limits/)

### Memory-Augmented Transformers
- [Memory-Augmented Transformers: A Systematic Review (arXiv)](https://arxiv.org/abs/2508.10824)
- [Memory-Augmented Transformers: Systematic Review v2 (arXiv HTML)](https://arxiv.org/html/2508.10824v2)
- [Large Memory Model (LM2) - MarkTechPost](https://www.marktechpost.com/2025/02/12/convergence-labs-introduces-the-large-memory-model-lm2-a-memory-augmented-transformer-architecture-designed-to-address-long-context-reasoning-challenges/)
- [Recurrent Memory-Augmented Transformers (arXiv)](https://arxiv.org/abs/2507.00453)
- [Memory-Augmented Transformers: Tackling Long-Context Tasks](https://medium.com/@khayyam.h/memory-augmented-transformers-tackling-long-context-tasks-without-blowing-up-ram-c0e85648773c)
- [MemATr: Memory-augmented Transformer for Video Anomaly Detection](https://dl.acm.org/doi/10.1145/3719203)

### Episodic Memory in LLMs
- [Position: Episodic Memory is the Missing Piece for Long-Term LLM Agents (arXiv)](https://arxiv.org/abs/2502.06975)
- [EM-LLM: Human-inspired Episodic Memory for Infinite Context LLMs](https://em-llm.github.io/)
- [EM-LLM GitHub Repository](https://github.com/em-llm/EM-LLM-model)
- [Towards Large Language Models with Human-like Episodic Memory](https://www.sciencedirect.com/science/article/abs/pii/S1364661325001792)
- [Episodic Memories Generation and Evaluation Benchmark (arXiv)](https://arxiv.org/html/2501.13121v1)
- [Human-inspired Episodic Memory for Infinite Context LLMs (arXiv)](https://arxiv.org/abs/2407.09450)
- [Assessing Episodic Memory in LLMs with SORT](https://openreview.net/forum?id=LLtUtzSOL5)
- [Agent Memory Paper List - GitHub](https://github.com/Shichun-Liu/Agent-Memory-Paper-List)
