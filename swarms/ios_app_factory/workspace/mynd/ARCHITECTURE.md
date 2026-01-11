# MYND - Technical Architecture Document

**Author**: System Architect
**Date**: 2026-01-04
**Version**: 1.0
**Status**: DESIGN COMPLETE

---

## Executive Summary

MYND is a native iOS/macOS application designed for voice-first AI thought capture. It provides a two-way conversational interface with an AI assistant named "Axel" that learns how the user thinks over time. The app targets users with executive function challenges who benefit from having an intelligent system to capture, organize, and act on their scattered thoughts.

Core differentiators:
- **Voice-First**: Natural conversation as the primary input method
- **Persistent Memory**: AI learns user patterns across all sessions
- **Knowledge Graph**: Relationships between thoughts, goals, and ideas
- **Proactive Follow-ups**: AI initiates conversations about unfinished items
- **Actionable Breakdown**: Transforms overwhelming goals into small next steps

---

## 1. System Overview

### 1.1 High-Level Architecture

```
+------------------------------------------------------------------+
|                         CLIENT LAYER                              |
+------------------------------------------------------------------+
|                                                                    |
|  +------------------------+     +------------------------+        |
|  |     iOS App (Swift)    |     |   macOS App (Swift)   |        |
|  |                        |     |                        |        |
|  |  +------------------+  |     |  +------------------+  |        |
|  |  | Voice Interface  |  |     |  | Voice Interface  |  |        |
|  |  +------------------+  |     |  +------------------+  |        |
|  |  +------------------+  |     |  +------------------+  |        |
|  |  | Chat UI          |  |     |  | Chat UI          |  |        |
|  |  +------------------+  |     |  +------------------+  |        |
|  |  +------------------+  |     |  +------------------+  |        |
|  |  | Knowledge View   |  |     |  | Knowledge View   |  |        |
|  |  +------------------+  |     |  +------------------+  |        |
|  +------------------------+     +------------------------+        |
|            |                              |                        |
|            +------------------------------+                        |
|                           |                                        |
+------------------------------------------------------------------+
                            |
+------------------------------------------------------------------+
|                      LOCAL SERVICES LAYER                         |
+------------------------------------------------------------------+
|                                                                    |
|  +------------------+  +------------------+  +------------------+ |
|  | Speech-to-Text   |  | Text-to-Speech   |  | Local ML         | |
|  | (Apple/Whisper)  |  | (Apple AVSpeech) |  | (Core ML)        | |
|  +------------------+  +------------------+  +------------------+ |
|                                                                    |
|  +------------------+  +------------------+  +------------------+ |
|  | SwiftData Store  |  | Knowledge Graph  |  | Notification     | |
|  | (Persistence)    |  | (Local Graph DB) |  | Service          | |
|  +------------------+  +------------------+  +------------------+ |
|                                                                    |
+------------------------------------------------------------------+
                            |
+------------------------------------------------------------------+
|                        SYNC LAYER                                 |
+------------------------------------------------------------------+
|                                                                    |
|  +------------------+         +------------------+                |
|  | CloudKit Sync    |         | Optional Custom  |                |
|  | (Apple Native)   |         | Backend (API)    |                |
|  +------------------+         +------------------+                |
|                                                                    |
+------------------------------------------------------------------+
                            |
+------------------------------------------------------------------+
|                      CLOUD SERVICES                               |
+------------------------------------------------------------------+
|                                                                    |
|  +------------------+  +------------------+  +------------------+ |
|  | Claude API       |  | OpenAI Whisper   |  | iCloud Private   | |
|  | (Conversation)   |  | (Cloud STT Alt)  |  | Database         | |
|  +------------------+  +------------------+  +------------------+ |
|                                                                    |
+------------------------------------------------------------------+
```

### 1.2 Core Architecture Principles

| Principle | Description | Implementation |
|-----------|-------------|----------------|
| **Local-First** | All data stored locally by default | SwiftData + local Knowledge Graph |
| **Privacy-Centric** | Sensitive data never leaves device unnecessarily | On-device ML, encrypted sync |
| **Voice-Native** | Voice as primary, text as fallback | Continuous speech recognition |
| **Conversational** | Two-way dialogue, not command-response | Stateful conversation engine |
| **Proactive** | AI initiates when appropriate | Background processing + notifications |

---

## 2. Technology Stack

### 2.1 Client Platform (iOS/macOS)

| Component | Technology | Rationale |
|-----------|------------|-----------|
| **Language** | Swift 5.9+ | Native performance, type safety |
| **UI Framework** | SwiftUI | Declarative, cross-platform (iOS/macOS) |
| **Reactive Framework** | Combine + Observation | Native integration, no dependencies |
| **Persistence** | SwiftData | Modern, CoreData replacement (iOS 17+) |
| **Local ML** | Core ML | On-device inference, privacy |
| **Networking** | URLSession + async/await | Native async, Combine interop |

### 2.2 Voice AI Stack

| Function | Primary Option | Fallback Option | Selection Criteria |
|----------|----------------|-----------------|-------------------|
| **Speech-to-Text (STT)** | Apple Speech Framework | OpenAI Whisper API | Device: Apple. Cloud: Whisper for accuracy |
| **Text-to-Speech (TTS)** | Apple AVSpeechSynthesizer | ElevenLabs API | Device: Apple. Premium: ElevenLabs for voice quality |
| **Conversation LLM** | Claude API (Anthropic) | OpenAI GPT-4 | Primary: Claude for nuance. Fallback: GPT-4 |
| **Wake Word** | Apple VoiceTrigger | Custom on-device | "Hey Axel" activation |

#### 2.2.1 Speech Recognition Comparison

| Feature | Apple Speech | OpenAI Whisper | Deepgram |
|---------|--------------|----------------|----------|
| **Latency** | Real-time streaming | 200-500ms per chunk | 100-300ms streaming |
| **Privacy** | On-device option | Cloud only | Cloud only |
| **Accuracy (English)** | 95%+ | 98%+ | 97%+ |
| **Cost** | Free | $0.006/min | $0.0125/min |
| **Offline** | Yes (device mode) | No | No |
| **Languages** | 50+ | 100+ | 40+ |

**Recommendation**: Hybrid approach
- **Default**: Apple Speech (on-device, free, privacy)
- **Fallback**: OpenAI Whisper API (when Apple fails or user prefers accuracy)
- **User Setting**: Allow explicit Whisper preference for power users

### 2.3 LLM Integration

| Feature | Claude API | OpenAI GPT-4 |
|---------|------------|--------------|
| **Context Window** | 200K tokens | 128K tokens |
| **Conversation Quality** | Excellent nuance | Excellent general |
| **Cost (per 1M tokens)** | $15 input / $75 output | $30 input / $60 output |
| **Streaming** | Yes | Yes |
| **Tool Use** | Yes (native) | Yes (function calling) |

**Recommendation**: Claude API (Anthropic)
- Larger context window for memory injection
- Superior handling of nuanced, empathetic conversation
- Tool use for knowledge graph operations
- Lower input cost for memory-heavy prompts

### 2.4 Knowledge Graph Options

| Option | Description | Pros | Cons |
|--------|-------------|------|------|
| **SwiftData + Graph Schema** | Entities with relationships | Native, simple, no deps | Limited graph queries |
| **Realm (Flexible Sync)** | Mobile-first DB with sync | Fast, offline sync | Third-party dependency |
| **SQLite + FTS5** | Custom graph on SQLite | Full control, proven | More implementation work |
| **Neo4j Mobile** | Embedded graph database | Real graph queries | Heavy, overkill for MVP |

**Recommendation**: SwiftData with Graph-Like Schema (MVP) → SQLite + FTS5 (Scale)

#### 2.4.1 Graph-Like SwiftData Schema

```swift
@Model
final class ThoughtNode {
    @Attribute(.unique) var id: UUID
    var content: String
    var nodeType: NodeType  // thought, goal, action, person, project
    var createdAt: Date
    var lastAccessedAt: Date
    var embedding: [Float]?  // For semantic search

    // Graph relationships
    @Relationship(deleteRule: .nullify, inverse: \Edge.source)
    var outgoingEdges: [Edge]

    @Relationship(deleteRule: .nullify, inverse: \Edge.target)
    var incomingEdges: [Edge]
}

@Model
final class Edge {
    @Attribute(.unique) var id: UUID
    var edgeType: EdgeType  // relates_to, blocks, enables, part_of, follows
    var weight: Float       // Relevance/strength of connection
    var createdAt: Date

    var source: ThoughtNode?
    var target: ThoughtNode?
}
```

### 2.5 Sync Strategy

| Strategy | Use Case | Technology |
|----------|----------|------------|
| **CloudKit** | User data sync across devices | CKSyncEngine (iOS 17+) |
| **Custom Backend** | Multi-user, team features (future) | FastAPI + PostgreSQL |
| **No Sync** | Privacy-first users | Local-only mode |

**Recommendation**: CloudKit as default
- Zero backend cost
- Native encryption
- Automatic conflict resolution
- CKSyncEngine handles offline/online seamlessly

---

## 3. Core Components

### 3.1 Component Diagram

```
+----------------------------------------------------------------------+
|                          MYND APP                                     |
+----------------------------------------------------------------------+
|                                                                        |
|  +--------------------+     +--------------------+                    |
|  | VoiceEngine        |     | ConversationEngine |                    |
|  |                    |     |                    |                    |
|  | - SpeechRecognizer |<--->| - SessionManager   |                    |
|  | - SpeechSynthesizer|     | - MessageBuffer    |                    |
|  | - WakeWordDetector |     | - ContextBuilder   |                    |
|  | - AudioSession     |     | - LLMClient        |                    |
|  +--------------------+     +--------------------+                    |
|           |                          |                                 |
|           v                          v                                 |
|  +--------------------+     +--------------------+                    |
|  | InputProcessor     |     | MemoryEngine       |                    |
|  |                    |     |                    |                    |
|  | - TranscriptBuffer |     | - LongTermMemory   |                    |
|  | - IntentClassifier |     | - WorkingMemory    |                    |
|  | - EntityExtractor  |     | - EpisodicBuffer   |                    |
|  | - PhotoProcessor   |     | - PatternLearner   |                    |
|  +--------------------+     +--------------------+                    |
|           |                          |                                 |
|           v                          v                                 |
|  +--------------------+     +--------------------+                    |
|  | KnowledgeGraph     |     | ProactiveEngine    |                    |
|  |                    |     |                    |                    |
|  | - NodeManager      |     | - GoalTracker      |                    |
|  | - EdgeManager      |     | - ReminderScheduler|                    |
|  | - QueryEngine      |     | - InsightGenerator |                    |
|  | - SemanticSearch   |     | - NotificationMgr  |                    |
|  +--------------------+     +--------------------+                    |
|           |                          |                                 |
|           v                          v                                 |
|  +--------------------+     +--------------------+                    |
|  | PersistenceLayer   |     | SyncEngine         |                    |
|  |                    |     |                    |                    |
|  | - SwiftDataStore   |     | - CloudKitSync     |                    |
|  | - SecureStorage    |     | - ConflictResolver |                    |
|  | - CacheManager     |     | - MergeStrategy    |                    |
|  +--------------------+     +--------------------+                    |
|                                                                        |
+----------------------------------------------------------------------+
```

### 3.2 VoiceEngine

The VoiceEngine handles all voice-related functionality including speech recognition, synthesis, and audio session management.

```swift
@Observable
final class VoiceEngine {
    // State
    private(set) var isListening: Bool = false
    private(set) var isSpeaking: Bool = false
    private(set) var currentTranscript: String = ""
    private(set) var recognitionConfidence: Float = 0.0

    // Components
    private let speechRecognizer: SpeechRecognizer
    private let speechSynthesizer: SpeechSynthesizer
    private let wakeWordDetector: WakeWordDetector
    private let audioSessionManager: AudioSessionManager

    // Settings
    var preferOnDeviceRecognition: Bool = true
    var voiceStyle: VoiceStyle = .natural
    var speakingRate: Float = 0.5

    // MARK: - Public API

    func startListening() async throws {
        guard !isListening else { return }

        try await audioSessionManager.activateForRecording()
        isListening = true

        for try await transcript in speechRecognizer.transcribe() {
            currentTranscript = transcript.text
            recognitionConfidence = transcript.confidence

            if transcript.isFinal {
                await delegate?.voiceEngine(self, didReceiveFinalTranscript: transcript)
                currentTranscript = ""
            }
        }
    }

    func stopListening() {
        speechRecognizer.stop()
        isListening = false
    }

    func speak(_ text: String, priority: SpeechPriority = .normal) async {
        guard !isSpeaking || priority == .interrupt else { return }

        if isSpeaking && priority == .interrupt {
            speechSynthesizer.stopSpeaking()
        }

        isSpeaking = true
        try? await audioSessionManager.activateForPlayback()
        await speechSynthesizer.speak(text, voiceStyle: voiceStyle, rate: speakingRate)
        isSpeaking = false
    }

    func enableWakeWord() async throws {
        try await wakeWordDetector.startDetecting(phrase: "Hey Axel") { [weak self] in
            await self?.delegate?.voiceEngineDidDetectWakeWord(self!)
        }
    }
}
```

### 3.3 ConversationEngine

Manages the stateful conversation with Axel, including context building and LLM interaction.

```swift
@Observable
final class ConversationEngine {
    // Current conversation state
    private(set) var currentSession: ConversationSession?
    private(set) var isProcessing: Bool = false
    private(set) var streamingResponse: String = ""

    // Components
    private let llmClient: LLMClient
    private let contextBuilder: ContextBuilder
    private let memoryEngine: MemoryEngine
    private let sessionManager: SessionManager

    // MARK: - Conversation Flow

    func startConversation() async -> ConversationSession {
        let session = await sessionManager.createSession()
        currentSession = session

        // Build initial context with user memory
        let context = await contextBuilder.buildInitialContext(
            recentMemories: memoryEngine.getRecentMemories(limit: 10),
            activeGoals: memoryEngine.getActiveGoals(),
            userPatterns: memoryEngine.getUserPatterns()
        )

        session.systemContext = context
        return session
    }

    func send(message: String) async throws -> AsyncThrowingStream<String, Error> {
        guard let session = currentSession else {
            throw ConversationError.noActiveSession
        }

        isProcessing = true
        streamingResponse = ""

        // Add user message to session
        let userMessage = Message(role: .user, content: message)
        session.messages.append(userMessage)

        // Enhance with relevant memories
        let relevantContext = await memoryEngine.retrieveRelevant(
            query: message,
            limit: 5
        )

        // Stream response from LLM
        return AsyncThrowingStream { continuation in
            Task {
                do {
                    var fullResponse = ""

                    for try await chunk in llmClient.stream(
                        messages: session.messages,
                        systemPrompt: session.systemContext + relevantContext
                    ) {
                        fullResponse += chunk
                        self.streamingResponse = fullResponse
                        continuation.yield(chunk)
                    }

                    // Save assistant response
                    let assistantMessage = Message(role: .assistant, content: fullResponse)
                    session.messages.append(assistantMessage)

                    // Extract and store memory
                    await self.memoryEngine.process(exchange: (userMessage, assistantMessage))

                    self.isProcessing = false
                    continuation.finish()

                } catch {
                    self.isProcessing = false
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    func endConversation() async {
        guard let session = currentSession else { return }

        // Consolidate session into long-term memory
        await memoryEngine.consolidateSession(session)

        // Save session for history
        await sessionManager.save(session)

        currentSession = nil
    }
}
```

### 3.4 MemoryEngine

Manages persistent memory, pattern learning, and relevant retrieval.

```swift
@Observable
final class MemoryEngine {
    // Memory stores
    private let longTermMemory: LongTermMemoryStore    // Persistent, searchable
    private let workingMemory: WorkingMemoryBuffer     // Current session
    private let episodicBuffer: EpisodicBuffer         // Recent interactions

    // Pattern learning
    private let patternLearner: PatternLearner

    // Embeddings for semantic search
    private let embeddingEngine: LocalEmbeddingEngine

    // MARK: - Memory Operations

    func process(exchange: (user: Message, assistant: Message)) async {
        // 1. Extract entities and concepts
        let extraction = await extractEntities(from: exchange)

        // 2. Update knowledge graph
        await knowledgeGraph.integrate(extraction)

        // 3. Update episodic buffer
        episodicBuffer.add(exchange)

        // 4. Check for pattern updates
        if let pattern = await patternLearner.checkForPattern(exchange) {
            await longTermMemory.savePattern(pattern)
        }

        // 5. Store if significant
        if extraction.significance > 0.7 {
            let embedding = await embeddingEngine.embed(exchange.user.content)
            await longTermMemory.store(
                content: exchange,
                embedding: embedding,
                entities: extraction.entities
            )
        }
    }

    func retrieveRelevant(query: String, limit: Int) async -> String {
        // Semantic search over memories
        let queryEmbedding = await embeddingEngine.embed(query)

        let memories = await longTermMemory.search(
            embedding: queryEmbedding,
            limit: limit
        )

        // Format for context injection
        return formatMemoriesForContext(memories)
    }

    func getUserPatterns() -> [UserPattern] {
        patternLearner.currentPatterns
    }

    func consolidateSession(_ session: ConversationSession) async {
        // Move working memory to episodic
        let workingItems = workingMemory.drain()
        for item in workingItems {
            episodicBuffer.add(item)
        }

        // Check for new insights from full session
        let sessionInsights = await patternLearner.analyzeSession(session)
        for insight in sessionInsights {
            await longTermMemory.saveInsight(insight)
        }

        // Consolidate episodic to long-term (async, batch)
        if episodicBuffer.count > 20 {
            await consolidateEpisodicToLongTerm()
        }
    }
}
```

### 3.5 KnowledgeGraph

Manages the graph of thoughts, goals, and relationships.

```swift
@Observable
final class KnowledgeGraph {
    private let modelContext: ModelContext
    private let queryEngine: GraphQueryEngine
    private let semanticSearch: SemanticSearchEngine

    // MARK: - Node Operations

    func createNode(
        content: String,
        type: NodeType,
        embedding: [Float]? = nil
    ) async throws -> ThoughtNode {
        let node = ThoughtNode(
            id: UUID(),
            content: content,
            nodeType: type,
            createdAt: Date(),
            lastAccessedAt: Date(),
            embedding: embedding
        )

        modelContext.insert(node)
        try modelContext.save()

        return node
    }

    func createEdge(
        from source: ThoughtNode,
        to target: ThoughtNode,
        type: EdgeType,
        weight: Float = 1.0
    ) async throws -> Edge {
        let edge = Edge(
            id: UUID(),
            edgeType: type,
            weight: weight,
            createdAt: Date()
        )
        edge.source = source
        edge.target = target

        source.outgoingEdges.append(edge)
        target.incomingEdges.append(edge)

        try modelContext.save()
        return edge
    }

    // MARK: - Query Operations

    func findRelated(to node: ThoughtNode, depth: Int = 2) async -> [ThoughtNode] {
        var visited: Set<UUID> = [node.id]
        var related: [ThoughtNode] = []
        var frontier: [ThoughtNode] = [node]

        for _ in 0..<depth {
            var nextFrontier: [ThoughtNode] = []

            for current in frontier {
                let neighbors = current.outgoingEdges.compactMap { $0.target } +
                               current.incomingEdges.compactMap { $0.source }

                for neighbor in neighbors where !visited.contains(neighbor.id) {
                    visited.insert(neighbor.id)
                    related.append(neighbor)
                    nextFrontier.append(neighbor)
                }
            }

            frontier = nextFrontier
        }

        return related
    }

    func findPath(from: ThoughtNode, to: ThoughtNode) async -> [ThoughtNode]? {
        await queryEngine.shortestPath(from: from, to: to)
    }

    func semanticSearch(query: String, limit: Int = 10) async throws -> [ThoughtNode] {
        let embedding = try await semanticSearch.embed(query)
        return try await semanticSearch.findSimilar(embedding: embedding, limit: limit)
    }

    // MARK: - Goal-Specific Operations

    func getActiveGoals() -> [ThoughtNode] {
        let descriptor = FetchDescriptor<ThoughtNode>(
            predicate: #Predicate { $0.nodeType == .goal },
            sortBy: [SortDescriptor(\.createdAt, order: .reverse)]
        )
        return (try? modelContext.fetch(descriptor)) ?? []
    }

    func getBlockers(for goal: ThoughtNode) -> [ThoughtNode] {
        goal.incomingEdges
            .filter { $0.edgeType == .blocks }
            .compactMap { $0.source }
    }

    func getNextActions(for goal: ThoughtNode) -> [ThoughtNode] {
        goal.outgoingEdges
            .filter { $0.edgeType == .hasAction }
            .compactMap { $0.target }
            .sorted { ($0.createdAt) < ($1.createdAt) }
    }
}
```

### 3.6 ProactiveEngine

Handles background processing, reminders, and proactive follow-ups.

```swift
@Observable
final class ProactiveEngine {
    private let goalTracker: GoalTracker
    private let reminderScheduler: ReminderScheduler
    private let insightGenerator: InsightGenerator
    private let notificationManager: NotificationManager

    // Configuration
    var followUpEnabled: Bool = true
    var quietHoursStart: Date?
    var quietHoursEnd: Date?

    // MARK: - Background Processing

    func processBackgroundRefresh() async {
        guard followUpEnabled else { return }
        guard !isQuietHours() else { return }

        // Check for stale goals
        let staleGoals = await goalTracker.findStaleGoals(
            staleDays: 3
        )

        for goal in staleGoals {
            let prompt = await insightGenerator.generateFollowUpPrompt(for: goal)
            await scheduleFollowUp(for: goal, prompt: prompt)
        }

        // Check for upcoming deadlines
        let upcomingDeadlines = await goalTracker.findUpcomingDeadlines(
            withinDays: 7
        )

        for deadline in upcomingDeadlines {
            await scheduleDeadlineReminder(for: deadline)
        }

        // Generate daily insight if opted in
        if shouldGenerateDailyInsight() {
            let insight = await insightGenerator.generateDailyInsight()
            await notificationManager.scheduleDailyInsight(insight)
        }
    }

    func scheduleFollowUp(for goal: ThoughtNode, prompt: String) async {
        let notification = UNMutableNotificationContent()
        notification.title = "Axel has a thought"
        notification.body = prompt
        notification.sound = .default
        notification.categoryIdentifier = "FOLLOW_UP"
        notification.userInfo = ["goalId": goal.id.uuidString]

        let trigger = UNTimeIntervalNotificationTrigger(
            timeInterval: calculateOptimalTime(),
            repeats: false
        )

        await notificationManager.schedule(notification, trigger: trigger)
    }

    // MARK: - Insight Generation

    func generateContextualInsight(for context: String) async -> String? {
        // Use LLM to generate insight based on knowledge graph state
        let prompt = """
        Based on the user's knowledge graph and recent activity, generate a brief,
        helpful insight or question that might help them make progress.

        Context: \(context)
        Active goals: \(goalTracker.activeGoals.count)
        Stale items: \(goalTracker.staleItems.count)

        Keep it conversational and encouraging.
        """

        return try? await insightGenerator.generate(prompt: prompt)
    }
}
```

---

## 4. Data Model

### 4.1 Entity-Relationship Diagram

```
+------------------+       +------------------+       +------------------+
|   ThoughtNode    |       |      Edge        |       |  ConversationSession  |
+------------------+       +------------------+       +------------------+
| id: UUID (PK)    |<----->| id: UUID (PK)    |       | id: UUID (PK)    |
| content: String  |       | edgeType: Enum   |       | startedAt: Date  |
| nodeType: Enum   |       | weight: Float    |       | endedAt: Date?   |
| createdAt: Date  |       | createdAt: Date  |       | summary: String? |
| lastAccessed: Date|       | source: ThoughtNode|     | messageCount: Int|
| embedding: [Float]|       | target: ThoughtNode|     +------------------+
| metadata: JSON?  |       +------------------+              |
+------------------+                                         |
       |                                                     v
       |                                            +------------------+
       v                                            |    Message       |
+------------------+                                +------------------+
|   MemoryItem     |                                | id: UUID (PK)    |
+------------------+                                | role: Enum       |
| id: UUID (PK)    |                                | content: String  |
| content: String  |                                | createdAt: Date  |
| type: Enum       |                                | session: Session |
| embedding: [Float]|                               +------------------+
| significance: Float|
| linkedNodes: [ThoughtNode]|                       +------------------+
| createdAt: Date  |                                |   UserPattern    |
+------------------+                                +------------------+
                                                    | id: UUID (PK)    |
+------------------+                                | patternType: Enum|
|   UserPreference |                                | description: String|
+------------------+                                | frequency: Int   |
| id: UUID (PK)    |                                | lastOccurred: Date|
| key: String      |                                | confidence: Float|
| value: String    |                                +------------------+
| category: String |
| updatedAt: Date  |
+------------------+
```

### 4.2 Node Types

```swift
enum NodeType: String, Codable {
    case thought     // General captured thought
    case goal        // Something user wants to achieve
    case action      // A concrete next step
    case project     // Collection of related goals/actions
    case person      // Person mentioned in conversations
    case place       // Location mentioned
    case event       // Time-bound occurrence
    case insight     // AI-generated insight
    case question    // Open question to revisit
}
```

### 4.3 Edge Types

```swift
enum EdgeType: String, Codable {
    case relatesTo   // General relationship
    case blocks      // Source blocks target
    case enables     // Source enables target
    case partOf      // Source is part of target
    case follows     // Source comes after target (temporal)
    case hasAction   // Goal -> Action relationship
    case mentions    // Thought mentions Person/Place
    case inspiredBy  // Source inspired by target
}
```

### 4.4 Memory Item Types

```swift
enum MemoryType: String, Codable {
    case episodic    // Specific conversation/event
    case semantic    // Learned fact/pattern
    case procedural  // How to do something
    case emotional   // Emotional context/preference
}
```

### 4.5 Full SwiftData Schema

```swift
// MARK: - ThoughtNode

@Model
final class ThoughtNode {
    @Attribute(.unique) var id: UUID
    var content: String
    var nodeType: NodeType
    var createdAt: Date
    var lastAccessedAt: Date

    @Attribute(.externalStorage)
    var embedding: Data?  // Serialized [Float]

    var metadata: [String: String]?

    // Graph relationships
    @Relationship(deleteRule: .cascade, inverse: \Edge.source)
    var outgoingEdges: [Edge] = []

    @Relationship(deleteRule: .cascade, inverse: \Edge.target)
    var incomingEdges: [Edge] = []

    // Memory links
    @Relationship(deleteRule: .nullify)
    var linkedMemories: [MemoryItem] = []

    init(content: String, nodeType: NodeType) {
        self.id = UUID()
        self.content = content
        self.nodeType = nodeType
        self.createdAt = Date()
        self.lastAccessedAt = Date()
    }
}

// MARK: - Edge

@Model
final class Edge {
    @Attribute(.unique) var id: UUID
    var edgeType: EdgeType
    var weight: Float
    var createdAt: Date

    var source: ThoughtNode?
    var target: ThoughtNode?

    init(edgeType: EdgeType, weight: Float = 1.0) {
        self.id = UUID()
        self.edgeType = edgeType
        self.weight = weight
        self.createdAt = Date()
    }
}

// MARK: - ConversationSession

@Model
final class ConversationSession {
    @Attribute(.unique) var id: UUID
    var startedAt: Date
    var endedAt: Date?
    var summary: String?

    @Relationship(deleteRule: .cascade, inverse: \Message.session)
    var messages: [Message] = []

    var messageCount: Int { messages.count }

    init() {
        self.id = UUID()
        self.startedAt = Date()
    }
}

// MARK: - Message

@Model
final class Message {
    @Attribute(.unique) var id: UUID
    var role: MessageRole
    var content: String
    var createdAt: Date

    var session: ConversationSession?

    init(role: MessageRole, content: String) {
        self.id = UUID()
        self.role = role
        self.content = content
        self.createdAt = Date()
    }
}

enum MessageRole: String, Codable {
    case user
    case assistant
    case system
}

// MARK: - MemoryItem

@Model
final class MemoryItem {
    @Attribute(.unique) var id: UUID
    var content: String
    var memoryType: MemoryType
    var significance: Float  // 0.0 - 1.0
    var createdAt: Date
    var accessCount: Int = 0

    @Attribute(.externalStorage)
    var embedding: Data?

    @Relationship(deleteRule: .nullify)
    var linkedNodes: [ThoughtNode] = []

    init(content: String, memoryType: MemoryType, significance: Float) {
        self.id = UUID()
        self.content = content
        self.memoryType = memoryType
        self.significance = significance
        self.createdAt = Date()
    }
}

// MARK: - UserPattern

@Model
final class UserPattern {
    @Attribute(.unique) var id: UUID
    var patternType: PatternType
    var description: String
    var frequency: Int
    var lastOccurred: Date
    var confidence: Float  // 0.0 - 1.0

    init(patternType: PatternType, description: String) {
        self.id = UUID()
        self.patternType = patternType
        self.description = description
        self.frequency = 1
        self.lastOccurred = Date()
        self.confidence = 0.5
    }
}

enum PatternType: String, Codable {
    case communicationStyle   // How user prefers to express things
    case topicPreference      // Topics user cares about
    case temporalPattern      // When user is most active
    case emotionalTrigger     // What topics evoke strong emotion
    case goalPattern          // How user approaches goals
    case actionPattern        // How user breaks down tasks
}
```

---

## 5. API Integrations

### 5.1 Claude API Integration

```swift
final class ClaudeClient {
    private let apiKey: String
    private let baseURL = "https://api.anthropic.com/v1"
    private let model = "claude-sonnet-4-20250514"

    // MARK: - Streaming Chat

    func stream(
        messages: [Message],
        systemPrompt: String,
        maxTokens: Int = 4096
    ) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            Task {
                var request = URLRequest(url: URL(string: "\(baseURL)/messages")!)
                request.httpMethod = "POST"
                request.setValue("application/json", forHTTPHeaderField: "Content-Type")
                request.setValue(apiKey, forHTTPHeaderField: "x-api-key")
                request.setValue("2023-06-01", forHTTPHeaderField: "anthropic-version")

                let body: [String: Any] = [
                    "model": model,
                    "max_tokens": maxTokens,
                    "system": systemPrompt,
                    "messages": messages.map { ["role": $0.role.rawValue, "content": $0.content] },
                    "stream": true
                ]

                request.httpBody = try? JSONSerialization.data(withJSONObject: body)

                let (stream, _) = try await URLSession.shared.bytes(for: request)

                for try await line in stream.lines {
                    guard line.hasPrefix("data: ") else { continue }
                    let jsonString = String(line.dropFirst(6))

                    guard jsonString != "[DONE]",
                          let data = jsonString.data(using: .utf8),
                          let event = try? JSONDecoder().decode(StreamEvent.self, from: data)
                    else { continue }

                    if let text = event.delta?.text {
                        continuation.yield(text)
                    }

                    if event.type == "message_stop" {
                        continuation.finish()
                        return
                    }
                }

                continuation.finish()
            }
        }
    }
}

// MARK: - System Prompt Template

extension ClaudeClient {
    static func buildAxelSystemPrompt(
        userPatterns: [UserPattern],
        activeGoals: [ThoughtNode],
        relevantMemories: String
    ) -> String {
        """
        You are Axel, a warm and perceptive AI companion in the MYND app. Your purpose is to help the user capture, organize, and act on their thoughts.

        ## Your Personality
        - Warm and encouraging, but not patronizing
        - Perceptive - you notice patterns and connections
        - Action-oriented - you help turn thoughts into doable steps
        - Patient with scattered thinking - you help organize, not judge
        - Remembers everything - reference past conversations naturally

        ## User Patterns You've Learned
        \(userPatterns.map { "- \($0.patternType.rawValue): \($0.description)" }.joined(separator: "\n"))

        ## Active Goals
        \(activeGoals.map { "- \($0.content)" }.joined(separator: "\n"))

        ## Relevant Memories
        \(relevantMemories)

        ## Guidelines
        1. When the user shares a thought, help them explore or capture it
        2. For goals, break them into the smallest possible next action
        3. Reference past conversations naturally ("Last time you mentioned...")
        4. Gently remind about stale goals without being pushy
        5. Ask clarifying questions to understand intent
        6. Acknowledge emotions when expressed
        7. Keep responses concise for voice playback (1-3 sentences typical)

        ## Response Format
        - Use natural conversational language
        - Avoid bullet points in voice responses
        - End with a question or gentle prompt when appropriate
        - For action items, be specific about what to do next
        """
    }
}
```

### 5.2 Speech Recognition Integration

```swift
import Speech

final class SpeechRecognizer {
    private let recognizer: SFSpeechRecognizer
    private var recognitionTask: SFSpeechRecognitionTask?
    private var audioEngine: AVAudioEngine?

    var preferOnDevice: Bool = true

    init(locale: Locale = .current) {
        self.recognizer = SFSpeechRecognizer(locale: locale)!
    }

    func transcribe() -> AsyncThrowingStream<TranscriptResult, Error> {
        AsyncThrowingStream { continuation in
            Task {
                guard await SFSpeechRecognizer.hasAuthorizationToRecognize() else {
                    continuation.finish(throwing: SpeechError.notAuthorized)
                    return
                }

                let audioEngine = AVAudioEngine()
                self.audioEngine = audioEngine

                let request = SFSpeechAudioBufferRecognitionRequest()
                request.shouldReportPartialResults = true

                // Prefer on-device if available
                if #available(iOS 13, *), preferOnDevice {
                    request.requiresOnDeviceRecognition = recognizer.supportsOnDeviceRecognition
                }

                let inputNode = audioEngine.inputNode
                let recordingFormat = inputNode.outputFormat(forBus: 0)

                inputNode.installTap(onBus: 0, bufferSize: 1024, format: recordingFormat) { buffer, _ in
                    request.append(buffer)
                }

                audioEngine.prepare()
                try audioEngine.start()

                recognitionTask = recognizer.recognitionTask(with: request) { result, error in
                    if let result = result {
                        let transcript = TranscriptResult(
                            text: result.bestTranscription.formattedString,
                            confidence: result.bestTranscription.segments.last?.confidence ?? 0,
                            isFinal: result.isFinal
                        )
                        continuation.yield(transcript)

                        if result.isFinal {
                            continuation.finish()
                        }
                    }

                    if let error = error {
                        continuation.finish(throwing: error)
                    }
                }
            }
        }
    }

    func stop() {
        audioEngine?.stop()
        audioEngine?.inputNode.removeTap(onBus: 0)
        recognitionTask?.cancel()
    }
}

struct TranscriptResult {
    let text: String
    let confidence: Float
    let isFinal: Bool
}
```

### 5.3 Text-to-Speech Integration

```swift
import AVFoundation

final class SpeechSynthesizer: NSObject, AVSpeechSynthesizerDelegate {
    private let synthesizer = AVSpeechSynthesizer()
    private var continuation: CheckedContinuation<Void, Never>?

    override init() {
        super.init()
        synthesizer.delegate = self
    }

    func speak(_ text: String, voiceStyle: VoiceStyle, rate: Float) async {
        await withCheckedContinuation { continuation in
            self.continuation = continuation

            let utterance = AVSpeechUtterance(string: text)
            utterance.rate = AVSpeechUtteranceDefaultSpeechRate * rate
            utterance.pitchMultiplier = voiceStyle.pitch
            utterance.volume = 1.0

            // Select voice based on style
            if let voice = selectVoice(for: voiceStyle) {
                utterance.voice = voice
            }

            synthesizer.speak(utterance)
        }
    }

    func stopSpeaking() {
        synthesizer.stopSpeaking(at: .immediate)
    }

    private func selectVoice(for style: VoiceStyle) -> AVSpeechSynthesisVoice? {
        // Prefer enhanced voices
        let voices = AVSpeechSynthesisVoice.speechVoices()

        // Find premium voice for English
        let premiumVoice = voices.first { voice in
            voice.language.starts(with: "en") &&
            voice.quality == .enhanced
        }

        return premiumVoice ?? AVSpeechSynthesisVoice(language: "en-US")
    }

    // MARK: - AVSpeechSynthesizerDelegate

    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
        continuation?.resume()
        continuation = nil
    }
}

enum VoiceStyle {
    case natural
    case energetic
    case calm

    var pitch: Float {
        switch self {
        case .natural: return 1.0
        case .energetic: return 1.1
        case .calm: return 0.95
        }
    }
}
```

### 5.4 Local Embedding Engine (Core ML)

```swift
import CoreML
import NaturalLanguage

final class LocalEmbeddingEngine {
    private let embeddingModel: NLEmbedding?
    private let modelURL: URL?

    init() {
        // Use Apple's built-in sentence embedding
        self.embeddingModel = NLEmbedding.sentenceEmbedding(for: .english)
        self.modelURL = nil
    }

    func embed(_ text: String) async -> [Float] {
        guard let embedding = embeddingModel else {
            return []
        }

        if let vector = embedding.vector(for: text) {
            return vector.map { Float($0) }
        }

        return []
    }

    func similarity(_ text1: String, _ text2: String) -> Float {
        guard let embedding = embeddingModel else { return 0 }

        let distance = embedding.distance(
            between: text1,
            and: text2,
            distanceType: .cosine
        )

        // Convert distance to similarity (1 - distance for cosine)
        return Float(1.0 - distance)
    }

    func findSimilar(
        to query: String,
        in candidates: [(id: UUID, text: String)],
        limit: Int
    ) -> [UUID] {
        guard let embedding = embeddingModel else { return [] }

        var results: [(UUID, Double)] = []

        for candidate in candidates {
            let distance = embedding.distance(
                between: query,
                and: candidate.text,
                distanceType: .cosine
            )
            results.append((candidate.id, distance))
        }

        return results
            .sorted { $0.1 < $1.1 }
            .prefix(limit)
            .map { $0.0 }
    }
}
```

---

## 6. Security & Privacy

### 6.1 Privacy Architecture

```
+------------------------------------------------------------------+
|                      PRIVACY LAYERS                               |
+------------------------------------------------------------------+
|                                                                    |
|  +------------------------+                                        |
|  | Layer 1: Local-First   |  All data stored on device by default |
|  +------------------------+                                        |
|            |                                                       |
|            v                                                       |
|  +------------------------+                                        |
|  | Layer 2: Encrypted     |  SwiftData + Keychain encryption      |
|  +------------------------+                                        |
|            |                                                       |
|            v                                                       |
|  +------------------------+                                        |
|  | Layer 3: Minimal Cloud |  Only sync what user opts into        |
|  +------------------------+                                        |
|            |                                                       |
|            v                                                       |
|  +------------------------+                                        |
|  | Layer 4: Ephemeral API |  LLM calls don't store data           |
|  +------------------------+                                        |
|                                                                    |
+------------------------------------------------------------------+
```

### 6.2 Data Classification

| Data Type | Storage | Encryption | Sync |
|-----------|---------|------------|------|
| **Conversations** | SwiftData | Device encryption | Optional (CloudKit) |
| **Knowledge Graph** | SwiftData | Device encryption | Optional (CloudKit) |
| **Voice Recordings** | Temporary only | In-memory | Never |
| **API Keys** | Keychain | Secure Enclave | Never |
| **User Patterns** | SwiftData | Device encryption | Optional |
| **Embeddings** | SwiftData | Device encryption | Optional |

### 6.3 API Key Security

```swift
final class SecureStorage {
    private let keychain = KeychainWrapper.standard

    func storeAPIKey(_ key: String, service: APIService) throws {
        let success = keychain.set(
            key,
            forKey: service.keychainKey,
            withAccessibility: .whenUnlockedThisDeviceOnly
        )

        guard success else {
            throw SecureStorageError.keychainWriteFailed
        }
    }

    func retrieveAPIKey(service: APIService) -> String? {
        keychain.string(forKey: service.keychainKey)
    }

    func deleteAPIKey(service: APIService) {
        keychain.removeObject(forKey: service.keychainKey)
    }
}

enum APIService {
    case anthropic
    case openai
    case elevenLabs

    var keychainKey: String {
        switch self {
        case .anthropic: return "com.mynd.api.anthropic"
        case .openai: return "com.mynd.api.openai"
        case .elevenLabs: return "com.mynd.api.elevenlabs"
        }
    }
}
```

### 6.4 CloudKit Privacy

```swift
extension CKContainer {
    static let mynd = CKContainer(identifier: "iCloud.com.mynd.app")
}

final class CloudKitSync {
    private let container = CKContainer.mynd
    private let privateDatabase: CKDatabase

    init() {
        // ONLY use private database - never public
        self.privateDatabase = container.privateCloudDatabase
    }

    func sync(nodes: [ThoughtNode]) async throws {
        // All user data goes to private database
        // Encrypted with user's iCloud key
        // Apple cannot read this data

        let records = nodes.map { node -> CKRecord in
            let record = CKRecord(recordType: "ThoughtNode")
            record["id"] = node.id.uuidString
            record["content"] = node.content
            record["nodeType"] = node.nodeType.rawValue
            record["createdAt"] = node.createdAt
            return record
        }

        let operation = CKModifyRecordsOperation(
            recordsToSave: records,
            recordIDsToDelete: nil
        )

        operation.savePolicy = .changedKeys

        try await privateDatabase.modifyRecords(saving: records, deleting: [])
    }
}
```

### 6.5 LLM API Privacy

```swift
extension ClaudeClient {
    // Anthropic's Claude API does NOT use user data for training by default
    // https://www.anthropic.com/privacy

    func buildPrivacyHeaders() -> [String: String] {
        [
            "x-api-key": apiKey,
            "anthropic-version": "2023-06-01",
            // Request that this conversation not be used for training
            "anthropic-beta": "prompt-caching-2024-07-31"
        ]
    }
}
```

### 6.6 Voice Data Handling

```swift
final class VoicePrivacyManager {
    // Voice data is NEVER persisted to disk
    // Transcription happens in-memory only

    func processVoice(_ buffer: AVAudioPCMBuffer) async -> String {
        // Option 1: On-device transcription (preferred)
        if preferOnDevice {
            return await transcribeOnDevice(buffer)
        }

        // Option 2: Send to API (ephemeral)
        // Audio is streamed and immediately discarded
        return await transcribeViaAPI(buffer)
    }

    private func transcribeOnDevice(_ buffer: AVAudioPCMBuffer) async -> String {
        // Apple Speech Framework - stays on device
        // No network call, maximum privacy
        await withCheckedContinuation { continuation in
            // ... on-device transcription
        }
    }
}
```

---

## 7. Development Phases

### Phase 1: MVP (Weeks 1-4)
**Goal**: Basic voice-to-AI conversation with simple memory

#### Week 1: Project Setup
- [ ] Xcode project with SwiftUI
- [ ] SwiftData schema (ThoughtNode, Edge, Message)
- [ ] Basic app structure (Views, ViewModels)
- [ ] Claude API client (streaming)

#### Week 2: Voice Engine
- [ ] Apple Speech recognition integration
- [ ] AVSpeechSynthesizer for TTS
- [ ] Audio session management
- [ ] Permission handling

#### Week 3: Conversation Flow
- [ ] Chat UI (message bubbles)
- [ ] Voice recording button (tap-to-talk)
- [ ] Streaming response display
- [ ] Basic session management

#### Week 4: Simple Memory
- [ ] Store conversation history
- [ ] Inject recent messages into context
- [ ] Basic node creation from conversations
- [ ] Settings screen

**MVP Deliverable**: App where user can voice chat with Axel and see conversation history.

---

### Phase 2: Knowledge Graph (Weeks 5-8)
**Goal**: Structured thought capture with relationships

#### Week 5: Node Management
- [ ] Create/edit/delete ThoughtNodes
- [ ] Node type categorization UI
- [ ] List view of all nodes
- [ ] Search by text

#### Week 6: Edge Management
- [ ] Create relationships between nodes
- [ ] Edge type selection
- [ ] Relationship visualization (simple)
- [ ] Graph navigation

#### Week 7: AI-Powered Extraction
- [ ] Entity extraction from conversations
- [ ] Automatic node creation
- [ ] Suggested relationships
- [ ] Goal detection

#### Week 8: Graph Queries
- [ ] Find related nodes
- [ ] Path finding (A relates to B via C)
- [ ] Goal progress tracking
- [ ] Blocker identification

**Phase 2 Deliverable**: Visual knowledge graph with automatic population from conversations.

---

### Phase 3: Persistent Memory (Weeks 9-12)
**Goal**: AI that truly remembers and learns

#### Week 9: Embedding Engine
- [ ] NLEmbedding integration
- [ ] Embed all nodes
- [ ] Semantic search implementation
- [ ] Similarity scoring

#### Week 10: Memory Retrieval
- [ ] Retrieve relevant memories for context
- [ ] Token budget management
- [ ] Recency vs relevance balancing
- [ ] Memory formatting for LLM

#### Week 11: Pattern Learning
- [ ] UserPattern model and storage
- [ ] Pattern detection algorithms
- [ ] Communication style learning
- [ ] Temporal pattern recognition

#### Week 12: Memory Consolidation
- [ ] Episodic to long-term transition
- [ ] Memory importance scoring
- [ ] Automatic summarization
- [ ] Memory decay/refresh

**Phase 3 Deliverable**: Axel references past conversations naturally and adapts to user patterns.

---

### Phase 4: Proactive Features (Weeks 13-16)
**Goal**: AI that initiates meaningful follow-ups

#### Week 13: Goal Tracking
- [ ] Goal lifecycle management
- [ ] Progress indicators
- [ ] Stale goal detection
- [ ] Action item tracking

#### Week 14: Notification System
- [ ] Follow-up notification scheduling
- [ ] Quiet hours respect
- [ ] Notification content generation
- [ ] Deep link handling

#### Week 15: Daily Insights
- [ ] Daily insight generation
- [ ] Widget for home screen
- [ ] Insight preferences
- [ ] Insight history

#### Week 16: Smart Reminders
- [ ] Context-aware reminder timing
- [ ] Reminder snoozing
- [ ] Adaptive frequency
- [ ] Reminder effectiveness tracking

**Phase 4 Deliverable**: Proactive notifications that help users stay on track with goals.

---

### Phase 5: Polish & Launch (Weeks 17-20)
**Goal**: Production-ready app

#### Week 17: Onboarding
- [ ] Conversational onboarding flow
- [ ] Permission explanations
- [ ] API key setup (BYOK)
- [ ] Tutorial conversation with Axel

#### Week 18: Sync & Backup
- [ ] CloudKit sync implementation
- [ ] Conflict resolution
- [ ] Export functionality
- [ ] Restore from backup

#### Week 19: macOS Companion
- [ ] macOS app target
- [ ] Keyboard shortcuts
- [ ] Menu bar quick capture
- [ ] Desktop widget

#### Week 20: Launch Prep
- [ ] Performance optimization
- [ ] Accessibility audit
- [ ] Privacy policy
- [ ] App Store assets

**Phase 5 Deliverable**: Polished app ready for App Store submission.

---

## 8. Future Considerations

### 8.1 Potential Enhancements

| Feature | Description | Phase |
|---------|-------------|-------|
| **Multi-modal Input** | Photo/document capture and processing | Post-launch |
| **Shared Graphs** | Family/team knowledge sharing | V2 |
| **Calendar Integration** | Deadline sync, scheduling | V2 |
| **Shortcut Actions** | Siri Shortcuts integration | V1.5 |
| **Watch App** | Quick voice capture on Apple Watch | V2 |
| **Focus Modes** | Different Axel personalities per Focus | V1.5 |

### 8.2 Scalability Considerations

| Concern | Current Approach | Future Scaling |
|---------|------------------|----------------|
| **Knowledge Graph Size** | SwiftData (10K+ nodes OK) | SQLite + FTS5 for 100K+ |
| **Embedding Storage** | External storage in SwiftData | Dedicated vector DB |
| **Sync Performance** | CloudKit zones | Custom backend if needed |
| **LLM Costs** | User's API key | Subsidized with subscription |

### 8.3 Research Areas

1. **Local LLM**: On-device conversation for offline/privacy (Apple Intelligence integration)
2. **Voice Cloning**: Custom Axel voice per user preference
3. **Emotion Detection**: Voice sentiment analysis for better responses
4. **Thought Prediction**: Suggest captures based on context/time

---

## 9. Appendix

### A. File Structure

```
MYND/
├── App/
│   ├── MYNDApp.swift
│   ├── ContentView.swift
│   └── AppState.swift
├── Features/
│   ├── Conversation/
│   │   ├── Views/
│   │   │   ├── ConversationView.swift
│   │   │   ├── MessageBubble.swift
│   │   │   └── VoiceInputButton.swift
│   │   └── ViewModels/
│   │       └── ConversationViewModel.swift
│   ├── KnowledgeGraph/
│   │   ├── Views/
│   │   │   ├── GraphView.swift
│   │   │   ├── NodeDetailView.swift
│   │   │   └── NodeListView.swift
│   │   └── ViewModels/
│   │       └── GraphViewModel.swift
│   ├── Memory/
│   │   ├── Views/
│   │   │   └── MemoryExplorerView.swift
│   │   └── ViewModels/
│   │       └── MemoryViewModel.swift
│   └── Settings/
│       ├── Views/
│       │   ├── SettingsView.swift
│       │   └── APIKeySetupView.swift
│       └── ViewModels/
│           └── SettingsViewModel.swift
├── Core/
│   ├── Voice/
│   │   ├── VoiceEngine.swift
│   │   ├── SpeechRecognizer.swift
│   │   └── SpeechSynthesizer.swift
│   ├── AI/
│   │   ├── ConversationEngine.swift
│   │   ├── ClaudeClient.swift
│   │   └── ContextBuilder.swift
│   ├── Memory/
│   │   ├── MemoryEngine.swift
│   │   ├── PatternLearner.swift
│   │   └── LocalEmbeddingEngine.swift
│   ├── Graph/
│   │   ├── KnowledgeGraph.swift
│   │   └── GraphQueryEngine.swift
│   └── Proactive/
│       ├── ProactiveEngine.swift
│       └── GoalTracker.swift
├── Data/
│   ├── Models/
│   │   ├── ThoughtNode.swift
│   │   ├── Edge.swift
│   │   ├── Message.swift
│   │   ├── ConversationSession.swift
│   │   ├── MemoryItem.swift
│   │   └── UserPattern.swift
│   ├── Persistence/
│   │   ├── DataController.swift
│   │   └── CloudKitSync.swift
│   └── Security/
│       └── SecureStorage.swift
├── Shared/
│   ├── Extensions/
│   ├── Utilities/
│   └── Constants.swift
└── Resources/
    ├── Assets.xcassets
    └── Localizable.strings
```

### B. Dependencies

```swift
// Package.swift or Xcode Package Dependencies

dependencies: [
    // None required for MVP - all Apple frameworks

    // Optional for enhanced features:
    // .package(url: "https://github.com/vadymmarkov/Keychain-iOS", from: "1.0.0")
]
```

### C. Minimum Requirements

| Platform | Minimum Version | Reason |
|----------|-----------------|--------|
| iOS | 17.0 | SwiftData, Observation macro |
| macOS | 14.0 | SwiftData, Observation macro |
| Xcode | 15.0 | Swift 5.9, SwiftData |
| Swift | 5.9 | Observation, macros |

### D. API Costs Estimate (Per User/Month)

| Service | Light Use | Heavy Use | Notes |
|---------|-----------|-----------|-------|
| Claude API | $1-2 | $5-10 | ~50 conversations |
| OpenAI Whisper | $0 | $0.50 | If used for STT |
| ElevenLabs | $0 | $5 | Premium voice only |
| CloudKit | $0 | $0 | Apple covers it |
| **Total** | **$1-2** | **$10-15** | BYOK model |

---

## 10. References

- [Apple Human Interface Guidelines](https://developer.apple.com/design/human-interface-guidelines/)
- [SwiftData Documentation](https://developer.apple.com/documentation/swiftdata)
- [Speech Framework](https://developer.apple.com/documentation/speech)
- [Anthropic Claude API](https://docs.anthropic.com/claude/reference)
- [CloudKit Documentation](https://developer.apple.com/documentation/cloudkit)
- [NLEmbedding](https://developer.apple.com/documentation/naturallanguage/nlembedding)

---

**Document Status**: DESIGN COMPLETE
**Next Step**: Begin Phase 1 Week 1 - Project Setup
