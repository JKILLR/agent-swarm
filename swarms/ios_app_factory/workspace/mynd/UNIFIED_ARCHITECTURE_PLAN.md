# MYND: Unified Architecture & Implementation Plan

**Version**: 1.0
**Date**: 2026-01-04
**Status**: DEFINITIVE REFERENCE FOR INCREMENTAL BUILD

---

## 1. Executive Summary

### What is MYND?

MYND is a native iOS app that helps ADHD professionals capture, organize, and explore their thoughts. Unlike traditional note apps that require filing and organization, MYND:

1. **Captures thoughts** via text (primary) or voice input
2. **Visualizes thoughts** as nodes in a 2D mind map
3. **Organizes automatically** using AI to find connections and clusters
4. **Provides an AI companion (Axel)** that helps explore and understand thought patterns

### Target User

ADHD professionals aged 25-45 who:
- Have fleeting thoughts they need to capture quickly
- Struggle with traditional organization systems
- Want to see visual connections between their ideas
- Need a non-judgmental tool that works with their brain, not against it

### Core Value Proposition

| Problem | MYND Solution |
|---------|---------------|
| Thoughts disappear before being captured | Quick text/voice capture with instant feedback |
| Note apps require too many decisions | AI organizes automatically - no folders or tags required |
| Can't see how ideas connect | 2D mind map visualization shows relationships |
| Productivity tools feel judgmental | Axel: warm, non-judgmental AI companion |

### Key Differentiators

1. **2D Mind Map Visualization** - See all thoughts as connected nodes (not 3D - simpler, faster, mobile-friendly)
2. **ADHD-Designed** - Text-first input (validated as better for ADHD than voice-first), instant feedback
3. **Semantic Organization** - AI understands meaning, finds connections automatically
4. **Privacy-First** - Local processing where possible, user controls what goes to cloud

---

## 2. Technical Architecture

### 2.1 High-Level Stack

```
┌─────────────────────────────────────────────────────────────────────┐
│                         iOS CLIENT                                   │
├─────────────────────────────────────────────────────────────────────┤
│  SwiftUI          │  SpriteKit        │  SwiftData    │  WhisperKit │
│  (UI Layer)       │  (2D Visualization)│  (Persistence)│  (Voice)    │
├─────────────────────────────────────────────────────────────────────┤
│                    Local Processing Layer                            │
│  • MobileBERT (embeddings)  • Offline search  • Queue for sync      │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ HTTPS/WebSocket
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         BACKEND (Supabase)                          │
├─────────────────────────────────────────────────────────────────────┤
│  PostgreSQL + pgvector  │  Auth  │  Storage  │  Edge Functions      │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         AI SERVICES                                  │
├─────────────────────────────────────────────────────────────────────┤
│  Claude Haiku (fast)  │  Claude Sonnet (deep analysis)              │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 iOS Technology Choices

| Component | Technology | Rationale |
|-----------|------------|-----------|
| **UI Framework** | SwiftUI | Modern, reactive, Apple's future |
| **2D Visualization** | SpriteKit | Best iOS 2D engine, built-in physics for force-directed layout |
| **Persistence** | SwiftData | Latest Apple persistence, works with iCloud |
| **Voice Input** | WhisperKit | Best Whisper implementation for iOS |
| **Local ML** | CoreML + MobileBERT | On-device embeddings for offline semantic search |

### 2.3 Why SpriteKit for 2D Mind Map?

Based on UI research (ARCHITECTURE_OPTIONS_2D.md):

| Approach | Performance | Complexity | Verdict |
|----------|-------------|------------|---------|
| SwiftUI Canvas | 60fps @ 100 nodes | Low | Too slow for large graphs |
| Core Animation | 60fps @ 300 nodes | High | Good but complex |
| **SpriteKit** | **60fps @ 500+ nodes** | **Medium** | **Best balance** |
| Metal | 60fps @ 1000+ nodes | Very High | Overkill for MVP |

**SpriteKit advantages:**
- Built-in physics for force-directed layout
- 60fps guaranteed up to 500 nodes
- Native gesture handling
- Easy SwiftUI integration via `SpriteView`

---

## 3. Data Model

### 3.1 Core Entities (SwiftData)

```swift
@Model
class Thought {
    @Attribute(.unique) var id: UUID
    var content: String
    var summary: String?           // AI-generated
    var category: String?          // AI-assigned: idea, task, reflection, etc.
    var embedding: [Float]?        // 384-dim vector for semantic search
    var position: CGPoint?         // Position in 2D graph (nil = auto-layout)
    var createdAt: Date
    var updatedAt: Date
    var sourceType: String         // "text" or "voice"
    var isArchived: Bool

    @Relationship var cluster: Cluster?
    @Relationship var connections: [Connection]?
}

@Model
class Cluster {
    @Attribute(.unique) var id: UUID
    var name: String               // AI-generated or user-defined
    var summary: String?           // AI summary of cluster
    var color: String              // Hex color for visualization
    var createdAt: Date

    @Relationship(inverse: \Thought.cluster) var thoughts: [Thought]?
}

@Model
class Connection {
    @Attribute(.unique) var id: UUID
    var sourceThoughtId: UUID
    var targetThoughtId: UUID
    var connectionType: String     // "related", "followUp", "contradicts", "supports"
    var strength: Float            // 0-1, AI confidence
    var isUserCreated: Bool        // true if user manually connected
    var createdAt: Date
}
```

### 3.2 Graph Visualization Model

```swift
// Used by SpriteKit scene
struct GraphNode {
    let thought: Thought
    var position: CGPoint
    var velocity: CGPoint = .zero
    var isSelected: Bool = false
    var isDragging: Bool = false
}

struct GraphEdge {
    let connection: Connection
    let sourceNode: GraphNode
    let targetNode: GraphNode
}
```

---

## 4. 2D Visualization Specification

### 4.1 SpriteKit Graph Scene

```swift
class MindMapScene: SKScene {
    // Nodes
    private var thoughtSprites: [UUID: ThoughtNodeSprite] = [:]
    private var edgeLines: [UUID: SKShapeNode] = [:]

    // Physics settings for force-directed layout
    private let repulsionStrength: CGFloat = 5000
    private let attractionStrength: CGFloat = 0.01
    private let damping: CGFloat = 0.9

    override func didMove(to view: SKView) {
        physicsWorld.gravity = .zero  // Floating nodes
        backgroundColor = .clear
    }

    func addThought(_ thought: Thought) {
        let sprite = ThoughtNodeSprite(thought: thought)
        sprite.position = thought.position ?? randomPosition()

        // Physics body for force simulation
        sprite.physicsBody = SKPhysicsBody(circleOfRadius: sprite.radius)
        sprite.physicsBody?.mass = 1.0
        sprite.physicsBody?.linearDamping = 0.8
        sprite.physicsBody?.allowsRotation = false

        addChild(sprite)
        thoughtSprites[thought.id] = sprite
    }

    // Force-directed layout runs in update() loop
    override func update(_ currentTime: TimeInterval) {
        applyRepulsionForces()   // Nodes push apart
        applyAttractionForces()  // Connected nodes pull together
        applyCenteringForce()    // Keeps graph centered
        updateEdgePositions()    // Redraw connection lines
    }
}
```

### 4.2 Node Visual Design

```
┌──────────────────────────────────────────────────┐
│                   THOUGHT NODE                    │
├──────────────────────────────────────────────────┤
│                                                   │
│    ┌─────────────────────────────────────────┐   │
│    │  ● "Meeting with Sarah about project"   │   │
│    │                                         │   │
│    │  [idea] ─────── Category badge          │   │
│    └─────────────────────────────────────────┘   │
│                                                   │
│    Size: Based on content length (50-150pt)      │
│    Color: Based on cluster membership            │
│    Border: Highlighted when selected             │
│    Shadow: Subtle for depth                      │
│                                                   │
└──────────────────────────────────────────────────┘
```

### 4.3 Gesture Handling

| Gesture | Action |
|---------|--------|
| **Tap node** | Select node, show details panel |
| **Tap canvas** | Deselect, optionally create new node |
| **Drag node** | Move node (locks position from auto-layout) |
| **Pinch** | Zoom in/out (0.5x to 3x) |
| **Two-finger pan** | Pan entire canvas |
| **Long-press node** | Show context menu (edit, delete, connect) |
| **Drag from node edge** | Create connection to another node |

---

## 5. AI Pipeline

### 5.1 Multi-Model Strategy

| Model | Use Case | Latency Target | Cost |
|-------|----------|----------------|------|
| **Claude Haiku** | Acknowledgments, quick categorization | <500ms | ~$0.00025/interaction |
| **Claude Sonnet** | Deep analysis, connections, insights | 1-3s | ~$0.003/interaction |
| **Local MobileBERT** | Embeddings for semantic search | <50ms | Free |

### 5.2 Request Routing

```swift
func routeRequest(_ request: AIRequest) -> Model {
    switch request.type {
    case .acknowledgment, .quickCategory:
        return .claudeHaiku
    case .deepAnalysis, .findConnections, .generateInsight:
        return .claudeSonnet
    case .embedding, .localSearch:
        return .localMobileBERT
    }
}
```

### 5.3 Axel Personality

**Core traits:**
- Warm but not effusive
- Non-judgmental (CRITICAL for ADHD users)
- Brief responses (1-2 sentences for acknowledgments)
- References user's actual thoughts in responses

**Personality toggle** (in settings):
- "Warm & Supportive" (default)
- "Direct & Efficient"
- "Balanced"

---

## 6. Incremental Implementation Phases

### Design Principles

1. **Each phase produces working software** - Never have broken builds
2. **Phase N is usable before Phase N+1** - Users can benefit at each stage
3. **Later phases enhance, don't replace** - Build on what works
4. **Risk checkpoints at each phase** - Know when to pivot

---

### Phase 0: Foundation (Weeks 1-2)

**Goal**: Basic thought capture working, static 2D canvas, local persistence

#### Deliverables
- [ ] Xcode project with proper structure
- [ ] SwiftData models (Thought, Cluster, Connection)
- [ ] Main UI: text input field + list of thoughts
- [ ] Basic SpriteKit canvas showing thoughts as circles
- [ ] Drag to move nodes (no physics yet)
- [ ] Local persistence (thoughts survive app restart)

#### Files to Create
```
MYND/
├── App/
│   ├── MYNDApp.swift
│   └── ContentView.swift
├── Models/
│   ├── Thought.swift
│   ├── Cluster.swift
│   └── Connection.swift
├── Views/
│   ├── ThoughtInputView.swift
│   ├── ThoughtListView.swift
│   └── MindMapView.swift
├── Graph/
│   ├── MindMapScene.swift
│   └── ThoughtNodeSprite.swift
└── Services/
    └── DataService.swift
```

#### Test Criteria
- [ ] Can type a thought and submit
- [ ] Thought appears in list AND as node on canvas
- [ ] Can drag node to new position
- [ ] Kill app, reopen - thoughts are still there

#### Risk Checkpoint
- If SpriteKit performance is poor with 20 nodes → investigate SwiftUI Canvas
- If SwiftData has issues → fallback to Core Data

---

### Phase 1: Core MVP (Weeks 3-6)

**Goal**: Force-directed layout, basic AI integration, polished capture experience

#### Deliverables
- [ ] Force-directed graph layout (nodes auto-arrange)
- [ ] Visual edges between connected thoughts
- [ ] Claude Haiku integration for acknowledgments
- [ ] Immediate visual feedback on thought capture
- [ ] Zoom and pan gestures
- [ ] Node selection and detail panel

#### New Files
```
Services/
├── AIService.swift          # Claude API integration
├── ForceLayoutEngine.swift  # Physics simulation
Views/
├── ThoughtDetailSheet.swift # Selected node details
Graph/
└── EdgeNode.swift           # Connection line rendering
```

#### Test Criteria
- [ ] 50 nodes render at 60fps
- [ ] Nodes spread out naturally, don't overlap
- [ ] Connected nodes stay closer together
- [ ] AI acknowledges each thought within 1 second
- [ ] Pinch to zoom works smoothly

#### Risk Checkpoint
- If Claude API latency >3s consistently → implement local fallback messages
- If force layout is jittery → reduce simulation steps, add damping

---

### Phase 2: AI Organization (Weeks 7-10)

**Goal**: AI understands thoughts, creates clusters, finds connections

#### Deliverables
- [ ] Auto-categorization (idea, task, reflection, etc.)
- [ ] Semantic embeddings (local MobileBERT)
- [ ] AI-suggested connections between thoughts
- [ ] Auto-clustering of related thoughts
- [ ] Visual cluster grouping (background colors)
- [ ] Basic search (text + semantic)

#### New Files
```
Services/
├── EmbeddingService.swift   # MobileBERT integration
├── ClusteringService.swift  # AI clustering logic
├── SearchService.swift      # Hybrid search
Views/
├── SearchView.swift
└── ClusterView.swift
```

#### Test Criteria
- [ ] 80%+ categorization accuracy (manual review of 50 thoughts)
- [ ] Related thoughts cluster visually
- [ ] Search finds semantically related thoughts (not just keyword match)
- [ ] Can tap cluster to see all thoughts in it

#### Risk Checkpoint
- If AI clustering <60% accurate → simplify to user-created tags
- If MobileBERT too slow → server-side embeddings only

---

### Phase 3: Voice & Polish (Weeks 11-14)

**Goal**: Voice capture, visual refinements, delightful UX

#### Deliverables
- [ ] Voice input with WhisperKit transcription
- [ ] Live transcription display during recording
- [ ] Edit transcription before sending
- [ ] Manual edge creation (connect any two nodes)
- [ ] Animations and transitions
- [ ] Onboarding flow
- [ ] Settings screen

#### New Files
```
Services/
└── VoiceService.swift       # WhisperKit integration
Views/
├── VoiceInputView.swift
├── OnboardingView.swift
└── SettingsView.swift
```

#### Test Criteria
- [ ] Voice transcription >90% accuracy
- [ ] Transcription appears in <500ms after speech ends
- [ ] Smooth animations throughout app
- [ ] New users understand app in <2 minutes

#### Risk Checkpoint
- If voice feels awkward → deprioritize, keep text as primary
- If onboarding confuses users → simplify drastically

---

### Phase 4: Cloud & Advanced (Weeks 15-20)

**Goal**: Full Axel personality, cloud sync, monetization

#### Deliverables
- [ ] Supabase backend integration
- [ ] User authentication
- [ ] Cloud sync across devices
- [ ] Full Axel conversational AI
- [ ] Deep analysis and insights
- [ ] StoreKit subscription tiers
- [ ] Privacy controls

#### New Files
```
Services/
├── AuthService.swift
├── SyncService.swift
├── SubscriptionService.swift
└── AxelConversationService.swift
Views/
├── LoginView.swift
├── SubscriptionView.swift
└── PrivacySettingsView.swift
```

#### Test Criteria
- [ ] Sign up, sign in flows work
- [ ] Thoughts sync between iPhone and iPad
- [ ] Axel provides meaningful insights about patterns
- [ ] Can upgrade to paid tier via in-app purchase

#### Risk Checkpoint
- If sync conflicts common → implement CRDT or simplify to manual sync
- If conversion <1% → extend trial, consider freemium pivot

---

## 7. Key Technical Decisions

### Decision 1: SpriteKit over Metal for MVP

**Rationale**:
- SpriteKit handles 500+ nodes at 60fps - sufficient for MVP
- Built-in physics saves 2+ weeks of development
- Metal is overkill and adds significant complexity
- Can upgrade to Metal in v2 if needed

### Decision 2: Text-First over Voice-First

**Rationale** (from ASSUMPTION_VALIDATION.md):
- ADHD users have "delay aversion" - they hate waiting
- Voice has latency (transcription + AI response)
- Text provides instant feedback
- Voice remains available but not primary

### Decision 3: SwiftData over Core Data

**Rationale**:
- SwiftData is Apple's modern solution
- Better SwiftUI integration
- Simpler API, less boilerplate
- iCloud sync built-in

### Decision 4: Supabase over Custom Backend

**Rationale**:
- PostgreSQL + Auth + Storage in one platform
- pgvector for semantic search
- Row Level Security for user isolation
- Edge Functions for custom logic
- Faster time to market than custom backend

### Decision 5: Claude over OpenAI

**Rationale**:
- Better personality/character consistency
- Multi-model strategy (Haiku for speed, Sonnet for depth)
- Better at warm, non-judgmental responses
- Lower hallucination rate

---

## 8. Kill Gates & Success Metrics

### Kill Gates (When to Stop/Pivot)

| Phase | Gate | Pivot Action |
|-------|------|--------------|
| **Phase 1** | <30fps with 50 nodes | Try SwiftUI Canvas or simplify to list view |
| **Phase 1** | Claude latency consistently >5s | Add aggressive caching, local acknowledgments |
| **Phase 2** | AI clustering <50% accurate | Remove auto-clustering, use manual tags |
| **Phase 3** | Voice feels awkward to 70%+ testers | Deprioritize voice, focus on text |
| **Phase 4** | <0.5% conversion after 30 days | Pivot to B2B or ad-supported model |

### Success Metrics by Phase

| Phase | Metric | Target |
|-------|--------|--------|
| **Phase 0** | App compiles and runs | 100% |
| **Phase 1** | 50 nodes @ 60fps | Pass |
| **Phase 1** | AI response time | <3 seconds |
| **Phase 2** | Categorization accuracy | >70% |
| **Phase 2** | Semantic search relevance | >80% top-5 |
| **Phase 3** | Voice transcription accuracy | >90% |
| **Phase 4** | D7 retention (beta) | >25% |
| **Phase 4** | Trial to paid conversion | >2% |

---

## 9. File Structure Reference

```
MYND/
├── App/
│   ├── MYNDApp.swift              # App entry point
│   └── ContentView.swift          # Root view
│
├── Models/                        # SwiftData models
│   ├── Thought.swift
│   ├── Cluster.swift
│   ├── Connection.swift
│   └── UserSettings.swift
│
├── Views/                         # SwiftUI views
│   ├── Input/
│   │   ├── ThoughtInputView.swift
│   │   └── VoiceInputView.swift
│   ├── MindMap/
│   │   ├── MindMapView.swift      # SwiftUI wrapper
│   │   └── MindMapControlsView.swift
│   ├── Detail/
│   │   ├── ThoughtDetailSheet.swift
│   │   └── ClusterDetailSheet.swift
│   ├── Search/
│   │   └── SearchView.swift
│   ├── Settings/
│   │   ├── SettingsView.swift
│   │   └── PrivacySettingsView.swift
│   └── Onboarding/
│       └── OnboardingView.swift
│
├── Graph/                         # SpriteKit visualization
│   ├── MindMapScene.swift         # Main SpriteKit scene
│   ├── ThoughtNodeSprite.swift    # Node rendering
│   ├── EdgeNode.swift             # Connection lines
│   └── ForceLayoutEngine.swift    # Physics simulation
│
├── Services/                      # Business logic
│   ├── DataService.swift          # SwiftData operations
│   ├── AIService.swift            # Claude integration
│   ├── EmbeddingService.swift     # MobileBERT
│   ├── SearchService.swift        # Hybrid search
│   ├── ClusteringService.swift    # AI clustering
│   ├── VoiceService.swift         # WhisperKit
│   ├── SyncService.swift          # Supabase sync
│   └── AuthService.swift          # Authentication
│
├── Utilities/
│   ├── Extensions/
│   └── Helpers/
│
└── Resources/
    ├── Assets.xcassets
    └── ML Models/
        └── MobileBERT.mlmodel
```

---

## 10. Quick Reference Commands

### Create Xcode Project
```bash
# Create new iOS App project named "MYND"
# SwiftUI lifecycle, SwiftData enabled
```

### Add Dependencies (Package.swift or SPM)
```swift
dependencies: [
    .package(url: "https://github.com/argmaxinc/WhisperKit", from: "0.6.0"),
    .package(url: "https://github.com/supabase-community/supabase-swift", from: "2.0.0")
]
```

### Run Tests
```bash
xcodebuild test -scheme MYND -destination 'platform=iOS Simulator,name=iPhone 15'
```

---

## 11. Document Cross-References

| Document | Purpose | Location |
|----------|---------|----------|
| **ARCHITECTURE_V3.md** | Full technical architecture | workspace/mynd/ |
| **UI_VISUALIZATION_RESEARCH.md** | 2D visualization research | workspace/research/ |
| **ARCHITECTURE_OPTIONS_2D.md** | SpriteKit vs alternatives | workspace/research/ |
| **ASSUMPTION_VALIDATION.md** | Research on ADHD assumptions | workspace/mynd/ |
| **CRITIQUE_V3.md** | Risk analysis | workspace/mynd/ |
| **POSITIONING_STRATEGY.md** | Marketing positioning | workspace/mynd/ |
| **REMEDIATION_COMPLETE.md** | Risk mitigations | workspace/mynd/ |

---

*This is THE definitive reference for building MYND. All implementation should follow this plan.*

*Last updated: 2026-01-04*
