# Architecture Plan Code Review

**Document Reviewed**: UNIFIED_ARCHITECTURE_PLAN.md
**Reviewer**: iOS Architecture Agent
**Date**: 2026-01-04
**Severity Levels**: 🔴 Critical | 🟠 Major | 🟡 Minor | 🟢 Suggestion

---

## Executive Summary

The architecture plan is well-structured and demonstrates solid understanding of iOS development. However, there are **significant issues** in the Swift code samples that would cause compilation failures and runtime crashes. The SwiftData models have design flaws that will cause problems at scale. These must be addressed before implementation begins.

**Overall Assessment**: Good foundation, but code samples need substantial revision.

---

## 1. Swift Code Issues

### 🔴 CRITICAL: SwiftData Model Errors

#### Issue 1.1: CGPoint is not Codable by default

```swift
// ❌ BROKEN - Line 118
var position: CGPoint?  // CGPoint is NOT Codable in SwiftData
```

**Problem**: SwiftData requires all stored properties to be `Codable`. `CGPoint` is NOT Codable by default and will cause a crash or compilation error.

**Fix**:
```swift
// ✅ CORRECT - Store as separate components or use a wrapper
var positionX: Double?
var positionY: Double?

// Computed property for convenience
var position: CGPoint? {
    get {
        guard let x = positionX, let y = positionY else { return nil }
        return CGPoint(x: x, y: y)
    }
    set {
        positionX = newValue?.x
        positionY = newValue?.y
    }
}
```

Or create a `@Codable` wrapper struct stored as `Data`.

---

#### Issue 1.2: [Float] Array Storage in SwiftData

```swift
// ⚠️ PROBLEMATIC - Line 117
var embedding: [Float]?  // 384-dim vector
```

**Problem**: SwiftData can handle arrays, but a 384-element `[Float]` array for every thought is:
1. Memory-intensive when loaded
2. Not indexed/searchable by SwiftData
3. Will cause performance issues with large datasets

**Recommendation**:
```swift
// ✅ BETTER - Store as Data, decode lazily
var embeddingData: Data?

var embedding: [Float]? {
    get {
        guard let data = embeddingData else { return nil }
        return data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }
    set {
        embeddingData = newValue.flatMap { Data(buffer: UnsafeBufferPointer(start: $0, count: $0.count)) }
    }
}
```

---

#### Issue 1.3: Relationship Declaration Missing `deleteRule`

```swift
// ❌ INCOMPLETE - Line 125
@Relationship var cluster: Cluster?
@Relationship var connections: [Connection]?
```

**Problem**: No `deleteRule` specified. When a `Cluster` is deleted, what happens to its `Thought` objects? This can cause orphaned data or crashes.

**Fix**:
```swift
// ✅ CORRECT
@Relationship(deleteRule: .nullify) var cluster: Cluster?
@Relationship(deleteRule: .cascade) var connections: [Connection]?
```

---

#### Issue 1.4: Connection Model Design Flaw

```swift
// ❌ POOR DESIGN - Lines 143-144
var sourceThoughtId: UUID
var targetThoughtId: UUID
```

**Problem**: Storing `UUID`s instead of using `@Relationship` breaks SwiftData's graph capabilities and requires manual lookups. This is an anti-pattern.

**Fix**:
```swift
// ✅ CORRECT - Use proper relationships
@Model
class Connection {
    @Attribute(.unique) var id: UUID

    @Relationship var sourceThought: Thought
    @Relationship var targetThought: Thought

    var connectionType: ConnectionType  // Use enum, not String
    var strength: Float
    var isUserCreated: Bool
    var createdAt: Date

    init(source: Thought, target: Thought, type: ConnectionType, strength: Float = 0.5, isUserCreated: Bool = false) {
        self.id = UUID()
        self.sourceThought = source
        self.targetThought = target
        self.connectionType = type
        self.strength = strength
        self.isUserCreated = isUserCreated
        self.createdAt = Date()
    }
}

enum ConnectionType: String, Codable {
    case related, followUp, contradicts, supports
}
```

---

### 🟠 MAJOR: SpriteKit Scene Issues

#### Issue 1.5: Missing Required Initializers

```swift
// ❌ INCOMPLETE - Line 177
class MindMapScene: SKScene {
    // No init(size:) or required init(coder:)
}
```

**Problem**: `SKScene` requires initializers. The code sample will not compile.

**Fix**:
```swift
// ✅ CORRECT
class MindMapScene: SKScene {
    private var thoughtSprites: [UUID: ThoughtNodeSprite] = [:]
    private var edgeLines: [UUID: SKShapeNode] = [:]

    override init(size: CGSize) {
        super.init(size: size)
    }

    required init?(coder aDecoder: NSCoder) {
        super.init(coder: aDecoder)
    }

    override func didMove(to view: SKView) {
        physicsWorld.gravity = .zero
        backgroundColor = .clear
    }
}
```

---

#### Issue 1.6: Force Layout Mixing with SKPhysicsBody

```swift
// ⚠️ CONFLICTING - Lines 197-200 and 207-210
sprite.physicsBody = SKPhysicsBody(...)  // Uses SpriteKit physics

override func update(_ currentTime: TimeInterval) {
    applyRepulsionForces()   // Manual force application
    applyAttractionForces()
}
```

**Problem**: The code defines `SKPhysicsBody` but then also calls manual force methods in `update()`. This creates two competing physics systems. You should choose one approach:

1. **Pure SpriteKit Physics**: Use `SKPhysicsBody`, `SKFieldNode` for repulsion/attraction
2. **Manual Force-Directed**: No `SKPhysicsBody`, update positions directly

**Recommendation**: For a force-directed graph, manual control is better. SpriteKit physics is designed for collision simulation, not graph layouts.

```swift
// ✅ CORRECT - Manual approach (recommended)
class MindMapScene: SKScene {
    override func update(_ currentTime: TimeInterval) {
        let dt = currentTime - lastUpdateTime
        lastUpdateTime = currentTime

        // Apply forces manually
        for (id, node) in thoughtSprites {
            var force = CGVector.zero
            force += calculateRepulsionForce(for: node)
            force += calculateAttractionForce(for: node)
            force += calculateCenteringForce(for: node)

            node.velocity = node.velocity * damping + force * CGFloat(dt)
            node.position = node.position + node.velocity * CGFloat(dt)
        }
        updateEdgePositions()
    }
}
```

---

#### Issue 1.7: ThoughtNodeSprite Not Defined

```swift
// ❌ UNDEFINED - Line 193
let sprite = ThoughtNodeSprite(thought: thought)
```

**Problem**: `ThoughtNodeSprite` is referenced but never defined. The file structure mentions it but provides no implementation guidance.

**Minimum Required Definition**:
```swift
class ThoughtNodeSprite: SKNode {
    let thoughtId: UUID
    let radius: CGFloat
    var velocity: CGVector = .zero

    private let circleNode: SKShapeNode
    private let labelNode: SKLabelNode

    init(thought: Thought) {
        self.thoughtId = thought.id
        self.radius = Self.calculateRadius(for: thought.content)

        self.circleNode = SKShapeNode(circleOfRadius: radius)
        self.labelNode = SKLabelNode(text: thought.summary ?? String(thought.content.prefix(30)))

        super.init()

        circleNode.fillColor = .systemBlue
        circleNode.strokeColor = .white
        addChild(circleNode)

        labelNode.fontSize = 14
        labelNode.verticalAlignmentMode = .center
        addChild(labelNode)
    }

    required init?(coder aDecoder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    private static func calculateRadius(for content: String) -> CGFloat {
        return CGFloat(min(max(content.count, 50), 150)) / 2
    }
}
```

---

### 🟠 MAJOR: AI Service Code Issues

#### Issue 1.8: Incomplete Enum/Model Definition

```swift
// ❌ INCOMPLETE - Lines 264-273
func routeRequest(_ request: AIRequest) -> Model {
    switch request.type {
    case .acknowledgment, .quickCategory:
        return .claudeHaiku
    // ...
    }
}
```

**Problem**: `AIRequest`, `Model`, and the case types are undefined. This is pseudocode, not Swift.

**Fix**:
```swift
// ✅ CORRECT - Provide actual types
enum AIModel {
    case claudeHaiku
    case claudeSonnet
    case localMobileBERT

    var modelId: String {
        switch self {
        case .claudeHaiku: return "claude-3-haiku-20240307"
        case .claudeSonnet: return "claude-3-sonnet-20240229"
        case .localMobileBERT: return "local"
        }
    }
}

enum AIRequestType {
    case acknowledgment
    case quickCategory
    case deepAnalysis
    case findConnections
    case generateInsight
    case embedding
    case localSearch
}

struct AIRequest {
    let type: AIRequestType
    let content: String
    let context: [Thought]?
}

func routeRequest(_ request: AIRequest) -> AIModel {
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

---

### 🟡 MINOR: GraphNode/GraphEdge Issues

#### Issue 1.9: Struct Contains Reference to Class

```swift
// ⚠️ PROBLEMATIC - Lines 154-168
struct GraphNode {
    let thought: Thought  // Thought is a @Model class
    // ...
}
```

**Problem**: `GraphNode` is a `struct` but contains a reference to a `@Model` class (`Thought`). This creates confusing semantics around value/reference types and can cause unexpected behavior with SwiftData's observation.

**Recommendation**: Either:
1. Make `GraphNode` a class
2. Store only `thought.id` and look up the `Thought` when needed
3. Keep `Thought` reference but document the implications

```swift
// ✅ OPTION 1 - ID-based (preferred for separation of concerns)
struct GraphNode: Identifiable {
    let id: UUID  // The Thought's ID
    var position: CGPoint
    var velocity: CGVector = .zero
    var isSelected: Bool = false
    var isDragging: Bool = false
}

// ✅ OPTION 2 - Class-based
class GraphNode: Identifiable {
    let thought: Thought
    var position: CGPoint
    // ...
}
```

---

## 2. SwiftData Model Design Issues

### 🟠 MAJOR: String-Based Type Storage

#### Issue 2.1: Using Strings for Enums

```swift
// ❌ BAD PRACTICE
var category: String?          // "idea", "task", "reflection", etc.
var sourceType: String         // "text" or "voice"
var connectionType: String     // "related", "followUp", etc.
```

**Problem**: String-based types are:
1. Not type-safe (typos compile but fail at runtime)
2. Harder to refactor
3. Can't leverage Swift's exhaustive switch checking

**Fix**:
```swift
// ✅ CORRECT - Use proper enums
enum ThoughtCategory: String, Codable, CaseIterable {
    case idea
    case task
    case reflection
    case question
    case memory
    case goal
}

enum SourceType: String, Codable {
    case text
    case voice
}

@Model
class Thought {
    var category: ThoughtCategory?
    var sourceType: SourceType = .text
    // ...
}
```

---

### 🟡 MINOR: Missing Default Values

#### Issue 2.2: No Initializers with Defaults

```swift
// ❌ INCOMPLETE
@Model
class Thought {
    @Attribute(.unique) var id: UUID
    var content: String
    var createdAt: Date
    // No init provided
}
```

**Problem**: SwiftData models should have explicit initializers with sensible defaults.

**Fix**:
```swift
// ✅ CORRECT
@Model
class Thought {
    @Attribute(.unique) var id: UUID
    var content: String
    var summary: String?
    var category: ThoughtCategory?
    var embeddingData: Data?
    var positionX: Double?
    var positionY: Double?
    var createdAt: Date
    var updatedAt: Date
    var sourceType: SourceType
    var isArchived: Bool

    @Relationship(deleteRule: .nullify) var cluster: Cluster?
    @Relationship(deleteRule: .cascade, inverse: \Connection.sourceThought) var outgoingConnections: [Connection]
    @Relationship(deleteRule: .cascade, inverse: \Connection.targetThought) var incomingConnections: [Connection]

    init(content: String, sourceType: SourceType = .text) {
        self.id = UUID()
        self.content = content
        self.sourceType = sourceType
        self.createdAt = Date()
        self.updatedAt = Date()
        self.isArchived = false
        self.outgoingConnections = []
        self.incomingConnections = []
    }
}
```

---

### 🟠 MAJOR: Bidirectional Relationship Issue

#### Issue 2.3: Connection Relationships Are Unidirectional

```swift
// Current design
class Connection {
    var sourceThoughtId: UUID
    var targetThoughtId: UUID
}

class Thought {
    @Relationship var connections: [Connection]?  // Which connections? Source? Target? Both?
}
```

**Problem**: The relationship is ambiguous. A thought can be the source OR target of a connection. The current model doesn't distinguish these.

**Fix**:
```swift
// ✅ CORRECT - Explicit bidirectional relationships
@Model
class Thought {
    // ...

    @Relationship(deleteRule: .cascade, inverse: \Connection.sourceThought)
    var outgoingConnections: [Connection] = []

    @Relationship(deleteRule: .cascade, inverse: \Connection.targetThought)
    var incomingConnections: [Connection] = []

    // Convenience property
    var allConnections: [Connection] {
        outgoingConnections + incomingConnections
    }
}

@Model
class Connection {
    @Attribute(.unique) var id: UUID

    @Relationship var sourceThought: Thought
    @Relationship var targetThought: Thought

    // ...
}
```

---

## 3. File Structure Issues

### 🟢 SUGGESTION: Missing Key Files

The file structure is good but missing some important files:

```
MYND/
├── Services/
│   ├── DataService.swift          ✅
│   ├── AIService.swift            ✅
│   └── NetworkService.swift       ❌ MISSING - Needed for API calls
│
├── ViewModels/                    ❌ MISSING ENTIRE FOLDER
│   ├── MindMapViewModel.swift     - Manages graph state
│   ├── ThoughtInputViewModel.swift - Handles input logic
│   └── SearchViewModel.swift      - Search state management
│
├── Graph/
│   ├── MindMapScene.swift         ✅
│   └── GraphLayoutEngine.swift    ❌ Should be in Graph/, not Services/
│
├── Configuration/                  ❌ MISSING
│   ├── AppConfig.swift            - API keys, feature flags
│   └── Constants.swift            - App-wide constants
│
└── Protocols/                      ❌ MISSING
    ├── AIServiceProtocol.swift    - For testing/mocking
    └── DataServiceProtocol.swift
```

---

### 🟡 MINOR: Inconsistent Naming

- `ForceLayoutEngine.swift` in `Services/` but it's visualization logic
- `ThoughtNodeSprite.swift` but `EdgeNode.swift` (inconsistent: `*Node` vs `*Sprite`)

**Recommendation**:
- Rename `EdgeNode.swift` → `EdgeSprite.swift` for consistency
- Move `ForceLayoutEngine.swift` to `Graph/`

---

## 4. Implementation Pitfalls

### 🔴 CRITICAL: SpriteKit + SwiftUI Integration

The plan mentions using `SpriteView` but doesn't address major integration challenges:

1. **State Synchronization**: SwiftData changes need to update SpriteKit nodes
2. **Gesture Conflicts**: SpriteKit and SwiftUI gesture recognizers can conflict
3. **Performance**: Re-rendering SpriteView on SwiftUI state changes can be expensive

**Recommendation**: Use `UIViewRepresentable` wrapping an `SKView` for better control:

```swift
struct MindMapView: UIViewRepresentable {
    @Bindable var viewModel: MindMapViewModel

    func makeUIView(context: Context) -> SKView {
        let skView = SKView()
        skView.presentScene(MindMapScene(size: skView.bounds.size))
        skView.ignoresSiblingOrder = true
        return skView
    }

    func updateUIView(_ skView: SKView, context: Context) {
        guard let scene = skView.scene as? MindMapScene else { return }
        scene.updateThoughts(viewModel.thoughts)
    }
}
```

---

### 🟠 MAJOR: MobileBERT Performance Concern

The plan states local embeddings in <50ms. MobileBERT on iPhone 12 or newer can achieve this, but:

1. **First load latency**: Model loading takes 2-5 seconds
2. **Memory usage**: ~100MB when loaded
3. **Background processing**: CoreML can be deprioritized by iOS

**Recommendation**:
- Load model at app launch (splash screen time)
- Keep model in memory (don't reload)
- Add fallback for older devices
- Consider quantized models for smaller memory footprint

---

### 🟠 MAJOR: Force-Directed Layout CPU Usage

Running physics simulation every frame (`update()`) will:
1. Drain battery quickly
2. Heat up the device
3. Be unnecessary when graph is stable

**Recommendation**: Implement simulation cooldown:

```swift
class MindMapScene: SKScene {
    private var totalKineticEnergy: CGFloat = 0
    private var isSimulating: Bool = true
    private let energyThreshold: CGFloat = 0.01

    override func update(_ currentTime: TimeInterval) {
        guard isSimulating else { return }

        applyForces()
        totalKineticEnergy = calculateTotalKineticEnergy()

        if totalKineticEnergy < energyThreshold {
            isSimulating = false
            // Wake up on new node added or node dragged
        }
    }

    func wakeSimulation() {
        isSimulating = true
    }
}
```

---

### 🟡 MINOR: Missing Error Handling

No error handling patterns defined for:
- Network failures
- AI API rate limits
- SwiftData save failures
- Voice transcription failures

**Recommendation**: Define a consistent error handling strategy:

```swift
enum MYNDError: LocalizedError {
    case networkUnavailable
    case aiServiceError(underlying: Error)
    case dataCorruption
    case transcriptionFailed

    var errorDescription: String? {
        switch self {
        case .networkUnavailable:
            return "No internet connection. Your thoughts are saved locally."
        // ...
        }
    }
}
```

---

### 🟡 MINOR: Cluster Color as String

```swift
// ❌
var color: String  // Hex color for visualization
```

**Problem**: Hex string parsing is error-prone and verbose.

**Fix**:
```swift
// ✅ Store as codable color components
struct CodableColor: Codable {
    var red: Double
    var green: Double
    var blue: Double
    var alpha: Double = 1.0

    var color: Color {
        Color(red: red, green: green, blue: blue, opacity: alpha)
    }
}

// Or store as Data using UIColor archiving
var colorData: Data?
```

---

## 5. Architecture Suggestions

### 🟢 Consider MVVM Pattern

The document mentions Services but no ViewModels. For SwiftUI, MVVM is standard:

```swift
@Observable
class MindMapViewModel {
    private let dataService: DataServiceProtocol
    private let aiService: AIServiceProtocol

    var thoughts: [Thought] = []
    var selectedThought: Thought?
    var isLoading: Bool = false

    func addThought(_ content: String) async {
        isLoading = true
        defer { isLoading = false }

        let thought = Thought(content: content)
        dataService.save(thought)

        // Get AI acknowledgment
        if let response = await aiService.acknowledge(thought) {
            // Handle response
        }
    }
}
```

---

### 🟢 Consider Protocol-Based Services

For testability:

```swift
protocol DataServiceProtocol {
    func fetchAllThoughts() -> [Thought]
    func save(_ thought: Thought) throws
    func delete(_ thought: Thought) throws
}

// Real implementation
class DataService: DataServiceProtocol { /* ... */ }

// Mock for testing
class MockDataService: DataServiceProtocol { /* ... */ }
```

---

## Summary of Required Changes

### Before Implementation Begins:

| Priority | Issue | Action Required |
|----------|-------|-----------------|
| 🔴 Critical | CGPoint storage | Use positionX/positionY components |
| 🔴 Critical | Connection model | Use @Relationship, not UUID storage |
| 🔴 Critical | Missing initializers | Add required inits to all classes |
| 🟠 Major | String enums | Convert to proper Swift enums |
| 🟠 Major | Physics conflict | Choose manual OR SpriteKit physics |
| 🟠 Major | Bidirectional relationships | Add outgoing/incoming connection arrays |
| 🟠 Major | SpriteKit integration | Use UIViewRepresentable, not SpriteView |
| 🟠 Major | Force simulation CPU | Add simulation cooldown |
| 🟡 Minor | Missing ViewModels | Add MVVM layer |
| 🟡 Minor | File structure | Add Protocols/, Configuration/ folders |
| 🟢 Suggestion | Error handling | Define error types and handling patterns |

---

## Recommended Next Steps

1. **Revise SwiftData models** with corrected relationships and types
2. **Create complete class definitions** for SpriteKit components
3. **Define all enums and protocols** before implementation
4. **Add ViewModel layer** to file structure
5. **Document error handling strategy**

---

*Review completed: 2026-01-04*
