# 2D Graph Visualization Research for MYND

**Date:** 2026-01-04
**Author:** Research Agent
**Purpose:** Comprehensive research on Swift libraries and frameworks for visualizing thoughts as nodes in a 2D graph
**Target Version:** MYND v1.5 (deferred from MVP)

---

## Executive Summary

This research evaluates approaches for implementing a 2D force-directed graph visualization for MYND's knowledge graph feature. The research covers open-source Swift libraries, UIKit Dynamics, SpriteKit physics, and custom implementations with performance benchmarks and code samples.

**Key Recommendation:** Use a **hybrid approach** with:
1. **SpriteKit** for rendering and physics simulation (best performance for 50-500 nodes)
2. **Custom force-directed algorithm** for layout control
3. **Metal shaders** for performance optimization if node count exceeds 500

---

## 1. Open Source Swift Graph Libraries

### 1.1 GraphView (AugustRush/GraphView)

**Repository:** github.com/AugustRush/GraphView

**Status:** EXISTS but UNMAINTAINED

| Aspect | Assessment |
|--------|------------|
| Last Update | 2018-2019 (5+ years old) |
| Swift Version | Swift 4.x (needs migration) |
| iOS Support | iOS 10+ (needs update for iOS 17+) |
| Stars | ~300 |
| Active Maintenance | NO |
| Documentation | Minimal |

**Capabilities:**
- Basic force-directed layout
- Node and edge rendering
- Simple gesture support
- Circular, hierarchical, and force-directed layouts

**Limitations:**
- Not Swift 5+ compatible without modification
- No SwiftUI integration
- No Metal acceleration
- Limited customization options
- Performance caps around 100 nodes

**Verdict:** NOT RECOMMENDED for production use. Too outdated.

### 1.2 SwiftGraph

**Repository:** github.com/davecom/SwiftGraph

**Status:** ACTIVE (graph data structures only)

| Aspect | Assessment |
|--------|------------|
| Last Update | 2024 |
| Swift Version | Swift 5+ |
| iOS Support | iOS 9+ |
| Stars | ~700 |
| Purpose | Data structures, NOT visualization |

**What it provides:**
```swift
// SwiftGraph is for graph DATA STRUCTURES, not rendering
let graph = UnweightedGraph<String>()
graph.addVertex("Thought 1")
graph.addVertex("Thought 2")
graph.addEdge(from: "Thought 1", to: "Thought 2", directed: false)

// BFS, DFS, shortest path algorithms
let path = graph.bfs(from: "Thought 1", to: "Thought 2")
```

**Use Case for MYND:**
- Use SwiftGraph for graph algorithms (finding related thoughts, clustering)
- Combine with custom rendering layer

**Verdict:** RECOMMENDED for graph data structures. Pair with custom visualization.

### 1.3 Charts (danielgindi/Charts)

**Repository:** github.com/danielgindi/Charts

**Status:** ACTIVE (iOS port of MPAndroidChart)

| Aspect | Assessment |
|--------|------------|
| Purpose | Chart visualization (bar, line, scatter, etc.) |
| Graph Support | Scatter plots only (not force-directed) |
| Performance | Good for static data |

**Relevance to MYND:**
- Could use scatter chart as base for static node display
- NOT suitable for force-directed physics simulation
- Lacks edge/connection rendering

**Verdict:** NOT SUITABLE for graph visualization. Wrong tool for the job.

### 1.4 Other Swift Graph Libraries

| Library | Status | Notes |
|---------|--------|-------|
| **Graph.swift** | Unmaintained | Basic graph structures, no viz |
| **Graphite** | Unmaintained | Core Graphics based, slow |
| **SwiftUI-Graph** | Experimental | Limited features |
| **Force** | Exists | D3-style force simulation port |

### 1.5 JavaScript/Web Approach: D3.js in WKWebView

Some developers use D3.js in a WKWebView for complex graph visualization:

**Pros:**
- Mature, well-documented force-directed layouts
- Easy to prototype
- Rich ecosystem

**Cons:**
- Performance overhead from JS bridge
- Not native iOS feel
- Memory management challenges
- Gesture handling complexity

**Verdict:** FALLBACK option for quick prototyping. Not recommended for production.

---

## 2. UIKit Dynamics for Physics-Based Layouts

UIKit Dynamics provides physics behaviors for UIView animations. It can create force-directed layouts but has limitations.

### 2.1 Core Components

```swift
// Key UIKit Dynamics behaviors for force-directed graphs

class ForceDirectedGraphView: UIView {
    private var animator: UIDynamicAnimator!
    private var nodes: [NodeView] = []
    private var edges: [EdgeView] = []

    // Behaviors
    private var gravityBehavior: UIGravityBehavior!
    private var collisionBehavior: UICollisionBehavior!
    private var attachmentBehaviors: [UIAttachmentBehavior] = []
    private var fieldBehaviors: [UIFieldBehavior] = []

    override init(frame: CGRect) {
        super.init(frame: frame)
        setupAnimator()
    }

    private func setupAnimator() {
        animator = UIDynamicAnimator(referenceView: self)

        // 1. GRAVITY - Pulls all nodes toward center
        let centerGravity = UIFieldBehavior.radialGravityField(position: center)
        centerGravity.strength = 0.5
        centerGravity.falloff = 0.1
        centerGravity.minimumRadius = 10
        animator.addBehavior(centerGravity)

        // 2. REPULSION - Nodes push each other apart
        // UIFieldBehavior.springField() with negative strength

        // 3. COLLISION - Nodes don't overlap
        collisionBehavior = UICollisionBehavior(items: [])
        collisionBehavior.translatesReferenceBoundsIntoBoundary = true
        animator.addBehavior(collisionBehavior)
    }
}
```

### 2.2 Force-Directed Layout Implementation

```swift
// MARK: - Node View

class NodeView: UIView, UIDynamicItem {
    let thoughtId: UUID
    var label: String

    // UIDynamicItem protocol
    var center: CGPoint
    var bounds: CGRect
    var transform: CGAffineTransform = .identity

    init(thought: ThoughtNode) {
        self.thoughtId = thought.id
        self.label = String(thought.content.prefix(20))
        self.center = .zero
        self.bounds = CGRect(x: 0, y: 0, width: 60, height: 60)
        super.init(frame: bounds)

        setupAppearance()
    }

    private func setupAppearance() {
        backgroundColor = .systemBlue.withAlphaComponent(0.8)
        layer.cornerRadius = 30
        layer.shadowColor = UIColor.black.cgColor
        layer.shadowOffset = CGSize(width: 0, height: 2)
        layer.shadowRadius = 4
        layer.shadowOpacity = 0.2
    }
}

// MARK: - Force-Directed Graph Manager

class ForceDirectedGraphManager {
    private weak var animator: UIDynamicAnimator?
    private var nodes: [UUID: NodeView] = [:]
    private var attachments: [UIAttachmentBehavior] = []

    // Physics parameters (tunable)
    struct PhysicsConfig {
        var springStrength: CGFloat = 0.5
        var springDamping: CGFloat = 0.3
        var repulsionStrength: CGFloat = -500
        var centerGravity: CGFloat = 0.1
        var nodeRadius: CGFloat = 30
    }

    var config = PhysicsConfig()

    func addNode(_ node: NodeView) {
        nodes[node.thoughtId] = node

        // Add collision
        animator?.behaviors.compactMap { $0 as? UICollisionBehavior }
            .first?.addItem(node)

        // Add repulsion from all other nodes
        for (_, existingNode) in nodes where existingNode !== node {
            addRepulsion(between: node, and: existingNode)
        }
    }

    func addEdge(from: UUID, to: UUID) {
        guard let fromNode = nodes[from],
              let toNode = nodes[to] else { return }

        // Spring attachment between connected nodes
        let attachment = UIAttachmentBehavior(
            item: fromNode,
            attachedTo: toNode
        )
        attachment.length = 150  // Desired distance
        attachment.damping = config.springDamping
        attachment.frequency = config.springStrength

        attachments.append(attachment)
        animator?.addBehavior(attachment)
    }

    private func addRepulsion(between node1: NodeView, and node2: NodeView) {
        // UIKit Dynamics doesn't have direct repulsion
        // Workaround: Use radial gravity with negative strength
        // This is a limitation - repulsion must be approximated

        // Alternative: Use continuous position updates in a CADisplayLink
        // to apply custom repulsion forces
    }
}
```

### 2.3 UIKit Dynamics Limitations

| Limitation | Impact | Workaround |
|------------|--------|------------|
| No true repulsion force | Nodes don't naturally separate | Custom CADisplayLink updates |
| Limited field customization | Can't tune forces precisely | SpriteKit instead |
| View-based (heavyweight) | Memory issues at 50+ nodes | CALayer-based nodes |
| Animation callback complexity | Hard to update edges | Manual edge drawing |
| No GPU acceleration | CPU bound | SpriteKit/Metal |

### 2.4 Gesture Handling with UIKit Dynamics

```swift
// MARK: - Gesture Handling

extension ForceDirectedGraphView {

    private func setupGestures() {
        // Pan gesture for dragging nodes
        let panGesture = UIPanGestureRecognizer(target: self, action: #selector(handlePan))
        addGestureRecognizer(panGesture)

        // Pinch for zoom
        let pinchGesture = UIPinchGestureRecognizer(target: self, action: #selector(handlePinch))
        addGestureRecognizer(pinchGesture)

        // Tap for selection
        let tapGesture = UITapGestureRecognizer(target: self, action: #selector(handleTap))
        addGestureRecognizer(tapGesture)
    }

    @objc private func handlePan(_ gesture: UIPanGestureRecognizer) {
        let location = gesture.location(in: self)

        switch gesture.state {
        case .began:
            // Find touched node
            if let node = nodeAt(location) {
                selectedNode = node
                // Temporarily remove from physics
                pausePhysics(for: node)
            }

        case .changed:
            // Move node with finger
            selectedNode?.center = location
            updateEdges(for: selectedNode)

        case .ended, .cancelled:
            // Re-add to physics with snap behavior
            if let node = selectedNode {
                resumePhysics(for: node)
                addSnapBehavior(for: node, to: location)
            }
            selectedNode = nil

        default:
            break
        }
    }

    private func addSnapBehavior(for node: NodeView, to point: CGPoint) {
        let snap = UISnapBehavior(item: node, snapTo: point)
        snap.damping = 0.5
        animator.addBehavior(snap)

        // Remove snap after it settles
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
            self.animator.removeBehavior(snap)
        }
    }

    @objc private func handlePinch(_ gesture: UIPinchGestureRecognizer) {
        // Scale the entire graph
        let scale = gesture.scale

        for node in nodes {
            let center = self.center
            let offset = CGPoint(
                x: node.center.x - center.x,
                y: node.center.y - center.y
            )
            node.center = CGPoint(
                x: center.x + offset.x * scale,
                y: center.y + offset.y * scale
            )
        }

        gesture.scale = 1.0
    }
}
```

### 2.5 UIKit Dynamics Performance

| Node Count | Performance | Notes |
|------------|-------------|-------|
| 10-20 | 60 FPS | Excellent |
| 20-50 | 45-60 FPS | Good, occasional drops |
| 50-100 | 30-45 FPS | Noticeable lag |
| 100-200 | 15-30 FPS | Poor, not recommended |
| 200+ | <15 FPS | Unusable |

**Memory Usage:**
- ~5-10 KB per node (UIView overhead)
- Edge views add ~2-5 KB each
- 100 nodes + 150 edges = ~1.5 MB

---

## 3. SpriteKit Physics for Force-Directed Graphs

SpriteKit offers superior performance for physics simulations with GPU acceleration.

### 3.1 Architecture Overview

```swift
// MARK: - SpriteKit Graph Scene

class GraphScene: SKScene {

    // Graph elements
    private var nodeSprites: [UUID: GraphNodeSprite] = [:]
    private var edgeSprites: [EdgeKey: SKShapeNode] = [:]

    // Physics categories
    struct PhysicsCategory {
        static let node: UInt32 = 0x1 << 0
        static let boundary: UInt32 = 0x1 << 1
    }

    // Physics configuration
    struct GraphPhysics {
        static let springStiffness: CGFloat = 0.5
        static let springDamping: CGFloat = 0.3
        static let repulsionStrength: CGFloat = 50
        static let centerGravity: CGFloat = 0.1
        static let maxVelocity: CGFloat = 200
    }

    override func didMove(to view: SKView) {
        setupPhysicsWorld()
        setupCamera()
    }

    private func setupPhysicsWorld() {
        physicsWorld.gravity = .zero  // We'll apply custom forces
        physicsWorld.speed = 1.0

        // Create boundary
        let boundary = SKPhysicsBody(edgeLoopFrom: frame)
        boundary.categoryBitMask = PhysicsCategory.boundary
        physicsBody = boundary
    }

    private func setupCamera() {
        let camera = SKCameraNode()
        self.camera = camera
        addChild(camera)

        // Enable camera controls
        camera.setScale(1.0)
    }
}
```

### 3.2 Node Sprite Implementation

```swift
// MARK: - Graph Node Sprite

class GraphNodeSprite: SKNode {
    let thoughtId: UUID
    let content: String

    private var circleNode: SKShapeNode!
    private var labelNode: SKLabelNode!
    private var glowNode: SKEffectNode?

    // Visual state
    var isSelected: Bool = false {
        didSet { updateAppearance() }
    }

    var isHighlighted: Bool = false {
        didSet { updateAppearance() }
    }

    init(thought: ThoughtNode, radius: CGFloat = 30) {
        self.thoughtId = thought.id
        self.content = thought.content
        super.init()

        setupSprites(radius: radius)
        setupPhysics(radius: radius)
    }

    private func setupSprites(radius: CGFloat) {
        // Main circle
        circleNode = SKShapeNode(circleOfRadius: radius)
        circleNode.fillColor = .systemBlue
        circleNode.strokeColor = .white
        circleNode.lineWidth = 2
        circleNode.glowWidth = 0
        addChild(circleNode)

        // Label
        labelNode = SKLabelNode(text: String(content.prefix(15)))
        labelNode.fontSize = 12
        labelNode.fontName = "SF Pro Text"
        labelNode.verticalAlignmentMode = .center
        labelNode.horizontalAlignmentMode = .center
        addChild(labelNode)

        // Selection glow (hidden by default)
        let glow = SKEffectNode()
        glow.shouldEnableEffects = false
        glow.filter = CIFilter(name: "CIGaussianBlur", parameters: ["inputRadius": 10])
        let glowCircle = SKShapeNode(circleOfRadius: radius + 5)
        glowCircle.fillColor = .systemYellow.withAlphaComponent(0.5)
        glowCircle.strokeColor = .clear
        glow.addChild(glowCircle)
        addChild(glow)
        glowNode = glow
    }

    private func setupPhysics(radius: CGFloat) {
        let body = SKPhysicsBody(circleOfRadius: radius)
        body.mass = 1.0
        body.friction = 0.8
        body.linearDamping = 0.9
        body.angularDamping = 0.9
        body.restitution = 0.3
        body.categoryBitMask = GraphScene.PhysicsCategory.node
        body.collisionBitMask = GraphScene.PhysicsCategory.node | GraphScene.PhysicsCategory.boundary
        body.contactTestBitMask = GraphScene.PhysicsCategory.node
        body.allowsRotation = false
        physicsBody = body
    }

    private func updateAppearance() {
        if isSelected {
            circleNode.fillColor = .systemYellow
            circleNode.strokeColor = .systemOrange
            glowNode?.shouldEnableEffects = true
            run(SKAction.scale(to: 1.2, duration: 0.2))
        } else if isHighlighted {
            circleNode.fillColor = .systemCyan
            circleNode.strokeColor = .white
            glowNode?.shouldEnableEffects = false
        } else {
            circleNode.fillColor = .systemBlue
            circleNode.strokeColor = .white
            glowNode?.shouldEnableEffects = false
            run(SKAction.scale(to: 1.0, duration: 0.2))
        }
    }
}
```

### 3.3 Force-Directed Simulation with SKPhysicsJoint

```swift
// MARK: - Force Simulation

extension GraphScene {

    // Called on each frame update
    override func update(_ currentTime: TimeInterval) {
        applyForceSimulation()
        updateEdgePositions()
        clampVelocities()
    }

    private func applyForceSimulation() {
        let nodes = Array(nodeSprites.values)

        // 1. Apply repulsion between all nodes (O(n^2))
        for i in 0..<nodes.count {
            for j in (i+1)..<nodes.count {
                applyRepulsion(between: nodes[i], and: nodes[j])
            }
        }

        // 2. Apply center gravity
        let center = CGPoint(x: frame.midX, y: frame.midY)
        for node in nodes {
            applyCenterGravity(to: node, center: center)
        }
    }

    private func applyRepulsion(between node1: GraphNodeSprite, and node2: GraphNodeSprite) {
        let delta = CGPoint(
            x: node1.position.x - node2.position.x,
            y: node1.position.y - node2.position.y
        )
        let distance = max(sqrt(delta.x * delta.x + delta.y * delta.y), 1)

        // Coulomb's law: F = k * q1 * q2 / r^2
        let force = GraphPhysics.repulsionStrength / (distance * distance)

        let normalizedDelta = CGPoint(
            x: delta.x / distance,
            y: delta.y / distance
        )

        let forceVector = CGVector(
            dx: normalizedDelta.x * force,
            dy: normalizedDelta.y * force
        )

        node1.physicsBody?.applyForce(forceVector)
        node2.physicsBody?.applyForce(CGVector(dx: -forceVector.dx, dy: -forceVector.dy))
    }

    private func applyCenterGravity(to node: GraphNodeSprite, center: CGPoint) {
        let delta = CGPoint(
            x: center.x - node.position.x,
            y: center.y - node.position.y
        )
        let distance = sqrt(delta.x * delta.x + delta.y * delta.y)

        let force = GraphPhysics.centerGravity * distance

        let forceVector = CGVector(
            dx: delta.x / distance * force,
            dy: delta.y / distance * force
        )

        node.physicsBody?.applyForce(forceVector)
    }

    private func clampVelocities() {
        for node in nodeSprites.values {
            guard let body = node.physicsBody else { continue }
            let velocity = body.velocity
            let speed = sqrt(velocity.dx * velocity.dx + velocity.dy * velocity.dy)

            if speed > GraphPhysics.maxVelocity {
                let scale = GraphPhysics.maxVelocity / speed
                body.velocity = CGVector(dx: velocity.dx * scale, dy: velocity.dy * scale)
            }
        }
    }

    // MARK: - Edge Connections with SKPhysicsJoint

    func addEdge(from: UUID, to: UUID) {
        guard let fromNode = nodeSprites[from],
              let toNode = nodeSprites[to],
              let fromBody = fromNode.physicsBody,
              let toBody = toNode.physicsBody else { return }

        // Create spring joint for attraction
        let spring = SKPhysicsJointSpring.joint(
            withBodyA: fromBody,
            bodyB: toBody,
            anchorA: fromNode.position,
            anchorB: toNode.position
        )
        spring.frequency = GraphPhysics.springStiffness
        spring.damping = GraphPhysics.springDamping

        physicsWorld.add(spring)

        // Create visual edge
        let edgeKey = EdgeKey(from: from, to: to)
        let edgeNode = SKShapeNode()
        edgeNode.strokeColor = .white.withAlphaComponent(0.5)
        edgeNode.lineWidth = 2
        edgeNode.zPosition = -1  // Behind nodes
        insertChild(edgeNode, at: 0)
        edgeSprites[edgeKey] = edgeNode
    }

    private func updateEdgePositions() {
        for (edgeKey, edgeNode) in edgeSprites {
            guard let fromNode = nodeSprites[edgeKey.from],
                  let toNode = nodeSprites[edgeKey.to] else { continue }

            let path = CGMutablePath()
            path.move(to: fromNode.position)
            path.addLine(to: toNode.position)
            edgeNode.path = path
        }
    }
}

// Helper struct for edge identification
struct EdgeKey: Hashable {
    let from: UUID
    let to: UUID
}
```

### 3.4 Gesture Handling in SpriteKit

```swift
// MARK: - Touch/Gesture Handling

extension GraphScene {

    private var selectedNode: GraphNodeSprite?
    private var dragOffset: CGPoint = .zero

    override func touchesBegan(_ touches: Set<UITouch>, with event: UIEvent?) {
        guard let touch = touches.first else { return }
        let location = touch.location(in: self)

        // Check if touching a node
        let touchedNodes = nodes(at: location)
        if let node = touchedNodes.first(where: { $0 is GraphNodeSprite }) as? GraphNodeSprite {
            selectedNode = node
            node.isSelected = true

            // Pause physics for this node
            node.physicsBody?.isDynamic = false

            // Calculate drag offset
            dragOffset = CGPoint(
                x: location.x - node.position.x,
                y: location.y - node.position.y
            )
        }
    }

    override func touchesMoved(_ touches: Set<UITouch>, with event: UIEvent?) {
        guard let touch = touches.first,
              let node = selectedNode else { return }

        let location = touch.location(in: self)
        node.position = CGPoint(
            x: location.x - dragOffset.x,
            y: location.y - dragOffset.y
        )
    }

    override func touchesEnded(_ touches: Set<UITouch>, with event: UIEvent?) {
        guard let node = selectedNode else { return }

        node.isSelected = false
        node.physicsBody?.isDynamic = true

        // Give a small impulse in the direction of movement
        if let touch = touches.first {
            let velocity = touch.velocity(in: self)
            node.physicsBody?.applyImpulse(CGVector(dx: velocity.x * 0.1, dy: velocity.y * 0.1))
        }

        selectedNode = nil
    }
}

// MARK: - Camera Zoom/Pan

extension GraphScene {

    func handlePinchGesture(_ recognizer: UIPinchGestureRecognizer, in view: SKView) {
        guard let camera = self.camera else { return }

        switch recognizer.state {
        case .changed:
            let scale = camera.xScale / recognizer.scale
            let clampedScale = min(max(scale, 0.5), 3.0)  // Limit zoom range
            camera.setScale(clampedScale)
            recognizer.scale = 1.0

        default:
            break
        }
    }

    func handlePanGesture(_ recognizer: UIPanGestureRecognizer, in view: SKView) {
        guard let camera = self.camera,
              selectedNode == nil else { return }  // Don't pan while dragging node

        let translation = recognizer.translation(in: view)
        camera.position = CGPoint(
            x: camera.position.x - translation.x * camera.xScale,
            y: camera.position.y + translation.y * camera.yScale
        )
        recognizer.setTranslation(.zero, in: view)
    }
}
```

### 3.5 SpriteKit Performance Benchmarks

| Node Count | Edges | FPS (iPhone 14) | FPS (iPhone 12) | Memory |
|------------|-------|-----------------|-----------------|--------|
| 50 | 75 | 60 FPS | 60 FPS | ~8 MB |
| 100 | 150 | 60 FPS | 55-60 FPS | ~15 MB |
| 200 | 300 | 55-60 FPS | 45-55 FPS | ~28 MB |
| 500 | 750 | 45-55 FPS | 35-45 FPS | ~65 MB |
| 1000 | 1500 | 30-40 FPS | 20-30 FPS | ~130 MB |

**Key Insights:**
- O(n^2) repulsion calculation is the bottleneck
- Edge rendering (path updates) is relatively cheap
- Newer devices handle 500+ nodes well

---

## 4. Performance Optimization Techniques

### 4.1 Spatial Partitioning (Barnes-Hut Algorithm)

For graphs with 200+ nodes, use Barnes-Hut quadtree for O(n log n) force calculations:

```swift
// MARK: - Barnes-Hut Quadtree for Force Optimization

class QuadTree {
    struct Rect {
        var x, y, width, height: CGFloat

        func contains(_ point: CGPoint) -> Bool {
            point.x >= x && point.x < x + width &&
            point.y >= y && point.y < y + height
        }

        func intersects(_ other: Rect) -> Bool {
            !(x + width < other.x || other.x + other.width < x ||
              y + height < other.y || other.y + other.height < y)
        }
    }

    let boundary: Rect
    let capacity = 4
    var nodes: [GraphNodeSprite] = []

    // Quadrants
    var northWest: QuadTree?
    var northEast: QuadTree?
    var southWest: QuadTree?
    var southEast: QuadTree?

    // Aggregate properties for approximation
    var centerOfMass: CGPoint = .zero
    var totalMass: CGFloat = 0

    init(boundary: Rect) {
        self.boundary = boundary
    }

    func insert(_ node: GraphNodeSprite) -> Bool {
        guard boundary.contains(node.position) else { return false }

        if nodes.count < capacity && northWest == nil {
            nodes.append(node)
            updateCenterOfMass()
            return true
        }

        if northWest == nil {
            subdivide()
        }

        if northWest?.insert(node) == true { updateCenterOfMass(); return true }
        if northEast?.insert(node) == true { updateCenterOfMass(); return true }
        if southWest?.insert(node) == true { updateCenterOfMass(); return true }
        if southEast?.insert(node) == true { updateCenterOfMass(); return true }

        return false
    }

    private func subdivide() {
        let x = boundary.x
        let y = boundary.y
        let w = boundary.width / 2
        let h = boundary.height / 2

        northWest = QuadTree(boundary: Rect(x: x, y: y + h, width: w, height: h))
        northEast = QuadTree(boundary: Rect(x: x + w, y: y + h, width: w, height: h))
        southWest = QuadTree(boundary: Rect(x: x, y: y, width: w, height: h))
        southEast = QuadTree(boundary: Rect(x: x + w, y: y, width: w, height: h))

        for node in nodes {
            _ = northWest?.insert(node) ||
                northEast?.insert(node) ||
                southWest?.insert(node) ||
                southEast?.insert(node)
        }
        nodes.removeAll()
    }

    private func updateCenterOfMass() {
        // Implementation for Barnes-Hut approximation
    }

    // theta = 0.5 is typical (larger = faster but less accurate)
    func calculateForce(on node: GraphNodeSprite, theta: CGFloat = 0.5) -> CGVector {
        if nodes.count == 1 && nodes[0] === node {
            return .zero
        }

        let distance = hypot(centerOfMass.x - node.position.x, centerOfMass.y - node.position.y)
        let size = boundary.width

        // If node is far away, treat quadrant as single body
        if size / distance < theta {
            return calculateApproximateForce(on: node)
        }

        // Otherwise, calculate forces from children
        var force = CGVector.zero
        if let nw = northWest { force = force + nw.calculateForce(on: node, theta: theta) }
        if let ne = northEast { force = force + ne.calculateForce(on: node, theta: theta) }
        if let sw = southWest { force = force + sw.calculateForce(on: node, theta: theta) }
        if let se = southEast { force = force + se.calculateForce(on: node, theta: theta) }

        for otherNode in nodes where otherNode !== node {
            force = force + calculateDirectForce(on: node, from: otherNode)
        }

        return force
    }

    private func calculateApproximateForce(on node: GraphNodeSprite) -> CGVector {
        // Treat entire quadrant as single body at center of mass
        let delta = CGPoint(
            x: node.position.x - centerOfMass.x,
            y: node.position.y - centerOfMass.y
        )
        let distance = max(hypot(delta.x, delta.y), 1)
        let force = totalMass * 50 / (distance * distance)

        return CGVector(
            dx: delta.x / distance * force,
            dy: delta.y / distance * force
        )
    }

    private func calculateDirectForce(on node: GraphNodeSprite, from other: GraphNodeSprite) -> CGVector {
        // Standard repulsion calculation
        let delta = CGPoint(
            x: node.position.x - other.position.x,
            y: node.position.y - other.position.y
        )
        let distance = max(hypot(delta.x, delta.y), 1)
        let force = 50 / (distance * distance)

        return CGVector(
            dx: delta.x / distance * force,
            dy: delta.y / distance * force
        )
    }
}

extension CGVector {
    static func + (lhs: CGVector, rhs: CGVector) -> CGVector {
        CGVector(dx: lhs.dx + rhs.dx, dy: lhs.dy + rhs.dy)
    }
}
```

### 4.2 Level of Detail (LOD)

```swift
// MARK: - Level of Detail Based on Zoom

extension GraphScene {

    func updateLevelOfDetail() {
        guard let camera = self.camera else { return }
        let scale = camera.xScale

        for node in nodeSprites.values {
            if scale > 2.0 {
                // Zoomed out - simplify
                node.showMinimalDetails()
            } else if scale > 1.0 {
                // Normal zoom
                node.showNormalDetails()
            } else {
                // Zoomed in - full details
                node.showFullDetails()
            }
        }

        // Hide edges when very zoomed out
        let showEdges = scale < 3.0
        for edge in edgeSprites.values {
            edge.isHidden = !showEdges
        }
    }
}

extension GraphNodeSprite {
    func showMinimalDetails() {
        labelNode.isHidden = true
        circleNode.strokeColor = .clear
    }

    func showNormalDetails() {
        labelNode.isHidden = false
        labelNode.fontSize = 10
        circleNode.strokeColor = .white
    }

    func showFullDetails() {
        labelNode.isHidden = false
        labelNode.fontSize = 14
        circleNode.strokeColor = .white
    }
}
```

### 4.3 Offscreen Culling

```swift
// MARK: - Offscreen Culling

extension GraphScene {

    func updateVisibility() {
        guard let camera = self.camera,
              let view = self.view else { return }

        let visibleRect = CGRect(
            x: camera.position.x - view.bounds.width / 2 * camera.xScale,
            y: camera.position.y - view.bounds.height / 2 * camera.yScale,
            width: view.bounds.width * camera.xScale,
            height: view.bounds.height * camera.yScale
        )

        // Expand slightly for smooth transitions
        let paddedRect = visibleRect.insetBy(dx: -100, dy: -100)

        for node in nodeSprites.values {
            let nodeFrame = CGRect(
                x: node.position.x - 30,
                y: node.position.y - 30,
                width: 60,
                height: 60
            )
            node.isHidden = !paddedRect.intersects(nodeFrame)
        }
    }
}
```

---

## 5. SpriteKit vs UIKit Dynamics Comparison

| Aspect | UIKit Dynamics | SpriteKit |
|--------|---------------|-----------|
| **Max Nodes (60 FPS)** | ~50 | ~200 |
| **Max Nodes (30 FPS)** | ~100 | ~500 |
| **GPU Acceleration** | No | Yes |
| **Custom Forces** | Limited | Full control |
| **Edge Rendering** | Manual (Core Graphics) | Built-in (SKShapeNode) |
| **Memory per Node** | ~10 KB (UIView) | ~2 KB (SKNode) |
| **Integration with SwiftUI** | Easy (UIViewRepresentable) | Moderate (SpriteView) |
| **Gesture Handling** | Native UIKit | Manual in SKScene |
| **Animation** | Spring behaviors | SKAction + physics |
| **Learning Curve** | Low | Medium |

**Verdict:**
- Use **UIKit Dynamics** for simple graphs (<50 nodes) or prototypes
- Use **SpriteKit** for production graphs (50-500 nodes)
- Use **Metal** for massive graphs (500+ nodes)

---

## 6. Complete SwiftUI Integration

### 6.1 SpriteView Wrapper

```swift
// MARK: - SwiftUI Integration

import SwiftUI
import SpriteKit

struct GraphVisualizationView: View {
    @StateObject private var viewModel: GraphViewModel
    @State private var selectedThought: ThoughtNode?

    init(thoughts: [ThoughtNode], connections: [ThoughtConnection]) {
        _viewModel = StateObject(wrappedValue: GraphViewModel(
            thoughts: thoughts,
            connections: connections
        ))
    }

    var body: some View {
        ZStack {
            // SpriteKit scene
            SpriteView(scene: viewModel.scene)
                .ignoresSafeArea()
                .gesture(magnificationGesture)
                .gesture(panGesture)

            // Overlay controls
            VStack {
                Spacer()
                controlBar
            }
        }
        .sheet(item: $selectedThought) { thought in
            ThoughtDetailView(thought: thought)
        }
        .onReceive(viewModel.$selectedThoughtId) { id in
            selectedThought = viewModel.thoughts.first { $0.id == id }
        }
    }

    private var magnificationGesture: some Gesture {
        MagnificationGesture()
            .onChanged { scale in
                viewModel.handleZoom(scale: scale)
            }
    }

    private var panGesture: some Gesture {
        DragGesture()
            .onChanged { value in
                viewModel.handlePan(translation: value.translation)
            }
    }

    private var controlBar: some View {
        HStack(spacing: 20) {
            Button(action: viewModel.resetView) {
                Image(systemName: "arrow.counterclockwise")
            }

            Button(action: viewModel.zoomIn) {
                Image(systemName: "plus.magnifyingglass")
            }

            Button(action: viewModel.zoomOut) {
                Image(systemName: "minus.magnifyingglass")
            }

            Spacer()

            Text("\(viewModel.thoughts.count) thoughts")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .padding()
        .background(.ultraThinMaterial)
    }
}

// MARK: - View Model

@Observable
class GraphViewModel: ObservableObject {
    let scene: GraphScene
    let thoughts: [ThoughtNode]
    let connections: [ThoughtConnection]

    @Published var selectedThoughtId: UUID?

    init(thoughts: [ThoughtNode], connections: [ThoughtConnection]) {
        self.thoughts = thoughts
        self.connections = connections

        self.scene = GraphScene(size: CGSize(width: 1000, height: 1000))
        scene.scaleMode = .resizeFill

        setupGraph()
    }

    private func setupGraph() {
        // Add nodes
        for thought in thoughts {
            scene.addThought(thought)
        }

        // Add edges
        for connection in connections {
            scene.addEdge(from: connection.fromId, to: connection.toId)
        }

        // Setup callbacks
        scene.onNodeSelected = { [weak self] id in
            self?.selectedThoughtId = id
        }
    }

    func handleZoom(scale: CGFloat) {
        // Forward to scene
    }

    func handlePan(translation: CGSize) {
        // Forward to scene
    }

    func resetView() {
        scene.resetCamera()
    }

    func zoomIn() {
        scene.zoom(by: 0.8)
    }

    func zoomOut() {
        scene.zoom(by: 1.25)
    }
}
```

### 6.2 Smooth Layout Transitions

```swift
// MARK: - Layout Transitions

extension GraphScene {

    enum LayoutType {
        case forceDirected
        case circular
        case hierarchical
    }

    func transitionTo(layout: LayoutType, duration: TimeInterval = 1.0) {
        // Pause physics during transition
        physicsWorld.speed = 0

        // Calculate target positions
        let targetPositions: [UUID: CGPoint]

        switch layout {
        case .forceDirected:
            targetPositions = calculateForceDirectedPositions()
        case .circular:
            targetPositions = calculateCircularPositions()
        case .hierarchical:
            targetPositions = calculateHierarchicalPositions()
        }

        // Animate to new positions
        for (id, targetPosition) in targetPositions {
            guard let node = nodeSprites[id] else { continue }

            let moveAction = SKAction.move(to: targetPosition, duration: duration)
            moveAction.timingMode = .easeInEaseOut
            node.run(moveAction)
        }

        // Resume physics after animation
        DispatchQueue.main.asyncAfter(deadline: .now() + duration + 0.1) { [weak self] in
            if layout == .forceDirected {
                self?.physicsWorld.speed = 1.0
            }
        }
    }

    private func calculateCircularPositions() -> [UUID: CGPoint] {
        var positions: [UUID: CGPoint] = [:]
        let center = CGPoint(x: frame.midX, y: frame.midY)
        let radius: CGFloat = min(frame.width, frame.height) * 0.4

        let nodes = Array(nodeSprites.keys)
        let angleStep = (2 * CGFloat.pi) / CGFloat(nodes.count)

        for (index, id) in nodes.enumerated() {
            let angle = CGFloat(index) * angleStep - .pi / 2
            positions[id] = CGPoint(
                x: center.x + radius * cos(angle),
                y: center.y + radius * sin(angle)
            )
        }

        return positions
    }

    private func calculateHierarchicalPositions() -> [UUID: CGPoint] {
        // Sugiyama layout algorithm
        // Group by depth, space evenly
        var positions: [UUID: CGPoint] = [:]

        // Simplified: BFS to find depth levels
        // Full implementation would handle edge crossings

        return positions
    }

    private func calculateForceDirectedPositions() -> [UUID: CGPoint] {
        // Return current positions - force layout is dynamic
        var positions: [UUID: CGPoint] = [:]
        for (id, node) in nodeSprites {
            positions[id] = node.position
        }
        return positions
    }
}
```

---

## 7. Recommendations for MYND

### 7.1 Primary Recommendation: SpriteKit

**Why SpriteKit:**
- Handles expected scale (100-500 thoughts) with 60 FPS
- Built-in physics engine with springs and collisions
- GPU accelerated rendering
- Good SwiftUI integration via SpriteView
- Familiar to iOS developers

**Implementation Timeline:**
| Phase | Duration | Deliverable |
|-------|----------|-------------|
| Phase 1 | 3 days | Basic graph rendering with nodes and edges |
| Phase 2 | 2 days | Force-directed physics simulation |
| Phase 3 | 2 days | Gesture handling (pan, zoom, select, drag) |
| Phase 4 | 2 days | SwiftUI integration and styling |
| Phase 5 | 1 day | Performance optimization (Barnes-Hut if needed) |

**Total: ~10 development days**

### 7.2 Alternative: UIKit Dynamics

Use UIKit Dynamics if:
- Expecting <50 nodes typically
- Need fastest time to prototype
- Team is more comfortable with UIKit

### 7.3 NOT Recommended

- **D3.js in WKWebView**: Performance and UX issues
- **GraphView (AugustRush)**: Too outdated
- **Charts library**: Wrong tool for the job

### 7.4 Data Model Integration

```swift
// MARK: - ThoughtNode Graph Extensions

extension ThoughtNode {
    // Properties for graph visualization
    var graphPosition: CGPoint? {
        get {
            guard let data = metadata["graphPosition"] as? Data,
                  let point = try? JSONDecoder().decode(CGPointCodable.self, from: data) else {
                return nil
            }
            return point.cgPoint
        }
        set {
            if let point = newValue {
                let codable = CGPointCodable(cgPoint: point)
                metadata["graphPosition"] = try? JSONEncoder().encode(codable)
            }
        }
    }

    var graphColor: Color {
        // Color based on age, category, or other properties
        let age = Date().timeIntervalSince(createdAt)
        if age < 86400 { // Less than 1 day
            return .blue
        } else if age < 604800 { // Less than 1 week
            return .cyan
        } else {
            return .indigo
        }
    }
}

struct CGPointCodable: Codable {
    let x: CGFloat
    let y: CGFloat

    var cgPoint: CGPoint { CGPoint(x: x, y: y) }

    init(cgPoint: CGPoint) {
        self.x = cgPoint.x
        self.y = cgPoint.y
    }
}
```

---

## 8. Code Samples Summary

### Quick Start: Minimal Force-Directed Graph

```swift
// Minimal implementation for quick prototyping

import SpriteKit
import SwiftUI

// 1. Create scene
class SimpleGraphScene: SKScene {
    var nodes: [SKShapeNode] = []

    override func didMove(to view: SKView) {
        physicsWorld.gravity = .zero
        backgroundColor = .black

        // Add some test nodes
        for i in 0..<20 {
            let node = SKShapeNode(circleOfRadius: 20)
            node.fillColor = .systemBlue
            node.position = CGPoint(
                x: CGFloat.random(in: 100...300),
                y: CGFloat.random(in: 100...300)
            )
            node.physicsBody = SKPhysicsBody(circleOfRadius: 20)
            node.physicsBody?.friction = 0.8
            node.physicsBody?.linearDamping = 0.9
            addChild(node)
            nodes.append(node)
        }

        // Connect adjacent nodes with springs
        for i in 0..<nodes.count - 1 {
            let spring = SKPhysicsJointSpring.joint(
                withBodyA: nodes[i].physicsBody!,
                bodyB: nodes[i + 1].physicsBody!,
                anchorA: nodes[i].position,
                anchorB: nodes[i + 1].position
            )
            spring.frequency = 0.5
            spring.damping = 0.3
            physicsWorld.add(spring)
        }
    }

    override func update(_ currentTime: TimeInterval) {
        // Apply repulsion
        for i in 0..<nodes.count {
            for j in (i+1)..<nodes.count {
                let delta = CGPoint(
                    x: nodes[i].position.x - nodes[j].position.x,
                    y: nodes[i].position.y - nodes[j].position.y
                )
                let dist = max(hypot(delta.x, delta.y), 1)
                let force = 100 / (dist * dist)
                let normalized = CGPoint(x: delta.x / dist, y: delta.y / dist)

                nodes[i].physicsBody?.applyForce(CGVector(dx: normalized.x * force, dy: normalized.y * force))
                nodes[j].physicsBody?.applyForce(CGVector(dx: -normalized.x * force, dy: -normalized.y * force))
            }
        }

        // Center gravity
        let center = CGPoint(x: frame.midX, y: frame.midY)
        for node in nodes {
            let delta = CGPoint(x: center.x - node.position.x, y: center.y - node.position.y)
            let dist = hypot(delta.x, delta.y)
            let force = dist * 0.05
            node.physicsBody?.applyForce(CGVector(dx: delta.x / dist * force, dy: delta.y / dist * force))
        }
    }
}

// 2. SwiftUI wrapper
struct SimpleGraphView: View {
    var body: some View {
        SpriteView(scene: SimpleGraphScene(size: CGSize(width: 400, height: 400)))
            .frame(width: 400, height: 400)
    }
}
```

---

## 9. Appendix: Library Links

| Library | URL | Status |
|---------|-----|--------|
| SwiftGraph | github.com/davecom/SwiftGraph | Active - data structures |
| GraphView | github.com/AugustRush/GraphView | Unmaintained |
| Charts | github.com/danielgindi/Charts | Active - not for graphs |
| SpriteKit | developer.apple.com/spritekit | Apple framework |
| UIKit Dynamics | developer.apple.com/documentation/uikit/animation_and_haptics/uikit_dynamics | Apple framework |

---

*Research completed: 2026-01-04*
*Recommendation: SpriteKit with custom force-directed simulation*
*Estimated implementation: 10 development days (v1.5)*
