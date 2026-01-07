//
//  MindMapScene.swift
//  MYND
//
//  Created on 2026-01-04.
//  iOS 17+, Swift 5.9+
//

import SpriteKit
import SwiftUI

/// SKScene subclass for 2D mind map visualization
/// Uses SpriteKit for performant rendering of thought nodes
class MindMapScene: SKScene {

    // MARK: - Properties

    /// Reference to the view model for data binding
    weak var viewModel: MindMapViewModel?

    /// Dictionary mapping thought IDs to their sprite nodes
    private var thoughtSprites: [UUID: ThoughtNodeSprite] = [:]

    /// Dictionary mapping connection IDs to their edge sprites
    private var connectionEdges: [UUID: ConnectionEdgeSprite] = [:]

    /// Currently selected node
    private var selectedNode: ThoughtNodeSprite?

    /// Node being dragged
    private var draggedNode: ThoughtNodeSprite?

    /// Camera node for pan/zoom
    private var cameraNode: SKCameraNode!

    /// Background grid node (for subtle grid lines)
    private var backgroundGrid: SKShapeNode?

    /// Container node for star field
    private var starFieldContainer: SKNode?

    // MARK: - Layout Constants

    private let nodeSpacing: CGFloat = 150

    // Removed minZoom and maxZoom stored properties as per instructions

    // MARK: - Lifecycle

    override func didMove(to view: SKView) {
        super.didMove(to: view)

        setupScene()
        setupCamera()
        setupGestures(for: view)
    }

    override func didChangeSize(_ oldSize: CGSize) {
        super.didChangeSize(oldSize)
        rebuildBackground()
    }

    // MARK: - Setup

    private func setupScene() {
        backgroundColor = UIColor.systemBackground
        
        // Add subtle background visual (backdrop)
        let backdrop = SKShapeNode(rectOf: size)
        backdrop.position = CGPoint(x: size.width / 2, y: size.height / 2)
        backdrop.fillColor = UIColor.secondarySystemBackground
        backdrop.alpha = 0.2
        backdrop.zPosition = -100
        backdrop.blendMode = .alpha
        addChild(backdrop)
        
        physicsWorld.gravity = .zero
        
        // Build background grid and star field
        rebuildBackground()
    }

    private func rebuildBackground() {
        // Remove existing grid and stars if any
        backgroundGrid?.removeFromParent()
        starFieldContainer?.removeFromParent()

        // Create grid node
        let grid = SKShapeNode()
        let path = CGMutablePath()
        let lineSpacing: CGFloat = 100

        let maxX = size.width
        let maxY = size.height

        // Vertical lines
        var x: CGFloat = 0
        while x <= maxX {
            path.move(to: CGPoint(x: x, y: 0))
            path.addLine(to: CGPoint(x: x, y: maxY))
            x += lineSpacing
        }
        // Horizontal lines
        var y: CGFloat = 0
        while y <= maxY {
            path.move(to: CGPoint(x: 0, y: y))
            path.addLine(to: CGPoint(x: maxX, y: y))
            y += lineSpacing
        }

        grid.path = path
        grid.strokeColor = UIColor.label.withAlphaComponent(0.05)
        grid.lineWidth = 0.5
        grid.zPosition = -90
        grid.name = "background_grid"
        grid.position = CGPoint(x: size.width / 2, y: size.height / 2)
        addChild(grid)
        backgroundGrid = grid

        // Create star field container
        let stars = SKNode()
        stars.zPosition = -95
        stars.position = .zero

        let starCount = 100
        for _ in 0..<starCount {
            let star = SKShapeNode(circleOfRadius: 1.0)
            star.fillColor = UIColor.white.withAlphaComponent(CGFloat.random(in: 0.05...0.12))
            star.strokeColor = .clear
            let starX = CGFloat.random(in: 0...maxX)
            let starY = CGFloat.random(in: 0...maxY)
            star.position = CGPoint(x: starX, y: starY)
            stars.addChild(star)
        }

        addChild(stars)
        starFieldContainer = stars
    }

    private func setupCamera() {
        cameraNode = SKCameraNode()
        cameraNode.position = CGPoint(x: size.width / 2, y: size.height / 2)
        addChild(cameraNode)
        camera = cameraNode
        
        // Clamp initial scale to AppConfig min/max zoom scale
        cameraNode.setScale(1.0)
    }

    private func setupGestures(for view: SKView) {
        // Pan gesture for canvas dragging
        let panGesture = UIPanGestureRecognizer(target: self, action: #selector(handlePan(_:)))
        panGesture.minimumNumberOfTouches = 2
        view.addGestureRecognizer(panGesture)

        // Pinch gesture for zoom
        let pinchGesture = UIPinchGestureRecognizer(target: self, action: #selector(handlePinch(_:)))
        view.addGestureRecognizer(pinchGesture)
    }

    // MARK: - Sync with ViewModel

    /// Synchronizes the scene with the current view model state
    func syncWithViewModel() {
        guard let viewModel = viewModel else { return }

        syncThoughtNodes(with: viewModel.thoughts)
        syncConnectionEdges(with: viewModel.connections)
    }

    private func syncThoughtNodes(with thoughts: [Thought]) {
        let currentThoughtIds = Set(thoughts.map { $0.id })
        let existingSpriteIds = Set(thoughtSprites.keys)

        // Add new thoughts
        let newIds = currentThoughtIds.subtracting(existingSpriteIds)
        for thought in thoughts where newIds.contains(thought.id) {
            addThoughtNode(thought)
        }

        // Remove deleted thoughts
        let removedIds = existingSpriteIds.subtracting(currentThoughtIds)
        for id in removedIds {
            removeThoughtNode(id: id)
        }

        // Update existing thoughts
        for thought in thoughts {
            if let sprite = thoughtSprites[thought.id] {
                sprite.updateContent(thought.content)
                sprite.updateCategory(thought.category)
                
                // Update position from model if different and non-zero
                let target = CGPoint(x: thought.positionX, y: thought.positionY)
                if sprite.position != target && target != .zero {
                    sprite.position = target
                }
            }
        }
    }

    private func syncConnectionEdges(with connections: [Connection]) {
        let currentConnectionIds = Set(connections.map { $0.id })
        let existingEdgeIds = Set(connectionEdges.keys)

        // Add new connections
        let newIds = currentConnectionIds.subtracting(existingEdgeIds)
        for connection in connections where newIds.contains(connection.id) {
            addConnectionEdge(connection)
        }

        // Remove deleted connections
        let removedIds = existingEdgeIds.subtracting(currentConnectionIds)
        for id in removedIds {
            removeConnectionEdge(id: id)
        }
    }

    // MARK: - Node Management

    private func addThoughtNode(_ thought: Thought) {
        let sprite = ThoughtNodeSprite(thought: thought)

        // Position: use stored position or calculate new one
        let position = CGPoint(x: thought.positionX, y: thought.positionY)
        if position == .zero {
            sprite.position = calculateNewNodePosition()
        } else {
            sprite.position = position
        }

        addChild(sprite)
        thoughtSprites[thought.id] = sprite

        // Animate appearance
        sprite.alpha = 0
        sprite.setScale(0.5)
        sprite.run(SKAction.group([
            SKAction.fadeIn(withDuration: 0.3),
            SKAction.scale(to: 1.0, duration: 0.3)
        ]))
    }

    private func removeThoughtNode(id: UUID) {
        guard let sprite = thoughtSprites[id] else { return }

        sprite.run(SKAction.sequence([
            SKAction.group([
                SKAction.fadeOut(withDuration: 0.2),
                SKAction.scale(to: 0.5, duration: 0.2)
            ]),
            SKAction.removeFromParent()
        ]))

        thoughtSprites.removeValue(forKey: id)
    }

    private func addConnectionEdge(_ connection: Connection) {
        guard let sourceId = connection.sourceThought?.id,
              let targetId = connection.targetThought?.id,
              let sourceSprite = thoughtSprites[sourceId],
              let targetSprite = thoughtSprites[targetId] else {
            return
        }

        let edge = ConnectionEdgeSprite(
            connectionId: connection.id,
            source: sourceSprite,
            target: targetSprite,
            strength: CGFloat(connection.strength)
        )

        addChild(edge)
        connectionEdges[connection.id] = edge
    }

    private func removeConnectionEdge(id: UUID) {
        guard let edge = connectionEdges[id] else { return }

        edge.run(SKAction.sequence([
            SKAction.fadeOut(withDuration: 0.2),
            SKAction.removeFromParent()
        ]))

        connectionEdges.removeValue(forKey: id)
    }

    private func calculateNewNodePosition() -> CGPoint {
        // Place new nodes in a spiral pattern from center
        let nodeCount = thoughtSprites.count
        let angle = CGFloat(nodeCount) * 0.5
        let radius = AppConfig.defaultNodeSpacing + CGFloat(nodeCount) * 20

        let centerX = size.width / 2
        let centerY = size.height / 2

        return CGPoint(
            x: centerX + cos(angle) * radius,
            y: centerY + sin(angle) * radius
        )
    }

    // MARK: - Touch Handling

    override func touchesBegan(_ touches: Set<UITouch>, with event: UIEvent?) {
        guard let touch = touches.first else { return }
        let location = touch.location(in: self)

        // Check if touching a node
        if let node = nodes(at: location).first(where: { $0 is ThoughtNodeSprite }) as? ThoughtNodeSprite {
            draggedNode = node
            node.setSelected(true)

            // Deselect previous selection
            if selectedNode != node {
                selectedNode?.setSelected(false)
            }
            selectedNode = node

            // Highlight connected edges
            for edge in connectionEdges.values {
                if edge.connects(thoughtId: node.thoughtId) {
                    edge.setHighlighted(true)
                } else {
                    edge.setHighlighted(false)
                }
            }

            // Update selected thought in view model
            viewModel?.selectedThought = viewModel?.thought(with: node.thoughtId)
        } else {
            // Tapped empty space - deselect
            selectedNode?.setSelected(false)
            selectedNode = nil
            viewModel?.selectedThought = nil

            // Remove all highlights
            for edge in connectionEdges.values {
                edge.setHighlighted(false)
            }
        }
    }

    override func touchesMoved(_ touches: Set<UITouch>, with event: UIEvent?) {
        guard let touch = touches.first, let node = draggedNode else { return }

        let location = touch.location(in: self)
        node.position = location

        // Update connection edges
        for edge in connectionEdges.values {
            edge.updatePath()
        }

        // Update position in view model
        if let thought = viewModel?.thought(with: node.thoughtId) {
            viewModel?.updatePosition(for: thought, position: location)
        }
    }

    override func touchesEnded(_ touches: Set<UITouch>, with event: UIEvent?) {
        draggedNode = nil
    }

    override func touchesCancelled(_ touches: Set<UITouch>, with event: UIEvent?) {
        draggedNode = nil
    }

    // MARK: - Gesture Handlers

    @objc private func handlePan(_ gesture: UIPanGestureRecognizer) {
        let translation = gesture.translation(in: view)

        cameraNode.position = CGPoint(
            x: cameraNode.position.x - translation.x,
            y: cameraNode.position.y + translation.y
        )

        gesture.setTranslation(.zero, in: view)
    }

    @objc private func handlePinch(_ gesture: UIPinchGestureRecognizer) {
        guard gesture.state == .changed else { return }

        let newScale = cameraNode.xScale / gesture.scale
        let clampedScale = max(AppConfig.minZoomScale, min(AppConfig.maxZoomScale, newScale))

        cameraNode.setScale(clampedScale)
        gesture.scale = 1.0
    }

    // MARK: - Camera Utilities

    /// Smoothly centers the camera on the currently selected node, if any
    func centerOnSelectedNode() {
        guard let selected = selectedNode else { return }
        let target = selected.position
        let move = SKAction.move(to: target, duration: 0.25)
        move.timingMode = .easeInEaseOut
        cameraNode.run(move)
    }

    /// Zooms and pans the camera to fit all nodes with padding
    func zoomToFitAllNodes(padding: CGFloat = 80) {
        let sprites = Array(thoughtSprites.values)
        guard !sprites.isEmpty else { return }

        // Compute bounding box of all sprites
        var minX = CGFloat.greatestFiniteMagnitude
        var minY = CGFloat.greatestFiniteMagnitude
        var maxX = -CGFloat.greatestFiniteMagnitude
        var maxY = -CGFloat.greatestFiniteMagnitude

        for sprite in sprites {
            let frame = sprite.calculateAccumulatedFrame()
            minX = min(minX, frame.minX)
            minY = min(minY, frame.minY)
            maxX = max(maxX, frame.maxX)
            maxY = max(maxY, frame.maxY)
        }

        // Apply padding
        minX -= padding
        minY -= padding
        maxX += padding
        maxY += padding

        let contentSize = CGSize(width: maxX - minX, height: maxY - minY)
        let contentCenter = CGPoint(x: (minX + maxX) / 2, y: (minY + maxY) / 2)

        // Determine scale to fit content within current scene view size
        guard contentSize.width > 0, contentSize.height > 0 else { return }

        let scaleX = size.width / contentSize.width
        let scaleY = size.height / contentSize.height
        // Use the smaller scale to fit both dimensions
        var targetScale = 1.0 / min(scaleX, scaleY)
        // Clamp scale to allowed range
        targetScale = max(AppConfig.minZoomScale, min(AppConfig.maxZoomScale, targetScale))

        let move = SKAction.move(to: contentCenter, duration: 0.28)
        move.timingMode = .easeInEaseOut
        let scale = SKAction.scale(to: targetScale, duration: 0.28)
        scale.timingMode = .easeInEaseOut
        cameraNode.run(SKAction.group([move, scale]))
    }

    // MARK: - Update Loop

    override func update(_ currentTime: TimeInterval) {
        // Update all connection edge positions
        for edge in connectionEdges.values {
            edge.update()
        }

        // Parallax effect for background grid and star field
        if let cameraPos = cameraNode?.position {
            backgroundGrid?.position = CGPoint(x: cameraPos.x * 0.05, y: cameraPos.y * 0.05)
            starFieldContainer?.position = CGPoint(x: cameraPos.x * 0.02, y: cameraPos.y * 0.02)
        }
    }
}
