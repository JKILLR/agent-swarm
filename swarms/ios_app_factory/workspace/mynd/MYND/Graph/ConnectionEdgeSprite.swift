//
//  ConnectionEdgeSprite.swift
//  MYND
//
//  Created on 2026-01-04.
//  iOS 17+, Swift 5.9+
//

import SpriteKit
import SwiftUI

/// SKNode subclass representing a connection edge between two thought nodes
/// Draws a curved line with configurable appearance based on connection properties
class ConnectionEdgeSprite: SKNode {

    // MARK: - Properties

    /// The ID of the connection this sprite represents
    let connectionId: UUID

    /// Source node reference
    private weak var sourceNode: ThoughtNodeSprite?

    /// Target node reference
    private weak var targetNode: ThoughtNodeSprite?

    /// The curve shape node
    private var curveNode: SKShapeNode!

    /// Connection strength (0.0 - 1.0)
    private var strength: CGFloat

    /// Connection type for coloring
    private var connectionType: String?

    // MARK: - Constants

    private let minLineWidth: CGFloat = 1.5
    private let maxLineWidth: CGFloat = 6.0
    private let curveControlPointOffset: CGFloat = 50.0
    private let animationDuration: TimeInterval = 0.4

    // MARK: - Initialization

    init(connectionId: UUID, source: ThoughtNodeSprite, target: ThoughtNodeSprite, strength: CGFloat = 0.5, type: String? = nil) {
        self.connectionId = connectionId
        self.sourceNode = source
        self.targetNode = target
        self.strength = min(1.0, max(0.0, strength))
        self.connectionType = type
        super.init()

        setupCurveNode()
        updatePath()
        animateCreation()

        name = "connection_\(connectionId.uuidString)"
        zPosition = -10 // Draw behind nodes
    }

    required init?(coder aDecoder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    // MARK: - Setup

    private func setupCurveNode() {
        curveNode = SKShapeNode()
        curveNode.strokeColor = connectionColor(for: connectionType)
        curveNode.lineWidth = lineWidth(for: strength)
        curveNode.fillColor = .clear
        curveNode.lineCap = .round
        curveNode.lineJoin = .round
        curveNode.alpha = 0 // Start invisible for animation

        addChild(curveNode)
    }

    // MARK: - Public Methods

    /// Updates the edge path when nodes move
    func updatePath() {
        guard let source = sourceNode, let target = targetNode else {
            removeFromParent()
            return
        }

        let startPoint = source.position
        let endPoint = target.position

        let path = createCurvedPath(from: startPoint, to: endPoint)
        curveNode.path = path
    }

    /// Updates the connection strength and line thickness
    func updateStrength(_ newStrength: CGFloat) {
        strength = min(1.0, max(0.0, newStrength))
        let targetWidth = lineWidth(for: strength)

        let action = SKAction.customAction(withDuration: 0.2) { [weak self] _, _ in
            self?.curveNode.lineWidth = targetWidth
        }
        curveNode.run(action)
    }

    /// Updates the connection type and color
    func updateType(_ newType: String?) {
        connectionType = newType
        let targetColor = connectionColor(for: newType)

        let colorAction = SKAction.customAction(withDuration: 0.3) { [weak self] node, elapsedTime in
            guard let shapeNode = self?.curveNode else { return }
            let progress = elapsedTime / 0.3
            shapeNode.strokeColor = shapeNode.strokeColor.interpolate(to: targetColor, progress: progress)
        }
        curveNode.run(colorAction)
    }

    /// Highlights the edge (e.g., when connected node is selected)
    func setHighlighted(_ highlighted: Bool) {
        let targetAlpha: CGFloat = highlighted ? 1.0 : 0.7
        let targetWidth = highlighted ? lineWidth(for: strength) * 1.5 : lineWidth(for: strength)

        let fadeAction = SKAction.fadeAlpha(to: targetAlpha, duration: 0.15)
        let widthAction = SKAction.customAction(withDuration: 0.15) { [weak self] _, _ in
            self?.curveNode.lineWidth = targetWidth
        }
        curveNode.run(SKAction.group([fadeAction, widthAction]))
    }

    /// Called every frame to keep edge aligned with nodes
    func update() {
        updatePath()
    }

    /// Returns true if this edge connects to the given thought ID
    func connects(thoughtId: UUID) -> Bool {
        return sourceNode?.thoughtId == thoughtId || targetNode?.thoughtId == thoughtId
    }

    // MARK: - Private Methods

    private func createCurvedPath(from start: CGPoint, to end: CGPoint) -> CGPath {
        let path = CGMutablePath()
        path.move(to: start)

        // Calculate control points for a smooth curve
        let midX = (start.x + end.x) / 2
        let midY = (start.y + end.y) / 2

        // Offset control point perpendicular to the line
        let dx = end.x - start.x
        let dy = end.y - start.y
        let distance = sqrt(dx * dx + dy * dy)

        // Scale curve offset based on distance
        let curveOffset = min(curveControlPointOffset, distance * 0.2)

        // Perpendicular offset direction
        let perpX = -dy / distance * curveOffset
        let perpY = dx / distance * curveOffset

        let controlPoint = CGPoint(x: midX + perpX, y: midY + perpY)

        path.addQuadCurve(to: end, control: controlPoint)

        return path
    }

    private func lineWidth(for strength: CGFloat) -> CGFloat {
        return minLineWidth + (maxLineWidth - minLineWidth) * strength
    }

    private func connectionColor(for type: String?) -> UIColor {
        switch type?.lowercased() {
        case "causal":
            return UIColor(red: 0.29, green: 0.56, blue: 0.89, alpha: 1.0) // Blue
        case "temporal":
            return UIColor(red: 0.35, green: 0.78, blue: 0.62, alpha: 1.0) // Green
        case "associative":
            return UIColor(red: 0.69, green: 0.47, blue: 0.82, alpha: 1.0) // Purple
        case "contradictory":
            return UIColor(red: 0.89, green: 0.35, blue: 0.35, alpha: 1.0) // Red
        case "supportive":
            return UIColor(red: 0.95, green: 0.77, blue: 0.25, alpha: 1.0) // Yellow
        default:
            return UIColor(red: 0.6, green: 0.65, blue: 0.7, alpha: 1.0) // Default gray
        }
    }

    private func animateCreation() {
        // Fade in
        let fadeIn = SKAction.fadeAlpha(to: 0.7, duration: animationDuration)
        fadeIn.timingMode = .easeOut

        // Draw animation - start with zero line width and grow
        curveNode.lineWidth = 0
        let targetWidth = lineWidth(for: strength)

        let drawAction = SKAction.customAction(withDuration: animationDuration) { [weak self] _, elapsedTime in
            guard let self = self else { return }
            let progress = elapsedTime / self.animationDuration
            self.curveNode.lineWidth = targetWidth * progress
        }

        curveNode.run(SKAction.group([fadeIn, drawAction]))
    }
}
