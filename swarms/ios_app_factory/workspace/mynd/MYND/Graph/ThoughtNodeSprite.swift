//
//  ThoughtNodeSprite.swift
//  MYND
//
//  Created on 2026-01-04.
//  iOS 17+, Swift 5.9+
//

import SpriteKit
import SwiftUI

/// SKNode subclass representing a thought bubble in the mind map
/// Displays the thought content with visual styling based on category
class ThoughtNodeSprite: SKNode {

    // MARK: - Properties

    /// The ID of the thought this sprite represents
    let thoughtId: UUID

    /// Current category for color updates
    private var currentCategory: ThoughtCategory

    /// Background shape node
    private var backgroundNode: SKShapeNode!

    /// Inner highlight overlay node for gradient effect
    private var highlightNode: SKShapeNode?

    /// Text label node
    private var labelNode: SKLabelNode!

    /// Selection indicator
    private var selectionRing: SKShapeNode?

    /// Current selection state
    private(set) var isSelected: Bool = false

    // MARK: - Constants

    private let minRadius: CGFloat = 40
    private let maxRadius: CGFloat = 80
    private let padding: CGFloat = 16
    private let fontSize: CGFloat = 14
    private let maxLabelWidth: CGFloat = 120

    // MARK: - Computed Properties

    /// The radius of the node based on content length
    var radius: CGFloat {
        backgroundNode != nil ? backgroundNode.frame.width / 2 : minRadius
    }

    // MARK: - Initialization

    init(thought: Thought) {
        self.thoughtId = thought.id
        self.currentCategory = thought.category
        super.init()

        setupBackground(content: thought.content, category: thought.category)
        setupLabel(with: thought.content)
        setupPhysicsBody()

        name = "thought_\(thought.id.uuidString)"
    }

    required init?(coder aDecoder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    // MARK: - Setup

    private func setupBackground(content: String, category: ThoughtCategory) {
        // Calculate size based on content
        let contentLength = content.count
        let sizeMultiplier = min(1.0, max(0.5, CGFloat(contentLength) / 100.0))
        let nodeRadius = minRadius + (maxRadius - minRadius) * sizeMultiplier

        // Create circular background
        backgroundNode = SKShapeNode(circleOfRadius: nodeRadius)
        backgroundNode.fillColor = categoryColor(for: category)
        backgroundNode.strokeColor = UIColor.white.withAlphaComponent(0.15)
        backgroundNode.lineWidth = 1

        // Add subtle shadow effect
        backgroundNode.glowWidth = 4
        backgroundNode.alpha = 0.98

        addChild(backgroundNode)

        // Create inner highlight node with lighter color for gradient-like effect
        let highlight = SKShapeNode(circleOfRadius: nodeRadius)
        let baseColor = categoryColor(for: category)
        let lighterColor = baseColor.interpolate(to: .white, progress: 0.3)
        highlight.fillColor = lighterColor
        highlight.alpha = 0.35
        highlight.blendMode = .add
        highlight.strokeColor = .clear
        highlight.zPosition = backgroundNode.zPosition + 1

        addChild(highlight)
        highlightNode = highlight
    }

    private func setupLabel(with content: String) {
        labelNode = SKLabelNode(fontNamed: "SF Pro Text")
        labelNode.fontSize = fontSize
        labelNode.fontColor = .white
        labelNode.verticalAlignmentMode = .center
        labelNode.horizontalAlignmentMode = .center

        // Truncate long text
        updateLabelText(content)

        addChild(labelNode)
    }

    private func setupPhysicsBody() {
        let radius = backgroundNode.frame.width / 2
        physicsBody = SKPhysicsBody(circleOfRadius: radius)
        physicsBody?.isDynamic = true
        physicsBody?.mass = 1.0
        physicsBody?.linearDamping = 0.8
        physicsBody?.allowsRotation = false
        physicsBody?.affectedByGravity = false
        physicsBody?.categoryBitMask = 1
        physicsBody?.collisionBitMask = 1
        physicsBody?.contactTestBitMask = 0
    }

    // MARK: - Public Methods

    /// Updates the displayed content
    func updateContent(_ content: String) {
        updateLabelText(content)
    }

    /// Updates the category and background color
    func updateCategory(_ category: ThoughtCategory) {
        guard category != currentCategory else { return }
        currentCategory = category

        let newColor = categoryColor(for: category)
        let newHighlightColor = newColor.interpolate(to: .white, progress: 0.3)

        // Animate color change for backgroundNode
        let colorAction = SKAction.customAction(withDuration: 0.3) { [weak self] _, elapsedTime in
            guard let self = self else { return }
            let progress = elapsedTime / 0.3

            // Background fill color interpolate
            let oldColor = self.backgroundNode.fillColor
            self.backgroundNode.fillColor = oldColor.interpolate(to: newColor, progress: progress)

            // Highlight fill color interpolate
            if let highlight = self.highlightNode {
                let oldHighlightColor = highlight.fillColor
                highlight.fillColor = oldHighlightColor.interpolate(to: newHighlightColor, progress: progress)
            }
        }
        backgroundNode.run(colorAction)
    }

    /// Sets the selection state
    func setSelected(_ selected: Bool) {
        isSelected = selected

        if selected {
            addSelectionRing()
            animateSelection()
        } else {
            removeSelectionRing()
        }
    }

    // MARK: - Private Methods

    private func updateLabelText(_ content: String) {
        // Truncate and add ellipsis if needed
        let maxChars = 30
        if content.count > maxChars {
            let index = content.index(content.startIndex, offsetBy: maxChars - 3)
            labelNode.text = String(content[..<index]) + "..."
        } else {
            labelNode.text = content
        }

        // Removed unsupported properties:
        // labelNode.preferredMaxLayoutWidth = maxLabelWidth
        // labelNode.numberOfLines = 2
    }

    private func categoryColor(for category: ThoughtCategory) -> UIColor {
        switch category {
        case .idea:
            return UIColor(red: 0.29, green: 0.56, blue: 0.89, alpha: 1.0) // Blue
        case .task:
            return UIColor(red: 0.35, green: 0.78, blue: 0.62, alpha: 1.0) // Green
        case .reflection:
            return UIColor(red: 0.69, green: 0.47, blue: 0.82, alpha: 1.0) // Purple
        case .question:
            return UIColor(red: 0.95, green: 0.61, blue: 0.25, alpha: 1.0) // Orange
        case .insight:
            return UIColor(red: 0.95, green: 0.77, blue: 0.25, alpha: 1.0) // Yellow
        case .memory:
            return UIColor(red: 0.35, green: 0.78, blue: 0.62, alpha: 1.0) // Green
        case .goal:
            return UIColor(red: 0.89, green: 0.35, blue: 0.35, alpha: 1.0) // Red
        case .note:
            return UIColor(red: 0.45, green: 0.55, blue: 0.68, alpha: 1.0) // Default gray-blue
        }
    }

    private func addSelectionRing() {
        guard selectionRing == nil else { return }

        let ringRadius = (backgroundNode.frame.width / 2) + 4
        selectionRing = SKShapeNode(circleOfRadius: ringRadius)
        selectionRing?.strokeColor = .white
        selectionRing?.lineWidth = 3
        selectionRing?.fillColor = .clear
        selectionRing?.zPosition = -1

        addChild(selectionRing!)
    }

    private func removeSelectionRing() {
        selectionRing?.removeFromParent()
        selectionRing = nil
    }

    private func animateSelection() {
        // Subtle pulse animation with easing and more pronounced scale
        let scaleUp = SKAction.scale(to: 1.08, duration: 0.18)
        scaleUp.timingMode = .easeInEaseOut
        let scaleDown = SKAction.scale(to: 1.0, duration: 0.2)
        scaleDown.timingMode = .easeInEaseOut
        run(SKAction.sequence([scaleUp, scaleDown]))
    }
}

// MARK: - UIColor Extension

extension UIColor {
    func interpolate(to color: UIColor, progress: CGFloat) -> UIColor {
        var r1: CGFloat = 0, g1: CGFloat = 0, b1: CGFloat = 0, a1: CGFloat = 0
        var r2: CGFloat = 0, g2: CGFloat = 0, b2: CGFloat = 0, a2: CGFloat = 0

        self.getRed(&r1, green: &g1, blue: &b1, alpha: &a1)
        color.getRed(&r2, green: &g2, blue: &b2, alpha: &a2)

        let clampedProgress = min(1.0, max(0.0, progress))

        return UIColor(
            red: r1 + (r2 - r1) * clampedProgress,
            green: g1 + (g2 - g1) * clampedProgress,
            blue: b1 + (b2 - b1) * clampedProgress,
            alpha: a1 + (a2 - a1) * clampedProgress
        )
    }
}
