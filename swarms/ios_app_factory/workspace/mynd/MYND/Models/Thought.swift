//
//  Thought.swift
//  MYND
//
//  Created on 2026-01-04.
//  iOS 17+, Swift 5.9+
//

import Foundation
import SwiftData

/// Core thought entity for the mind map
/// Represents a single captured thought with position and category
@Model
final class Thought {

    // MARK: - Properties

    /// Unique identifier
    @Attribute(.unique) var id: UUID

    /// The thought content text
    var content: String

    /// X position on the canvas (stored as Double, not CGPoint)
    var positionX: Double

    /// Y position on the canvas (stored as Double, not CGPoint)
    var positionY: Double

    /// Category for organization
    var category: ThoughtCategory

    /// When the thought was created
    var createdAt: Date

    /// Connections where this thought is the source
    @Relationship(deleteRule: .cascade, inverse: \Connection.sourceThought)
    var outgoingConnections: [Connection] = []

    /// Connections where this thought is the target
    @Relationship(deleteRule: .cascade, inverse: \Connection.targetThought)
    var incomingConnections: [Connection] = []

    /// Cluster this thought belongs to (if any)
    @Relationship(inverse: \Cluster.thoughts)
    var cluster: Cluster?

    // MARK: - Initialization

    init(
        id: UUID = UUID(),
        content: String,
        positionX: Double = 0.0,
        positionY: Double = 0.0,
        category: ThoughtCategory = .note,
        createdAt: Date = Date()
    ) {
        self.id = id
        self.content = content
        self.positionX = positionX
        self.positionY = positionY
        self.category = category
        self.createdAt = createdAt
    }

    // MARK: - Computed Properties

    /// All connections (both incoming and outgoing)
    var allConnections: [Connection] {
        outgoingConnections + incomingConnections
    }

    /// All connected thoughts
    var connectedThoughts: [Thought] {
        let outgoing = outgoingConnections.compactMap { $0.targetThought }
        let incoming = incomingConnections.compactMap { $0.sourceThought }
        return outgoing + incoming
    }
}
