//
//  Connection.swift
//  MYND
//
//  Created on 2026-01-04.
//  iOS 17+, Swift 5.9+
//

import Foundation
import SwiftData

/// Represents a connection between two thoughts
/// Used to create the graph structure in the mind map
@Model
final class Connection {

    // MARK: - Properties

    /// Unique identifier
    @Attribute(.unique) var id: UUID

    /// The source thought of this connection
    var sourceThought: Thought?

    /// The target thought of this connection
    var targetThought: Thought?

    /// Connection strength (0.0 to 1.0)
    /// Higher values indicate stronger relationships
    var strength: Double

    /// When the connection was created
    var createdAt: Date

    // MARK: - Initialization

    init(
        id: UUID = UUID(),
        sourceThought: Thought? = nil,
        targetThought: Thought? = nil,
        strength: Double = 0.5,
        createdAt: Date = Date()
    ) {
        self.id = id
        self.sourceThought = sourceThought
        self.targetThought = targetThought
        self.strength = strength
        self.createdAt = createdAt
    }

    // MARK: - Computed Properties

    /// Whether this connection is valid (has both source and target)
    var isValid: Bool {
        sourceThought != nil && targetThought != nil
    }
}
