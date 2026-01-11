//
//  Cluster.swift
//  MYND
//
//  Created on 2026-01-04.
//  iOS 17+, Swift 5.9+
//

import Foundation
import SwiftData

// MARK: - Cluster Model

/// Represents a group of related thoughts
/// Clusters are AI-generated based on semantic similarity
@Model
final class Cluster {

    // MARK: - Properties

    /// Unique identifier
    @Attribute(.unique)
    var id: UUID

    /// AI-generated cluster name/theme
    var name: String

    /// AI-generated description of the cluster theme
    var summary: String?

    /// Display color (hex string)
    var colorHex: String

    /// Center X position on canvas
    var centerX: Double

    /// Center Y position on canvas
    var centerY: Double

    /// Radius for cluster boundary visualization
    var radius: Double

    /// Creation timestamp
    var createdAt: Date

    /// Last updated timestamp
    var updatedAt: Date

    /// Whether this cluster is expanded in UI
    var isExpanded: Bool

    // MARK: - Relationships

    /// Thoughts belonging to this cluster
    @Relationship(inverse: \Thought.cluster) var thoughts: [Thought] = []

    // MARK: - Computed Properties

    /// Center position as CGPoint
    var center: CGPoint {
        get { CGPoint(x: centerX, y: centerY) }
        set {
            centerX = newValue.x
            centerY = newValue.y
        }
    }

    /// Number of thoughts in cluster
    var thoughtCount: Int {
        thoughts.count
    }

    /// Whether this cluster is empty
    var isEmpty: Bool {
        thoughts.isEmpty
    }

    // MARK: - Initialization

    init(
        id: UUID = UUID(),
        name: String,
        summary: String? = nil,
        colorHex: String = "#728C9E",
        centerX: Double = 0,
        centerY: Double = 0,
        radius: Double = 150,
        isExpanded: Bool = true
    ) {
        self.id = id
        self.name = name
        self.summary = summary
        self.colorHex = colorHex
        self.centerX = centerX
        self.centerY = centerY
        self.radius = radius
        self.createdAt = Date()
        self.updatedAt = Date()
        self.isExpanded = isExpanded
    }

    // MARK: - Convenience Methods

    /// Adds a thought to this cluster
    func addThought(_ thought: Thought) {
        guard !thoughts.contains(where: { $0.id == thought.id }) else { return }
        thoughts.append(thought)
        thought.cluster = self
        updatedAt = Date()
        recalculateCenter()
    }

    /// Removes a thought from this cluster
    func removeThought(_ thought: Thought) {
        thoughts.removeAll { $0.id == thought.id }
        thought.cluster = nil
        updatedAt = Date()
        recalculateCenter()
    }

    /// Recalculates center based on thought positions
    func recalculateCenter() {
        guard !thoughts.isEmpty else { return }

        let totalX = thoughts.reduce(0.0) { $0 + $1.positionX }
        let totalY = thoughts.reduce(0.0) { $0 + $1.positionY }

        centerX = totalX / Double(thoughts.count)
        centerY = totalY / Double(thoughts.count)

        // Recalculate radius to encompass all thoughts
        let maxDistance = thoughts.map { thought in
            let dx = thought.positionX - centerX
            let dy = thought.positionY - centerY
            return sqrt(dx * dx + dy * dy)
        }.max() ?? 100

        radius = maxDistance + 50 // Add padding
    }

    /// Updates the cluster name and summary
    func update(name: String, summary: String?) {
        self.name = name
        self.summary = summary
        self.updatedAt = Date()
    }
}

// MARK: - Hashable

extension Cluster: Hashable {
    static func == (lhs: Cluster, rhs: Cluster) -> Bool {
        lhs.id == rhs.id
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(id)
    }
}

