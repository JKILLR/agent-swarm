//
//  MindMapViewModel.swift
//  MYND
//
//  Created on 2026-01-04.
//  iOS 17+, Swift 5.9+
//

import Foundation
import SwiftUI
import SwiftData
import Observation

/// Observable view model managing the mind map state
/// Bridges between SwiftData persistence and SpriteKit visualization
@Observable
@MainActor
final class MindMapViewModel {

    // MARK: - Dependencies

    private var modelContext: ModelContext?

    // MARK: - Published State

    /// All captured thoughts
    private(set) var thoughts: [Thought] = []

    /// All connections
    private(set) var connections: [Connection] = []

    /// Currently selected thought
    var selectedThought: Thought?

    /// Whether the graph is currently loading
    var isLoading: Bool = false

    /// Error message to display
    var errorMessage: String?

    // MARK: - Initialization

    init(modelContext: ModelContext? = nil) {
        self.modelContext = modelContext
        if modelContext != nil {
            loadData()
        }
    }

    /// Configure with model context (call from view)
    func configure(with context: ModelContext) {
        self.modelContext = context
        DataService.shared.configure(with: context)
        loadData()
    }

    // MARK: - Thought Operations

    /// Adds a new thought with the given content
    func addThought(content: String, category: ThoughtCategory = .note) {
        guard let context = modelContext else {
            errorMessage = "Database not configured"
            return
        }

        // Calculate position for new thought
        let position = calculateNewThoughtPosition()

        let thought = Thought(
            content: content,
            positionX: position.x,
            positionY: position.y,
            category: category
        )

        context.insert(thought)

        do {
            try context.save()
            thoughts.insert(thought, at: 0)
        } catch {
            errorMessage = "Failed to save thought: \(error.localizedDescription)"
        }
    }

    /// Deletes a thought
    func deleteThought(_ thought: Thought) {
        guard let context = modelContext else { return }

        context.delete(thought)

        do {
            try context.save()
            thoughts.removeAll { $0.id == thought.id }

            if selectedThought?.id == thought.id {
                selectedThought = nil
            }
        } catch {
            errorMessage = "Failed to delete thought: \(error.localizedDescription)"
        }
    }

    /// Updates a thought's position
    func updatePosition(for thought: Thought, position: CGPoint) {
        thought.positionX = position.x
        thought.positionY = position.y

        do {
            try modelContext?.save()
        } catch {
            errorMessage = "Failed to save position: \(error.localizedDescription)"
        }
    }

    /// Updates a thought's category
    func updateCategory(for thought: Thought, category: ThoughtCategory) {
        thought.category = category

        do {
            try modelContext?.save()
        } catch {
            errorMessage = "Failed to save category: \(error.localizedDescription)"
        }
    }

    /// Updates a thought's content
    func updateContent(for thought: Thought, content: String) {
        thought.content = content

        do {
            try modelContext?.save()
        } catch {
            errorMessage = "Failed to save content: \(error.localizedDescription)"
        }
    }

    // MARK: - Connection Operations

    /// Creates a connection between two thoughts
    func createConnection(from source: Thought, to target: Thought, strength: Double = 0.5) {
        guard let context = modelContext else { return }

        // Prevent duplicate connections
        let existingConnection = connections.first { conn in
            (conn.sourceThought?.id == source.id && conn.targetThought?.id == target.id) ||
            (conn.sourceThought?.id == target.id && conn.targetThought?.id == source.id)
        }
        guard existingConnection == nil else { return }

        let connection = Connection(
            sourceThought: source,
            targetThought: target,
            strength: strength
        )

        context.insert(connection)

        do {
            try context.save()
            connections.append(connection)
        } catch {
            errorMessage = "Failed to create connection: \(error.localizedDescription)"
        }
    }

    /// Deletes a connection
    func deleteConnection(_ connection: Connection) {
        guard let context = modelContext else { return }

        context.delete(connection)

        do {
            try context.save()
            connections.removeAll { $0.id == connection.id }
        } catch {
            errorMessage = "Failed to delete connection: \(error.localizedDescription)"
        }
    }

    // MARK: - Data Loading

    /// Loads all data from persistence
    func loadData() {
        isLoading = true

        do {
            thoughts = try DataService.shared.fetchAllThoughts()
            connections = try DataService.shared.fetchAllConnections()
        } catch {
            errorMessage = "Failed to load data: \(error.localizedDescription)"
        }

        isLoading = false
    }

    /// Refreshes data from persistence
    func refresh() {
        loadData()
    }

    // MARK: - Position Calculation

    /// Calculates a position for a new thought node
    private func calculateNewThoughtPosition() -> CGPoint {
        // Spiral pattern for new nodes
        let nodeCount = thoughts.count
        let angle = Double(nodeCount) * 0.5
        let baseRadius: Double = 150
        let radius = baseRadius + Double(nodeCount) * 20

        // Default center position (will be adjusted by scene)
        let centerX: Double = 400
        let centerY: Double = 400

        return CGPoint(
            x: centerX + cos(angle) * radius,
            y: centerY + sin(angle) * radius
        )
    }

    // MARK: - Utilities

    /// Clears any error message
    func clearError() {
        errorMessage = nil
    }

    /// Gets a thought by ID
    func thought(with id: UUID) -> Thought? {
        thoughts.first { $0.id == id }
    }

    /// Gets connections for a specific thought
    func connections(for thought: Thought) -> [Connection] {
        connections.filter { conn in
            conn.sourceThought?.id == thought.id || conn.targetThought?.id == thought.id
        }
    }
}
