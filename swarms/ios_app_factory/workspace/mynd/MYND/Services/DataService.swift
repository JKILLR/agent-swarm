//
//  DataService.swift
//  MYND
//
//  Created on 2026-01-04.
//  iOS 17+, Swift 5.9+
//

import Foundation
import SwiftData

/// Service handling all SwiftData CRUD operations
/// Provides a clean interface for persistence operations
@MainActor
final class DataService {

    // MARK: - Singleton

    static let shared = DataService()

    // MARK: - Properties

    private var modelContext: ModelContext?

    // MARK: - Initialization

    private init() {}

    // MARK: - Configuration

    /// Configure the service with a model context
    func configure(with context: ModelContext) {
        self.modelContext = context
    }

    // MARK: - Thought Operations

    /// Fetches all thoughts, sorted by creation date (newest first)
    func fetchAllThoughts() throws -> [Thought] {
        guard let context = modelContext else {
            throw DataServiceError.notConfigured
        }

        let descriptor = FetchDescriptor<Thought>(
            sortBy: [SortDescriptor(\.createdAt, order: .reverse)]
        )
        return try context.fetch(descriptor)
    }

    /// Fetches thoughts matching a search query
    func searchThoughts(query: String) throws -> [Thought] {
        guard let context = modelContext else {
            throw DataServiceError.notConfigured
        }

        let predicate = #Predicate<Thought> { thought in
            thought.content.localizedStandardContains(query)
        }

        let descriptor = FetchDescriptor<Thought>(
            predicate: predicate,
            sortBy: [SortDescriptor(\.createdAt, order: .reverse)]
        )
        return try context.fetch(descriptor)
    }

    /// Creates a new thought with the given content
    @discardableResult
    func createThought(
        content: String,
        category: ThoughtCategory = .note,
        positionX: Double = 0,
        positionY: Double = 0
    ) throws -> Thought {
        guard let context = modelContext else {
            throw DataServiceError.notConfigured
        }

        let thought = Thought(
            content: content,
            positionX: positionX,
            positionY: positionY,
            category: category
        )
        context.insert(thought)
        try context.save()
        return thought
    }

    /// Updates a thought's content
    func updateThoughtContent(_ thought: Thought, content: String) throws {
        guard let context = modelContext else {
            throw DataServiceError.notConfigured
        }

        thought.content = content
        try context.save()
    }

    /// Updates a thought's position
    func updateThoughtPosition(_ thought: Thought, x: Double, y: Double) throws {
        guard let context = modelContext else {
            throw DataServiceError.notConfigured
        }

        thought.positionX = x
        thought.positionY = y
        try context.save()
    }

    /// Updates a thought's category
    func updateThoughtCategory(_ thought: Thought, category: ThoughtCategory) throws {
        guard let context = modelContext else {
            throw DataServiceError.notConfigured
        }

        thought.category = category
        try context.save()
    }

    /// Deletes a thought
    func deleteThought(_ thought: Thought) throws {
        guard let context = modelContext else {
            throw DataServiceError.notConfigured
        }

        context.delete(thought)
        try context.save()
    }

    // MARK: - Connection Operations

    /// Fetches all connections
    func fetchAllConnections() throws -> [Connection] {
        guard let context = modelContext else {
            throw DataServiceError.notConfigured
        }

        let descriptor = FetchDescriptor<Connection>(
            sortBy: [SortDescriptor(\.createdAt, order: .reverse)]
        )
        return try context.fetch(descriptor)
    }

    /// Creates a connection between two thoughts
    @discardableResult
    func createConnection(
        source: Thought,
        target: Thought,
        strength: Double = 0.5
    ) throws -> Connection {
        guard let context = modelContext else {
            throw DataServiceError.notConfigured
        }

        let connection = Connection(
            sourceThought: source,
            targetThought: target,
            strength: strength
        )
        context.insert(connection)
        try context.save()
        return connection
    }

    /// Deletes a connection
    func deleteConnection(_ connection: Connection) throws {
        guard let context = modelContext else {
            throw DataServiceError.notConfigured
        }

        context.delete(connection)
        try context.save()
    }

    // MARK: - Cluster Operations

    /// Fetches all clusters
    func fetchAllClusters() throws -> [Cluster] {
        guard let context = modelContext else {
            throw DataServiceError.notConfigured
        }

        let descriptor = FetchDescriptor<Cluster>(
            sortBy: [SortDescriptor(\.createdAt, order: .reverse)]
        )
        return try context.fetch(descriptor)
    }

    /// Creates a new cluster
    @discardableResult
    func createCluster(name: String, colorHex: String = "#728C9E") throws -> Cluster {
        guard let context = modelContext else {
            throw DataServiceError.notConfigured
        }

        let cluster = Cluster(name: name, colorHex: colorHex)
        context.insert(cluster)
        try context.save()
        return cluster
    }

    /// Adds a thought to a cluster
    func addThoughtToCluster(_ thought: Thought, cluster: Cluster) throws {
        guard let context = modelContext else {
            throw DataServiceError.notConfigured
        }

        cluster.addThought(thought)
        try context.save()
    }

    /// Removes a thought from its cluster
    func removeThoughtFromCluster(_ thought: Thought, cluster: Cluster) throws {
        guard let context = modelContext else {
            throw DataServiceError.notConfigured
        }

        cluster.removeThought(thought)
        try context.save()
    }

    // MARK: - Batch Operations

    /// Saves all pending changes
    func save() throws {
        try modelContext?.save()
    }

    /// Deletes all data (for testing/reset)
    func deleteAllData() throws {
        guard let context = modelContext else {
            throw DataServiceError.notConfigured
        }

        try context.delete(model: Thought.self)
        try context.delete(model: Connection.self)
        try context.delete(model: Cluster.self)
    }
}

// MARK: - Error Types

enum DataServiceError: LocalizedError {
    case notConfigured
    case notFound
    case saveFailed

    var errorDescription: String? {
        switch self {
        case .notConfigured:
            return "DataService has not been configured with a ModelContext"
        case .notFound:
            return "The requested item was not found"
        case .saveFailed:
            return "Failed to save changes"
        }
    }
}
