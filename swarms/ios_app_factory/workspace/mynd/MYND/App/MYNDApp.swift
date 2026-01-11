//
//  MYNDApp.swift
//  MYND
//
//  Created on 2026-01-04.
//  iOS 17+, Swift 5.9+
//

import SwiftUI
import SwiftData

@main
struct MYNDApp: App {

    // MARK: - SwiftData Configuration

    /// The shared model container for all SwiftData persistence
    var sharedModelContainer: ModelContainer = {
        let schema = Schema([
            Thought.self,
            Connection.self,
            Cluster.self,
        ])

        let modelConfiguration = ModelConfiguration(
            schema: schema,
            isStoredInMemoryOnly: false,
            allowsSave: true
        )

        do {
            return try ModelContainer(
                for: schema,
                configurations: [modelConfiguration]
            )
        } catch {
            fatalError("Could not create ModelContainer: \(error)")
        }
    }()

    // MARK: - Body

    var body: some Scene {
        WindowGroup {
            ContentView()
        }
        .modelContainer(sharedModelContainer)
    }
}
