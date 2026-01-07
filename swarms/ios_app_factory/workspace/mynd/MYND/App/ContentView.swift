//
//  ContentView.swift
//  MYND
//
//  Created on 2026-01-04.
//  iOS 17+, Swift 5.9+
//

import SwiftUI
import SwiftData

/// Root view with tab navigation between MindMap and List views
struct ContentView: View {

    // MARK: - Environment

    @Environment(\.modelContext) private var modelContext

    // MARK: - State

    @State private var selectedTab: Tab = .mindMap
    @State private var viewModel = MindMapViewModel()

    // MARK: - Tab Enum

    enum Tab: Hashable {
        case mindMap
        case list
    }

    // MARK: - Body

    var body: some View {
        TabView(selection: $selectedTab) {
            // Mind Map Tab
            NavigationStack {
                VStack(spacing: 0) {
                    MindMapView(viewModel: viewModel)
                    ThoughtInputView(viewModel: viewModel)
                }
                .navigationTitle("Mind Map")
                .navigationBarTitleDisplayMode(.inline)
            }
            .tabItem {
                Label("Mind Map", systemImage: "circle.hexagongrid")
            }
            .tag(Tab.mindMap)

            // List Tab
            NavigationStack {
                VStack(spacing: 0) {
                    ThoughtListView(viewModel: viewModel)
                    ThoughtInputView(viewModel: viewModel)
                }
                .navigationTitle("Thoughts")
                .navigationBarTitleDisplayMode(.inline)
            }
            .tabItem {
                Label("List", systemImage: "list.bullet")
            }
            .tag(Tab.list)
        }
        .onAppear {
            // Configure ViewModel with SwiftData context
            viewModel.configure(with: modelContext)
        }
    }
}

#Preview {
    ContentView()
        .modelContainer(for: [Thought.self, Connection.self, Cluster.self], inMemory: true)
}
