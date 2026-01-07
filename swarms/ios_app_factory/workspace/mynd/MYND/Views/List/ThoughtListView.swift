//
//  ThoughtListView.swift
//  MYND
//
//  Created on 2026-01-04.
//  iOS 17+, Swift 5.9+
//

import SwiftUI

/// Simple list view showing all captured thoughts
/// Sorted by creation date (newest first)
struct ThoughtListView: View {

    // MARK: - Properties

    @Bindable var viewModel: MindMapViewModel

    // MARK: - Body

    var body: some View {
        Group {
            if viewModel.thoughts.isEmpty {
                emptyStateView
            } else {
                thoughtsList
            }
        }
    }

    // MARK: - Subviews

    private var emptyStateView: some View {
        ContentUnavailableView(
            "No Thoughts Yet",
            systemImage: "brain.head.profile",
            description: Text("Start capturing your thoughts using the text field below.")
        )
    }

    private var thoughtsList: some View {
        List {
            ForEach(viewModel.thoughts) { thought in
                ThoughtRowView(thought: thought)
                    .swipeActions(edge: .trailing, allowsFullSwipe: true) {
                        Button(role: .destructive) {
                            viewModel.deleteThought(thought)
                        } label: {
                            Label("Delete", systemImage: "trash")
                        }
                    }
            }
        }
        .listStyle(.plain)
    }
}

// MARK: - Thought Row View

struct ThoughtRowView: View {
    let thought: Thought

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(thought.content)
                .font(.body)
                .lineLimit(3)

            HStack {
                Label(thought.category.displayName, systemImage: thought.category.iconName)
                    .font(.caption)
                    .foregroundStyle(thought.category.color)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 2)
                    .background(
                        Capsule()
                            .fill(thought.category.color.opacity(0.15))
                    )

                Spacer()

                Text(thought.createdAt, style: .relative)
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }
        }
        .padding(.vertical, 4)
    }
}

#Preview {
    NavigationStack {
        ThoughtListView(viewModel: MindMapViewModel())
    }
}
