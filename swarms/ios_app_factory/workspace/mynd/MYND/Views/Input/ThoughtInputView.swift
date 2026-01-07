//
//  ThoughtInputView.swift
//  MYND
//
//  Created on 2026-01-04.
//  iOS 17+, Swift 5.9+
//

import SwiftUI

/// Text field for quick thought capture
/// Designed for ADHD users - instant submission, minimal friction
struct ThoughtInputView: View {

    // MARK: - Properties

    @Bindable var viewModel: MindMapViewModel
    @State private var thoughtText: String = ""
    @FocusState private var isTextFieldFocused: Bool

    // MARK: - Body

    var body: some View {
        HStack(spacing: 12) {
            TextField("What's on your mind?", text: $thoughtText)
                .textFieldStyle(.plain)
                .padding(.horizontal, 16)
                .padding(.vertical, 12)
                .background(
                    RoundedRectangle(cornerRadius: 24)
                        .fill(Color(.systemGray6))
                )
                .focused($isTextFieldFocused)
                .submitLabel(.send)
                .onSubmit(submitThought)

            Button(action: submitThought) {
                Image(systemName: "arrow.up.circle.fill")
                    .font(.system(size: 32))
                    .foregroundStyle(thoughtText.isEmpty ? .gray : .blue)
            }
            .disabled(thoughtText.isEmpty)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
        .background(Color(.systemBackground))
    }

    // MARK: - Actions

    private func submitThought() {
        let trimmed = thoughtText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }

        viewModel.addThought(content: trimmed)
        thoughtText = ""

        // Keep keyboard up for rapid entry (ADHD-friendly)
        isTextFieldFocused = true
    }
}

#Preview {
    ThoughtInputView(viewModel: MindMapViewModel())
}
