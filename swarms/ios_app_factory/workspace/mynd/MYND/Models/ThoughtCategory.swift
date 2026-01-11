//
//  ThoughtCategory.swift
//  MYND
//
//  Created on 2026-01-04.
//  iOS 17+, Swift 5.9+
//

import Foundation
import SwiftUI

/// Categories for organizing thoughts
enum ThoughtCategory: String, Codable, CaseIterable {
    case idea = "idea"
    case task = "task"
    case note = "note"
    case question = "question"
    case insight = "insight"
    case memory = "memory"
    case goal = "goal"
    case reflection = "reflection"

    /// Display name for the category
    var displayName: String {
        rawValue.capitalized
    }

    /// Associated color for visual distinction
    var color: Color {
        switch self {
        case .idea: return .yellow
        case .task: return .blue
        case .note: return .gray
        case .question: return .purple
        case .insight: return .orange
        case .memory: return .green
        case .goal: return .red
        case .reflection: return .cyan
        }
    }

    /// SF Symbol icon name
    var iconName: String {
        switch self {
        case .idea: return "lightbulb.fill"
        case .task: return "checkmark.circle"
        case .note: return "note.text"
        case .question: return "questionmark.circle"
        case .insight: return "sparkles"
        case .memory: return "brain.head.profile"
        case .goal: return "target"
        case .reflection: return "bubble.left.and.bubble.right"
        }
    }
}
