//
//  AppConfig.swift
//  MYND
//
//  Created on 2026-01-04.
//  iOS 17+, Swift 5.9+
//

import Foundation

/// Application-wide configuration and feature flags
/// Centralizes all configurable settings for easy management
enum AppConfig {

    // MARK: - Debug Settings

    /// Shows FPS and node count in SpriteKit view
    static let showDebugInfo: Bool = {
        #if DEBUG
        return true
        #else
        return false
        #endif
    }()

    /// Enables verbose logging
    static let enableVerboseLogging: Bool = {
        #if DEBUG
        return true
        #else
        return false
        #endif
    }()

    // MARK: - Feature Flags

    /// Phase 0: Basic functionality
    static let enableBasicCapture = true
    static let enableMindMapView = true
    static let enableListView = true

    /// Phase 1: AI integration (disabled for Phase 0)
    static let enableAIAcknowledgments = false
    static let enableForceDirectedLayout = false

    /// Phase 2: AI organization (disabled for Phase 0)
    static let enableAutoCategories = false
    static let enableSemanticSearch = false
    static let enableAutoClustering = false

    /// Phase 3: Voice input (disabled for Phase 0)
    static let enableVoiceInput = false

    /// Phase 4: Cloud features (disabled for Phase 0)
    static let enableCloudSync = false
    static let enableUserAuth = false

    // MARK: - API Configuration

    /// Placeholder for Claude API key
    /// In production, this should be retrieved from Keychain
    static var claudeAPIKey: String? {
        // TODO: Implement secure key retrieval from Keychain
        return nil
    }

    /// Placeholder for Supabase configuration
    static var supabaseURL: String? {
        // TODO: Configure for production
        return nil
    }

    static var supabaseAnonKey: String? {
        // TODO: Configure for production
        return nil
    }

    // MARK: - UI Configuration

    /// Minimum nodes before enabling performance optimizations
    static let performanceOptimizationThreshold = 50

    /// Maximum nodes to render at once (for memory management)
    static let maxVisibleNodes = 500

    /// Default animation duration
    static let defaultAnimationDuration: TimeInterval = 0.3

    // MARK: - Graph Configuration

    /// Default spacing between nodes
    static let defaultNodeSpacing: CGFloat = 150

    /// Minimum zoom level
    static let minZoomScale: CGFloat = 0.5

    /// Maximum zoom level
    static let maxZoomScale: CGFloat = 3.0

    // MARK: - Persistence

    /// Whether to use in-memory storage (for testing)
    static let useInMemoryStorage: Bool = {
        #if DEBUG
        return ProcessInfo.processInfo.arguments.contains("-useInMemoryStorage")
        #else
        return false
        #endif
    }()
}
