# MYND - Technical Architecture Document v2.0

**Author**: System Architect
**Date**: 2026-01-04
**Version**: 2.0
**Status**: COMPREHENSIVE DESIGN - Addressing Critical Reviews

---

## Executive Summary

MYND is a native iOS/macOS application for voice-first AI thought capture with a persistent memory system. The AI assistant "Axel" engages in two-way conversation, learns user patterns over time, and proactively follows up on goals and thoughts.

### Core Value Proposition
- **Voice-First**: Natural conversation as primary input (not command-response)
- **Persistent Memory**: AI learns and remembers across all sessions
- **Knowledge Graph**: Relationships between thoughts, goals, and ideas
- **Proactive Follow-ups**: AI initiates conversations about unfinished items
- **Executive Function Support**: Designed for people with scattered thoughts

### Critical Design Decisions (from v1.0 Review)

| Issue | v1.0 Problem | v2.0 Solution |
|-------|--------------|---------------|
| Wake Word | iOS doesn't allow third-party always-listening | Push-to-talk + Siri Shortcuts integration |
| Voice Latency | 1-2s response time feels broken | Streaming + optimistic UI + "thinking" indicators |
| Knowledge Graph | SwiftData can't scale for graph operations | Hybrid: SwiftData for storage + in-memory graph for queries |
| BYOK Onboarding | Too complex for target audience | Tiered: Demo mode → Managed → BYOK for power users |

---

## 1. System Architecture

### 1.1 High-Level Architecture

```
+--------------------------------------------------------------------------+
|                              CLIENT LAYER                                  |
+--------------------------------------------------------------------------+
|                                                                            |
|   +---------------------------+      +---------------------------+         |
|   |      iOS App (Swift)      |      |     macOS App (Swift)     |         |
|   |                           |      |                           |         |
|   |  +---------------------+  |      |  +---------------------+  |         |
|   |  | Voice Interface     |  |      |  | Voice Interface     |  |         |
|   |  | (Push-to-Talk)      |  |      |  | (Push-to-Talk)      |  |         |
|   |  +---------------------+  |      |  +---------------------+  |         |
|   |  +---------------------+  |      |  +---------------------+  |         |
|   |  | Conversation UI     |  |      |  | Conversation UI     |  |         |
|   |  +---------------------+  |      |  +---------------------+  |         |
|   |  +---------------------+  |      |  +---------------------+  |         |
|   |  | Knowledge Graph     |  |      |  | Knowledge Graph     |  |         |
|   |  | Visualization       |  |      |  | Visualization       |  |         |
|   |  +---------------------+  |      |  +---------------------+  |         |
|   +---------------------------+      +---------------------------+         |
|               |                                  |                         |
|               +----------------------------------+                         |
|                               |                                            |
+--------------------------------------------------------------------------+
                                |
+--------------------------------------------------------------------------+
|                        LOCAL SERVICES LAYER                                |
+--------------------------------------------------------------------------+
|                                                                            |
|  +-------------------+  +-------------------+  +-------------------+       |
|  | Voice Engine      |  | Graph Engine      |  | Memory Engine     |       |
|  |                   |  |                   |  |                   |       |
|  | - Apple Speech    |  | - In-Memory Graph |  | - Embedding Store |       |
|  | - AVSpeech TTS    |  | - SwiftData Sync  |  | - Semantic Search |       |
|  | - Audio Session   |  | - Query Engine    |  | - Pattern Learning|       |
|  +-------------------+  +-------------------+  +-------------------+       |
|                                                                            |
|  +-------------------+  +-------------------+  +-------------------+       |
|  | Conversation      |  | Proactive Engine  |  | Sync Engine       |       |
|  | Engine            |  |                   |  |                   |       |
|  | - LLM Client      |  | - Goal Tracker    |  | - CloudKit        |       |
|  | - Context Builder |  | - Notifications   |  | - Conflict Merge  |       |
|  | - Stream Handler  |  | - Insights        |  | - Offline Queue   |       |
|  +-------------------+  +-------------------+  +-------------------+       |
|                                                                            |
+--------------------------------------------------------------------------+
                                |
+--------------------------------------------------------------------------+
|                          PERSISTENCE LAYER                                 |
+--------------------------------------------------------------------------+
|                                                                            |
|  +-------------------------------+  +-------------------------------+      |
|  | SwiftData Store              |  | In-Memory Graph Cache          |      |
|  | (Source of Truth)            |  | (Fast Queries)                 |      |
|  |                               |  |                                |      |
|  | - ThoughtNode (indexed)      |  | - Adjacency lists             |      |
|  | - Edge (indexed)             |  | - Precomputed paths           |      |
|  | - ConversationSession        |  | - Clustering cache            |      |
|  | - MemoryItem                 |  | - Rebuild on launch           |      |
|  +-------------------------------+  +-------------------------------+      |
|                                                                            |
+--------------------------------------------------------------------------+
                                |
+--------------------------------------------------------------------------+
|                            CLOUD SERVICES                                  |
+--------------------------------------------------------------------------+
|                                                                            |
|  +-------------------+  +-------------------+  +-------------------+       |
|  | Claude API        |  | CloudKit Private  |  | Optional:         |       |
|  | (Anthropic)       |  | Database          |  | ElevenLabs TTS    |       |
|  |                   |  |                   |  | OpenAI Whisper    |       |
|  | - Streaming       |  | - End-to-end      |  |                   |       |
|  | - Tool Use        |  |   encrypted       |  |                   |       |
|  +-------------------+  +-------------------+  +-------------------+       |
|                                                                            |
+--------------------------------------------------------------------------+
```

### 1.2 Architecture Patterns

| Pattern | Choice | Rationale |
|---------|--------|-----------|
| **App Architecture** | MVVM + Observation | SwiftUI native, testable, simple |
| **Dependency Injection** | Protocol-based DI container | Enables testing, swappable implementations |
| **Reactive** | Combine + @Observable | Native, no third-party dependencies |
| **Persistence** | Repository pattern over SwiftData | Enables future migration to SQLite |
| **Networking** | async/await with URLSession | Modern, built-in cancellation |
| **Graph Queries** | In-memory with SwiftData backing | Fast queries, persistent storage |

### 1.3 Module Structure

```
MYND/
├── App/
│   ├── MYNDApp.swift                    # Entry point
│   ├── AppState.swift                   # Global state container
│   └── DependencyContainer.swift        # DI registration
│
├── Features/
│   ├── Conversation/
│   │   ├── Views/
│   │   │   ├── ConversationView.swift
│   │   │   ├── MessageBubble.swift
│   │   │   ├── VoiceInputButton.swift
│   │   │   └── StreamingResponseView.swift
│   │   ├── ViewModels/
│   │   │   └── ConversationViewModel.swift
│   │   └── ConversationFeature.swift    # Feature entry point
│   │
│   ├── KnowledgeGraph/
│   │   ├── Views/
│   │   │   ├── GraphView.swift          # 2D visualization
│   │   │   ├── NodeDetailView.swift
│   │   │   ├── NodeListView.swift
│   │   │   └── GraphFilterView.swift
│   │   ├── ViewModels/
│   │   │   └── GraphViewModel.swift
│   │   └── Visualization/
│   │       ├── ForceDirectedLayout.swift
│   │       └── GraphRenderer.swift
│   │
│   ├── Proactive/
│   │   ├── Views/
│   │   │   ├── MorningBriefingView.swift
│   │   │   └── InsightCardView.swift
│   │   └── ViewModels/
│   │       └── ProactiveViewModel.swift
│   │
│   └── Settings/
│       ├── Views/
│       │   ├── SettingsView.swift
│       │   ├── VoiceSettingsView.swift
│       │   ├── PrivacySettingsView.swift
│       │   └── APIKeySetupView.swift
│       └── ViewModels/
│           └── SettingsViewModel.swift
│
├── Core/
│   ├── Voice/
│   │   ├── VoiceEngine.swift            # Orchestrates voice I/O
│   │   ├── SpeechRecognizer.swift       # Apple Speech wrapper
│   │   ├── SpeechRecognizerProtocol.swift
│   │   ├── SpeechSynthesizer.swift      # TTS wrapper
│   │   ├── SpeechSynthesizerProtocol.swift
│   │   ├── AudioSessionManager.swift
│   │   └── VoiceError.swift
│   │
│   ├── AI/
│   │   ├── ConversationEngine.swift     # LLM interaction orchestrator
│   │   ├── LLMClient.swift              # API abstraction
│   │   ├── LLMClientProtocol.swift
│   │   ├── ClaudeClient.swift           # Anthropic implementation
│   │   ├── ContextBuilder.swift         # Memory + context assembly
│   │   ├── SystemPromptBuilder.swift    # Axel personality
│   │   └── StreamingHandler.swift
│   │
│   ├── Memory/
│   │   ├── MemoryEngine.swift           # Memory orchestration
│   │   ├── EmbeddingEngine.swift        # Text → vector
│   │   ├── EmbeddingEngineProtocol.swift
│   │   ├── SemanticSearch.swift         # Vector similarity
│   │   ├── PatternLearner.swift         # User pattern detection
│   │   └── MemoryConsolidator.swift     # Episodic → long-term
│   │
│   ├── Graph/
│   │   ├── KnowledgeGraph.swift         # Graph operations
│   │   ├── InMemoryGraph.swift          # Fast query graph
│   │   ├── GraphQueryEngine.swift       # Traversal, pathfinding
│   │   ├── EntityExtractor.swift        # Extract entities from text
│   │   └── RelationshipDetector.swift   # Detect connections
│   │
│   └── Proactive/
│       ├── ProactiveEngine.swift        # Background processing
│       ├── GoalTracker.swift
│       ├── InsightGenerator.swift
│       ├── ReminderScheduler.swift
│       └── NotificationManager.swift
│
├── Data/
│   ├── Models/
│   │   ├── ThoughtNode.swift            # SwiftData model
│   │   ├── Edge.swift                   # SwiftData model
│   │   ├── Message.swift                # SwiftData model
│   │   ├── ConversationSession.swift    # SwiftData model
│   │   ├── MemoryItem.swift             # SwiftData model
│   │   ├── UserPattern.swift            # SwiftData model
│   │   └── Enums/
│   │       ├── NodeType.swift
│   │       ├── EdgeType.swift
│   │       ├── MemoryType.swift
│   │       └── PatternType.swift
│   │
│   ├── Repositories/
│   │   ├── NodeRepository.swift
│   │   ├── NodeRepositoryProtocol.swift
│   │   ├── EdgeRepository.swift
│   │   ├── SessionRepository.swift
│   │   └── MemoryRepository.swift
│   │
│   ├── Persistence/
│   │   ├── SwiftDataManager.swift       # SwiftData container
│   │   ├── MigrationManager.swift       # Schema versioning
│   │   └── ModelContainer+Extensions.swift
│   │
│   └── Sync/
│       ├── CloudKitSync.swift
│       ├── ConflictResolver.swift
│       └── OfflineQueue.swift
│
├── Security/
│   ├── SecureStorage.swift              # Keychain wrapper
│   ├── APIKeyManager.swift
│   └── EncryptionManager.swift
│
├── Shared/
│   ├── Extensions/
│   │   ├── Date+Extensions.swift
│   │   ├── String+Extensions.swift
│   │   └── View+Extensions.swift
│   ├── Utilities/
│   │   ├── TokenCounter.swift
│   │   └── Logger.swift
│   └── Constants.swift
│
├── Widgets/
│   ├── QuickCaptureWidget.swift         # Lock Screen widget
│   ├── GoalProgressWidget.swift         # Home Screen widget
│   └── WidgetData.swift
│
└── Resources/
    ├── Assets.xcassets
    ├── Localizable.strings
    └── Sounds/
        ├── thinking.wav                 # Feedback sounds
        └── complete.wav
```

---

## 2. Voice Conversation System

### 2.1 Design Philosophy

> **Critical Learning from v1.0 Review**: Wake word ("Hey Axel") is technically infeasible on iOS for third-party apps. Apple restricts background audio access. We must design around this constraint.

**Interaction Model**: Push-to-Talk with Natural Conversation

```
+------------------+     +-----------------+     +------------------+
|   User taps      | --> | Recognition     | --> | Claude processes |
|   microphone     |     | streams text    |     | + streams back   |
+------------------+     +-----------------+     +------------------+
        ^                                                  |
        |                                                  v
+------------------+     +-----------------+     +------------------+
|   User can       | <-- | TTS speaks      | <-- | Response ready   |
|   interrupt      |     | with streaming  |     |                  |
+------------------+     +-----------------+     +------------------+
```

### 2.2 Voice Latency Mitigation

**Problem**: Claude API cold start = 200-500ms. Full round-trip = 1-2s. This feels broken.

**Solution**: Multi-pronged latency reduction

```swift
// 1. Optimistic UI - Show activity immediately
enum ConversationState {
    case idle
    case listening(transcript: String)      // Real-time transcript
    case thinking                            // Show "thinking" indicator
    case streaming(partial: String)          // Stream response as it arrives
    case speaking                            // TTS playing
}

// 2. Audio Feedback - User knows something is happening
class LatencyMitigator {
    private let thinkingSound = SystemSoundID(1057)  // Subtle tick

    func beginProcessing() {
        // Play subtle audio cue
        AudioServicesPlaySystemSound(thinkingSound)

        // Haptic feedback
        let generator = UIImpactFeedbackGenerator(style: .soft)
        generator.impactOccurred()
    }
}

// 3. Pre-generated Acknowledgments - Immediate response
enum QuickAcknowledgment: String, CaseIterable {
    case gotIt = "Got it, let me think about that..."
    case interesting = "That's interesting, give me a moment..."
    case understood = "I hear you..."
    case processing = "Processing that thought..."

    static func random() -> String {
        allCases.randomElement()?.rawValue ?? "Hmm..."
    }
}

// 4. Streaming TTS - Speak as tokens arrive
class StreamingTTSHandler {
    private let synthesizer = AVSpeechSynthesizer()
    private var sentenceBuffer = ""

    func processStreamChunk(_ chunk: String) {
        sentenceBuffer += chunk

        // Speak complete sentences as they form
        if let sentence = extractCompleteSentence() {
            speakSentence(sentence)
        }
    }

    private func extractCompleteSentence() -> String? {
        // Extract and speak complete sentences immediately
        let sentenceEnders = [".", "!", "?"]
        for ender in sentenceEnders {
            if let range = sentenceBuffer.range(of: ender) {
                let sentence = String(sentenceBuffer[..<range.upperBound])
                sentenceBuffer = String(sentenceBuffer[range.upperBound...])
                return sentence.trimmingCharacters(in: .whitespaces)
            }
        }
        return nil
    }
}
```

### 2.3 Voice Engine Architecture

```swift
// MARK: - Protocols for Testing

protocol SpeechRecognizerProtocol {
    func transcribe() -> AsyncThrowingStream<TranscriptResult, Error>
    func stop()
    var isAvailable: Bool { get }
}

protocol SpeechSynthesizerProtocol {
    func speak(_ text: String, style: VoiceStyle, rate: Float) async
    func stopSpeaking()
    var isSpeaking: Bool { get }
}

// MARK: - Voice Engine

@Observable
final class VoiceEngine {
    // Dependencies (injected for testing)
    private let speechRecognizer: SpeechRecognizerProtocol
    private let speechSynthesizer: SpeechSynthesizerProtocol
    private let audioSessionManager: AudioSessionManager
    private let latencyMitigator: LatencyMitigator

    // State
    private(set) var state: VoiceState = .idle
    private(set) var currentTranscript: String = ""
    private(set) var recognitionConfidence: Float = 0.0

    // Configuration
    var voiceStyle: VoiceStyle = .natural
    var speakingRate: Float = 0.5  // User adjustable

    enum VoiceState: Equatable {
        case idle
        case listening
        case processing
        case speaking
        case error(String)
    }

    init(
        speechRecognizer: SpeechRecognizerProtocol = AppleSpeechRecognizer(),
        speechSynthesizer: SpeechSynthesizerProtocol = AppleSpeechSynthesizer(),
        audioSessionManager: AudioSessionManager = .shared,
        latencyMitigator: LatencyMitigator = LatencyMitigator()
    ) {
        self.speechRecognizer = speechRecognizer
        self.speechSynthesizer = speechSynthesizer
        self.audioSessionManager = audioSessionManager
        self.latencyMitigator = latencyMitigator
    }

    // MARK: - Public API

    func startListening() async throws {
        // If speaking, interrupt first
        if state == .speaking {
            stopSpeaking()
            // Small delay for audio session transition
            try await Task.sleep(nanoseconds: 100_000_000)
        }

        guard state != .listening else { return }

        try await audioSessionManager.activateForRecording()
        state = .listening
        currentTranscript = ""

        do {
            for try await result in speechRecognizer.transcribe() {
                currentTranscript = result.text
                recognitionConfidence = result.confidence

                if result.isFinal {
                    state = .processing
                    latencyMitigator.beginProcessing()
                }
            }
        } catch {
            state = .error(error.localizedDescription)
            throw error
        }
    }

    func stopListening() {
        speechRecognizer.stop()
        if state == .listening {
            state = .idle
        }
    }

    func speak(_ text: String, priority: SpeechPriority = .normal) async {
        guard !text.isEmpty else { return }

        // Handle interruption
        if state == .speaking && priority == .interrupt {
            stopSpeaking()
        }

        guard state != .speaking || priority == .interrupt else { return }

        do {
            try await audioSessionManager.activateForPlayback()
            state = .speaking
            await speechSynthesizer.speak(text, style: voiceStyle, rate: speakingRate)
            state = .idle
        } catch {
            state = .error(error.localizedDescription)
        }
    }

    func stopSpeaking() {
        speechSynthesizer.stopSpeaking()
        state = .idle
    }

    /// Stream response and speak sentences as they complete
    func streamAndSpeak(_ stream: AsyncThrowingStream<String, Error>) async throws {
        state = .speaking

        var buffer = ""
        let sentenceEnders: Set<Character> = [".", "!", "?"]

        try await audioSessionManager.activateForPlayback()

        for try await chunk in stream {
            buffer += chunk

            // Find complete sentences
            while let enderIndex = buffer.firstIndex(where: { sentenceEnders.contains($0) }) {
                let sentenceEnd = buffer.index(after: enderIndex)
                let sentence = String(buffer[..<sentenceEnd]).trimmingCharacters(in: .whitespaces)
                buffer = String(buffer[sentenceEnd...])

                if !sentence.isEmpty {
                    await speechSynthesizer.speak(sentence, style: voiceStyle, rate: speakingRate)
                }
            }
        }

        // Speak any remaining text
        let remaining = buffer.trimmingCharacters(in: .whitespaces)
        if !remaining.isEmpty {
            await speechSynthesizer.speak(remaining, style: voiceStyle, rate: speakingRate)
        }

        state = .idle
    }
}

// MARK: - Supporting Types

struct TranscriptResult {
    let text: String
    let confidence: Float
    let isFinal: Bool
}

enum SpeechPriority {
    case normal
    case interrupt  // Stops current speech
}

enum VoiceStyle: String, CaseIterable, Codable {
    case natural
    case calm
    case energetic

    var pitch: Float {
        switch self {
        case .natural: return 1.0
        case .calm: return 0.95
        case .energetic: return 1.05
        }
    }
}
```

### 2.4 Speech Recognition Options

| Option | Latency | Accuracy | Privacy | Offline | Cost |
|--------|---------|----------|---------|---------|------|
| **Apple Speech (on-device)** | Real-time | 90-95% | On-device | Yes | Free |
| **Apple Speech (server)** | 100-200ms | 95%+ | Cloud | No | Free |
| **OpenAI Whisper API** | 200-500ms | 98%+ | Cloud | No | $0.006/min |
| **On-device Whisper** | 500ms+ | 95%+ | On-device | Yes | Free |

**Recommendation**: Apple Speech (on-device) as default, with user option for Whisper API for better accuracy.

```swift
// MARK: - Apple Speech Recognizer Implementation

final class AppleSpeechRecognizer: SpeechRecognizerProtocol {
    private let recognizer: SFSpeechRecognizer
    private var recognitionTask: SFSpeechRecognitionTask?
    private var audioEngine: AVAudioEngine?

    var preferOnDevice: Bool = true

    var isAvailable: Bool {
        SFSpeechRecognizer.authorizationStatus() == .authorized &&
        (recognizer.isAvailable || recognizer.supportsOnDeviceRecognition)
    }

    init(locale: Locale = .current) {
        self.recognizer = SFSpeechRecognizer(locale: locale) ?? SFSpeechRecognizer()!
    }

    func transcribe() -> AsyncThrowingStream<TranscriptResult, Error> {
        AsyncThrowingStream { continuation in
            Task { [weak self] in
                guard let self else { return }

                guard await SFSpeechRecognizer.hasAuthorizationToRecognize() else {
                    continuation.finish(throwing: VoiceError.notAuthorized)
                    return
                }

                let audioEngine = AVAudioEngine()
                self.audioEngine = audioEngine

                let request = SFSpeechAudioBufferRecognitionRequest()
                request.shouldReportPartialResults = true

                // Prefer on-device for privacy
                if preferOnDevice && recognizer.supportsOnDeviceRecognition {
                    request.requiresOnDeviceRecognition = true
                }

                let inputNode = audioEngine.inputNode
                let recordingFormat = inputNode.outputFormat(forBus: 0)

                inputNode.installTap(onBus: 0, bufferSize: 1024, format: recordingFormat) { buffer, _ in
                    request.append(buffer)
                }

                audioEngine.prepare()
                try audioEngine.start()

                self.recognitionTask = recognizer.recognitionTask(with: request) { [weak self] result, error in
                    if let result {
                        let transcript = TranscriptResult(
                            text: result.bestTranscription.formattedString,
                            confidence: result.bestTranscription.segments.last?.confidence ?? 0,
                            isFinal: result.isFinal
                        )
                        continuation.yield(transcript)

                        if result.isFinal {
                            self?.cleanup()
                            continuation.finish()
                        }
                    }

                    if let error {
                        self?.cleanup()
                        continuation.finish(throwing: error)
                    }
                }
            }
        }
    }

    func stop() {
        recognitionTask?.cancel()
        cleanup()
    }

    private func cleanup() {
        audioEngine?.stop()
        audioEngine?.inputNode.removeTap(onBus: 0)
        audioEngine = nil
        recognitionTask = nil
    }
}
```

### 2.5 Text-to-Speech Options

| Option | Quality | Latency | Cost | Offline |
|--------|---------|---------|------|---------|
| **Apple AVSpeech (enhanced)** | Good | Instant | Free | Yes |
| **ElevenLabs API** | Excellent | 200-500ms | $0.30/1K chars | No |
| **OpenAI TTS** | Very Good | 200ms | $0.015/1K chars | No |

**Recommendation**: Apple AVSpeech for MVP (free, works offline). ElevenLabs as premium option for users who value natural voice.

```swift
// MARK: - Apple Speech Synthesizer

final class AppleSpeechSynthesizer: NSObject, SpeechSynthesizerProtocol, AVSpeechSynthesizerDelegate {
    private let synthesizer = AVSpeechSynthesizer()
    private var continuation: CheckedContinuation<Void, Never>?

    private(set) var isSpeaking: Bool = false

    override init() {
        super.init()
        synthesizer.delegate = self
    }

    func speak(_ text: String, style: VoiceStyle, rate: Float) async {
        guard !text.isEmpty else { return }

        await withCheckedContinuation { continuation in
            self.continuation = continuation
            isSpeaking = true

            let utterance = AVSpeechUtterance(string: text)
            utterance.rate = AVSpeechUtteranceDefaultSpeechRate * rate
            utterance.pitchMultiplier = style.pitch
            utterance.volume = 1.0

            // Use best available voice
            if let voice = selectBestVoice(for: style) {
                utterance.voice = voice
            }

            synthesizer.speak(utterance)
        }
    }

    func stopSpeaking() {
        synthesizer.stopSpeaking(at: .word)  // Stop at word boundary for naturalness
        finishContinuation()
    }

    private func selectBestVoice(for style: VoiceStyle) -> AVSpeechSynthesisVoice? {
        let voices = AVSpeechSynthesisVoice.speechVoices()

        // Prefer premium/enhanced voices
        let premiumVoice = voices.first { voice in
            voice.language.starts(with: "en") &&
            voice.quality == .premium
        }

        if let premium = premiumVoice {
            return premium
        }

        // Fallback to enhanced
        let enhancedVoice = voices.first { voice in
            voice.language.starts(with: "en") &&
            voice.quality == .enhanced
        }

        return enhancedVoice ?? AVSpeechSynthesisVoice(language: "en-US")
    }

    // MARK: - AVSpeechSynthesizerDelegate

    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
        finishContinuation()
    }

    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didCancel utterance: AVSpeechUtterance) {
        finishContinuation()
    }

    private func finishContinuation() {
        isSpeaking = false
        continuation?.resume()
        continuation = nil
    }
}
```

### 2.6 Interruption Handling

User should be able to interrupt Axel at any time by tapping the microphone.

```swift
// In ConversationViewModel

func handleMicrophoneTap() async throws {
    switch voiceEngine.state {
    case .idle:
        // Start listening
        try await voiceEngine.startListening()

    case .listening:
        // User is done talking, stop and process
        voiceEngine.stopListening()

    case .speaking:
        // User interrupts Axel - stop speaking, start listening
        voiceEngine.stopSpeaking()
        try await voiceEngine.startListening()

    case .processing:
        // Cancel current processing, start over
        cancelCurrentRequest()
        try await voiceEngine.startListening()

    case .error:
        // Retry
        try await voiceEngine.startListening()
    }
}
```

---

## 3. Knowledge Graph System

### 3.1 The Scalability Problem and Solution

**v1.0 Problem**: SwiftData relationships don't support efficient graph traversal. O(N) relationship loading, no native graph queries.

**v2.0 Solution**: Hybrid architecture

```
+---------------------------+      +---------------------------+
|     SwiftData             |      |     In-Memory Graph       |
|     (Persistence)         | <==> |     (Fast Queries)        |
+---------------------------+      +---------------------------+
|                           |      |                           |
|  - Source of truth        |      |  - Adjacency lists        |
|  - CloudKit sync          |      |  - O(1) neighbor lookup   |
|  - Indexed fields         |      |  - BFS/DFS traversal      |
|  - Disk persistence       |      |  - Clustering             |
|                           |      |  - Rebuilt on launch      |
+---------------------------+      +---------------------------+
```

### 3.2 SwiftData Models (Optimized)

```swift
import SwiftData
import Foundation

// MARK: - Node Types

enum NodeType: String, Codable, CaseIterable {
    case thought     // General captured thought
    case goal        // Something user wants to achieve
    case action      // A concrete next step
    case project     // Collection of related goals/actions
    case person      // Person mentioned
    case place       // Location
    case event       // Time-bound occurrence
    case insight     // AI-generated insight
    case question    // Open question to revisit
}

// MARK: - Edge Types

enum EdgeType: String, Codable, CaseIterable {
    case relatesTo   // General relationship
    case blocks      // Source blocks target
    case enables     // Source enables target
    case partOf      // Source is part of target
    case follows     // Source comes after target (temporal)
    case hasAction   // Goal -> Action relationship
    case mentions    // Thought mentions Person/Place
    case inspiredBy  // Source inspired by target
}

// MARK: - ThoughtNode Model

@Model
final class ThoughtNode {
    // Primary identifier
    @Attribute(.unique)
    var id: UUID

    // Core content
    var content: String
    var nodeType: NodeType

    // Timestamps (indexed for fast filtering)
    @Attribute(.indexed)
    var createdAt: Date

    @Attribute(.indexed)
    var lastAccessedAt: Date

    // Type index for filtering
    @Attribute(.indexed)
    var nodeTypeRaw: String  // For index-based filtering

    // Embedding for semantic search (stored externally for large data)
    @Attribute(.externalStorage)
    var embeddingData: Data?

    // Metadata
    var metadata: [String: String]?

    // Sync metadata
    var cloudKitRecordID: String?
    var lastSyncedAt: Date?
    var needsSync: Bool = false

    // Completion tracking (for goals/actions)
    var isCompleted: Bool = false
    var completedAt: Date?

    // Relationships - use ID-based for sync safety
    // Note: Actual graph traversal uses InMemoryGraph

    init(content: String, nodeType: NodeType) {
        self.id = UUID()
        self.content = content
        self.nodeType = nodeType
        self.nodeTypeRaw = nodeType.rawValue
        self.createdAt = Date()
        self.lastAccessedAt = Date()
    }

    // Computed property for embedding
    var embedding: [Float]? {
        get {
            guard let data = embeddingData else { return nil }
            return data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        }
        set {
            guard let floats = newValue else {
                embeddingData = nil
                return
            }
            embeddingData = floats.withUnsafeBytes { Data($0) }
        }
    }
}

// MARK: - Edge Model

@Model
final class Edge {
    @Attribute(.unique)
    var id: UUID

    var edgeType: EdgeType
    var weight: Float  // Strength of connection (0.0 - 1.0)

    @Attribute(.indexed)
    var createdAt: Date

    // ID-based references (safer for sync)
    @Attribute(.indexed)
    var sourceId: UUID

    @Attribute(.indexed)
    var targetId: UUID

    // Sync metadata
    var cloudKitRecordID: String?
    var needsSync: Bool = false

    init(edgeType: EdgeType, sourceId: UUID, targetId: UUID, weight: Float = 1.0) {
        self.id = UUID()
        self.edgeType = edgeType
        self.sourceId = sourceId
        self.targetId = targetId
        self.weight = weight
        self.createdAt = Date()
    }
}
```

### 3.3 In-Memory Graph for Fast Queries

```swift
import Foundation

/// Fast in-memory graph for O(1) neighbor lookups and graph algorithms
@Observable
final class InMemoryGraph {
    // Adjacency lists
    private var outgoing: [UUID: [(UUID, EdgeInfo)]] = [:]  // node -> [(neighbor, edge)]
    private var incoming: [UUID: [(UUID, EdgeInfo)]] = [:]  // node -> [(neighbor, edge)]

    // Node cache for quick access
    private var nodeCache: [UUID: NodeInfo] = [:]

    // Computed clusters (lazy rebuild)
    private var clusters: [[UUID]]?
    private var clustersDirty = true

    struct NodeInfo {
        let id: UUID
        let content: String
        let nodeType: NodeType
        var embedding: [Float]?
    }

    struct EdgeInfo {
        let id: UUID
        let edgeType: EdgeType
        let weight: Float
    }

    // MARK: - Building the Graph

    /// Rebuild graph from SwiftData (call on app launch)
    func rebuild(nodes: [ThoughtNode], edges: [Edge]) {
        outgoing.removeAll(keepingCapacity: true)
        incoming.removeAll(keepingCapacity: true)
        nodeCache.removeAll(keepingCapacity: true)

        // Index nodes
        for node in nodes {
            nodeCache[node.id] = NodeInfo(
                id: node.id,
                content: node.content,
                nodeType: node.nodeType,
                embedding: node.embedding
            )
            outgoing[node.id] = []
            incoming[node.id] = []
        }

        // Index edges
        for edge in edges {
            let info = EdgeInfo(id: edge.id, edgeType: edge.edgeType, weight: edge.weight)
            outgoing[edge.sourceId, default: []].append((edge.targetId, info))
            incoming[edge.targetId, default: []].append((edge.sourceId, info))
        }

        clustersDirty = true
    }

    /// Incrementally add a node
    func addNode(_ node: ThoughtNode) {
        nodeCache[node.id] = NodeInfo(
            id: node.id,
            content: node.content,
            nodeType: node.nodeType,
            embedding: node.embedding
        )
        outgoing[node.id] = []
        incoming[node.id] = []
        clustersDirty = true
    }

    /// Incrementally add an edge
    func addEdge(_ edge: Edge) {
        let info = EdgeInfo(id: edge.id, edgeType: edge.edgeType, weight: edge.weight)
        outgoing[edge.sourceId, default: []].append((edge.targetId, info))
        incoming[edge.targetId, default: []].append((edge.sourceId, info))
        clustersDirty = true
    }

    // MARK: - Query Operations

    /// O(1) neighbor lookup
    func neighbors(of nodeId: UUID, direction: Direction = .both) -> [(UUID, EdgeInfo)] {
        var result: [(UUID, EdgeInfo)] = []

        if direction == .outgoing || direction == .both {
            result.append(contentsOf: outgoing[nodeId] ?? [])
        }

        if direction == .incoming || direction == .both {
            result.append(contentsOf: incoming[nodeId] ?? [])
        }

        return result
    }

    enum Direction {
        case outgoing
        case incoming
        case both
    }

    /// BFS to find related nodes within depth
    func findRelated(to nodeId: UUID, depth: Int = 2, limit: Int = 50) -> [UUID] {
        var visited: Set<UUID> = [nodeId]
        var result: [UUID] = []
        var frontier: [UUID] = [nodeId]

        for _ in 0..<depth {
            var nextFrontier: [UUID] = []

            for current in frontier {
                for (neighbor, _) in neighbors(of: current) {
                    if !visited.contains(neighbor) {
                        visited.insert(neighbor)
                        result.append(neighbor)
                        nextFrontier.append(neighbor)

                        if result.count >= limit {
                            return result
                        }
                    }
                }
            }

            frontier = nextFrontier
        }

        return result
    }

    /// Find shortest path between two nodes
    func findPath(from sourceId: UUID, to targetId: UUID) -> [UUID]? {
        guard nodeCache[sourceId] != nil, nodeCache[targetId] != nil else {
            return nil
        }

        if sourceId == targetId {
            return [sourceId]
        }

        var visited: Set<UUID> = [sourceId]
        var parent: [UUID: UUID] = [:]
        var queue: [UUID] = [sourceId]

        while !queue.isEmpty {
            let current = queue.removeFirst()

            for (neighbor, _) in neighbors(of: current) {
                if !visited.contains(neighbor) {
                    visited.insert(neighbor)
                    parent[neighbor] = current

                    if neighbor == targetId {
                        // Reconstruct path
                        var path: [UUID] = [targetId]
                        var node = targetId
                        while let p = parent[node] {
                            path.insert(p, at: 0)
                            node = p
                        }
                        return path
                    }

                    queue.append(neighbor)
                }
            }
        }

        return nil  // No path found
    }

    /// Get all nodes of a specific type
    func nodes(ofType type: NodeType) -> [UUID] {
        nodeCache.values.filter { $0.nodeType == type }.map { $0.id }
    }

    /// Calculate node importance (PageRank-lite)
    func importance(of nodeId: UUID) -> Float {
        let inDegree = Float(incoming[nodeId]?.count ?? 0)
        let outDegree = Float(outgoing[nodeId]?.count ?? 0)
        return (inDegree * 2.0 + outDegree) / 3.0  // Weighted toward incoming
    }

    /// Get stale goals (goals not accessed recently)
    func staleGoals(olderThan date: Date, from nodes: [ThoughtNode]) -> [UUID] {
        let goalIds = self.nodes(ofType: .goal)
        return nodes
            .filter { goalIds.contains($0.id) && $0.lastAccessedAt < date && !$0.isCompleted }
            .map { $0.id }
    }

    // MARK: - Clustering

    /// Find connected components (clusters)
    func getClusters() -> [[UUID]] {
        if let clusters = clusters, !clustersDirty {
            return clusters
        }

        var visited: Set<UUID> = []
        var result: [[UUID]] = []

        for nodeId in nodeCache.keys {
            if visited.contains(nodeId) { continue }

            var cluster: [UUID] = []
            var stack: [UUID] = [nodeId]

            while !stack.isEmpty {
                let current = stack.removeLast()
                if visited.contains(current) { continue }

                visited.insert(current)
                cluster.append(current)

                for (neighbor, _) in neighbors(of: current) {
                    if !visited.contains(neighbor) {
                        stack.append(neighbor)
                    }
                }
            }

            if !cluster.isEmpty {
                result.append(cluster)
            }
        }

        clusters = result
        clustersDirty = false
        return result
    }

    // MARK: - Stats

    var nodeCount: Int { nodeCache.count }
    var edgeCount: Int { outgoing.values.reduce(0) { $0 + $1.count } }
}
```

### 3.4 Knowledge Graph Service (Orchestrator)

```swift
@Observable
final class KnowledgeGraph {
    private let nodeRepository: NodeRepositoryProtocol
    private let edgeRepository: EdgeRepositoryProtocol
    private let inMemoryGraph: InMemoryGraph
    private let embeddingEngine: EmbeddingEngineProtocol
    private let entityExtractor: EntityExtractor

    init(
        nodeRepository: NodeRepositoryProtocol,
        edgeRepository: EdgeRepositoryProtocol,
        embeddingEngine: EmbeddingEngineProtocol = NLEmbeddingEngine()
    ) {
        self.nodeRepository = nodeRepository
        self.edgeRepository = edgeRepository
        self.embeddingEngine = embeddingEngine
        self.entityExtractor = EntityExtractor()
        self.inMemoryGraph = InMemoryGraph()
    }

    // MARK: - Initialization

    func loadGraph() async throws {
        let nodes = try await nodeRepository.fetchAll()
        let edges = try await edgeRepository.fetchAll()
        inMemoryGraph.rebuild(nodes: nodes, edges: edges)
    }

    // MARK: - Node Operations

    func createNode(content: String, type: NodeType) async throws -> ThoughtNode {
        let node = ThoughtNode(content: content, nodeType: type)

        // Generate embedding
        if let embedding = await embeddingEngine.embed(content) {
            node.embedding = embedding
        }

        try await nodeRepository.save(node)
        inMemoryGraph.addNode(node)

        // Auto-extract entities and create connections
        let entities = await entityExtractor.extract(from: content)
        for entity in entities {
            try await linkToEntity(node: node, entity: entity)
        }

        return node
    }

    func createEdge(from source: ThoughtNode, to target: ThoughtNode, type: EdgeType, weight: Float = 1.0) async throws -> Edge {
        let edge = Edge(edgeType: type, sourceId: source.id, targetId: target.id, weight: weight)
        try await edgeRepository.save(edge)
        inMemoryGraph.addEdge(edge)
        return edge
    }

    // MARK: - Query Operations

    func findRelated(to node: ThoughtNode, depth: Int = 2, limit: Int = 20) -> [ThoughtNode] {
        let ids = inMemoryGraph.findRelated(to: node.id, depth: depth, limit: limit)
        return ids.compactMap { try? nodeRepository.fetchSync(id: $0) }
    }

    func findPath(from: ThoughtNode, to: ThoughtNode) -> [ThoughtNode]? {
        guard let path = inMemoryGraph.findPath(from: from.id, to: to.id) else {
            return nil
        }
        return path.compactMap { try? nodeRepository.fetchSync(id: $0) }
    }

    func semanticSearch(query: String, limit: Int = 10) async throws -> [ThoughtNode] {
        guard let queryEmbedding = await embeddingEngine.embed(query) else {
            return []
        }

        // Search all nodes with embeddings
        let nodes = try await nodeRepository.fetchAll()

        var scored: [(ThoughtNode, Float)] = []
        for node in nodes {
            guard let nodeEmbedding = node.embedding else { continue }
            let similarity = cosineSimilarity(queryEmbedding, nodeEmbedding)
            scored.append((node, similarity))
        }

        return scored
            .sorted { $0.1 > $1.1 }
            .prefix(limit)
            .map { $0.0 }
    }

    // MARK: - Goal-Specific Operations

    func getActiveGoals() async throws -> [ThoughtNode] {
        try await nodeRepository.fetchByType(.goal)
            .filter { !$0.isCompleted }
            .sorted { $0.createdAt > $1.createdAt }
    }

    func getStaleGoals(olderThan days: Int = 7) async throws -> [ThoughtNode] {
        let cutoff = Calendar.current.date(byAdding: .day, value: -days, to: Date())!
        let nodes = try await nodeRepository.fetchAll()
        let ids = inMemoryGraph.staleGoals(olderThan: cutoff, from: nodes)
        return nodes.filter { ids.contains($0.id) }
    }

    func getNextActions(for goal: ThoughtNode) -> [ThoughtNode] {
        let neighbors = inMemoryGraph.neighbors(of: goal.id, direction: .outgoing)
        let actionIds = neighbors
            .filter { $0.1.edgeType == .hasAction }
            .map { $0.0 }
        return actionIds.compactMap { try? nodeRepository.fetchSync(id: $0) }
            .filter { !$0.isCompleted }
            .sorted { $0.createdAt < $1.createdAt }
    }

    func getBlockers(for goal: ThoughtNode) -> [ThoughtNode] {
        let neighbors = inMemoryGraph.neighbors(of: goal.id, direction: .incoming)
        let blockerIds = neighbors
            .filter { $0.1.edgeType == .blocks }
            .map { $0.0 }
        return blockerIds.compactMap { try? nodeRepository.fetchSync(id: $0) }
    }

    // MARK: - Private Helpers

    private func linkToEntity(node: ThoughtNode, entity: ExtractedEntity) async throws {
        // Find or create entity node
        let existingNodes = try await nodeRepository.search(content: entity.text)

        let targetNode: ThoughtNode
        if let existing = existingNodes.first(where: { $0.nodeType == entity.type }) {
            targetNode = existing
        } else {
            targetNode = try await createNode(content: entity.text, type: entity.type)
        }

        // Create edge if not exists
        if inMemoryGraph.findPath(from: node.id, to: targetNode.id) == nil {
            _ = try await createEdge(from: node, to: targetNode, type: .mentions)
        }
    }

    private func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count, !a.isEmpty else { return 0 }

        var dotProduct: Float = 0
        var normA: Float = 0
        var normB: Float = 0

        for i in 0..<a.count {
            dotProduct += a[i] * b[i]
            normA += a[i] * a[i]
            normB += b[i] * b[i]
        }

        let magnitude = sqrt(normA) * sqrt(normB)
        return magnitude > 0 ? dotProduct / magnitude : 0
    }
}
```

### 3.5 Entity Extraction

```swift
import NaturalLanguage

struct ExtractedEntity {
    let text: String
    let type: NodeType
    let confidence: Float
}

final class EntityExtractor {
    private let tagger = NLTagger(tagSchemes: [.nameType, .lexicalClass])

    func extract(from text: String) async -> [ExtractedEntity] {
        tagger.string = text

        var entities: [ExtractedEntity] = []

        tagger.enumerateTags(
            in: text.startIndex..<text.endIndex,
            unit: .word,
            scheme: .nameType,
            options: [.omitWhitespace, .omitPunctuation, .joinNames]
        ) { tag, range in
            guard let tag = tag else { return true }

            let entityText = String(text[range])

            let nodeType: NodeType? = switch tag {
            case .personalName: .person
            case .placeName: .place
            case .organizationName: .project
            default: nil
            }

            if let type = nodeType {
                entities.append(ExtractedEntity(
                    text: entityText,
                    type: type,
                    confidence: 0.8
                ))
            }

            return true
        }

        return entities
    }
}
```

---

## 4. AI/LLM Integration

### 4.1 Architecture Overview

```
+------------------+     +------------------+     +------------------+
|  Conversation    |     |  Context         |     |  Memory          |
|  Engine          | --> |  Builder         | --> |  Engine          |
+------------------+     +------------------+     +------------------+
        |                        |                        |
        v                        v                        v
+------------------+     +------------------+     +------------------+
|  LLM Client      |     |  Token Counter   |     |  Semantic Search |
|  (Claude API)    |     |                  |     |                  |
+------------------+     +------------------+     +------------------+
        |
        v
+------------------+
|  Streaming       |
|  Handler         |
+------------------+
```

### 4.2 Conversation Engine

```swift
@Observable
final class ConversationEngine {
    // Dependencies
    private let llmClient: LLMClientProtocol
    private let contextBuilder: ContextBuilder
    private let memoryEngine: MemoryEngine
    private let sessionManager: SessionManager

    // State
    private(set) var currentSession: ConversationSession?
    private(set) var isProcessing: Bool = false
    private(set) var streamingResponse: String = ""

    // Current request for cancellation
    private var currentTask: Task<Void, Error>?

    init(
        llmClient: LLMClientProtocol,
        memoryEngine: MemoryEngine,
        sessionManager: SessionManager
    ) {
        self.llmClient = llmClient
        self.memoryEngine = memoryEngine
        self.sessionManager = sessionManager
        self.contextBuilder = ContextBuilder(memoryEngine: memoryEngine)
    }

    // MARK: - Session Management

    func startConversation() async throws -> ConversationSession {
        let session = ConversationSession()
        currentSession = session

        // Build initial context with memory
        let systemPrompt = await contextBuilder.buildSystemPrompt(
            userPatterns: memoryEngine.currentPatterns,
            activeGoals: try await memoryEngine.getActiveGoals(),
            recentContext: memoryEngine.recentContext
        )

        session.systemPrompt = systemPrompt

        try await sessionManager.save(session)
        return session
    }

    func endConversation() async {
        guard let session = currentSession else { return }

        // Consolidate session into memory
        await memoryEngine.consolidateSession(session)

        // Save final state
        session.endedAt = Date()
        try? await sessionManager.save(session)

        currentSession = nil
    }

    // MARK: - Message Handling

    func send(message: String) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { [weak self] continuation in
            self?.currentTask = Task {
                guard let self, let session = self.currentSession else {
                    continuation.finish(throwing: ConversationError.noActiveSession)
                    return
                }

                self.isProcessing = true
                self.streamingResponse = ""

                do {
                    // Add user message
                    let userMessage = Message(role: .user, content: message)
                    session.messages.append(userMessage)

                    // Get relevant context from memory
                    let relevantMemories = await self.memoryEngine.retrieveRelevant(
                        query: message,
                        limit: 5
                    )

                    // Augment system prompt with relevant memories
                    let augmentedPrompt = session.systemPrompt + "\n\n## Relevant Context\n" + relevantMemories

                    // Stream response
                    var fullResponse = ""

                    for try await chunk in self.llmClient.stream(
                        messages: session.messages.map { ($0.role, $0.content) },
                        systemPrompt: augmentedPrompt
                    ) {
                        try Task.checkCancellation()

                        fullResponse += chunk
                        self.streamingResponse = fullResponse
                        continuation.yield(chunk)
                    }

                    // Save assistant response
                    let assistantMessage = Message(role: .assistant, content: fullResponse)
                    session.messages.append(assistantMessage)

                    // Process for memory extraction (async, don't block response)
                    Task {
                        await self.memoryEngine.process(
                            userMessage: message,
                            assistantResponse: fullResponse
                        )
                    }

                    self.isProcessing = false
                    continuation.finish()

                } catch is CancellationError {
                    self.isProcessing = false
                    continuation.finish()
                } catch {
                    self.isProcessing = false
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    func cancel() {
        currentTask?.cancel()
        currentTask = nil
        isProcessing = false
    }
}

// MARK: - Supporting Types

enum ConversationError: Error, LocalizedError {
    case noActiveSession
    case apiError(String)
    case rateLimited
    case cancelled

    var errorDescription: String? {
        switch self {
        case .noActiveSession: return "No active conversation session"
        case .apiError(let msg): return "API error: \(msg)"
        case .rateLimited: return "Rate limited. Please wait a moment."
        case .cancelled: return "Request cancelled"
        }
    }
}
```

### 4.3 Claude API Client

```swift
protocol LLMClientProtocol {
    func stream(
        messages: [(MessageRole, String)],
        systemPrompt: String
    ) -> AsyncThrowingStream<String, Error>
}

final class ClaudeClient: LLMClientProtocol {
    private let apiKey: String
    private let baseURL = URL(string: "https://api.anthropic.com/v1/messages")!
    private let model = "claude-sonnet-4-20250514"
    private let maxTokens = 4096

    // Rate limiting
    private var lastRequestTime: Date?
    private let minRequestInterval: TimeInterval = 0.1  // 100ms minimum between requests

    init(apiKey: String) {
        self.apiKey = apiKey
    }

    func stream(
        messages: [(MessageRole, String)],
        systemPrompt: String
    ) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            Task {
                do {
                    // Basic rate limiting
                    if let lastTime = lastRequestTime {
                        let elapsed = Date().timeIntervalSince(lastTime)
                        if elapsed < minRequestInterval {
                            try await Task.sleep(nanoseconds: UInt64((minRequestInterval - elapsed) * 1_000_000_000))
                        }
                    }
                    lastRequestTime = Date()

                    var request = URLRequest(url: baseURL)
                    request.httpMethod = "POST"
                    request.setValue("application/json", forHTTPHeaderField: "Content-Type")
                    request.setValue(apiKey, forHTTPHeaderField: "x-api-key")
                    request.setValue("2023-06-01", forHTTPHeaderField: "anthropic-version")

                    let body: [String: Any] = [
                        "model": model,
                        "max_tokens": maxTokens,
                        "system": systemPrompt,
                        "messages": messages.map { ["role": $0.0.rawValue, "content": $0.1] },
                        "stream": true
                    ]

                    request.httpBody = try JSONSerialization.data(withJSONObject: body)

                    let (stream, response) = try await URLSession.shared.bytes(for: request)

                    guard let httpResponse = response as? HTTPURLResponse else {
                        throw ConversationError.apiError("Invalid response")
                    }

                    if httpResponse.statusCode == 429 {
                        throw ConversationError.rateLimited
                    }

                    if httpResponse.statusCode != 200 {
                        throw ConversationError.apiError("HTTP \(httpResponse.statusCode)")
                    }

                    for try await line in stream.lines {
                        try Task.checkCancellation()

                        guard line.hasPrefix("data: ") else { continue }
                        let jsonString = String(line.dropFirst(6))

                        guard jsonString != "[DONE]",
                              let data = jsonString.data(using: .utf8),
                              let event = try? JSONDecoder().decode(StreamEvent.self, from: data)
                        else { continue }

                        if let text = event.delta?.text {
                            continuation.yield(text)
                        }

                        if event.type == "message_stop" {
                            break
                        }
                    }

                    continuation.finish()

                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }
}

// MARK: - API Response Types

struct StreamEvent: Decodable {
    let type: String
    let delta: Delta?

    struct Delta: Decodable {
        let text: String?
    }
}

enum MessageRole: String, Codable {
    case user
    case assistant
    case system
}
```

### 4.4 Context Builder & Axel Personality

```swift
final class ContextBuilder {
    private let memoryEngine: MemoryEngine
    private let tokenCounter = TokenCounter()

    // Token budgets
    private let systemPromptBudget = 2000
    private let patternsBudget = 500
    private let goalsBudget = 1000
    private let memoryBudget = 2000

    init(memoryEngine: MemoryEngine) {
        self.memoryEngine = memoryEngine
    }

    func buildSystemPrompt(
        userPatterns: [UserPattern],
        activeGoals: [ThoughtNode],
        recentContext: String
    ) async -> String {
        var prompt = axelPersonality

        // Add patterns if within budget
        let patternsSection = formatPatterns(userPatterns)
        if tokenCounter.count(patternsSection) <= patternsBudget {
            prompt += "\n\n## User Patterns I've Learned\n" + patternsSection
        }

        // Add active goals if within budget
        let goalsSection = formatGoals(activeGoals)
        if tokenCounter.count(goalsSection) <= goalsBudget {
            prompt += "\n\n## Current Active Goals\n" + goalsSection
        }

        // Add recent context if within budget
        if tokenCounter.count(recentContext) <= memoryBudget {
            prompt += "\n\n## Recent Context\n" + recentContext
        }

        return prompt
    }

    private var axelPersonality: String {
        """
        You are Axel, a warm and perceptive AI companion in the MYND app. Your purpose is to help the user capture, organize, and act on their thoughts.

        ## Your Core Traits
        - **Warm but not sycophantic**: Genuinely supportive without excessive praise
        - **Perceptive**: You notice patterns and connections the user might miss
        - **Action-oriented**: You help turn vague thoughts into concrete next steps
        - **Patient**: You help organize scattered thinking without judgment
        - **Memory-aware**: You naturally reference past conversations when relevant

        ## Conversation Guidelines
        1. When the user shares a thought, help them explore or capture it
        2. For goals, break them into the smallest possible next action
        3. Reference past conversations naturally: "Last time you mentioned..."
        4. Gently follow up on stale goals without being pushy
        5. Ask clarifying questions to understand intent
        6. Acknowledge emotions when expressed
        7. Keep responses concise for voice playback (1-3 sentences typical)

        ## Voice-Optimized Responses
        - Use natural conversational language
        - Avoid bullet points in voice responses (use flowing sentences)
        - End with a question or gentle prompt when appropriate
        - For action items, be specific about what to do next
        - Vary your phrasing - don't repeat the same structures

        ## Important Boundaries
        - You help with thought organization, not therapy or medical advice
        - If the user seems distressed, acknowledge it and suggest professional support
        - You don't make decisions for the user, but help them think through options
        """
    }

    private func formatPatterns(_ patterns: [UserPattern]) -> String {
        guard !patterns.isEmpty else { return "None learned yet." }

        return patterns.map { pattern in
            "- \(pattern.patternType.description): \(pattern.description)"
        }.joined(separator: "\n")
    }

    private func formatGoals(_ goals: [ThoughtNode]) -> String {
        guard !goals.isEmpty else { return "No active goals." }

        return goals.prefix(5).map { goal in
            "- \(goal.content)"
        }.joined(separator: "\n")
    }
}

// MARK: - Token Counter

final class TokenCounter {
    // Approximate: 4 characters per token for English
    private let charsPerToken: Double = 4.0

    func count(_ text: String) -> Int {
        Int(ceil(Double(text.count) / charsPerToken))
    }

    func truncate(_ text: String, toTokens maxTokens: Int) -> String {
        let maxChars = Int(Double(maxTokens) * charsPerToken)
        if text.count <= maxChars {
            return text
        }
        return String(text.prefix(maxChars)) + "..."
    }
}
```

### 4.5 Memory Engine

```swift
@Observable
final class MemoryEngine {
    // Stores
    private let memoryRepository: MemoryRepositoryProtocol
    private let patternLearner: PatternLearner
    private let embeddingEngine: EmbeddingEngineProtocol
    private let knowledgeGraph: KnowledgeGraph

    // Working memory (current session context)
    private var workingMemory: [String] = []
    private let workingMemoryLimit = 10

    // Episodic buffer (recent interactions awaiting consolidation)
    private var episodicBuffer: [(user: String, assistant: String)] = []
    private let episodicBufferLimit = 20

    // Current patterns
    private(set) var currentPatterns: [UserPattern] = []

    // Recent context summary
    private(set) var recentContext: String = ""

    init(
        memoryRepository: MemoryRepositoryProtocol,
        embeddingEngine: EmbeddingEngineProtocol,
        knowledgeGraph: KnowledgeGraph
    ) {
        self.memoryRepository = memoryRepository
        self.embeddingEngine = embeddingEngine
        self.knowledgeGraph = knowledgeGraph
        self.patternLearner = PatternLearner()
    }

    // MARK: - Memory Processing

    func process(userMessage: String, assistantResponse: String) async {
        // Add to working memory
        workingMemory.append(userMessage)
        if workingMemory.count > workingMemoryLimit {
            workingMemory.removeFirst()
        }

        // Add to episodic buffer
        episodicBuffer.append((user: userMessage, assistant: assistantResponse))

        // Check for pattern updates
        if let pattern = await patternLearner.detectPattern(
            from: episodicBuffer,
            existingPatterns: currentPatterns
        ) {
            currentPatterns.append(pattern)
            try? await memoryRepository.savePattern(pattern)
        }

        // Extract significant information for knowledge graph
        let significance = calculateSignificance(userMessage)
        if significance > 0.6 {
            let embedding = await embeddingEngine.embed(userMessage)
            let memory = MemoryItem(
                content: userMessage,
                memoryType: .episodic,
                significance: significance
            )
            memory.embedding = embedding
            try? await memoryRepository.save(memory)

            // Also create a thought node if it seems like a significant thought
            _ = try? await knowledgeGraph.createNode(
                content: userMessage,
                type: inferNodeType(from: userMessage)
            )
        }

        // Trigger consolidation if buffer is full
        if episodicBuffer.count >= episodicBufferLimit {
            await consolidateEpisodicMemory()
        }

        // Update recent context summary
        updateRecentContext()
    }

    func retrieveRelevant(query: String, limit: Int = 5) async -> String {
        // Semantic search over memories
        guard let queryEmbedding = await embeddingEngine.embed(query) else {
            return ""
        }

        let memories = try? await memoryRepository.searchByEmbedding(
            queryEmbedding,
            limit: limit
        )

        guard let memories, !memories.isEmpty else {
            return ""
        }

        return memories.map { memory in
            "[\(memory.memoryType.rawValue)] \(memory.content)"
        }.joined(separator: "\n")
    }

    func getActiveGoals() async throws -> [ThoughtNode] {
        try await knowledgeGraph.getActiveGoals()
    }

    // MARK: - Session Consolidation

    func consolidateSession(_ session: ConversationSession) async {
        // Summarize the session
        let summary = summarizeSession(session)

        // Store session summary as semantic memory
        let memory = MemoryItem(
            content: summary,
            memoryType: .semantic,
            significance: 0.8
        )
        memory.embedding = await embeddingEngine.embed(summary)
        try? await memoryRepository.save(memory)

        // Check for session-level patterns
        let sessionPatterns = await patternLearner.analyzeSession(session)
        for pattern in sessionPatterns {
            currentPatterns.append(pattern)
            try? await memoryRepository.savePattern(pattern)
        }

        // Clear episodic buffer after consolidation
        episodicBuffer.removeAll()
    }

    // MARK: - Private Helpers

    private func calculateSignificance(_ text: String) -> Float {
        var score: Float = 0.0

        // Contains goal-related keywords
        let goalKeywords = ["want to", "need to", "should", "goal", "plan", "trying to"]
        for keyword in goalKeywords {
            if text.lowercased().contains(keyword) {
                score += 0.3
            }
        }

        // Contains named entities
        if text.contains(where: { $0.isUppercase }) {
            score += 0.1
        }

        // Minimum length threshold
        if text.count > 50 {
            score += 0.1
        }

        // Contains action verbs
        let actionVerbs = ["finish", "complete", "start", "build", "create", "write", "call", "email"]
        for verb in actionVerbs {
            if text.lowercased().contains(verb) {
                score += 0.2
            }
        }

        return min(score, 1.0)
    }

    private func inferNodeType(from text: String) -> NodeType {
        let lowered = text.lowercased()

        if lowered.contains("want to") || lowered.contains("goal") || lowered.contains("need to") {
            return .goal
        }

        if lowered.contains("should i") || lowered.contains("wondering") || lowered.contains("?") {
            return .question
        }

        return .thought
    }

    private func summarizeSession(_ session: ConversationSession) -> String {
        let messageCount = session.messages.count
        let topics = extractTopics(from: session.messages)

        return "Session with \(messageCount) messages discussing: \(topics.joined(separator: ", "))"
    }

    private func extractTopics(from messages: [Message]) -> [String] {
        // Simple topic extraction - in production, use NLP
        let allContent = messages.map { $0.content }.joined(separator: " ")
        let words = allContent.components(separatedBy: .whitespaces)

        // Find most common significant words (simplified)
        let stopWords: Set<String> = ["the", "a", "an", "is", "are", "was", "were", "i", "you", "it"]
        let significantWords = words.filter { $0.count > 4 && !stopWords.contains($0.lowercased()) }

        let wordCounts = Dictionary(grouping: significantWords, by: { $0.lowercased() })
            .mapValues { $0.count }
            .sorted { $0.value > $1.value }

        return Array(wordCounts.prefix(3).map { $0.key })
    }

    private func updateRecentContext() {
        recentContext = workingMemory.suffix(5).joined(separator: "\n")
    }

    private func consolidateEpisodicMemory() async {
        // Batch consolidate episodic memories into long-term storage
        for interaction in episodicBuffer {
            let combined = "User: \(interaction.user)\nAxel: \(interaction.assistant)"
            let memory = MemoryItem(
                content: combined,
                memoryType: .episodic,
                significance: 0.5
            )
            memory.embedding = await embeddingEngine.embed(combined)
            try? await memoryRepository.save(memory)
        }

        episodicBuffer.removeAll()
    }
}
```

### 4.6 Embedding Engine

```swift
import NaturalLanguage

protocol EmbeddingEngineProtocol {
    func embed(_ text: String) async -> [Float]?
    func similarity(_ a: [Float], _ b: [Float]) -> Float
}

final class NLEmbeddingEngine: EmbeddingEngineProtocol {
    private let embedding: NLEmbedding?

    init() {
        // Use Apple's sentence embedding (works offline, ~512 dimensions)
        self.embedding = NLEmbedding.sentenceEmbedding(for: .english)
    }

    func embed(_ text: String) async -> [Float]? {
        guard let embedding else { return nil }
        guard let vector = embedding.vector(for: text) else { return nil }
        return vector.map { Float($0) }
    }

    func similarity(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count, !a.isEmpty else { return 0 }

        var dotProduct: Float = 0
        var normA: Float = 0
        var normB: Float = 0

        for i in 0..<a.count {
            dotProduct += a[i] * b[i]
            normA += a[i] * a[i]
            normB += b[i] * b[i]
        }

        let magnitude = sqrt(normA) * sqrt(normB)
        return magnitude > 0 ? dotProduct / magnitude : 0
    }
}
```

---

## 5. Data Model

### 5.1 Entity-Relationship Overview

```
+------------------+          +------------------+
|   ThoughtNode    |          |      Edge        |
+------------------+          +------------------+
| id: UUID (PK)    |<-------->| id: UUID (PK)    |
| content: String  |          | edgeType: Enum   |
| nodeType: Enum   |          | weight: Float    |
| createdAt: Date  |          | sourceId: UUID   |
| lastAccessedAt   |          | targetId: UUID   |
| embedding: Data? |          | createdAt: Date  |
| isCompleted: Bool|          +------------------+
+------------------+
        |
        | (via MemoryItem.linkedNodeIds)
        v
+------------------+          +------------------+
|   MemoryItem     |          |   UserPattern    |
+------------------+          +------------------+
| id: UUID (PK)    |          | id: UUID (PK)    |
| content: String  |          | patternType: Enum|
| memoryType: Enum |          | description: Str |
| significance: Flt|          | frequency: Int   |
| embedding: Data? |          | confidence: Float|
| createdAt: Date  |          +------------------+
+------------------+

+------------------+          +------------------+
|ConversationSession|         |    Message       |
+------------------+          +------------------+
| id: UUID (PK)    |<-------->| id: UUID (PK)    |
| startedAt: Date  |          | role: Enum       |
| endedAt: Date?   |          | content: String  |
| systemPrompt: Str|          | createdAt: Date  |
| summary: String? |          | sessionId: UUID  |
+------------------+          +------------------+
```

### 5.2 Complete SwiftData Models

```swift
// See Section 3.2 for ThoughtNode and Edge models

// MARK: - Conversation Session

@Model
final class ConversationSession {
    @Attribute(.unique)
    var id: UUID

    var startedAt: Date
    var endedAt: Date?
    var systemPrompt: String
    var summary: String?

    // Sync
    var cloudKitRecordID: String?
    var needsSync: Bool = false

    @Relationship(deleteRule: .cascade, inverse: \Message.session)
    var messages: [Message] = []

    var messageCount: Int { messages.count }

    init() {
        self.id = UUID()
        self.startedAt = Date()
        self.systemPrompt = ""
    }
}

// MARK: - Message

@Model
final class Message {
    @Attribute(.unique)
    var id: UUID

    var role: MessageRole
    var content: String

    @Attribute(.indexed)
    var createdAt: Date

    var session: ConversationSession?

    init(role: MessageRole, content: String) {
        self.id = UUID()
        self.role = role
        self.content = content
        self.createdAt = Date()
    }
}

// MARK: - Memory Item

@Model
final class MemoryItem {
    @Attribute(.unique)
    var id: UUID

    var content: String
    var memoryType: MemoryType
    var significance: Float  // 0.0 - 1.0

    @Attribute(.indexed)
    var createdAt: Date

    var accessCount: Int = 0
    var lastAccessedAt: Date?

    @Attribute(.externalStorage)
    var embeddingData: Data?

    // Linked thought nodes (by ID for sync safety)
    var linkedNodeIds: [UUID] = []

    init(content: String, memoryType: MemoryType, significance: Float) {
        self.id = UUID()
        self.content = content
        self.memoryType = memoryType
        self.significance = significance
        self.createdAt = Date()
    }

    var embedding: [Float]? {
        get {
            guard let data = embeddingData else { return nil }
            return data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        }
        set {
            guard let floats = newValue else {
                embeddingData = nil
                return
            }
            embeddingData = floats.withUnsafeBytes { Data($0) }
        }
    }
}

enum MemoryType: String, Codable {
    case episodic    // Specific conversation/event
    case semantic    // Learned fact/pattern
    case procedural  // How to do something
}

// MARK: - User Pattern

@Model
final class UserPattern {
    @Attribute(.unique)
    var id: UUID

    var patternType: PatternType
    var description: String
    var frequency: Int = 1
    var lastOccurred: Date
    var confidence: Float  // 0.0 - 1.0

    init(patternType: PatternType, description: String) {
        self.id = UUID()
        self.patternType = patternType
        self.description = description
        self.lastOccurred = Date()
        self.confidence = 0.5
    }
}

enum PatternType: String, Codable, CaseIterable {
    case communicationStyle   // How user prefers to express things
    case topicPreference      // Topics user cares about
    case temporalPattern      // When user is most active
    case goalPattern          // How user approaches goals
    case actionPattern        // How user breaks down tasks

    var description: String {
        switch self {
        case .communicationStyle: return "Communication Style"
        case .topicPreference: return "Topic Preference"
        case .temporalPattern: return "Activity Pattern"
        case .goalPattern: return "Goal Approach"
        case .actionPattern: return "Task Pattern"
        }
    }
}
```

### 5.3 Schema Migration Strategy

```swift
// MARK: - Migration Manager

enum MYNDSchemaVersion: Int {
    case v1 = 1  // Initial release
    case v2 = 2  // Added UserPattern.confidence
    case v3 = 3  // Added sync metadata

    static var current: MYNDSchemaVersion { .v3 }
}

final class MigrationManager {
    static let shared = MigrationManager()

    func configure() -> ModelConfiguration {
        let schema = Schema([
            ThoughtNode.self,
            Edge.self,
            ConversationSession.self,
            Message.self,
            MemoryItem.self,
            UserPattern.self
        ])

        return ModelConfiguration(
            schema: schema,
            isStoredInMemoryOnly: false,
            allowsSave: true
        )
    }

    // Handle migrations as needed
    func migrate(from oldVersion: Int, to newVersion: Int, context: ModelContext) async throws {
        // Migration logic for each version upgrade
        // For now, SwiftData handles lightweight migrations automatically
    }
}
```

---

## 6. Privacy & Security

### 6.1 Privacy Architecture

```
+------------------------------------------------------------------+
|                      PRIVACY LAYERS                               |
+------------------------------------------------------------------+
|                                                                    |
|  Layer 1: LOCAL-FIRST                                             |
|  +------------------------------------------------------------+  |
|  | All data stored on device by default                        |  |
|  | Voice audio never persisted (processed in memory only)      |  |
|  +------------------------------------------------------------+  |
|                              |                                    |
|                              v                                    |
|  Layer 2: ENCRYPTED STORAGE                                       |
|  +------------------------------------------------------------+  |
|  | SwiftData uses iOS data protection (AES-256)               |  |
|  | API keys in Keychain (Secure Enclave when available)       |  |
|  | CloudKit uses end-to-end encryption in private database    |  |
|  +------------------------------------------------------------+  |
|                              |                                    |
|                              v                                    |
|  Layer 3: MINIMAL CLOUD EXPOSURE                                  |
|  +------------------------------------------------------------+  |
|  | Only Claude API calls leave device (required for AI)       |  |
|  | User's own API key = Anthropic's zero-retention policy     |  |
|  | CloudKit sync is opt-in, uses private database only        |  |
|  +------------------------------------------------------------+  |
|                              |                                    |
|                              v                                    |
|  Layer 4: USER CONTROL                                            |
|  +------------------------------------------------------------+  |
|  | Export all data (JSON/CSV)                                 |  |
|  | Delete all data with confirmation                          |  |
|  | Granular sync controls                                     |  |
|  +------------------------------------------------------------+  |
|                                                                    |
+------------------------------------------------------------------+
```

### 6.2 Data Classification

| Data Type | Storage | Encryption | Cloud Sync | Retention |
|-----------|---------|------------|------------|-----------|
| **Voice Audio** | Memory only | N/A | Never | 0 seconds (transient) |
| **Transcripts** | SwiftData | Device encryption | Optional | Until deleted |
| **Conversations** | SwiftData | Device encryption | Optional | Until deleted |
| **Knowledge Graph** | SwiftData | Device encryption | Optional | Until deleted |
| **API Keys** | Keychain | Secure Enclave | Never | Until deleted |
| **User Patterns** | SwiftData | Device encryption | Optional | Until deleted |
| **Embeddings** | SwiftData | Device encryption | Never (regeneratable) | Until deleted |

### 6.3 Secure Storage Implementation

```swift
import Security
import Foundation

final class SecureStorage {
    static let shared = SecureStorage()

    private init() {}

    // MARK: - API Key Management

    func storeAPIKey(_ key: String, for service: APIService) throws {
        let data = Data(key.utf8)

        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service.keychainKey,
            kSecAttrAccessible as String: kSecAttrAccessibleWhenUnlockedThisDeviceOnly,
            kSecValueData as String: data
        ]

        // Delete existing if present
        SecItemDelete(query as CFDictionary)

        let status = SecItemAdd(query as CFDictionary, nil)

        guard status == errSecSuccess else {
            throw SecureStorageError.keychainWriteFailed(status)
        }
    }

    func retrieveAPIKey(for service: APIService) -> String? {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service.keychainKey,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne
        ]

        var result: AnyObject?
        let status = SecItemCopyMatching(query as CFDictionary, &result)

        guard status == errSecSuccess,
              let data = result as? Data,
              let key = String(data: data, encoding: .utf8)
        else {
            return nil
        }

        return key
    }

    func deleteAPIKey(for service: APIService) {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service.keychainKey
        ]

        SecItemDelete(query as CFDictionary)
    }

    func hasAPIKey(for service: APIService) -> Bool {
        retrieveAPIKey(for: service) != nil
    }
}

// MARK: - Supporting Types

enum APIService: String, CaseIterable {
    case anthropic
    case openai
    case elevenLabs

    var keychainKey: String {
        "com.mynd.api.\(rawValue)"
    }

    var displayName: String {
        switch self {
        case .anthropic: return "Claude (Anthropic)"
        case .openai: return "OpenAI"
        case .elevenLabs: return "ElevenLabs"
        }
    }
}

enum SecureStorageError: Error, LocalizedError {
    case keychainWriteFailed(OSStatus)
    case keychainReadFailed

    var errorDescription: String? {
        switch self {
        case .keychainWriteFailed(let status):
            return "Failed to save to Keychain (status: \(status))"
        case .keychainReadFailed:
            return "Failed to read from Keychain"
        }
    }
}
```

### 6.4 CloudKit Privacy Configuration

```swift
import CloudKit

final class CloudKitSync {
    // ONLY use private database - never public
    private let container = CKContainer(identifier: "iCloud.com.mynd.app")
    private lazy var privateDatabase = container.privateCloudDatabase

    var isSyncEnabled: Bool = false

    // MARK: - Sync Operations

    func sync(nodes: [ThoughtNode]) async throws {
        guard isSyncEnabled else { return }

        let records = nodes.filter { $0.needsSync }.map { node -> CKRecord in
            let recordID = CKRecord.ID(recordName: node.id.uuidString)
            let record = CKRecord(recordType: "ThoughtNode", recordID: recordID)

            record["content"] = node.content
            record["nodeType"] = node.nodeType.rawValue
            record["createdAt"] = node.createdAt
            record["lastAccessedAt"] = node.lastAccessedAt
            record["isCompleted"] = node.isCompleted

            // Don't sync embeddings (regeneratable, saves space)

            return record
        }

        guard !records.isEmpty else { return }

        let operation = CKModifyRecordsOperation(
            recordsToSave: records,
            recordIDsToDelete: nil
        )

        operation.savePolicy = .changedKeys
        operation.isAtomic = false  // Allow partial success

        try await privateDatabase.modifyRecords(saving: records, deleting: [])
    }

    func fetchChanges() async throws -> [CKRecord] {
        guard isSyncEnabled else { return [] }

        // Fetch changes since last sync token
        // Implementation uses CKFetchRecordZoneChangesOperation
        // ...

        return []
    }
}
```

### 6.5 Voice Data Handling

```swift
/// Voice data is NEVER persisted - only processed in memory
final class VoicePrivacyManager {

    /// Transcribes audio buffer without persisting
    /// Audio data is released immediately after transcription
    func transcribe(_ buffer: AVAudioPCMBuffer) async -> String {
        // Audio stays in memory only
        // No disk writes
        // Transcription happens on-device by default

        defer {
            // Ensure buffer is released
            // (Swift ARC handles this, but be explicit)
        }

        return await transcribeOnDevice(buffer)
    }

    private func transcribeOnDevice(_ buffer: AVAudioPCMBuffer) async -> String {
        // Uses Apple Speech Framework
        // All processing on-device
        // No network call when using on-device recognition
        return ""
    }
}
```

### 6.6 Data Export & Deletion

```swift
final class DataPortabilityManager {
    private let nodeRepository: NodeRepositoryProtocol
    private let sessionRepository: SessionRepositoryProtocol
    private let memoryRepository: MemoryRepositoryProtocol

    // MARK: - Export

    func exportAllData() async throws -> Data {
        let export = MYNDExport(
            exportDate: Date(),
            version: "1.0",
            nodes: try await nodeRepository.fetchAll(),
            sessions: try await sessionRepository.fetchAll(),
            memories: try await memoryRepository.fetchAll()
        )

        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]

        return try encoder.encode(export)
    }

    // MARK: - Delete All

    func deleteAllData() async throws {
        // Confirm with user before calling this

        // Delete from repositories
        try await nodeRepository.deleteAll()
        try await sessionRepository.deleteAll()
        try await memoryRepository.deleteAll()

        // Clear Keychain (API keys)
        for service in APIService.allCases {
            SecureStorage.shared.deleteAPIKey(for: service)
        }

        // Clear CloudKit (if enabled)
        // ...
    }
}

struct MYNDExport: Codable {
    let exportDate: Date
    let version: String
    let nodes: [ThoughtNodeExport]
    let sessions: [SessionExport]
    let memories: [MemoryExport]
}

struct ThoughtNodeExport: Codable {
    let id: UUID
    let content: String
    let nodeType: String
    let createdAt: Date
    let isCompleted: Bool
}

struct SessionExport: Codable {
    let id: UUID
    let startedAt: Date
    let endedAt: Date?
    let messages: [MessageExport]
}

struct MessageExport: Codable {
    let role: String
    let content: String
    let createdAt: Date
}

struct MemoryExport: Codable {
    let id: UUID
    let content: String
    let memoryType: String
    let significance: Float
    let createdAt: Date
}
```

---

## 7. Proactive System

### 7.1 Proactive Engine

```swift
import UserNotifications
import BackgroundTasks

@Observable
final class ProactiveEngine {
    private let goalTracker: GoalTracker
    private let insightGenerator: InsightGenerator
    private let notificationManager: NotificationManager
    private let knowledgeGraph: KnowledgeGraph

    // Configuration
    var isEnabled: Bool = true
    var quietHoursStart: Int = 22  // 10 PM
    var quietHoursEnd: Int = 8     // 8 AM

    // MARK: - Background Processing

    /// Called from BGAppRefreshTask
    func processBackgroundRefresh() async {
        guard isEnabled else { return }
        guard !isQuietHours() else { return }

        // 1. Check for stale goals
        let staleGoals = try? await knowledgeGraph.getStaleGoals(olderThan: 3)
        for goal in staleGoals ?? [] {
            let prompt = await insightGenerator.generateFollowUpPrompt(for: goal)
            await scheduleFollowUp(for: goal, prompt: prompt)
        }

        // 2. Check for upcoming deadlines (if we track them)
        // ...

        // 3. Generate daily insight if due
        if shouldGenerateDailyInsight() {
            let insight = await insightGenerator.generateDailyInsight()
            await notificationManager.scheduleDailyInsight(insight)
        }
    }

    // MARK: - Follow-up Scheduling

    func scheduleFollowUp(for goal: ThoughtNode, prompt: String) async {
        let content = UNMutableNotificationContent()
        content.title = "Axel has a thought"
        content.body = prompt
        content.sound = .default
        content.categoryIdentifier = "FOLLOW_UP"
        content.userInfo = ["goalId": goal.id.uuidString]

        // Schedule for optimal time
        let trigger = UNTimeIntervalNotificationTrigger(
            timeInterval: calculateOptimalDelay(),
            repeats: false
        )

        let request = UNNotificationRequest(
            identifier: "followup-\(goal.id)",
            content: content,
            trigger: trigger
        )

        try? await UNUserNotificationCenter.current().add(request)
    }

    // MARK: - Helpers

    private func isQuietHours() -> Bool {
        let hour = Calendar.current.component(.hour, from: Date())

        if quietHoursStart > quietHoursEnd {
            // Spans midnight (e.g., 22:00 - 08:00)
            return hour >= quietHoursStart || hour < quietHoursEnd
        } else {
            return hour >= quietHoursStart && hour < quietHoursEnd
        }
    }

    private func calculateOptimalDelay() -> TimeInterval {
        // Simple: schedule for next non-quiet hour
        // Could be smarter based on user patterns

        if isQuietHours() {
            // Calculate time until quiet hours end
            let calendar = Calendar.current
            let now = Date()
            var components = calendar.dateComponents([.year, .month, .day], from: now)
            components.hour = quietHoursEnd
            components.minute = 0

            if let targetDate = calendar.date(from: components) {
                var delay = targetDate.timeIntervalSince(now)
                if delay < 0 {
                    delay += 86400  // Add a day
                }
                return delay
            }
        }

        // Default: 30 minutes from now
        return 30 * 60
    }

    private func shouldGenerateDailyInsight() -> Bool {
        // Check if we've already generated one today
        // Implementation depends on UserDefaults tracking
        return true
    }
}
```

### 7.2 Insight Generator

```swift
final class InsightGenerator {
    private let llmClient: LLMClientProtocol
    private let knowledgeGraph: KnowledgeGraph

    func generateFollowUpPrompt(for goal: ThoughtNode) async -> String {
        // Use LLM to generate a natural follow-up question
        let prompt = """
        Generate a brief, friendly follow-up prompt for this goal that hasn't been worked on recently:

        Goal: \(goal.content)
        Last accessed: \(goal.lastAccessedAt.formatted())

        Keep it under 100 characters. Be encouraging, not nagging. Example tone:
        "Still thinking about the garden project? What's the smallest step you could take today?"
        """

        var result = ""
        for try await chunk in llmClient.stream(
            messages: [(.user, prompt)],
            systemPrompt: "You generate brief, encouraging follow-up prompts."
        ) {
            result += chunk
        }

        return result.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    func generateDailyInsight() async -> String {
        let activeGoals = (try? await knowledgeGraph.getActiveGoals()) ?? []
        let staleGoals = (try? await knowledgeGraph.getStaleGoals()) ?? []

        let prompt = """
        Generate a brief daily insight for the user based on their current state:

        Active goals: \(activeGoals.count)
        Stale goals: \(staleGoals.count)

        Goal examples:
        \(activeGoals.prefix(3).map { "- \($0.content)" }.joined(separator: "\n"))

        Keep it under 150 characters. Be observational and helpful.
        Example: "You've got 3 active goals. The 'learn Spanish' one hasn't had attention in a while - worth revisiting?"
        """

        var result = ""
        for try await chunk in llmClient.stream(
            messages: [(.user, prompt)],
            systemPrompt: "You generate brief, helpful daily insights."
        ) {
            result += chunk
        }

        return result.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
```

---

## 8. iOS/macOS Cross-Platform Strategy

### 8.1 Shared Code Architecture

```
+------------------------------------------------------------------+
|                         SHARED CODE (~85%)                        |
+------------------------------------------------------------------+
|                                                                    |
|  +------------------+  +------------------+  +------------------+  |
|  | Core/            |  | Data/            |  | Features/        |  |
|  | - VoiceEngine    |  | - Models         |  | - ViewModels     |  |
|  | - ConversationEng|  | - Repositories   |  | - Business Logic |  |
|  | - MemoryEngine   |  | - Sync           |  |                  |  |
|  | - KnowledgeGraph |  |                  |  |                  |  |
|  +------------------+  +------------------+  +------------------+  |
|                                                                    |
+------------------------------------------------------------------+
                              |
              +---------------+---------------+
              |                               |
+---------------------------+   +---------------------------+
|     iOS-SPECIFIC (~10%)   |   |   macOS-SPECIFIC (~5%)    |
+---------------------------+   +---------------------------+
|                           |   |                           |
| - iOS Navigation          |   | - Menu bar widget         |
| - Widgets (WidgetKit)     |   | - Keyboard shortcuts      |
| - Lock Screen integration |   | - Window management       |
| - Haptics                 |   | - NSApplication lifecycle |
| - Compact layouts         |   | - macOS-specific UI       |
|                           |   |                           |
+---------------------------+   +---------------------------+
```

### 8.2 Conditional Compilation

```swift
// MARK: - Platform-Specific Extensions

#if os(iOS)
import UIKit

extension View {
    func applyIOSStyle() -> some View {
        self
            .listStyle(.insetGrouped)
    }
}

#elseif os(macOS)
import AppKit

extension View {
    func applyMacOSStyle() -> some View {
        self
            .listStyle(.sidebar)
    }
}
#endif

// MARK: - Shared View with Platform Adaptations

struct ConversationView: View {
    @StateObject var viewModel: ConversationViewModel

    var body: some View {
        VStack {
            messagesView
            inputView
        }
        #if os(iOS)
        .navigationBarTitleDisplayMode(.inline)
        #elseif os(macOS)
        .frame(minWidth: 400, minHeight: 600)
        #endif
    }

    private var messagesView: some View {
        ScrollView {
            LazyVStack {
                ForEach(viewModel.messages) { message in
                    MessageBubble(message: message)
                }
            }
        }
    }

    private var inputView: some View {
        HStack {
            VoiceInputButton(isRecording: viewModel.isRecording) {
                Task { await viewModel.toggleRecording() }
            }

            #if os(macOS)
            // macOS also shows text input field
            TextField("Or type here...", text: $viewModel.textInput)
                .textFieldStyle(.roundedBorder)
                .onSubmit { viewModel.sendTextMessage() }
            #endif
        }
        .padding()
    }
}
```

### 8.3 macOS Menu Bar Widget

```swift
#if os(macOS)
import SwiftUI
import AppKit

@main
struct MYNDMacApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate

    var body: some Scene {
        WindowGroup {
            ContentView()
        }

        // Menu bar extra for quick capture
        MenuBarExtra("MYND", systemImage: "brain") {
            QuickCaptureMenuView()
        }
        .menuBarExtraStyle(.window)
    }
}

struct QuickCaptureMenuView: View {
    @StateObject var captureVM = QuickCaptureViewModel()

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Quick Capture")
                .font(.headline)

            HStack {
                Button(action: { captureVM.startRecording() }) {
                    Label("Voice", systemImage: "mic.fill")
                }
                .keyboardShortcut("r", modifiers: [.command])

                Button(action: { captureVM.focusTextField() }) {
                    Label("Text", systemImage: "keyboard")
                }
                .keyboardShortcut("t", modifiers: [.command])
            }

            if captureVM.isRecording {
                Text("Listening...")
                    .foregroundColor(.secondary)
            }

            Divider()

            Button("Open MYND") {
                NSWorkspace.shared.open(URL(string: "mynd://open")!)
            }

            Button("Quit") {
                NSApplication.shared.terminate(nil)
            }
            .keyboardShortcut("q")
        }
        .padding()
        .frame(width: 200)
    }
}
#endif
```

---

## 9. Development Approach

### 9.1 Phase 1: Core Voice + Chat MVP

**Goals**: Basic voice conversation with Claude, simple chat history

**Deliverables**:
- Push-to-talk voice input with Apple Speech
- Claude API streaming responses
- Apple AVSpeech TTS output
- Basic chat UI with message history
- SwiftData persistence for messages
- Keychain storage for API key
- Settings screen with voice preferences

**Critical Path Items**:
- [ ] VoiceEngine with push-to-talk
- [ ] ClaudeClient with streaming
- [ ] ConversationView with message bubbles
- [ ] API key setup flow
- [ ] Basic error handling and retry

### 9.2 Phase 2: Knowledge Graph

**Goals**: Capture and connect thoughts

**Deliverables**:
- ThoughtNode and Edge models
- In-memory graph for fast queries
- Entity extraction from conversations
- Simple list views for nodes
- 2D graph visualization (basic)
- Semantic search with embeddings

### 9.3 Phase 3: Persistent Memory

**Goals**: AI that remembers and learns

**Deliverables**:
- MemoryEngine with semantic retrieval
- Pattern learning for user preferences
- Context injection into prompts
- Memory consolidation (episodic → long-term)
- Token budget management

### 9.4 Phase 4: Proactive Features

**Goals**: AI that follows up helpfully

**Deliverables**:
- Background refresh processing
- Stale goal detection
- Follow-up notifications
- Morning briefing / evening reflection
- Daily insights

### 9.5 Phase 5: Polish & Launch

**Goals**: Production-ready app

**Deliverables**:
- CloudKit sync
- Data export
- Onboarding flow
- Accessibility audit
- Performance optimization
- macOS companion app
- Widgets (Lock Screen, Home Screen)
- App Store preparation

---

## 10. Technical Requirements

### 10.1 Minimum Requirements

| Platform | Minimum Version | Rationale |
|----------|-----------------|-----------|
| iOS | 17.0 | SwiftData, Observation macro |
| macOS | 14.0 (Sonoma) | SwiftData, Observation macro |
| Xcode | 15.0+ | Swift 5.9, SwiftData |
| Swift | 5.9+ | Observation, macros |

### 10.2 Dependencies

```swift
// Package.swift (if using SPM for any packages)

dependencies: [
    // Core functionality - all Apple frameworks, no external deps

    // Optional for enhanced features:
    // .package(url: "https://github.com/openai/whisper", from: "1.0.0"),  // On-device Whisper
]
```

**Framework Dependencies** (all Apple-native):
- SwiftUI
- SwiftData
- Speech (STT)
- AVFoundation (TTS, Audio)
- NaturalLanguage (Embeddings, NLP)
- CloudKit (Sync)
- UserNotifications (Proactive)
- BackgroundTasks (Refresh)
- WidgetKit (Widgets)

### 10.3 API Cost Estimates

| Service | Light Use | Heavy Use | Notes |
|---------|-----------|-----------|-------|
| Claude API | $1-3/mo | $5-15/mo | With user's own key |
| OpenAI Whisper | $0 | $0.50/mo | If used for STT |
| ElevenLabs | $0 | $5/mo | Premium voice option |
| CloudKit | $0 | $0 | Apple covers it |
| **Total** | **$1-3** | **$10-20** | BYOK model |

---

## 11. References

- [Apple Human Interface Guidelines](https://developer.apple.com/design/human-interface-guidelines/)
- [SwiftData Documentation](https://developer.apple.com/documentation/swiftdata)
- [Speech Framework](https://developer.apple.com/documentation/speech)
- [Anthropic Claude API](https://docs.anthropic.com/claude/reference)
- [CloudKit Documentation](https://developer.apple.com/documentation/cloudkit)
- [NLEmbedding](https://developer.apple.com/documentation/naturallanguage/nlembedding)
- [Background Tasks](https://developer.apple.com/documentation/backgroundtasks)

---

## Appendix A: Key Decisions Summary

| Decision | Choice | Alternatives Considered | Rationale |
|----------|--------|------------------------|-----------|
| App Architecture | MVVM + Observation | TCA, VIPER | Simple, native, testable |
| Graph Storage | SwiftData + In-Memory | Neo4j, Realm, SQLite | Native sync, fast queries |
| Voice STT | Apple Speech (on-device) | Whisper, Deepgram | Free, private, offline |
| Voice TTS | Apple AVSpeech | ElevenLabs, OpenAI | Free, offline, premium optional |
| LLM | Claude API | GPT-4, Local LLM | Best conversation quality |
| Sync | CloudKit | Custom backend, Supabase | Zero cost, native, encrypted |
| Embeddings | NLEmbedding | sentence-transformers | On-device, no deps |
| Wake Word | Removed (Siri Shortcuts) | Custom detector | iOS limitation |

---

**Document Status**: COMPREHENSIVE DESIGN COMPLETE
**Addresses**: All critical issues from v1.0 review
**Next Step**: Begin Phase 1 implementation
