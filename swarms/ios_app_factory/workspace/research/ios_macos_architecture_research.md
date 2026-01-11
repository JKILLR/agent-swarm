# iOS/macOS App Architecture Research for MYND
## AI-Powered Thought Capture Application (2025-2026)

**Researcher**: Research Specialist Agent
**Date**: 2026-01-04
**Version**: 1.0
**Status**: COMPREHENSIVE RESEARCH COMPLETE

---

## Executive Summary

This research document provides actionable intelligence for building a sophisticated AI-powered thought capture app targeting iOS 17+ and macOS 14+. The research synthesizes current best practices across SwiftUI maturity, shared codebase strategies, architecture patterns, background processing, push notifications, data layer options, audio handling, and performance optimization.

**Key Finding**: The iOS ecosystem in 2025-2026 has reached sufficient maturity for complex apps, but requires careful architectural decisions around SwiftUI interop, data layer technology, and background processing limitations.

---

## 1. SwiftUI vs UIKit Considerations

### 1.1 SwiftUI Maturity Assessment (iOS 18+, 2025)

SwiftUI has matured significantly since its 2019 introduction. By iOS 17/18, it handles approximately 90-95% of typical app UI requirements natively.

| Capability | iOS 17 Status | iOS 18 Status | Recommendation |
|------------|---------------|---------------|----------------|
| **Basic Layouts** | Excellent | Excellent | Use SwiftUI exclusively |
| **Lists & Grids** | Excellent | Excellent | Use SwiftUI with LazyVStack/LazyHGrid |
| **Navigation** | Good (NavigationStack) | Excellent | Use NavigationStack, avoid NavigationView |
| **Animations** | Excellent | Excellent | Use withAnimation, matchedGeometryEffect |
| **Custom Gestures** | Good | Good | Custom gesture combiners work well |
| **Drag & Drop** | Good | Good | Works but complex scenarios need care |
| **Text Input** | Good | Improved | TextEditor improved, still some focus issues |
| **Modal Presentation** | Good | Excellent | sheet/fullScreenCover reliable |
| **Scroll Performance** | Good | Excellent | scrollPosition, scrollTargetBehavior added |
| **Accessibility** | Excellent | Excellent | Best-in-class VoiceOver support |

### 1.2 Complex Interactions Analysis

#### Drag and Drop on Graphs

For the knowledge graph visualization in MYND, complex drag and drop requires consideration:

```swift
// SwiftUI Drag & Drop Pattern for Graph Nodes
struct GraphNodeView: View {
    let node: ThoughtNode
    @State private var position: CGPoint
    @GestureState private var dragOffset: CGSize = .zero

    var body: some View {
        NodeShape(node: node)
            .position(
                x: position.x + dragOffset.width,
                y: position.y + dragOffset.height
            )
            .gesture(
                DragGesture()
                    .updating($dragOffset) { value, state, _ in
                        state = value.translation
                    }
                    .onEnded { value in
                        position.x += value.translation.width
                        position.y += value.translation.height
                        // Notify graph engine of new position
                        onPositionChange(node.id, position)
                    }
            )
    }
}
```

**Recommendation**: SwiftUI can handle basic node dragging. For complex graph interactions with hundreds of nodes, consider:
- Using Canvas for drawing edges (more performant than Path views)
- Implementing spatial partitioning for hit testing
- UIKit interop for very dense graphs (>500 nodes visible)

#### Multi-Touch Gestures

```swift
// Simultaneous Gesture Pattern
struct GraphCanvasView: View {
    @State private var scale: CGFloat = 1.0
    @State private var offset: CGSize = .zero

    var body: some View {
        GraphContent()
            .scaleEffect(scale)
            .offset(offset)
            .gesture(
                SimultaneousGesture(
                    MagnificationGesture()
                        .onChanged { value in
                            scale = value
                        },
                    DragGesture()
                        .onChanged { value in
                            offset = value.translation
                        }
                )
            )
    }
}
```

### 1.3 UIKit Interop Requirements for MYND

Based on the app requirements, UIKit interop will be needed for:

| Component | Reason | Interop Approach |
|-----------|--------|------------------|
| **Audio Visualizer** | AVAudioEngine real-time | UIViewRepresentable wrapping CADisplayLink |
| **Dense Graph View** | >500 nodes performance | UIViewRepresentable wrapping Metal view |
| **Rich Text Editor** | Advanced formatting | UIViewRepresentable wrapping UITextView |
| **Camera/Photo Picker** | PhotosUI has SwiftUI but limited | PHPickerViewController via UIViewControllerRepresentable |

**UIKit Bridge Pattern**:

```swift
// Pattern for wrapping UIKit views
struct AudioVisualizerView: UIViewRepresentable {
    @Binding var audioLevel: Float

    func makeUIView(context: Context) -> WaveformView {
        let view = WaveformView()
        return view
    }

    func updateUIView(_ uiView: WaveformView, context: Context) {
        uiView.updateLevel(audioLevel)
    }

    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }

    class Coordinator: NSObject {
        var parent: AudioVisualizerView
        init(_ parent: AudioVisualizerView) {
            self.parent = parent
        }
    }
}
```

### 1.4 Component Recommendations for MYND

| MYND Component | Framework | Rationale |
|----------------|-----------|-----------|
| **Conversation View** | SwiftUI | Chat bubbles, streaming text - perfect for SwiftUI |
| **Voice Input Button** | SwiftUI | Simple interaction, animations |
| **Breathing Wall Animation** | SwiftUI | TimelineView + Canvas ideal |
| **Thought List** | SwiftUI | LazyVStack with search |
| **Settings** | SwiftUI | Form views native |
| **Graph Visualization (MVP)** | SwiftUI + Canvas | Up to ~500 nodes |
| **Graph Visualization (Scale)** | UIKit + Metal | Beyond 500 nodes |
| **Audio Waveform** | UIKit wrapped | Real-time performance |
| **Onboarding Flow** | SwiftUI | Page-based navigation |

---

## 2. Shared Codebase Strategy

### 2.1 SwiftUI Multiplatform Approach

SwiftUI's multiplatform capabilities allow significant code sharing between iOS and macOS:

```
Project Structure:
MYND/
├── Shared/                 # 70-80% of code
│   ├── Core/              # All business logic
│   ├── Features/          # Feature modules (mostly shared)
│   ├── Data/              # Models, persistence
│   └── Services/          # API clients, utilities
├── iOS/                   # 10-15% of code
│   ├── App/               # iOS app entry point
│   ├── Views/             # iOS-specific views
│   └── Extensions/        # iOS-specific extensions
├── macOS/                 # 10-15% of code
│   ├── App/               # macOS app entry point
│   ├── Views/             # macOS-specific views
│   └── Extensions/        # macOS-specific extensions
└── Tests/                 # Shared test suite
```

### 2.2 Mac Catalyst vs Native macOS

| Factor | Mac Catalyst | Native macOS (AppKit/SwiftUI) |
|--------|--------------|-------------------------------|
| **Development Speed** | Fast (reuse iOS code) | Slower (separate views) |
| **UI Quality** | iPad-like, less native | True macOS feel |
| **Menu Bar** | Limited support | Full native support |
| **Keyboard Shortcuts** | Basic support | Full support |
| **Window Management** | Basic | Full control |
| **Performance** | Good | Excellent |
| **App Size** | Larger (UIKit included) | Smaller |
| **SwiftUI** | Works (iOS views) | Works (native macOS) |

**Recommendation for MYND**: Native macOS with SwiftUI

Rationale:
1. Voice-first app benefits from proper keyboard shortcuts (desktop users)
2. Menu bar quick capture is valuable for desktop workflow
3. SwiftUI already shares most code
4. Mac Catalyst feels foreign for productivity apps

### 2.3 Shared Business Logic Layer

```swift
// MARK: - Platform-Independent Core Layer

// Protocols define contracts, shared across platforms
protocol ConversationEngineProtocol {
    func startConversation() async -> ConversationSession
    func send(message: String) async throws -> AsyncThrowingStream<String, Error>
    func endConversation() async
}

protocol VoiceEngineProtocol {
    var isListening: Bool { get }
    var currentTranscript: String { get }
    func startListening() async throws
    func stopListening()
}

// Shared implementation
@Observable
final class ConversationEngine: ConversationEngineProtocol {
    private let llmClient: LLMClientProtocol
    private let memoryEngine: MemoryEngineProtocol

    // Dependency injection for testability
    init(
        llmClient: LLMClientProtocol = ClaudeClient(),
        memoryEngine: MemoryEngineProtocol = MemoryEngine()
    ) {
        self.llmClient = llmClient
        self.memoryEngine = memoryEngine
    }

    // Implementation shared across platforms
    func send(message: String) async throws -> AsyncThrowingStream<String, Error> {
        // ... identical on iOS and macOS
    }
}
```

### 2.4 Platform-Specific Adaptations

```swift
// MARK: - Platform Adaptations

// Shared View with Platform Variations
struct VoiceInputButton: View {
    @Environment(\.voiceEngine) var voiceEngine

    var body: some View {
        Button(action: toggleRecording) {
            voiceButtonContent
        }
        .buttonStyle(VoiceButtonStyle())
        #if os(iOS)
        .hoverEffect(.lift)  // iOS-only
        #elseif os(macOS)
        .keyboardShortcut(.space, modifiers: [])  // macOS-only
        #endif
    }

    @ViewBuilder
    private var voiceButtonContent: some View {
        #if os(iOS)
        // Larger touch target for iOS
        Circle()
            .fill(voiceEngine.isListening ? Color.red : Color.blue)
            .frame(width: 80, height: 80)
        #elseif os(macOS)
        // Smaller button for macOS
        Circle()
            .fill(voiceEngine.isListening ? Color.red : Color.blue)
            .frame(width: 44, height: 44)
        #endif
    }
}

// Platform-specific implementations
#if os(iOS)
extension VoiceEngine {
    func configureAudioSession() throws {
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.playAndRecord, mode: .default, options: [.defaultToSpeaker])
        try session.setActive(true)
    }
}
#elseif os(macOS)
extension VoiceEngine {
    func configureAudioSession() throws {
        // macOS doesn't use AVAudioSession in the same way
        // Configure via AudioDeviceID if needed
    }
}
#endif
```

### 2.5 Target Version Strategy

| Platform | Minimum | Recommended | Rationale |
|----------|---------|-------------|-----------|
| **iOS** | 17.0 | 17.4+ | SwiftData stability, Observation macro |
| **macOS** | 14.0 | 14.4+ | SwiftData stability |
| **watchOS** | 10.0 | - | Future companion app |
| **visionOS** | 1.0 | - | Future spatial experience |

---

## 3. App Architecture Patterns

### 3.1 The Composable Architecture (TCA) Analysis

**Overview**: TCA is a comprehensive architecture by Point-Free that emphasizes unidirectional data flow, testability, and modularity.

#### Pros

| Advantage | Description |
|-----------|-------------|
| **Extreme Testability** | All business logic is pure functions, easily unit tested |
| **State Isolation** | Each feature has isolated state, prevents cross-contamination |
| **Side Effect Management** | Effects are declarative and testable |
| **Dependency Management** | Built-in DI system |
| **Time Travel Debugging** | Can replay state changes |
| **Modular Composition** | Features compose cleanly |

#### Cons

| Disadvantage | Description |
|--------------|-------------|
| **Learning Curve** | Steep learning curve, especially for new Swift developers |
| **Boilerplate** | Significant ceremony for simple features |
| **Bundle Size** | Adds ~2MB to app binary |
| **Compile Times** | Can slow compilation for large features |
| **Opinionated** | Requires buy-in to entire approach |
| **SwiftUI Bindings** | Bindings require special handling |

#### TCA Pattern Example

```swift
// TCA Feature Example for MYND Conversation
import ComposableArchitecture

@Reducer
struct ConversationFeature {
    @ObservableState
    struct State: Equatable {
        var messages: IdentifiedArrayOf<Message> = []
        var inputText: String = ""
        var isLoading: Bool = false
        var streamingResponse: String = ""
        @Presents var alert: AlertState<Action.Alert>?
    }

    enum Action: BindableAction {
        case binding(BindingAction<State>)
        case sendTapped
        case responseStreamReceived(String)
        case responseCompleted
        case errorOccurred(Error)
        case alert(PresentationAction<Alert>)

        enum Alert: Equatable {
            case retryTapped
            case dismissed
        }
    }

    @Dependency(\.conversationClient) var conversationClient
    @Dependency(\.uuid) var uuid

    var body: some ReducerOf<Self> {
        BindingReducer()

        Reduce { state, action in
            switch action {
            case .sendTapped:
                guard !state.inputText.isEmpty else { return .none }

                let message = Message(id: uuid(), role: .user, content: state.inputText)
                state.messages.append(message)
                state.inputText = ""
                state.isLoading = true
                state.streamingResponse = ""

                return .run { [content = message.content] send in
                    for try await chunk in try await conversationClient.stream(content) {
                        await send(.responseStreamReceived(chunk))
                    }
                    await send(.responseCompleted)
                } catch: { error, send in
                    await send(.errorOccurred(error))
                }

            case let .responseStreamReceived(chunk):
                state.streamingResponse += chunk
                return .none

            case .responseCompleted:
                let response = Message(
                    id: uuid(),
                    role: .assistant,
                    content: state.streamingResponse
                )
                state.messages.append(response)
                state.streamingResponse = ""
                state.isLoading = false
                return .none

            case let .errorOccurred(error):
                state.isLoading = false
                state.alert = AlertState {
                    TextState("Error")
                } actions: {
                    ButtonState(action: .retryTapped) {
                        TextState("Retry")
                    }
                    ButtonState(role: .cancel, action: .dismissed) {
                        TextState("Cancel")
                    }
                } message: {
                    TextState(error.localizedDescription)
                }
                return .none

            case .binding, .alert:
                return .none
            }
        }
        .ifLet(\.$alert, action: \.alert)
    }
}

// TCA View
struct ConversationView: View {
    @Bindable var store: StoreOf<ConversationFeature>

    var body: some View {
        VStack {
            ScrollView {
                LazyVStack {
                    ForEach(store.messages) { message in
                        MessageBubble(message: message)
                    }

                    if !store.streamingResponse.isEmpty {
                        StreamingBubble(text: store.streamingResponse)
                    }
                }
            }

            HStack {
                TextField("Message", text: $store.inputText)

                Button("Send") {
                    store.send(.sendTapped)
                }
                .disabled(store.isLoading)
            }
        }
        .alert($store.scope(state: \.alert, action: \.alert))
    }
}
```

**Recommendation for MYND**: Consider TCA for Phase 2+ if team is experienced, but MVVM is sufficient for MVP.

### 3.2 MVVM with Combine/async-await

**Overview**: Modern MVVM leverages Swift's Observation framework (iOS 17+) with async/await, providing simpler architecture than TCA while maintaining testability.

```swift
// MARK: - Modern MVVM Pattern for MYND

// ViewModel with @Observable (iOS 17+)
@Observable
@MainActor
final class ConversationViewModel {
    // State
    private(set) var messages: [Message] = []
    private(set) var isLoading = false
    private(set) var streamingResponse = ""
    private(set) var errorMessage: String?
    var inputText = ""

    // Dependencies (injected)
    private let conversationEngine: ConversationEngineProtocol
    private let voiceEngine: VoiceEngineProtocol
    private let memoryEngine: MemoryEngineProtocol

    init(
        conversationEngine: ConversationEngineProtocol,
        voiceEngine: VoiceEngineProtocol,
        memoryEngine: MemoryEngineProtocol
    ) {
        self.conversationEngine = conversationEngine
        self.voiceEngine = voiceEngine
        self.memoryEngine = memoryEngine
    }

    // MARK: - Actions

    func sendMessage() async {
        let content = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !content.isEmpty else { return }

        inputText = ""
        isLoading = true
        errorMessage = nil

        let userMessage = Message(role: .user, content: content)
        messages.append(userMessage)

        do {
            streamingResponse = ""

            for try await chunk in try await conversationEngine.send(message: content) {
                streamingResponse += chunk
            }

            let assistantMessage = Message(role: .assistant, content: streamingResponse)
            messages.append(assistantMessage)
            streamingResponse = ""

            // Store in memory
            await memoryEngine.store(exchange: (userMessage, assistantMessage))

        } catch {
            errorMessage = error.localizedDescription
        }

        isLoading = false
    }

    func startVoiceInput() async {
        do {
            try await voiceEngine.startListening()
        } catch {
            errorMessage = "Voice recognition failed: \(error.localizedDescription)"
        }
    }

    func stopVoiceInput() {
        voiceEngine.stopListening()
        inputText = voiceEngine.currentTranscript
    }
}

// View
struct ConversationView: View {
    @State var viewModel: ConversationViewModel

    var body: some View {
        VStack {
            MessageList(messages: viewModel.messages, streaming: viewModel.streamingResponse)

            if let error = viewModel.errorMessage {
                ErrorBanner(message: error)
            }

            InputBar(
                text: $viewModel.inputText,
                isLoading: viewModel.isLoading,
                onSend: { Task { await viewModel.sendMessage() } },
                onVoiceStart: { Task { await viewModel.startVoiceInput() } },
                onVoiceStop: { viewModel.stopVoiceInput() }
            )
        }
    }
}
```

### 3.3 Clean Architecture Layers

```
┌─────────────────────────────────────────────────────────────┐
│                    PRESENTATION LAYER                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │   Views     │  │ ViewModels  │  │  Routers    │          │
│  │  (SwiftUI)  │  │ (@Observable)│ │ (Navigation)│          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                      DOMAIN LAYER                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │  Use Cases  │  │   Entities  │  │ Repositories│          │
│  │  (Interactors)│ │   (Models)  │ │ (Protocols) │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                       DATA LAYER                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │   API       │  │  Database   │  │   Cache     │          │
│  │  Clients    │  │ (SwiftData) │  │  Managers   │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
└─────────────────────────────────────────────────────────────┘
```

### 3.4 Dependency Injection Patterns

#### Option 1: Environment-Based DI (SwiftUI Native)

```swift
// Define environment key
private struct ConversationEngineKey: EnvironmentKey {
    static let defaultValue: ConversationEngineProtocol = ConversationEngine()
}

extension EnvironmentValues {
    var conversationEngine: ConversationEngineProtocol {
        get { self[ConversationEngineKey.self] }
        set { self[ConversationEngineKey.self] = newValue }
    }
}

// Usage in App
@main
struct MYNDApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
                .environment(\.conversationEngine, ConversationEngine())
                .environment(\.voiceEngine, VoiceEngine())
                .environment(\.memoryEngine, MemoryEngine())
        }
    }
}

// Usage in Views
struct ConversationView: View {
    @Environment(\.conversationEngine) var conversationEngine

    var body: some View { /* ... */ }
}
```

#### Option 2: Container-Based DI

```swift
// DI Container
@MainActor
final class AppContainer {
    // Singletons
    lazy var llmClient: LLMClientProtocol = ClaudeClient(
        apiKey: SecureStorage.shared.getAPIKey(.anthropic)
    )

    lazy var memoryEngine: MemoryEngineProtocol = MemoryEngine(
        storage: dataContainer.graphStore,
        embeddings: localEmbeddingEngine
    )

    lazy var voiceEngine: VoiceEngineProtocol = VoiceEngine()

    lazy var conversationEngine: ConversationEngineProtocol = ConversationEngine(
        llmClient: llmClient,
        memoryEngine: memoryEngine
    )

    // Data
    lazy var dataContainer: DataContainer = DataContainer()
    lazy var localEmbeddingEngine: LocalEmbeddingEngineProtocol = LocalEmbeddingEngine()

    // Factory methods for ViewModels
    func makeConversationViewModel() -> ConversationViewModel {
        ConversationViewModel(
            conversationEngine: conversationEngine,
            voiceEngine: voiceEngine,
            memoryEngine: memoryEngine
        )
    }

    func makeSettingsViewModel() -> SettingsViewModel {
        SettingsViewModel(
            secureStorage: SecureStorage.shared,
            voiceEngine: voiceEngine
        )
    }
}

// Usage
@main
struct MYNDApp: App {
    @State private var container = AppContainer()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environment(container)
        }
    }
}
```

### 3.5 State Management Approaches

| Approach | Use Case | Pros | Cons |
|----------|----------|------|------|
| **@State** | Local view state | Simple, built-in | Not shareable |
| **@Observable** | ViewModel state | Modern, automatic | iOS 17+ only |
| **@Environment** | Shared services | SwiftUI native | Requires setup |
| **AppStorage** | User preferences | Automatic UserDefaults sync | Limited types |
| **SwiftData** | Persistent data | Database with SwiftUI integration | iOS 17+ only |

**Recommended State Architecture for MYND**:

```swift
// Global app state (Observable)
@Observable
final class AppState {
    var currentUser: User?
    var subscriptionTier: SubscriptionTier = .demo
    var isOnboarded: Bool = false
}

// Feature-specific state (in ViewModels)
@Observable
final class ConversationViewModel {
    var messages: [Message] = []
    var isListening: Bool = false
    // ...
}

// Persistent state (SwiftData)
@Model
final class ThoughtNode {
    var content: String
    var nodeType: NodeType
    // ...
}

// User preferences (AppStorage)
struct SettingsView: View {
    @AppStorage("voiceSpeed") var voiceSpeed: Double = 1.0
    @AppStorage("preferOnDeviceSTT") var preferOnDeviceSTT: Bool = true
    // ...
}
```

---

## 4. Background Processing

### 4.1 BGTaskScheduler Deep Dive

iOS severely limits background execution. Understanding these limits is critical for MYND's proactive features.

#### Background Task Types

| Task Type | Class | Max Duration | Use Case |
|-----------|-------|--------------|----------|
| **App Refresh** | BGAppRefreshTask | ~30 seconds | Sync, fetch updates |
| **Processing** | BGProcessingTask | Minutes (device-dependent) | Heavy work while charging |

#### Registration and Scheduling

```swift
// MARK: - Background Task Configuration

// 1. Info.plist Configuration
/*
<key>BGTaskSchedulerPermittedIdentifiers</key>
<array>
    <string>com.mynd.refresh</string>
    <string>com.mynd.processing</string>
</array>
*/

// 2. Task Registration (in AppDelegate or App init)
final class BackgroundTaskManager {
    static let refreshTaskIdentifier = "com.mynd.refresh"
    static let processingTaskIdentifier = "com.mynd.processing"

    func registerTasks() {
        BGTaskScheduler.shared.register(
            forTaskWithIdentifier: Self.refreshTaskIdentifier,
            using: nil
        ) { task in
            self.handleAppRefresh(task: task as! BGAppRefreshTask)
        }

        BGTaskScheduler.shared.register(
            forTaskWithIdentifier: Self.processingTaskIdentifier,
            using: nil
        ) { task in
            self.handleProcessing(task: task as! BGProcessingTask)
        }
    }

    // 3. Schedule Tasks
    func scheduleAppRefresh() {
        let request = BGAppRefreshTaskRequest(identifier: Self.refreshTaskIdentifier)
        request.earliestBeginDate = Date(timeIntervalSinceNow: 15 * 60) // 15 min minimum

        do {
            try BGTaskScheduler.shared.submit(request)
        } catch {
            print("Could not schedule app refresh: \(error)")
        }
    }

    func scheduleProcessing() {
        let request = BGProcessingTaskRequest(identifier: Self.processingTaskIdentifier)
        request.requiresNetworkConnectivity = true
        request.requiresExternalPower = true  // Important for long tasks
        request.earliestBeginDate = Date(timeIntervalSinceNow: 60 * 60) // 1 hour

        do {
            try BGTaskScheduler.shared.submit(request)
        } catch {
            print("Could not schedule processing: \(error)")
        }
    }

    // 4. Handle Tasks
    private func handleAppRefresh(task: BGAppRefreshTask) {
        // Schedule next refresh
        scheduleAppRefresh()

        // Create operation
        let operation = RefreshOperation()

        task.expirationHandler = {
            operation.cancel()
        }

        operation.completionBlock = {
            task.setTaskCompleted(success: !operation.isCancelled)
        }

        OperationQueue().addOperation(operation)
    }

    private func handleProcessing(task: BGProcessingTask) {
        // For MYND: Generate morning oracle, consolidate memories
        Task {
            do {
                try await generateMorningInsights()
                try await consolidateMemories()
                task.setTaskCompleted(success: true)
            } catch {
                task.setTaskCompleted(success: false)
            }
        }
    }
}
```

### 4.2 Background App Refresh

```swift
// MARK: - Background Refresh for MYND Proactive Features

final class ProactiveRefreshService {
    private let insightGenerator: InsightGenerator
    private let goalTracker: GoalTracker
    private let notificationManager: NotificationManager

    func performRefresh() async throws {
        // 1. Check for stale goals (quick operation)
        let staleGoals = await goalTracker.findStaleGoals(staleDays: 3)

        // 2. If we have stale goals and haven't reminded today, schedule notification
        if !staleGoals.isEmpty && !hasRemindedToday() {
            let goal = staleGoals.first!
            await notificationManager.scheduleGoalReminder(
                title: "Axel noticed something",
                body: "You haven't updated '\(goal.content.prefix(50))' in a while"
            )
        }

        // 3. If connected to network and charging, generate insight
        if await canGenerateInsight() {
            let insight = try await insightGenerator.generateQuickInsight()
            await storeInsightForMorning(insight)
        }
    }

    private func canGenerateInsight() async -> Bool {
        // Check network
        let monitor = NWPathMonitor()
        let queue = DispatchQueue(label: "network")

        return await withCheckedContinuation { continuation in
            monitor.pathUpdateHandler = { path in
                continuation.resume(returning: path.status == .satisfied)
                monitor.cancel()
            }
            monitor.start(queue: queue)
        }
    }
}
```

### 4.3 BackgroundAssets Framework (iOS 16.1+)

For downloading AI models or voice assets:

```swift
// MARK: - Background Asset Downloads

import BackgroundAssets

extension BADownloadManager {
    static func scheduleModelDownload() {
        let download = BAURLDownload(
            identifier: "com.mynd.model.voice",
            request: URLRequest(url: URL(string: "https://cdn.mynd.app/voices/axel.mlmodel")!),
            applicationGroupIdentifier: "group.com.mynd"
        )

        BADownloadManager.shared.schedule(download)
    }
}

// Handle in App Intent
struct DownloadExtension: BADownloaderExtension {
    func downloads(
        for request: BAContentRequest,
        manifestURL: URL,
        extensionInfo: BAAppExtensionInfo
    ) async -> Set<BADownload> {
        // Return downloads needed
        return []
    }

    func download(_ download: BADownload, didReceive update: BADownloadState) {
        // Handle progress/completion
    }
}
```

### 4.4 Long-Running Audio Sessions

For MYND's voice capture, audio sessions provide background execution:

```swift
// MARK: - Audio Session Configuration

final class AudioSessionManager {
    func configureForVoiceCapture() throws {
        let session = AVAudioSession.sharedInstance()

        // Category: playAndRecord for voice capture + TTS playback
        try session.setCategory(
            .playAndRecord,
            mode: .spokenAudio,
            options: [
                .defaultToSpeaker,
                .allowBluetooth,
                .allowBluetoothA2DP,
                .mixWithOthers
            ]
        )

        // Activate session
        try session.setActive(true, options: .notifyOthersOnDeactivation)
    }

    func configureForBackgroundPlayback() throws {
        // For TTS playback when app is backgrounded
        // Requires "Audio, AirPlay, and Picture in Picture" background mode

        let session = AVAudioSession.sharedInstance()
        try session.setCategory(
            .playback,
            mode: .spokenAudio,
            options: [.mixWithOthers]
        )
    }
}

// Handle interruptions
extension AudioSessionManager {
    func setupInterruptionHandling() {
        NotificationCenter.default.addObserver(
            forName: AVAudioSession.interruptionNotification,
            object: nil,
            queue: .main
        ) { [weak self] notification in
            guard let info = notification.userInfo,
                  let typeValue = info[AVAudioSessionInterruptionTypeKey] as? UInt,
                  let type = AVAudioSession.InterruptionType(rawValue: typeValue) else {
                return
            }

            switch type {
            case .began:
                // Pause recording/playback
                self?.handleInterruptionBegan()
            case .ended:
                // Resume if needed
                if let optionsValue = info[AVAudioSessionInterruptionOptionKey] as? UInt {
                    let options = AVAudioSession.InterruptionOptions(rawValue: optionsValue)
                    if options.contains(.shouldResume) {
                        self?.handleInterruptionEnded()
                    }
                }
            @unknown default:
                break
            }
        }
    }
}
```

### 4.5 Background Processing Limits and Best Practices

| Constraint | Limit | MYND Impact |
|------------|-------|-------------|
| **BGAppRefreshTask** | ~30 seconds | Quick sync only, no LLM calls |
| **BGProcessingTask** | Minutes (varies) | Generate insights when charging |
| **Minimum interval** | 15 minutes | Cannot refresh more frequently |
| **Battery state** | System-dependent | Processing requires external power |
| **Network** | Required for API calls | Queue offline, sync when connected |
| **User behavior** | Learned by system | Frequent users get more background time |

**Best Practices for MYND**:

1. **Never call LLM APIs in BGAppRefreshTask** - too slow, will timeout
2. **Generate insights during overnight charging** (BGProcessingTask)
3. **Cache aggressively** - pre-compute what you can
4. **Use local notifications** instead of real-time background processing
5. **Queue operations** - sync when app foregrounds
6. **Respect battery** - heavy processing only when charging

---

## 5. Push Notification Architecture

### 5.1 APNs Configuration

```swift
// MARK: - Push Notification Setup

// 1. Request authorization
final class NotificationManager: NSObject {
    func requestAuthorization() async throws -> Bool {
        let center = UNUserNotificationCenter.current()

        let options: UNAuthorizationOptions = [
            .alert,
            .sound,
            .badge,
            .criticalAlert,  // Requires entitlement
            .provisional     // Quiet delivery initially
        ]

        let granted = try await center.requestAuthorization(options: options)

        if granted {
            await MainActor.run {
                UIApplication.shared.registerForRemoteNotifications()
            }
        }

        return granted
    }
}

// 2. Handle device token
extension AppDelegate: UIApplicationDelegate {
    func application(
        _ application: UIApplication,
        didRegisterForRemoteNotificationsWithDeviceToken deviceToken: Data
    ) {
        let token = deviceToken.map { String(format: "%02.2hhx", $0) }.joined()
        // Send token to your backend for push delivery
        Task {
            try await APIClient.shared.registerDevice(token: token)
        }
    }

    func application(
        _ application: UIApplication,
        didFailToRegisterForRemoteNotificationsWithError error: Error
    ) {
        print("Failed to register: \(error)")
    }
}
```

### 5.2 Rich Notifications with Actions

```swift
// MARK: - Rich Notifications for MYND

extension NotificationManager {
    func setupNotificationCategories() {
        let center = UNUserNotificationCenter.current()

        // Follow-up notification category
        let followUpCategory = UNNotificationCategory(
            identifier: "FOLLOW_UP",
            actions: [
                UNNotificationAction(
                    identifier: "OPEN_CONVERSATION",
                    title: "Talk to Axel",
                    options: [.foreground]
                ),
                UNNotificationAction(
                    identifier: "SNOOZE",
                    title: "Remind me later",
                    options: []
                ),
                UNNotificationAction(
                    identifier: "MARK_DONE",
                    title: "I handled it",
                    options: [.destructive]
                )
            ],
            intentIdentifiers: [],
            options: [.customDismissAction]
        )

        // Morning insight category
        let insightCategory = UNNotificationCategory(
            identifier: "MORNING_INSIGHT",
            actions: [
                UNNotificationAction(
                    identifier: "VIEW_INSIGHT",
                    title: "See more",
                    options: [.foreground]
                ),
                UNTextInputNotificationAction(
                    identifier: "QUICK_THOUGHT",
                    title: "Capture thought",
                    options: [],
                    textInputButtonTitle: "Send",
                    textInputPlaceholder: "What's on your mind?"
                )
            ],
            intentIdentifiers: [],
            options: []
        )

        center.setNotificationCategories([followUpCategory, insightCategory])
    }

    // Schedule a rich notification
    func scheduleFollowUp(for goal: ThoughtNode, prompt: String) async {
        let content = UNMutableNotificationContent()
        content.title = "Axel has a thought"
        content.body = prompt
        content.sound = .default
        content.categoryIdentifier = "FOLLOW_UP"
        content.userInfo = [
            "goalId": goal.id.uuidString,
            "type": "followUp"
        ]

        // Add attachment (thumbnail of related content)
        if let imageURL = await generateThumbnail(for: goal) {
            if let attachment = try? UNNotificationAttachment(
                identifier: "thumbnail",
                url: imageURL,
                options: nil
            ) {
                content.attachments = [attachment]
            }
        }

        // Schedule for optimal time (learned from user behavior)
        let trigger = UNTimeIntervalNotificationTrigger(
            timeInterval: calculateOptimalDelay(),
            repeats: false
        )

        let request = UNNotificationRequest(
            identifier: "followUp-\(goal.id)",
            content: content,
            trigger: trigger
        )

        try? await UNUserNotificationCenter.current().add(request)
    }
}

// Handle notification actions
extension NotificationManager: UNUserNotificationCenterDelegate {
    func userNotificationCenter(
        _ center: UNUserNotificationCenter,
        didReceive response: UNNotificationResponse
    ) async {
        let userInfo = response.notification.request.content.userInfo

        switch response.actionIdentifier {
        case "OPEN_CONVERSATION":
            if let goalId = userInfo["goalId"] as? String {
                await DeepLinkManager.shared.navigate(to: .conversation(goalId: goalId))
            }

        case "SNOOZE":
            // Reschedule for 1 hour later
            if let goalId = userInfo["goalId"] as? String,
               let goal = await ThoughtStore.shared.fetch(id: goalId) {
                await scheduleFollowUp(for: goal, prompt: "Still thinking about this?")
            }

        case "MARK_DONE":
            if let goalId = userInfo["goalId"] as? String {
                await ThoughtStore.shared.markCompleted(id: goalId)
            }

        case "QUICK_THOUGHT":
            if let textResponse = response as? UNTextInputNotificationResponse {
                let thought = textResponse.userText
                await ThoughtStore.shared.quickCapture(content: thought)
            }

        default:
            break
        }
    }
}
```

### 5.3 Notification Service Extension

For modifying notifications before display:

```swift
// NotificationServiceExtension/NotificationService.swift

import UserNotifications

class NotificationService: UNNotificationServiceExtension {
    var contentHandler: ((UNNotificationContent) -> Void)?
    var bestAttemptContent: UNMutableNotificationContent?

    override func didReceive(
        _ request: UNNotificationRequest,
        withContentHandler contentHandler: @escaping (UNNotificationContent) -> Void
    ) {
        self.contentHandler = contentHandler
        bestAttemptContent = (request.content.mutableCopy() as? UNMutableNotificationContent)

        guard let bestAttemptContent = bestAttemptContent else { return }

        // Personalize notification based on user data
        if let insightType = request.content.userInfo["insightType"] as? String {
            switch insightType {
            case "morning":
                bestAttemptContent.title = "Good morning! Axel has thoughts for you"
            case "goalReminder":
                // Fetch goal details from shared container
                if let goalId = request.content.userInfo["goalId"] as? String,
                   let goalTitle = fetchGoalTitle(id: goalId) {
                    bestAttemptContent.body = "How's '\(goalTitle)' going?"
                }
            default:
                break
            }
        }

        // Download and attach image if needed
        if let imageURLString = request.content.userInfo["imageURL"] as? String,
           let imageURL = URL(string: imageURLString) {
            downloadImage(from: imageURL) { localURL in
                if let localURL = localURL,
                   let attachment = try? UNNotificationAttachment(
                       identifier: "image",
                       url: localURL,
                       options: nil
                   ) {
                    bestAttemptContent.attachments = [attachment]
                }
                contentHandler(bestAttemptContent)
            }
            return
        }

        contentHandler(bestAttemptContent)
    }

    override func serviceExtensionTimeWillExpire() {
        // Deliver best attempt before timeout
        if let contentHandler = contentHandler,
           let bestAttemptContent = bestAttemptContent {
            contentHandler(bestAttemptContent)
        }
    }

    private func fetchGoalTitle(id: String) -> String? {
        // Access shared container (App Groups)
        let container = UserDefaults(suiteName: "group.com.mynd")
        return container?.string(forKey: "goal-\(id)-title")
    }

    private func downloadImage(from url: URL, completion: @escaping (URL?) -> Void) {
        URLSession.shared.downloadTask(with: url) { localURL, _, _ in
            completion(localURL)
        }.resume()
    }
}
```

### 5.4 Background Push for Silent Updates

```swift
// MARK: - Silent Push Handling

// For syncing data in background without user interaction
extension AppDelegate {
    func application(
        _ application: UIApplication,
        didReceiveRemoteNotification userInfo: [AnyHashable: Any],
        fetchCompletionHandler completionHandler: @escaping (UIBackgroundFetchResult) -> Void
    ) {
        // Check if this is a silent push
        guard let aps = userInfo["aps"] as? [String: Any],
              aps["content-available"] as? Int == 1 else {
            completionHandler(.noData)
            return
        }

        // Handle different sync types
        if let syncType = userInfo["syncType"] as? String {
            Task {
                switch syncType {
                case "insights":
                    // New insights available from server
                    let success = await InsightSyncService.shared.sync()
                    completionHandler(success ? .newData : .failed)

                case "graph":
                    // Graph updates from other devices
                    let success = await GraphSyncService.shared.sync()
                    completionHandler(success ? .newData : .failed)

                default:
                    completionHandler(.noData)
                }
            }
        } else {
            completionHandler(.noData)
        }
    }
}

// Silent push payload example:
/*
{
    "aps": {
        "content-available": 1
    },
    "syncType": "insights",
    "version": "2024-01-04T10:30:00Z"
}
*/
```

### 5.5 User Notification Preferences

```swift
// MARK: - Notification Preferences

struct NotificationPreferences: Codable {
    var followUpsEnabled: Bool = true
    var morningInsightEnabled: Bool = true
    var morningInsightTime: Date = Calendar.current.date(from: DateComponents(hour: 8, minute: 0))!
    var quietHoursEnabled: Bool = true
    var quietHoursStart: Date = Calendar.current.date(from: DateComponents(hour: 22, minute: 0))!
    var quietHoursEnd: Date = Calendar.current.date(from: DateComponents(hour: 7, minute: 0))!
    var maxNotificationsPerDay: Int = 5
}

@Observable
final class NotificationPreferencesManager {
    var preferences: NotificationPreferences {
        didSet {
            save()
            updateScheduledNotifications()
        }
    }

    init() {
        self.preferences = Self.load()
    }

    var isQuietHours: Bool {
        guard preferences.quietHoursEnabled else { return false }

        let now = Date()
        let calendar = Calendar.current
        let startComponents = calendar.dateComponents([.hour, .minute], from: preferences.quietHoursStart)
        let endComponents = calendar.dateComponents([.hour, .minute], from: preferences.quietHoursEnd)
        let nowComponents = calendar.dateComponents([.hour, .minute], from: now)

        let nowMinutes = (nowComponents.hour ?? 0) * 60 + (nowComponents.minute ?? 0)
        let startMinutes = (startComponents.hour ?? 0) * 60 + (startComponents.minute ?? 0)
        let endMinutes = (endComponents.hour ?? 0) * 60 + (endComponents.minute ?? 0)

        if startMinutes > endMinutes {
            // Quiet hours span midnight
            return nowMinutes >= startMinutes || nowMinutes <= endMinutes
        } else {
            return nowMinutes >= startMinutes && nowMinutes <= endMinutes
        }
    }

    private func updateScheduledNotifications() {
        Task {
            // Cancel all pending and reschedule based on new preferences
            let center = UNUserNotificationCenter.current()
            center.removeAllPendingNotificationRequests()

            if preferences.morningInsightEnabled {
                await scheduleMorningInsight()
            }
        }
    }

    private func scheduleMorningInsight() async {
        let content = UNMutableNotificationContent()
        content.title = "Your morning insight"
        content.body = "Tap to see what Axel noticed"
        content.categoryIdentifier = "MORNING_INSIGHT"

        var dateComponents = Calendar.current.dateComponents(
            [.hour, .minute],
            from: preferences.morningInsightTime
        )

        let trigger = UNCalendarNotificationTrigger(
            dateMatching: dateComponents,
            repeats: true
        )

        let request = UNNotificationRequest(
            identifier: "morningInsight",
            content: content,
            trigger: trigger
        )

        try? await UNUserNotificationCenter.current().add(request)
    }

    private static func load() -> NotificationPreferences {
        guard let data = UserDefaults.standard.data(forKey: "notificationPreferences"),
              let preferences = try? JSONDecoder().decode(NotificationPreferences.self, from: data) else {
            return NotificationPreferences()
        }
        return preferences
    }

    private func save() {
        if let data = try? JSONEncoder().encode(preferences) {
            UserDefaults.standard.set(data, forKey: "notificationPreferences")
        }
    }
}
```

---

## 6. Data Layer

### 6.1 SwiftData vs CoreData (2025 Maturity Assessment)

| Factor | SwiftData | Core Data |
|--------|-----------|-----------|
| **Maturity** | iOS 17+ (2 years) | iOS 3+ (17 years) |
| **Stability** | Good (iOS 17.4+) | Excellent |
| **SwiftUI Integration** | Excellent | Good (via wrappers) |
| **CloudKit Sync** | Built-in | Requires setup |
| **Migration** | Lightweight auto | Manual NSMigrationManager |
| **Performance** | Good | Excellent (more tuned) |
| **Documentation** | Growing | Extensive |
| **Community Support** | Limited | Extensive |
| **Debugging Tools** | Basic | Excellent |
| **Complex Queries** | Limited | Full NSPredicate |

**Recommendation for MYND**: SwiftData for MVP with migration path planned

**Caveats**:
- Test thoroughly on iOS 17.0-17.3 (earlier versions had bugs)
- Plan for migration to Core Data or SQLite if performance issues arise
- SwiftData's relationship handling has known issues with complex graphs

### 6.2 SwiftData Implementation for MYND

```swift
// MARK: - SwiftData Models

import SwiftData

@Model
final class ThoughtNode {
    @Attribute(.unique) var id: UUID
    var content: String
    var nodeType: NodeType
    var createdAt: Date
    var lastAccessedAt: Date
    var isCompleted: Bool = false

    // Store embedding as external data
    @Attribute(.externalStorage)
    var embedding: Data?

    // Metadata as JSON
    var metadata: [String: String]?

    // Relationships - use inverse relationships for proper sync
    @Relationship(deleteRule: .cascade, inverse: \Edge.source)
    var outgoingEdges: [Edge] = []

    @Relationship(deleteRule: .cascade, inverse: \Edge.target)
    var incomingEdges: [Edge] = []

    @Relationship(deleteRule: .nullify)
    var linkedMemories: [MemoryItem] = []

    init(content: String, nodeType: NodeType) {
        self.id = UUID()
        self.content = content
        self.nodeType = nodeType
        self.createdAt = Date()
        self.lastAccessedAt = Date()
    }

    // Computed for graph operations
    var allNeighbors: [ThoughtNode] {
        let outgoing = outgoingEdges.compactMap { $0.target }
        let incoming = incomingEdges.compactMap { $0.source }
        return outgoing + incoming
    }
}

@Model
final class Edge {
    @Attribute(.unique) var id: UUID
    var edgeType: EdgeType
    var weight: Float
    var createdAt: Date

    var source: ThoughtNode?
    var target: ThoughtNode?

    init(edgeType: EdgeType, weight: Float = 1.0) {
        self.id = UUID()
        self.edgeType = edgeType
        self.weight = weight
        self.createdAt = Date()
    }
}

// Enums as raw representable for SwiftData
enum NodeType: String, Codable {
    case thought, goal, action, project, person, event, insight, question
}

enum EdgeType: String, Codable {
    case relatesTo, blocks, enables, partOf, follows, hasAction, mentions, inspiredBy
}
```

### 6.3 CloudKit Integration with SwiftData

```swift
// MARK: - CloudKit Sync Configuration

import SwiftData
import CloudKit

@main
struct MYNDApp: App {
    let modelContainer: ModelContainer

    init() {
        do {
            let schema = Schema([
                ThoughtNode.self,
                Edge.self,
                ConversationSession.self,
                Message.self,
                MemoryItem.self,
                UserPattern.self
            ])

            let configuration = ModelConfiguration(
                schema: schema,
                isStoredInMemoryOnly: false,
                cloudKitDatabase: .private("iCloud.com.mynd.app")
            )

            modelContainer = try ModelContainer(for: schema, configurations: [configuration])
        } catch {
            fatalError("Failed to create ModelContainer: \(error)")
        }
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
        }
        .modelContainer(modelContainer)
    }
}

// Custom sync conflict resolution
extension ThoughtNode {
    func merge(with other: ThoughtNode) -> ThoughtNode {
        // Keep the most recently modified version
        if self.lastAccessedAt > other.lastAccessedAt {
            return self
        }

        // Merge content if both were modified
        if self.content != other.content &&
           self.lastAccessedAt.timeIntervalSince(other.lastAccessedAt) < 60 {
            // Both modified within a minute - keep both
            self.content = "\(self.content)\n---\n\(other.content)"
        }

        // Union of relationships (never lose edges)
        let allOutgoing = Set(self.outgoingEdges.map { $0.id }).union(other.outgoingEdges.map { $0.id })
        // ... merge edges

        return self
    }
}
```

### 6.4 Keychain for Sensitive Data

```swift
// MARK: - Keychain Wrapper for API Keys

import Security

final class SecureStorage {
    static let shared = SecureStorage()

    private let serviceName = "com.mynd.app"

    enum Key: String {
        case anthropicAPIKey = "anthropic-api-key"
        case openAIAPIKey = "openai-api-key"
        case userToken = "user-token"
    }

    func store(_ value: String, for key: Key) throws {
        let data = value.data(using: .utf8)!

        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: serviceName,
            kSecAttrAccount as String: key.rawValue,
            kSecValueData as String: data,
            kSecAttrAccessible as String: kSecAttrAccessibleWhenUnlockedThisDeviceOnly
        ]

        // Delete existing
        SecItemDelete(query as CFDictionary)

        // Add new
        let status = SecItemAdd(query as CFDictionary, nil)
        guard status == errSecSuccess else {
            throw KeychainError.unableToStore(status)
        }
    }

    func retrieve(for key: Key) throws -> String? {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: serviceName,
            kSecAttrAccount as String: key.rawValue,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne
        ]

        var result: AnyObject?
        let status = SecItemCopyMatching(query as CFDictionary, &result)

        guard status == errSecSuccess,
              let data = result as? Data,
              let value = String(data: data, encoding: .utf8) else {
            if status == errSecItemNotFound {
                return nil
            }
            throw KeychainError.unableToRetrieve(status)
        }

        return value
    }

    func delete(for key: Key) throws {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: serviceName,
            kSecAttrAccount as String: key.rawValue
        ]

        let status = SecItemDelete(query as CFDictionary)
        guard status == errSecSuccess || status == errSecItemNotFound else {
            throw KeychainError.unableToDelete(status)
        }
    }

    enum KeychainError: Error {
        case unableToStore(OSStatus)
        case unableToRetrieve(OSStatus)
        case unableToDelete(OSStatus)
    }
}
```

### 6.5 File-Based Storage Patterns

```swift
// MARK: - File Storage for Large Assets

final class FileStorage {
    static let shared = FileStorage()

    private let fileManager = FileManager.default

    private var documentsURL: URL {
        fileManager.urls(for: .documentDirectory, in: .userDomainMask)[0]
    }

    private var cachesURL: URL {
        fileManager.urls(for: .cachesDirectory, in: .userDomainMask)[0]
    }

    // For voice recordings (temporary)
    func saveTemporaryAudio(_ data: Data, id: UUID) throws -> URL {
        let url = cachesURL.appendingPathComponent("audio/\(id.uuidString).m4a")
        try fileManager.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try data.write(to: url)
        return url
    }

    // For exported data
    func exportGraph(_ nodes: [ThoughtNode], edges: [Edge]) throws -> URL {
        let export = GraphExport(
            nodes: nodes.map { NodeExport(from: $0) },
            edges: edges.map { EdgeExport(from: $0) },
            exportedAt: Date(),
            version: "1.0"
        )

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(export)

        let url = documentsURL.appendingPathComponent("exports/graph-\(Date().ISO8601Format()).json")
        try fileManager.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try data.write(to: url)

        return url
    }

    // Clean up old cache files
    func cleanupCache(olderThan days: Int = 7) {
        let cutoff = Date().addingTimeInterval(-Double(days * 24 * 60 * 60))

        guard let contents = try? fileManager.contentsOfDirectory(
            at: cachesURL,
            includingPropertiesForKeys: [.creationDateKey]
        ) else { return }

        for url in contents {
            guard let values = try? url.resourceValues(forKeys: [.creationDateKey]),
                  let creationDate = values.creationDate,
                  creationDate < cutoff else { continue }

            try? fileManager.removeItem(at: url)
        }
    }
}

// Export models
struct GraphExport: Codable {
    let nodes: [NodeExport]
    let edges: [EdgeExport]
    let exportedAt: Date
    let version: String
}

struct NodeExport: Codable {
    let id: UUID
    let content: String
    let nodeType: String
    let createdAt: Date
    let metadata: [String: String]?

    init(from node: ThoughtNode) {
        self.id = node.id
        self.content = node.content
        self.nodeType = node.nodeType.rawValue
        self.createdAt = node.createdAt
        self.metadata = node.metadata
    }
}

struct EdgeExport: Codable {
    let id: UUID
    let sourceId: UUID
    let targetId: UUID
    let edgeType: String
    let weight: Float

    init(from edge: Edge) {
        self.id = edge.id
        self.sourceId = edge.source?.id ?? UUID()
        self.targetId = edge.target?.id ?? UUID()
        self.edgeType = edge.edgeType.rawValue
        self.weight = edge.weight
    }
}
```

---

## 7. Audio/Media Handling

### 7.1 AVAudioEngine for Voice Capture

```swift
// MARK: - Voice Engine with AVAudioEngine

import AVFoundation
import Speech

final class VoiceEngine: NSObject, ObservableObject {
    // Published state
    @Published private(set) var isListening = false
    @Published private(set) var isSpeaking = false
    @Published private(set) var currentTranscript = ""
    @Published private(set) var audioLevel: Float = 0

    // Audio components
    private var audioEngine: AVAudioEngine?
    private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
    private var recognitionTask: SFSpeechRecognitionTask?
    private let speechRecognizer: SFSpeechRecognizer?
    private let speechSynthesizer = AVSpeechSynthesizer()

    // Settings
    var preferOnDevice: Bool = true
    var silenceThreshold: TimeInterval = 2.0

    private var silenceTimer: Timer?

    override init() {
        self.speechRecognizer = SFSpeechRecognizer(locale: .current)
        super.init()
        speechSynthesizer.delegate = self
    }

    // MARK: - Speech Recognition

    func startListening() async throws {
        guard !isListening else { return }

        // Request authorization
        guard await requestAuthorization() else {
            throw VoiceError.notAuthorized
        }

        // Configure audio session
        try configureAudioSession()

        // Setup audio engine
        let audioEngine = AVAudioEngine()
        self.audioEngine = audioEngine

        // Create recognition request
        let request = SFSpeechAudioBufferRecognitionRequest()
        request.shouldReportPartialResults = true

        if preferOnDevice, speechRecognizer?.supportsOnDeviceRecognition == true {
            request.requiresOnDeviceRecognition = true
        }

        self.recognitionRequest = request

        // Install tap on audio input
        let inputNode = audioEngine.inputNode
        let recordingFormat = inputNode.outputFormat(forBus: 0)

        inputNode.installTap(onBus: 0, bufferSize: 1024, format: recordingFormat) { [weak self] buffer, _ in
            self?.recognitionRequest?.append(buffer)
            self?.updateAudioLevel(from: buffer)
        }

        // Start audio engine
        audioEngine.prepare()
        try audioEngine.start()

        // Start recognition
        recognitionTask = speechRecognizer?.recognitionTask(with: request) { [weak self] result, error in
            guard let self = self else { return }

            if let result = result {
                self.currentTranscript = result.bestTranscription.formattedString
                self.resetSilenceTimer()

                if result.isFinal {
                    self.handleFinalTranscript(result.bestTranscription.formattedString)
                }
            }

            if let error = error {
                self.handleRecognitionError(error)
            }
        }

        await MainActor.run {
            isListening = true
        }
    }

    func stopListening() {
        silenceTimer?.invalidate()
        silenceTimer = nil

        audioEngine?.stop()
        audioEngine?.inputNode.removeTap(onBus: 0)
        recognitionRequest?.endAudio()
        recognitionTask?.cancel()

        audioEngine = nil
        recognitionRequest = nil
        recognitionTask = nil

        isListening = false
    }

    // MARK: - Audio Level Monitoring

    private func updateAudioLevel(from buffer: AVAudioPCMBuffer) {
        guard let channelData = buffer.floatChannelData?[0] else { return }

        let frameCount = Int(buffer.frameLength)
        var sum: Float = 0

        for i in 0..<frameCount {
            sum += abs(channelData[i])
        }

        let average = sum / Float(frameCount)
        let decibels = 20 * log10(average)

        // Normalize to 0-1 range (assuming -60 to 0 dB range)
        let normalized = max(0, min(1, (decibels + 60) / 60))

        DispatchQueue.main.async {
            self.audioLevel = normalized
        }
    }

    // MARK: - Silence Detection

    private func resetSilenceTimer() {
        silenceTimer?.invalidate()
        silenceTimer = Timer.scheduledTimer(
            withTimeInterval: silenceThreshold,
            repeats: false
        ) { [weak self] _ in
            self?.handleSilenceDetected()
        }
    }

    private func handleSilenceDetected() {
        // User has stopped speaking - finalize input
        recognitionRequest?.endAudio()
    }

    // MARK: - Text-to-Speech

    func speak(_ text: String) async {
        guard !isSpeaking else { return }

        await MainActor.run {
            isSpeaking = true
        }

        // Pause listening while speaking
        let wasListening = isListening
        if wasListening {
            stopListening()
        }

        let utterance = AVSpeechUtterance(string: text)
        utterance.rate = AVSpeechUtteranceDefaultSpeechRate
        utterance.voice = AVSpeechSynthesisVoice(language: "en-US")

        await withCheckedContinuation { continuation in
            speechContinuation = continuation
            speechSynthesizer.speak(utterance)
        }

        await MainActor.run {
            isSpeaking = false
        }

        // Resume listening if it was active
        if wasListening {
            try? await startListening()
        }
    }

    private var speechContinuation: CheckedContinuation<Void, Never>?

    // MARK: - Configuration

    private func configureAudioSession() throws {
        let session = AVAudioSession.sharedInstance()

        try session.setCategory(
            .playAndRecord,
            mode: .spokenAudio,
            options: [.defaultToSpeaker, .allowBluetooth]
        )

        try session.setActive(true, options: .notifyOthersOnDeactivation)
    }

    private func requestAuthorization() async -> Bool {
        // Speech recognition authorization
        let speechStatus = await withCheckedContinuation { continuation in
            SFSpeechRecognizer.requestAuthorization { status in
                continuation.resume(returning: status)
            }
        }

        guard speechStatus == .authorized else { return false }

        // Microphone authorization
        let micStatus = await AVAudioSession.sharedInstance().requestRecordPermission()
        return micStatus
    }

    private func handleFinalTranscript(_ transcript: String) {
        // Notify delegate or publish
        NotificationCenter.default.post(
            name: .voiceTranscriptFinalized,
            object: transcript
        )
    }

    private func handleRecognitionError(_ error: Error) {
        stopListening()
        // Handle specific errors
    }
}

// MARK: - AVSpeechSynthesizerDelegate

extension VoiceEngine: AVSpeechSynthesizerDelegate {
    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
        speechContinuation?.resume()
        speechContinuation = nil
    }
}

// MARK: - Errors

enum VoiceError: Error {
    case notAuthorized
    case audioEngineError
    case recognitionError
}

// MARK: - Notifications

extension Notification.Name {
    static let voiceTranscriptFinalized = Notification.Name("voiceTranscriptFinalized")
}
```

### 7.2 Audio Session Management

```swift
// MARK: - Audio Session Manager

final class AudioSessionManager {
    static let shared = AudioSessionManager()

    enum Mode {
        case voiceCapture       // Recording user voice
        case playback           // Playing AI response
        case background         // Background audio (if needed)
    }

    private var currentMode: Mode?

    func configure(for mode: Mode) throws {
        let session = AVAudioSession.sharedInstance()

        switch mode {
        case .voiceCapture:
            try session.setCategory(
                .playAndRecord,
                mode: .spokenAudio,
                options: [
                    .defaultToSpeaker,
                    .allowBluetooth,
                    .allowBluetoothA2DP,
                    .interruptSpokenAudioAndMixWithOthers
                ]
            )

        case .playback:
            try session.setCategory(
                .playback,
                mode: .spokenAudio,
                options: [.mixWithOthers]
            )

        case .background:
            try session.setCategory(
                .playback,
                mode: .default,
                options: []
            )
        }

        try session.setActive(true, options: .notifyOthersOnDeactivation)
        currentMode = mode
    }

    func deactivate() throws {
        let session = AVAudioSession.sharedInstance()
        try session.setActive(false, options: .notifyOthersOnDeactivation)
        currentMode = nil
    }

    // Handle interruptions (phone calls, Siri, etc.)
    func setupInterruptionHandling(
        onInterrupt: @escaping () -> Void,
        onResume: @escaping () -> Void
    ) {
        NotificationCenter.default.addObserver(
            forName: AVAudioSession.interruptionNotification,
            object: nil,
            queue: .main
        ) { notification in
            guard let info = notification.userInfo,
                  let typeValue = info[AVAudioSessionInterruptionTypeKey] as? UInt,
                  let type = AVAudioSession.InterruptionType(rawValue: typeValue) else {
                return
            }

            switch type {
            case .began:
                onInterrupt()

            case .ended:
                if let optionsValue = info[AVAudioSessionInterruptionOptionKey] as? UInt {
                    let options = AVAudioSession.InterruptionOptions(rawValue: optionsValue)
                    if options.contains(.shouldResume) {
                        onResume()
                    }
                }

            @unknown default:
                break
            }
        }
    }

    // Handle route changes (headphones connected/disconnected)
    func setupRouteChangeHandling(onChange: @escaping (AVAudioSession.RouteChangeReason) -> Void) {
        NotificationCenter.default.addObserver(
            forName: AVAudioSession.routeChangeNotification,
            object: nil,
            queue: .main
        ) { notification in
            guard let info = notification.userInfo,
                  let reasonValue = info[AVAudioSessionRouteChangeReasonKey] as? UInt,
                  let reason = AVAudioSession.RouteChangeReason(rawValue: reasonValue) else {
                return
            }

            onChange(reason)
        }
    }
}
```

### 7.3 Background Audio

```swift
// MARK: - Background Audio for TTS Playback

// Required in Info.plist:
/*
<key>UIBackgroundModes</key>
<array>
    <string>audio</string>
</array>
*/

extension VoiceEngine {
    func speakInBackground(_ text: String) async {
        // Configure for background playback
        try? AudioSessionManager.shared.configure(for: .background)

        // Enable background audio
        let session = AVAudioSession.sharedInstance()
        try? session.setCategory(.playback, mode: .spokenAudio)

        await speak(text)

        // Restore normal configuration
        try? AudioSessionManager.shared.configure(for: .voiceCapture)
    }
}

// NowPlaying info for Control Center
import MediaPlayer

extension VoiceEngine {
    func updateNowPlayingInfo(title: String, isPlaying: Bool) {
        var nowPlayingInfo = [String: Any]()
        nowPlayingInfo[MPMediaItemPropertyTitle] = "MYND - Axel"
        nowPlayingInfo[MPMediaItemPropertyArtist] = title
        nowPlayingInfo[MPNowPlayingInfoPropertyPlaybackRate] = isPlaying ? 1.0 : 0.0

        MPNowPlayingInfoCenter.default().nowPlayingInfo = nowPlayingInfo
    }

    func setupRemoteCommands() {
        let commandCenter = MPRemoteCommandCenter.shared()

        commandCenter.pauseCommand.addTarget { [weak self] _ in
            self?.speechSynthesizer.pauseSpeaking(at: .immediate)
            return .success
        }

        commandCenter.playCommand.addTarget { [weak self] _ in
            self?.speechSynthesizer.continueSpeaking()
            return .success
        }
    }
}
```

### 7.4 Photo/Document Import

```swift
// MARK: - Photo and Document Import

import PhotosUI
import UniformTypeIdentifiers

// Photo Picker (SwiftUI)
struct PhotoImportView: View {
    @State private var selectedItems: [PhotosPickerItem] = []
    @State private var importedImages: [UIImage] = []

    var onImport: ([UIImage]) -> Void

    var body: some View {
        PhotosPicker(
            selection: $selectedItems,
            maxSelectionCount: 5,
            matching: .images
        ) {
            Label("Add Photos", systemImage: "photo.on.rectangle.angled")
        }
        .onChange(of: selectedItems) { _, newItems in
            Task {
                importedImages = []
                for item in newItems {
                    if let data = try? await item.loadTransferable(type: Data.self),
                       let image = UIImage(data: data) {
                        importedImages.append(image)
                    }
                }
                onImport(importedImages)
            }
        }
    }
}

// Document Picker
struct DocumentImportView: UIViewControllerRepresentable {
    let types: [UTType]
    let onPick: ([URL]) -> Void

    func makeUIViewController(context: Context) -> UIDocumentPickerViewController {
        let picker = UIDocumentPickerViewController(forOpeningContentTypes: types)
        picker.allowsMultipleSelection = true
        picker.delegate = context.coordinator
        return picker
    }

    func updateUIViewController(_ uiViewController: UIDocumentPickerViewController, context: Context) {}

    func makeCoordinator() -> Coordinator {
        Coordinator(onPick: onPick)
    }

    class Coordinator: NSObject, UIDocumentPickerDelegate {
        let onPick: ([URL]) -> Void

        init(onPick: @escaping ([URL]) -> Void) {
            self.onPick = onPick
        }

        func documentPicker(_ controller: UIDocumentPickerViewController, didPickDocumentsAt urls: [URL]) {
            onPick(urls)
        }
    }
}

// Usage in MYND for attaching to thoughts
extension ConversationViewModel {
    func attachPhotos(_ images: [UIImage]) async {
        for image in images {
            // Compress and store
            if let data = image.jpegData(compressionQuality: 0.8) {
                let attachment = try? await FileStorage.shared.saveAttachment(data, type: .image)
                // Associate with current thought
            }
        }
    }

    func attachDocument(at url: URL) async throws {
        // Start accessing security-scoped resource
        guard url.startAccessingSecurityScopedResource() else {
            throw AttachmentError.accessDenied
        }

        defer { url.stopAccessingSecurityScopedResource() }

        let data = try Data(contentsOf: url)
        let attachment = try await FileStorage.shared.saveAttachment(data, type: .document)
        // Associate with current thought
    }
}

enum AttachmentError: Error {
    case accessDenied
    case tooLarge
    case unsupportedFormat
}
```

---

## 8. Performance and Optimization

### 8.1 Memory Management for Graphs

```swift
// MARK: - Memory-Efficient Graph Handling

// In-Memory Graph with Lazy Loading
final class InMemoryGraph {
    // Store only IDs and lightweight metadata
    private var adjacencyList: [UUID: Set<UUID>] = [:]
    private var nodeMetadata: [UUID: NodeMetadata] = [:]

    // Weak cache for full node objects
    private var nodeCache = NSMapTable<NSUUID, ThoughtNode>.strongToWeakObjects()

    struct NodeMetadata {
        let nodeType: NodeType
        let contentPreview: String  // First 100 chars
        let lastAccessed: Date
    }

    // Load from SwiftData on launch
    func loadFromStore(_ store: ModelContext) async {
        let descriptor = FetchDescriptor<ThoughtNode>()

        guard let nodes = try? store.fetch(descriptor) else { return }

        for node in nodes {
            let metadata = NodeMetadata(
                nodeType: node.nodeType,
                contentPreview: String(node.content.prefix(100)),
                lastAccessed: node.lastAccessedAt
            )
            nodeMetadata[node.id] = metadata

            // Build adjacency list
            adjacencyList[node.id] = Set(node.allNeighbors.map { $0.id })
        }
    }

    // Get neighbors (O(1) lookup)
    func neighbors(of nodeId: UUID) -> Set<UUID> {
        adjacencyList[nodeId] ?? []
    }

    // BFS traversal
    func findPath(from: UUID, to: UUID, maxDepth: Int = 10) -> [UUID]? {
        var queue: [(UUID, [UUID])] = [(from, [from])]
        var visited: Set<UUID> = [from]

        while !queue.isEmpty {
            let (current, path) = queue.removeFirst()

            if current == to {
                return path
            }

            if path.count >= maxDepth {
                continue
            }

            for neighbor in neighbors(of: current) where !visited.contains(neighbor) {
                visited.insert(neighbor)
                queue.append((neighbor, path + [neighbor]))
            }
        }

        return nil
    }

    // Get full node (lazy load from store)
    func getNode(_ id: UUID, from store: ModelContext) -> ThoughtNode? {
        // Check cache first
        if let cached = nodeCache.object(forKey: id as NSUUID) {
            return cached
        }

        // Load from store
        let descriptor = FetchDescriptor<ThoughtNode>(
            predicate: #Predicate { $0.id == id }
        )

        guard let node = try? store.fetch(descriptor).first else {
            return nil
        }

        // Cache weakly
        nodeCache.setObject(node, forKey: id as NSUUID)

        return node
    }

    // Memory pressure handling
    func handleMemoryWarning() {
        // Clear the cache - nodes will be reloaded on demand
        nodeCache.removeAllObjects()
    }
}
```

### 8.2 Lazy Loading Strategies

```swift
// MARK: - Lazy Loading Patterns

// Paginated fetch for thought list
@Observable
final class ThoughtListViewModel {
    private(set) var thoughts: [ThoughtNode] = []
    private(set) var isLoading = false
    private(set) var hasMore = true

    private let pageSize = 50
    private var currentOffset = 0

    private let store: ModelContext

    init(store: ModelContext) {
        self.store = store
    }

    func loadNextPage() async {
        guard !isLoading, hasMore else { return }

        isLoading = true

        var descriptor = FetchDescriptor<ThoughtNode>(
            sortBy: [SortDescriptor(\.createdAt, order: .reverse)]
        )
        descriptor.fetchOffset = currentOffset
        descriptor.fetchLimit = pageSize

        do {
            let newThoughts = try store.fetch(descriptor)

            await MainActor.run {
                thoughts.append(contentsOf: newThoughts)
                currentOffset += newThoughts.count
                hasMore = newThoughts.count == pageSize
                isLoading = false
            }
        } catch {
            isLoading = false
        }
    }

    func refresh() async {
        currentOffset = 0
        hasMore = true
        thoughts = []
        await loadNextPage()
    }
}

// Lazy view loading in SwiftUI
struct ThoughtListView: View {
    @State var viewModel: ThoughtListViewModel

    var body: some View {
        List {
            ForEach(viewModel.thoughts) { thought in
                ThoughtRow(thought: thought)
                    .onAppear {
                        // Load more when approaching end
                        if thought.id == viewModel.thoughts.last?.id {
                            Task { await viewModel.loadNextPage() }
                        }
                    }
            }

            if viewModel.isLoading {
                ProgressView()
            }
        }
        .refreshable {
            await viewModel.refresh()
        }
    }
}
```

### 8.3 Main Thread Protection

```swift
// MARK: - Main Thread Safety

// Use @MainActor for UI-related classes
@MainActor
@Observable
final class ConversationViewModel {
    // All property access is on main thread
    private(set) var messages: [Message] = []

    func addMessage(_ message: Message) {
        // Guaranteed main thread
        messages.append(message)
    }
}

// Background processing with main thread updates
actor BackgroundProcessor {
    func processIntensiveTask() async -> Result {
        // Heavy computation happens off main thread
        let result = await performComputation()
        return result
    }
}

// Usage pattern
class SomeViewModel {
    private let processor = BackgroundProcessor()

    func doWork() async {
        // Start on whatever thread
        let result = await processor.processIntensiveTask()

        // Update UI on main thread
        await MainActor.run {
            self.updateUI(with: result)
        }
    }
}

// Async stream processing
extension ConversationEngine {
    func streamResponse() async throws -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            Task.detached(priority: .userInitiated) {
                // Network/processing on background thread
                do {
                    for try await chunk in await self.llmClient.stream() {
                        // Yield can happen on any thread
                        continuation.yield(chunk)
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }
}
```

### 8.4 Instruments Profiling Guide

```swift
// MARK: - Performance Monitoring

import os.signpost

// Create signpost for tracing
private let performanceLog = OSLog(subsystem: "com.mynd.app", category: "Performance")

extension ConversationEngine {
    func send(message: String) async throws {
        let signpostID = OSSignpostID(log: performanceLog)

        // Start interval
        os_signpost(.begin, log: performanceLog, name: "Send Message", signpostID: signpostID)

        defer {
            // End interval
            os_signpost(.end, log: performanceLog, name: "Send Message", signpostID: signpostID)
        }

        // Mark API call
        os_signpost(.event, log: performanceLog, name: "API Call Start")

        // ... perform work
    }
}

// Memory tracking
import Foundation

final class MemoryMonitor {
    static let shared = MemoryMonitor()

    var currentMemoryUsage: UInt64 {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4

        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }

        return result == KERN_SUCCESS ? info.resident_size : 0
    }

    func logMemoryUsage(label: String) {
        let usage = currentMemoryUsage
        let mb = Double(usage) / 1_000_000
        print("[\(label)] Memory: \(String(format: "%.2f", mb)) MB")
    }

    func startMemoryWarningObserver(onWarning: @escaping () -> Void) {
        NotificationCenter.default.addObserver(
            forName: UIApplication.didReceiveMemoryWarningNotification,
            object: nil,
            queue: .main
        ) { _ in
            onWarning()
        }
    }
}

// Performance test utilities
final class PerformanceTestHelper {
    static func measure<T>(label: String, block: () async throws -> T) async rethrows -> T {
        let start = CFAbsoluteTimeGetCurrent()
        let result = try await block()
        let end = CFAbsoluteTimeGetCurrent()
        print("[\(label)] Duration: \(String(format: "%.3f", end - start))s")
        return result
    }
}

// Usage
// await PerformanceTestHelper.measure(label: "Graph Load") {
//     await graph.loadFromStore(context)
// }
```

---

## 9. Architecture Decision Matrix

### For MYND Specifically

| Decision | Recommendation | Alternatives Considered | Rationale |
|----------|----------------|------------------------|-----------|
| **UI Framework** | SwiftUI + UIKit interop | Pure SwiftUI | Graph viz needs Metal/UIKit for scale |
| **Architecture** | MVVM + @Observable | TCA, VIPER | Simpler for MVP, sufficient testability |
| **Data Layer** | SwiftData + in-memory graph | Pure SwiftData, Core Data | SwiftData for sync, in-memory for queries |
| **Sync** | CloudKit via SwiftData | Custom backend | Zero cost, automatic encryption |
| **Voice** | Apple Speech + AVFoundation | Third-party SDKs | Privacy, no cost, sufficient quality |
| **LLM** | Claude API (BYOK) | OpenAI, local LLM | Context window, tool use, streaming |
| **DI Pattern** | Environment + Container | Pure Environment | Container for complex dependencies |
| **Background** | BGTaskScheduler + Notifications | None | Limited but sufficient for proactive |
| **Multiplatform** | Native macOS (not Catalyst) | Mac Catalyst | Better desktop experience |

---

## 10. Recommended Reading

### Apple Documentation
- SwiftUI: https://developer.apple.com/documentation/swiftui
- SwiftData: https://developer.apple.com/documentation/swiftdata
- CloudKit: https://developer.apple.com/documentation/cloudkit
- Speech: https://developer.apple.com/documentation/speech
- AVFoundation: https://developer.apple.com/documentation/avfoundation
- BackgroundTasks: https://developer.apple.com/documentation/backgroundtasks

### Architecture Resources
- Point-Free (TCA): https://www.pointfree.co
- Swift by Sundell: https://www.swiftbysundell.com
- Hacking with Swift: https://www.hackingwithswift.com

### Performance
- WWDC Sessions on Performance
- Instruments documentation
- Apple's Energy Efficiency Guide

---

## Key Findings Summary

### Most Important Discoveries

1. **SwiftUI is production-ready for 90%+ of MYND's UI**, with UIKit interop needed only for dense graph visualization and audio waveforms.

2. **SwiftData works for MVP** but has fundamental limitations for graph queries; the hybrid approach (SwiftData for storage, in-memory for queries) is correct.

3. **Background processing is severely limited on iOS** - proactive AI features must work within BGTaskScheduler constraints (30 seconds for refresh, minutes for processing while charging).

4. **Voice latency cannot be eliminated** with current Claude API - the "thoughtful companion" reframe with immediate acknowledgments is the right solution.

5. **Native macOS > Mac Catalyst** for a productivity app requiring keyboard shortcuts and menu bar integration.

---

*Research completed by Research Specialist Agent*
*Document version: 1.0*
*Last updated: 2026-01-04*
