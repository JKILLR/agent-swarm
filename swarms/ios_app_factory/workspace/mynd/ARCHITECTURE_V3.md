# MYND - Architecture v3.0: Strategic Resilience Edition

**Version**: 3.0
**Date**: 2026-01-04
**Status**: STRATEGIC ARCHITECTURE - Responding to Adversarial Critique
**Purpose**: Build a system that survives competition, pivots gracefully, and ships faster

---

## Executive Summary

This architecture addresses four critical strategic adjustments identified in the v2 critique:

1. **Multi-model LLM architecture** - Not locked to Anthropic
2. **Accelerated 20-24 week timeline** - Beat WWDC 2026
3. **Text-first UX** - Voice as enhancement, not dependency
4. **Plan B architecture** - Pivot-ready design patterns

### Key Principles

| Principle | Implementation |
|-----------|----------------|
| **Provider Independence** | Abstract all external APIs behind protocols |
| **Graceful Degradation** | Every premium feature has a free fallback |
| **Scope Discipline** | Cut features, not quality |
| **Pivot Ready** | Architecture supports 3 different business models |

---

## 1. Multi-Model LLM Architecture

### 1.1 Why Multi-Model Matters

| Risk | Probability | Multi-Model Mitigation |
|------|-------------|------------------------|
| Claude API costs double | 40% | Switch to GPT-4o-mini or Gemini Flash |
| Anthropic rate limits during launch | 60% | Fallback to OpenAI |
| Apple requires on-device AI | 20% | Local model support ready |
| User demands model choice | 30% | BYOK with any provider |

### 1.2 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           APPLICATION LAYER                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│  ConversationEngine  │  ProactiveEngine  │  InsightGenerator               │
└──────────────────────┴───────────────────┴──────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         LLM ABSTRACTION LAYER                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                        LLMProviderProtocol                              │ │
│  │                                                                         │ │
│  │  func complete(messages:, tools:, stream:) async throws -> LLMResponse │ │
│  │  func embed(text:) async throws -> [Float]                             │ │
│  │  var capabilities: LLMCapabilities { get }                              │ │
│  │  var costPerToken: CostEstimate { get }                                 │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                         │
│         ┌──────────────┬───────────┼───────────┬──────────────┐             │
│         ▼              ▼           ▼           ▼              ▼             │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌───────────┐ │
│  │  Claude    │ │  GPT-4     │ │  Gemini    │ │  Local     │ │  Apple    │ │
│  │  Provider  │ │  Provider  │ │  Provider  │ │  MLX       │ │  Intel.   │ │
│  │            │ │            │ │            │ │  Provider  │ │  Provider │ │
│  │ Haiku/     │ │ 4o-mini/   │ │ Flash/     │ │ Llama3/    │ │ (Future)  │ │
│  │ Sonnet/    │ │ 4o/4       │ │ Pro        │ │ Mistral    │ │           │ │
│  │ Opus       │ │            │ │            │ │            │ │           │ │
│  └────────────┘ └────────────┘ └────────────┘ └────────────┘ └───────────┘ │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                         ROUTING & FALLBACK LAYER                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                        LLMRouter                                        │ │
│  │                                                                         │ │
│  │  - Primary/Secondary/Tertiary provider chain                           │ │
│  │  - Automatic failover on error/timeout                                 │ │
│  │  - Cost-based routing (cheap for simple, expensive for complex)        │ │
│  │  - Rate limit detection and provider switching                         │ │
│  │  - Usage tracking per provider                                          │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.3 Swift Protocol Definitions

```swift
// MARK: - Core LLM Protocol

/// Protocol that all LLM providers must implement
protocol LLMProviderProtocol: Sendable {
    /// Unique identifier for this provider (e.g., "claude", "openai", "local")
    var providerId: String { get }

    /// Human-readable name for settings UI
    var displayName: String { get }

    /// What this provider can do
    var capabilities: LLMCapabilities { get }

    /// Cost estimation for budgeting
    var costEstimate: LLMCostEstimate { get }

    /// Whether provider is currently available (API key set, network up, etc.)
    var isAvailable: Bool { get async }

    /// Complete a conversation with optional streaming
    func complete(
        messages: [LLMMessage],
        systemPrompt: String,
        tools: [LLMTool]?,
        options: LLMCompletionOptions
    ) async throws -> LLMResponse

    /// Stream completion tokens
    func streamComplete(
        messages: [LLMMessage],
        systemPrompt: String,
        tools: [LLMTool]?,
        options: LLMCompletionOptions
    ) -> AsyncThrowingStream<LLMStreamChunk, Error>

    /// Generate text embeddings for semantic search
    func embed(texts: [String]) async throws -> [[Float]]

    /// Cancel any in-flight requests
    func cancelAll()
}

// MARK: - Supporting Types

struct LLMCapabilities: Sendable {
    let maxContextTokens: Int
    let supportsToolUse: Bool
    let supportsStreaming: Bool
    let supportsVision: Bool
    let supportsEmbeddings: Bool
    let isOnDevice: Bool
    let latencyClass: LatencyClass

    enum LatencyClass: Sendable {
        case instant      // <100ms - local models
        case fast         // 100-500ms - optimized cloud
        case standard     // 500-2000ms - typical cloud
        case slow         // 2000ms+ - complex reasoning
    }
}

struct LLMCostEstimate: Sendable {
    let inputTokenCost: Decimal      // Cost per 1M tokens
    let outputTokenCost: Decimal     // Cost per 1M tokens
    let embeddingCost: Decimal       // Cost per 1M tokens

    func estimateCost(inputTokens: Int, outputTokens: Int) -> Decimal {
        let inputCost = Decimal(inputTokens) * inputTokenCost / 1_000_000
        let outputCost = Decimal(outputTokens) * outputTokenCost / 1_000_000
        return inputCost + outputCost
    }
}

struct LLMMessage: Sendable, Codable {
    enum Role: String, Sendable, Codable {
        case system
        case user
        case assistant
        case tool
    }

    let role: Role
    let content: String
    let toolCallId: String?
    let toolCalls: [LLMToolCall]?
}

struct LLMCompletionOptions: Sendable {
    var temperature: Float = 0.7
    var maxTokens: Int = 2048
    var stopSequences: [String] = []
    var timeout: TimeInterval = 30
}

struct LLMResponse: Sendable {
    let content: String
    let toolCalls: [LLMToolCall]
    let usage: LLMUsage
    let finishReason: FinishReason

    enum FinishReason: Sendable {
        case completed
        case maxTokens
        case toolUse
        case contentFilter
        case error(String)
    }
}

struct LLMUsage: Sendable {
    let inputTokens: Int
    let outputTokens: Int
    let totalTokens: Int
    let estimatedCost: Decimal
}

struct LLMStreamChunk: Sendable {
    let delta: String
    let toolCallDelta: LLMToolCallDelta?
    let isComplete: Bool
    let usage: LLMUsage?
}
```

### 1.4 Provider Implementations

```swift
// MARK: - Claude Provider

final class ClaudeProvider: LLMProviderProtocol {
    let providerId = "claude"
    let displayName = "Claude (Anthropic)"

    var capabilities: LLMCapabilities {
        LLMCapabilities(
            maxContextTokens: 200_000,
            supportsToolUse: true,
            supportsStreaming: true,
            supportsVision: true,
            supportsEmbeddings: false,  // Use Voyage AI instead
            isOnDevice: false,
            latencyClass: .standard
        )
    }

    var costEstimate: LLMCostEstimate {
        // Claude 3.5 Haiku pricing
        LLMCostEstimate(
            inputTokenCost: 0.25,
            outputTokenCost: 1.25,
            embeddingCost: 0
        )
    }

    private let apiKey: () -> String?
    private let session: URLSession
    private let baseURL = URL(string: "https://api.anthropic.com/v1")!

    var isAvailable: Bool {
        get async {
            guard apiKey() != nil else { return false }
            // Could add health check here
            return true
        }
    }

    init(apiKeyProvider: @escaping () -> String?) {
        self.apiKey = apiKeyProvider
        self.session = URLSession(configuration: .default)
    }

    func complete(
        messages: [LLMMessage],
        systemPrompt: String,
        tools: [LLMTool]?,
        options: LLMCompletionOptions
    ) async throws -> LLMResponse {
        guard let key = apiKey() else {
            throw LLMError.notConfigured("Claude API key not set")
        }

        var request = URLRequest(url: baseURL.appendingPathComponent("messages"))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue(key, forHTTPHeaderField: "x-api-key")
        request.setValue("2023-06-01", forHTTPHeaderField: "anthropic-version")
        request.timeoutInterval = options.timeout

        let body = ClaudeRequestBody(
            model: "claude-3-5-haiku-20241022",
            maxTokens: options.maxTokens,
            system: systemPrompt,
            messages: messages.map { ClaudeMessage(from: $0) },
            tools: tools?.map { ClaudeTool(from: $0) }
        )

        request.httpBody = try JSONEncoder().encode(body)

        let (data, response) = try await session.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse else {
            throw LLMError.invalidResponse
        }

        guard httpResponse.statusCode == 200 else {
            throw LLMError.apiError(
                statusCode: httpResponse.statusCode,
                message: String(data: data, encoding: .utf8) ?? "Unknown error"
            )
        }

        let claudeResponse = try JSONDecoder().decode(ClaudeResponse.self, from: data)
        return claudeResponse.toLLMResponse()
    }

    func streamComplete(
        messages: [LLMMessage],
        systemPrompt: String,
        tools: [LLMTool]?,
        options: LLMCompletionOptions
    ) -> AsyncThrowingStream<LLMStreamChunk, Error> {
        AsyncThrowingStream { continuation in
            Task {
                // SSE streaming implementation
                // ... (similar to complete but with stream: true)
            }
        }
    }

    func embed(texts: [String]) async throws -> [[Float]] {
        throw LLMError.notSupported("Claude doesn't support embeddings directly")
    }

    func cancelAll() {
        session.getAllTasks { tasks in
            tasks.forEach { $0.cancel() }
        }
    }
}

// MARK: - OpenAI Provider

final class OpenAIProvider: LLMProviderProtocol {
    let providerId = "openai"
    let displayName = "GPT-4 (OpenAI)"

    var capabilities: LLMCapabilities {
        LLMCapabilities(
            maxContextTokens: 128_000,
            supportsToolUse: true,
            supportsStreaming: true,
            supportsVision: true,
            supportsEmbeddings: true,
            isOnDevice: false,
            latencyClass: .fast
        )
    }

    var costEstimate: LLMCostEstimate {
        // GPT-4o-mini pricing
        LLMCostEstimate(
            inputTokenCost: 0.15,
            outputTokenCost: 0.60,
            embeddingCost: 0.02  // text-embedding-3-small
        )
    }

    private let apiKey: () -> String?
    private let baseURL = URL(string: "https://api.openai.com/v1")!

    var isAvailable: Bool {
        get async { apiKey() != nil }
    }

    init(apiKeyProvider: @escaping () -> String?) {
        self.apiKey = apiKeyProvider
    }

    func complete(
        messages: [LLMMessage],
        systemPrompt: String,
        tools: [LLMTool]?,
        options: LLMCompletionOptions
    ) async throws -> LLMResponse {
        // OpenAI Chat Completions API implementation
        // Similar structure to Claude but different request/response format
        fatalError("Implementation omitted for brevity")
    }

    func streamComplete(
        messages: [LLMMessage],
        systemPrompt: String,
        tools: [LLMTool]?,
        options: LLMCompletionOptions
    ) -> AsyncThrowingStream<LLMStreamChunk, Error> {
        AsyncThrowingStream { continuation in
            // SSE streaming implementation
        }
    }

    func embed(texts: [String]) async throws -> [[Float]] {
        guard let key = apiKey() else {
            throw LLMError.notConfigured("OpenAI API key not set")
        }

        // text-embedding-3-small implementation
        fatalError("Implementation omitted for brevity")
    }

    func cancelAll() { }
}

// MARK: - Gemini Provider

final class GeminiProvider: LLMProviderProtocol {
    let providerId = "gemini"
    let displayName = "Gemini (Google)"

    var capabilities: LLMCapabilities {
        LLMCapabilities(
            maxContextTokens: 1_000_000,  // Gemini 1.5 Pro
            supportsToolUse: true,
            supportsStreaming: true,
            supportsVision: true,
            supportsEmbeddings: true,
            isOnDevice: false,
            latencyClass: .fast
        )
    }

    var costEstimate: LLMCostEstimate {
        // Gemini 1.5 Flash pricing (very cheap)
        LLMCostEstimate(
            inputTokenCost: 0.075,
            outputTokenCost: 0.30,
            embeddingCost: 0.00
        )
    }

    var isAvailable: Bool {
        get async { apiKey() != nil }
    }

    private let apiKey: () -> String?

    init(apiKeyProvider: @escaping () -> String?) {
        self.apiKey = apiKeyProvider
    }

    // ... Implementation similar to others
    func complete(messages: [LLMMessage], systemPrompt: String, tools: [LLMTool]?, options: LLMCompletionOptions) async throws -> LLMResponse {
        fatalError("Implementation omitted for brevity")
    }

    func streamComplete(messages: [LLMMessage], systemPrompt: String, tools: [LLMTool]?, options: LLMCompletionOptions) -> AsyncThrowingStream<LLMStreamChunk, Error> {
        AsyncThrowingStream { _ in }
    }

    func embed(texts: [String]) async throws -> [[Float]] {
        fatalError("Implementation omitted for brevity")
    }

    func cancelAll() { }
}

// MARK: - Local MLX Provider (On-Device)

final class LocalMLXProvider: LLMProviderProtocol {
    let providerId = "local"
    let displayName = "On-Device (Private)"

    var capabilities: LLMCapabilities {
        LLMCapabilities(
            maxContextTokens: 8_000,
            supportsToolUse: false,  // Most local models don't
            supportsStreaming: true,
            supportsVision: false,
            supportsEmbeddings: true,
            isOnDevice: true,
            latencyClass: .instant
        )
    }

    var costEstimate: LLMCostEstimate {
        // Free!
        LLMCostEstimate(
            inputTokenCost: 0,
            outputTokenCost: 0,
            embeddingCost: 0
        )
    }

    var isAvailable: Bool {
        get async {
            // Check if model is downloaded
            await modelManager.isModelAvailable
        }
    }

    private let modelManager: LocalModelManager

    init(modelManager: LocalModelManager = .shared) {
        self.modelManager = modelManager
    }

    func complete(
        messages: [LLMMessage],
        systemPrompt: String,
        tools: [LLMTool]?,
        options: LLMCompletionOptions
    ) async throws -> LLMResponse {
        // MLX/llama.cpp inference
        let prompt = formatPrompt(messages: messages, system: systemPrompt)
        let output = try await modelManager.generate(prompt: prompt, maxTokens: options.maxTokens)

        return LLMResponse(
            content: output,
            toolCalls: [],
            usage: LLMUsage(inputTokens: 0, outputTokens: 0, totalTokens: 0, estimatedCost: 0),
            finishReason: .completed
        )
    }

    func streamComplete(
        messages: [LLMMessage],
        systemPrompt: String,
        tools: [LLMTool]?,
        options: LLMCompletionOptions
    ) -> AsyncThrowingStream<LLMStreamChunk, Error> {
        AsyncThrowingStream { continuation in
            Task {
                let prompt = formatPrompt(messages: messages, system: systemPrompt)
                for try await token in modelManager.streamGenerate(prompt: prompt, maxTokens: options.maxTokens) {
                    continuation.yield(LLMStreamChunk(delta: token, toolCallDelta: nil, isComplete: false, usage: nil))
                }
                continuation.finish()
            }
        }
    }

    func embed(texts: [String]) async throws -> [[Float]] {
        // Use Apple's NaturalLanguage framework or local embedding model
        try await modelManager.embed(texts: texts)
    }

    func cancelAll() {
        modelManager.cancelGeneration()
    }

    private func formatPrompt(messages: [LLMMessage], system: String) -> String {
        // Format for Llama-style chat template
        var prompt = "<|system|>\n\(system)</s>\n"
        for message in messages {
            switch message.role {
            case .user:
                prompt += "<|user|>\n\(message.content)</s>\n"
            case .assistant:
                prompt += "<|assistant|>\n\(message.content)</s>\n"
            default:
                break
            }
        }
        prompt += "<|assistant|>\n"
        return prompt
    }
}
```

### 1.5 LLM Router with Fallback Chain

```swift
// MARK: - LLM Router

@Observable
final class LLMRouter {
    // MARK: - Configuration

    struct ProviderConfig: Sendable {
        let provider: any LLMProviderProtocol
        let priority: Int  // Lower = higher priority
        let maxRetries: Int
        let isEnabled: Bool
    }

    // MARK: - State

    private var providers: [ProviderConfig] = []
    private var usageByProvider: [String: LLMUsage] = [:]
    private var failuresByProvider: [String: Int] = [:]

    private(set) var currentProviderId: String = ""
    private(set) var isProcessing: Bool = false

    // MARK: - Configuration

    func registerProvider(_ provider: any LLMProviderProtocol, priority: Int, maxRetries: Int = 2) {
        providers.append(ProviderConfig(
            provider: provider,
            priority: priority,
            maxRetries: maxRetries,
            isEnabled: true
        ))
        providers.sort { $0.priority < $1.priority }
    }

    func setProviderEnabled(_ providerId: String, enabled: Bool) {
        if let index = providers.firstIndex(where: { $0.provider.providerId == providerId }) {
            providers[index] = ProviderConfig(
                provider: providers[index].provider,
                priority: providers[index].priority,
                maxRetries: providers[index].maxRetries,
                isEnabled: enabled
            )
        }
    }

    // MARK: - Routing

    func complete(
        messages: [LLMMessage],
        systemPrompt: String,
        tools: [LLMTool]? = nil,
        options: LLMCompletionOptions = LLMCompletionOptions(),
        preferredProvider: String? = nil
    ) async throws -> LLMResponse {
        isProcessing = true
        defer { isProcessing = false }

        let availableProviders = await getAvailableProviders(preferredProvider: preferredProvider)

        guard !availableProviders.isEmpty else {
            throw LLMError.noProvidersAvailable
        }

        var lastError: Error?

        for config in availableProviders {
            let provider = config.provider
            currentProviderId = provider.providerId

            for attempt in 1...config.maxRetries {
                do {
                    let response = try await provider.complete(
                        messages: messages,
                        systemPrompt: systemPrompt,
                        tools: tools,
                        options: options
                    )

                    // Track usage
                    trackUsage(provider: provider.providerId, usage: response.usage)
                    resetFailures(provider: provider.providerId)

                    return response

                } catch let error as LLMError {
                    lastError = error

                    switch error {
                    case .rateLimited:
                        // Immediately try next provider
                        trackFailure(provider: provider.providerId)
                        break

                    case .apiError(let code, _) where code >= 500:
                        // Server error - retry with backoff
                        try? await Task.sleep(nanoseconds: UInt64(attempt) * 500_000_000)
                        continue

                    case .timeout:
                        // Retry once, then next provider
                        if attempt < config.maxRetries {
                            continue
                        }
                        trackFailure(provider: provider.providerId)
                        break

                    default:
                        // Non-retryable error
                        throw error
                    }
                } catch {
                    lastError = error
                    trackFailure(provider: provider.providerId)
                }
            }
        }

        throw lastError ?? LLMError.allProvidersFailed
    }

    func streamComplete(
        messages: [LLMMessage],
        systemPrompt: String,
        tools: [LLMTool]? = nil,
        options: LLMCompletionOptions = LLMCompletionOptions(),
        preferredProvider: String? = nil
    ) -> AsyncThrowingStream<LLMStreamChunk, Error> {
        AsyncThrowingStream { continuation in
            Task {
                do {
                    let availableProviders = await getAvailableProviders(preferredProvider: preferredProvider)

                    guard let config = availableProviders.first else {
                        continuation.finish(throwing: LLMError.noProvidersAvailable)
                        return
                    }

                    currentProviderId = config.provider.providerId

                    for try await chunk in config.provider.streamComplete(
                        messages: messages,
                        systemPrompt: systemPrompt,
                        tools: tools,
                        options: options
                    ) {
                        continuation.yield(chunk)

                        if chunk.isComplete, let usage = chunk.usage {
                            trackUsage(provider: config.provider.providerId, usage: usage)
                        }
                    }

                    resetFailures(provider: config.provider.providerId)
                    continuation.finish()

                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    // MARK: - Cost-Based Routing

    /// Route simple queries to cheap providers, complex to expensive
    func smartRoute(
        messages: [LLMMessage],
        systemPrompt: String,
        complexity: QueryComplexity
    ) async throws -> LLMResponse {
        let preferredProvider: String?

        switch complexity {
        case .simple:
            // Use cheapest available (Gemini Flash > GPT-4o-mini > Local)
            preferredProvider = await cheapestAvailableProvider()
        case .moderate:
            // Use mid-tier (Claude Haiku, GPT-4o-mini)
            preferredProvider = "claude"
        case .complex:
            // Use best quality (Claude Sonnet, GPT-4o)
            preferredProvider = nil  // Use priority order
        }

        return try await complete(
            messages: messages,
            systemPrompt: systemPrompt,
            preferredProvider: preferredProvider
        )
    }

    enum QueryComplexity {
        case simple     // Quick acknowledgments, simple follow-ups
        case moderate   // Normal conversation
        case complex    // Deep analysis, insight generation
    }

    // MARK: - Private Helpers

    private func getAvailableProviders(preferredProvider: String?) async -> [ProviderConfig] {
        var available: [ProviderConfig] = []

        // If preferred, put it first
        if let preferred = preferredProvider,
           let config = providers.first(where: { $0.provider.providerId == preferred && $0.isEnabled }),
           await config.provider.isAvailable {
            available.append(config)
        }

        // Add remaining in priority order
        for config in providers where config.isEnabled {
            if config.provider.providerId != preferredProvider {
                if await config.provider.isAvailable {
                    available.append(config)
                }
            }
        }

        return available
    }

    private func cheapestAvailableProvider() async -> String? {
        var cheapest: (String, Decimal)?

        for config in providers where config.isEnabled {
            if await config.provider.isAvailable {
                let cost = config.provider.costEstimate.outputTokenCost
                if cheapest == nil || cost < cheapest!.1 {
                    cheapest = (config.provider.providerId, cost)
                }
            }
        }

        return cheapest?.0
    }

    private func trackUsage(provider: String, usage: LLMUsage) {
        if let existing = usageByProvider[provider] {
            usageByProvider[provider] = LLMUsage(
                inputTokens: existing.inputTokens + usage.inputTokens,
                outputTokens: existing.outputTokens + usage.outputTokens,
                totalTokens: existing.totalTokens + usage.totalTokens,
                estimatedCost: existing.estimatedCost + usage.estimatedCost
            )
        } else {
            usageByProvider[provider] = usage
        }
    }

    private func trackFailure(provider: String) {
        failuresByProvider[provider, default: 0] += 1
    }

    private func resetFailures(provider: String) {
        failuresByProvider[provider] = 0
    }
}

// MARK: - Error Types

enum LLMError: Error, LocalizedError {
    case notConfigured(String)
    case notSupported(String)
    case invalidResponse
    case apiError(statusCode: Int, message: String)
    case rateLimited
    case timeout
    case noProvidersAvailable
    case allProvidersFailed
    case cancelled

    var errorDescription: String? {
        switch self {
        case .notConfigured(let msg): return "Not configured: \(msg)"
        case .notSupported(let msg): return "Not supported: \(msg)"
        case .invalidResponse: return "Invalid response from API"
        case .apiError(let code, let msg): return "API error (\(code)): \(msg)"
        case .rateLimited: return "Rate limited"
        case .timeout: return "Request timed out"
        case .noProvidersAvailable: return "No AI providers available"
        case .allProvidersFailed: return "All AI providers failed"
        case .cancelled: return "Request cancelled"
        }
    }
}
```

### 1.6 Provider Registration and Switching UI

```swift
// MARK: - Dependency Container Setup

extension DependencyContainer {
    func configureLLMProviders() {
        let keyManager = resolve(APIKeyManager.self)

        let router = LLMRouter()

        // Register providers in priority order
        router.registerProvider(
            ClaudeProvider(apiKeyProvider: { keyManager.getKey(for: .claude) }),
            priority: 1
        )

        router.registerProvider(
            OpenAIProvider(apiKeyProvider: { keyManager.getKey(for: .openai) }),
            priority: 2
        )

        router.registerProvider(
            GeminiProvider(apiKeyProvider: { keyManager.getKey(for: .gemini) }),
            priority: 3
        )

        router.registerProvider(
            LocalMLXProvider(),
            priority: 4
        )

        register(router)
    }
}

// MARK: - Settings UI for Provider Selection

struct AIProviderSettingsView: View {
    @Environment(LLMRouter.self) private var router
    @Environment(APIKeyManager.self) private var keyManager

    var body: some View {
        Form {
            Section("AI Providers") {
                ForEach(ProviderType.allCases, id: \.self) { provider in
                    ProviderRow(
                        provider: provider,
                        isConfigured: keyManager.hasKey(for: provider),
                        isEnabled: router.isProviderEnabled(provider.id)
                    )
                }
            }

            Section("Usage This Month") {
                ForEach(router.usageByProvider.sorted(by: { $0.key < $1.key }), id: \.key) { provider, usage in
                    HStack {
                        Text(provider.capitalized)
                        Spacer()
                        Text("$\(usage.estimatedCost, format: .number.precision(.fractionLength(4)))")
                            .foregroundStyle(.secondary)
                    }
                }
            }
        }
        .navigationTitle("AI Providers")
    }
}

enum ProviderType: String, CaseIterable {
    case claude
    case openai
    case gemini
    case local

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .claude: return "Claude (Anthropic)"
        case .openai: return "GPT-4 (OpenAI)"
        case .gemini: return "Gemini (Google)"
        case .local: return "On-Device (Free)"
        }
    }

    var requiresAPIKey: Bool {
        self != .local
    }
}
```

---

## 2. Accelerated MVP Scope (20-24 Weeks)

### 2.1 Why Accelerate?

| Deadline | Event | Impact if MYND Not Launched |
|----------|-------|----------------------------|
| June 2026 | WWDC 2026 | Apple announces competing features |
| Q2 2026 | Sesame glasses launch | Voice companion market saturated |
| Q3 2026 | Pi/Microsoft memory features | Memory differentiator gone |

**Current timeline**: 32 weeks = August 2026 launch (AFTER all threats)
**Target timeline**: 20-24 weeks = April-May 2026 launch (BEFORE threats)

### 2.2 Scope Reduction Strategy

#### Features CUT from MVP

| Feature | Original Phase | Savings | Rationale |
|---------|---------------|---------|-----------|
| Knowledge Graph Visualization | v1.0 | 3 weeks | Defer to v1.5 |
| CloudKit Sync | v1.0 | 2 weeks | Local-only for MVP |
| ElevenLabs TTS | v1.0 | 1 week | AVSpeech sufficient |
| Morning Oracle | v1.0 | 2 weeks | Reactive only in MVP |
| Goal Tracking | v1.0 | 1.5 weeks | Simple notes in MVP |
| Apple Watch Widget | v1.5 | - | Already deferred |
| macOS App | v1.0 | 4 weeks | iOS only for MVP |
| Demo Mode AI | v1.0 | 1 week | Demo = local responses only |

**Total savings: 14.5 weeks**

#### Features KEPT (Core Value)

| Feature | Weeks | Why Essential |
|---------|-------|---------------|
| Text + Voice Input | 2 | Core capture mechanism |
| Claude Integration | 2 | AI differentiation |
| Thought List (Timeline) | 1.5 | Basic organization |
| Basic Search | 0.5 | Findability |
| Subscription/BYOK | 2 | Revenue |
| Lock Screen Widget | 1 | Quick capture |
| Settings | 1 | Configuration |
| Onboarding | 1 | First-time experience |

**Core build: 11 weeks**

### 2.3 Accelerated Timeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ACCELERATED 20-WEEK TIMELINE                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  PHASE 0: FOUNDATION (2 weeks)                                              │
│  ├── Design system, project setup                                           │
│  ├── LLM abstraction layer                                                  │
│  └── User interview synthesis (use existing research)                       │
│                                                                              │
│  PHASE 1: CORE CAPTURE (4 weeks)                                            │
│  ├── Text input with streaming AI response                                  │
│  ├── Voice input (Apple Speech)                                             │
│  ├── TTS playback (AVSpeech)                                                │
│  ├── SwiftData persistence (local only)                                     │
│  └── Thought timeline view                                                  │
│                                                                              │
│  PHASE 2: POLISH & MONETIZATION (3 weeks)                                   │
│  ├── Subscription flow (StoreKit 2)                                         │
│  ├── BYOK settings                                                          │
│  ├── Lock Screen widget                                                     │
│  ├── Onboarding flow                                                        │
│  └── Basic search                                                           │
│                                                                              │
│  PHASE 3: TESTING (2 weeks)                                                 │
│  ├── Unit tests for core flows                                              │
│  ├── UI tests for critical paths                                            │
│  ├── Performance profiling                                                  │
│  └── Accessibility audit                                                    │
│                                                                              │
│  PHASE 4: PRIVATE BETA (4 weeks)                                            │
│  ├── TestFlight with 50 users                                               │
│  ├── Bug fixes                                                              │
│  ├── Expand to 100 users                                                    │
│  └── Final polish                                                           │
│                                                                              │
│  PHASE 5: LAUNCH (1 week)                                                   │
│  ├── App Store submission                                                   │
│  ├── Marketing launch                                                       │
│  └── Monitor and hotfix                                                     │
│                                                                              │
│  BUFFER: 4 weeks                                                            │
│  └── Unexpected issues, App Store rejection, additional polish              │
│                                                                              │
│  TOTAL: 20 weeks (with 4-week buffer = 24 weeks worst case)                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.4 MVP v1.0 Feature Matrix

| Feature | Status | Notes |
|---------|--------|-------|
| **INPUT** | | |
| Text input | IN | Primary input method |
| Voice input (push-to-talk) | IN | Apple Speech |
| Photo input | OUT | v1.5 |
| **OUTPUT** | | |
| Text response (streaming) | IN | Claude/GPT streaming |
| TTS playback | IN | AVSpeech only |
| Premium voice | OUT | ElevenLabs in v1.5 |
| **STORAGE** | | |
| Local SwiftData | IN | Source of truth |
| CloudKit sync | OUT | v1.5 (risky, complex) |
| Export to JSON | IN | Data portability |
| **AI** | | |
| Multi-model support | IN | Claude, GPT, Gemini, Local |
| Streaming responses | IN | Essential for latency |
| Tool use (structured) | OUT | v1.5 |
| **ORGANIZATION** | | |
| Timeline view | IN | Chronological list |
| Basic search | IN | Text matching |
| Knowledge graph | OUT | v1.5 |
| Tags | OUT | v1.5 |
| **PROACTIVE** | | |
| Morning Oracle | OUT | v1.5 |
| Goal tracking | OUT | v1.5 |
| Notifications | OUT | v1.5 |
| **MONETIZATION** | | |
| Subscription tiers | IN | Starter, Pro |
| BYOK option | IN | Power users |
| Lifetime purchase | OUT | v1.5 (validate pricing first) |
| **PLATFORM** | | |
| iOS app | IN | Primary |
| macOS app | OUT | v1.5 |
| Apple Watch | OUT | v2.0 |
| Lock Screen widget | IN | Quick capture |
| Home Screen widget | OUT | v1.5 |

### 2.5 What Gets Cut vs Deferred

**CUT (Never Building)**
- Wake word activation (iOS doesn't allow)
- Always-on listening (battery/privacy)
- Real-time collaborative editing

**DEFERRED to v1.5 (6 weeks after launch)**
- CloudKit sync
- Knowledge graph visualization
- Goal tracking & Morning Oracle
- Premium voice (ElevenLabs)
- macOS app
- Photo/document input

**DEFERRED to v2.0 (12 weeks after launch)**
- Apple Watch
- Android
- Team/family features
- API for integrations

---

## 3. Text-First UX Architecture

### 3.1 Why Text-First?

The critique identified that "voice-first" is the wrong bet:

| Voice Limitation | Impact | Text Solution |
|-----------------|--------|---------------|
| 1-3s latency | Feels broken | Instant typing feedback |
| Social context | Can't use in public | Text works anywhere |
| ADHD users edit while thinking | Voice is one-take | Text allows backtracking |
| Transcription errors | Friction to re-speak | Type what you mean |
| Accent/noise issues | High error rate | Text is precise |

**New Philosophy**: Text is the reliable primary. Voice is a delightful enhancement for when conditions are right.

### 3.2 UI Flow Redesign

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TEXT-FIRST UX FLOW                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                         MAIN CONVERSATION VIEW                       │    │
│  │                                                                      │    │
│  │  ┌──────────────────────────────────────────────────────────────┐   │    │
│  │  │                    MESSAGE HISTORY                            │   │    │
│  │  │                                                               │   │    │
│  │  │   [User message in soft bubble]                               │   │    │
│  │  │                                                               │   │    │
│  │  │                        [Axel response with streaming]         │   │    │
│  │  │                                                               │   │    │
│  │  └──────────────────────────────────────────────────────────────┘   │    │
│  │                                                                      │    │
│  │  ┌──────────────────────────────────────────────────────────────┐   │    │
│  │  │  ┌──────────────────────────────────┐  ┌──────────────────┐  │   │    │
│  │  │  │  What's on your mind?            │  │ 🎤 │  ⬆️  │       │   │    │
│  │  │  │  [Text input - PRIMARY]          │  │    │     │       │   │    │
│  │  │  └──────────────────────────────────┘  └──────────────────┘  │   │    │
│  │  │                                                               │   │    │
│  │  │  [Mic button is SECONDARY - smaller, to the right]           │   │    │
│  │  └──────────────────────────────────────────────────────────────┘   │    │
│  │                                                                      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  INPUT MODES:                                                               │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                              │
│  1. TEXT (Default)                                                          │
│     ├── Tap input field → keyboard appears                                  │
│     ├── Type thought                                                        │
│     ├── Tap send → immediate local echo                                     │
│     └── Claude streams response → text displays instantly                   │
│                                                                              │
│  2. VOICE (Enhancement)                                                     │
│     ├── Tap mic button → "Listening..." state                               │
│     ├── Speak → real-time transcription appears in text field              │
│     ├── Tap again or pause → sends message                                  │
│     └── Response streams → optional TTS playback                            │
│                                                                              │
│  3. VOICE CONTINUATION (Hands-free mode)                                    │
│     ├── After TTS finishes, "Continue speaking?" prompt                     │
│     ├── User speaks → continues conversation                                │
│     └── Tap anywhere → exits voice mode                                     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.3 Input Component Architecture

```swift
// MARK: - Text-First Input View

struct ConversationInputView: View {
    @State private var inputText: String = ""
    @State private var inputMode: InputMode = .text
    @State private var isExpanded: Bool = false

    @Environment(VoiceEngine.self) private var voiceEngine
    @Environment(ConversationViewModel.self) private var viewModel

    enum InputMode {
        case text
        case voice
        case voiceContinuation  // Hands-free after TTS
    }

    var body: some View {
        VStack(spacing: 0) {
            // Voice transcription preview (when in voice mode)
            if inputMode == .voice {
                VoiceTranscriptionView(transcript: voiceEngine.currentTranscript)
                    .transition(.move(edge: .bottom).combined(with: .opacity))
            }

            // Main input bar
            HStack(spacing: 12) {
                // Text field (PRIMARY)
                TextField("What's on your mind?", text: $inputText, axis: .vertical)
                    .textFieldStyle(.plain)
                    .padding(.horizontal, 16)
                    .padding(.vertical, 12)
                    .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 24))
                    .lineLimit(1...6)
                    .disabled(inputMode == .voice)
                    .onSubmit { sendMessage() }

                // Voice button (SECONDARY - smaller)
                VoiceInputButton(
                    mode: $inputMode,
                    isListening: voiceEngine.state == .listening
                )

                // Send button (only visible when text entered)
                if !inputText.isEmpty {
                    SendButton(action: sendMessage)
                        .transition(.scale.combined(with: .opacity))
                }
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 8)
        }
        .animation(.spring(response: 0.3), value: inputMode)
        .animation(.spring(response: 0.3), value: inputText.isEmpty)
    }

    private func sendMessage() {
        let message = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !message.isEmpty else { return }

        Task {
            inputText = ""
            await viewModel.sendMessage(message)
        }
    }
}

// MARK: - Voice Input Button (Secondary)

struct VoiceInputButton: View {
    @Binding var mode: ConversationInputView.InputMode
    let isListening: Bool

    var body: some View {
        Button {
            toggleVoiceMode()
        } label: {
            Image(systemName: isListening ? "waveform" : "mic")
                .font(.system(size: 20, weight: .medium))
                .foregroundStyle(isListening ? .red : .secondary)
                .frame(width: 44, height: 44)
                .background(
                    Circle()
                        .fill(isListening ? Color.red.opacity(0.1) : Color.secondary.opacity(0.1))
                )
        }
        .buttonStyle(.plain)
        .accessibilityLabel(isListening ? "Stop listening" : "Voice input")
    }

    private func toggleVoiceMode() {
        if mode == .voice {
            mode = .text
        } else {
            mode = .voice
        }
    }
}

// MARK: - Voice Transcription View

struct VoiceTranscriptionView: View {
    let transcript: String

    var body: some View {
        HStack {
            Image(systemName: "waveform")
                .foregroundStyle(.red)

            Text(transcript.isEmpty ? "Listening..." : transcript)
                .foregroundStyle(transcript.isEmpty ? .secondary : .primary)
                .lineLimit(3)

            Spacer()
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16))
        .padding(.horizontal, 16)
        .padding(.bottom, 8)
    }
}
```

### 3.4 Response Display Architecture

```swift
// MARK: - Message Display

struct MessageBubble: View {
    let message: Message
    let isStreaming: Bool

    @Environment(TTSController.self) private var tts

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            if message.role == .assistant {
                // Axel avatar
                AxelAvatar(isThinking: isStreaming && message.content.isEmpty)
            }

            VStack(alignment: message.role == .user ? .trailing : .leading, spacing: 4) {
                // Message content
                Text(message.content)
                    .padding(.horizontal, 16)
                    .padding(.vertical, 12)
                    .background(bubbleBackground)
                    .foregroundStyle(message.role == .user ? .white : .primary)

                // Action buttons for assistant messages
                if message.role == .assistant && !isStreaming {
                    HStack(spacing: 16) {
                        // Speak button - plays TTS
                        Button {
                            Task { await tts.speak(message.content) }
                        } label: {
                            Label("Listen", systemImage: "speaker.wave.2")
                                .font(.caption)
                        }
                        .buttonStyle(.plain)
                        .foregroundStyle(.secondary)

                        // Save as thought button
                        Button {
                            // Save this response as a thought node
                        } label: {
                            Label("Save", systemImage: "bookmark")
                                .font(.caption)
                        }
                        .buttonStyle(.plain)
                        .foregroundStyle(.secondary)
                    }
                }
            }

            if message.role == .user {
                Spacer()
            }
        }
        .padding(.horizontal, 16)
    }

    private var bubbleBackground: some ShapeStyle {
        if message.role == .user {
            return AnyShapeStyle(Color.accentColor)
        } else {
            return AnyShapeStyle(Color.secondary.opacity(0.1))
        }
    }
}

// MARK: - Axel Avatar

struct AxelAvatar: View {
    let isThinking: Bool

    var body: some View {
        ZStack {
            Circle()
                .fill(
                    LinearGradient(
                        colors: [.blue.opacity(0.3), .purple.opacity(0.3)],
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    )
                )
                .frame(width: 36, height: 36)

            if isThinking {
                // Breathing animation while thinking
                BreathingIndicator()
            } else {
                Image(systemName: "brain.head.profile")
                    .font(.system(size: 16))
                    .foregroundStyle(.primary)
            }
        }
    }
}

// MARK: - Breathing Indicator (Thinking State)

struct BreathingIndicator: View {
    @State private var scale: CGFloat = 0.8

    var body: some View {
        Circle()
            .fill(.blue.opacity(0.5))
            .frame(width: 20, height: 20)
            .scaleEffect(scale)
            .onAppear {
                withAnimation(.easeInOut(duration: 1.5).repeatForever(autoreverses: true)) {
                    scale = 1.2
                }
            }
    }
}
```

### 3.5 TTS as Optional Enhancement

```swift
// MARK: - TTS Controller

@Observable
final class TTSController {
    enum State {
        case idle
        case speaking
        case paused
    }

    private(set) var state: State = .idle
    private(set) var progress: Double = 0  // 0-1

    private let synthesizer: SpeechSynthesizerProtocol
    private var userPrefersTTS: Bool = false  // User setting

    init(synthesizer: SpeechSynthesizerProtocol = AppleSpeechSynthesizer()) {
        self.synthesizer = synthesizer
    }

    /// Speak text aloud - called explicitly by user or auto-play setting
    func speak(_ text: String) async {
        guard state != .speaking else { return }

        state = .speaking
        await synthesizer.speak(text, style: .natural, rate: 0.5)
        state = .idle
    }

    /// Auto-speak if user has enabled the preference
    func autoSpeakIfEnabled(_ text: String) async {
        guard userPrefersTTS else { return }
        await speak(text)
    }

    func stop() {
        synthesizer.stopSpeaking()
        state = .idle
    }

    func setAutoSpeak(_ enabled: Bool) {
        userPrefersTTS = enabled
    }
}
```

### 3.6 Conversation Flow State Machine

```swift
// MARK: - Conversation State Machine

@Observable
final class ConversationViewModel {
    enum State {
        case idle
        case composingText
        case composingVoice(transcript: String)
        case sendingMessage
        case receivingResponse(partial: String)
        case speakingResponse
        case error(String)
    }

    private(set) var state: State = .idle
    private(set) var messages: [Message] = []

    private let llmRouter: LLMRouter
    private let voiceEngine: VoiceEngine
    private let ttsController: TTSController
    private let systemPromptBuilder: SystemPromptBuilder

    func sendMessage(_ text: String) async {
        let userMessage = Message(role: .user, content: text)
        messages.append(userMessage)

        state = .sendingMessage

        do {
            let systemPrompt = systemPromptBuilder.build(context: buildContext())

            // Add assistant placeholder for streaming
            let assistantMessage = Message(role: .assistant, content: "")
            messages.append(assistantMessage)

            state = .receivingResponse(partial: "")

            // Stream response
            for try await chunk in llmRouter.streamComplete(
                messages: messages.dropLast().map { $0.toLLMMessage() },
                systemPrompt: systemPrompt
            ) {
                if var lastMessage = messages.last, lastMessage.role == .assistant {
                    lastMessage.content += chunk.delta
                    messages[messages.count - 1] = lastMessage
                    state = .receivingResponse(partial: lastMessage.content)
                }
            }

            // Auto-speak if enabled
            if let response = messages.last?.content {
                await ttsController.autoSpeakIfEnabled(response)
            }

            state = .idle

        } catch {
            state = .error(error.localizedDescription)
        }
    }

    func startVoiceInput() async throws {
        state = .composingVoice(transcript: "")

        for try await result in voiceEngine.transcribe() {
            state = .composingVoice(transcript: result.text)

            if result.isFinal {
                // Auto-send after voice input completes
                await sendMessage(result.text)
            }
        }
    }

    func cancelVoiceInput() {
        voiceEngine.stopListening()
        state = .idle
    }

    private func buildContext() -> ConversationContext {
        // Build context from recent messages, memory, etc.
        ConversationContext(
            recentMessages: Array(messages.suffix(10)),
            userProfile: UserProfile.current,
            currentTime: Date()
        )
    }
}
```

---

## 4. Plan B Architecture

### 4.1 Pivot Scenarios and Responses

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PLAN B: PIVOT SCENARIOS                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  SCENARIO A: Apple Announces Competing Features (WWDC 2026)                 │
│  ────────────────────────────────────────────────────────────────────────── │
│  Probability: 70%                                                            │
│  Detection: WWDC keynote announcement                                        │
│  Timeline: Within 24 hours of announcement                                   │
│                                                                              │
│  Response:                                                                   │
│  ├── Messaging pivot: "MYND: Your AI, Your Data, Your Choice"              │
│  ├── Emphasize: BYOK, export, no lock-in, cross-platform future            │
│  ├── Technical: Add Apple Intelligence as a provider option                 │
│  └── Positioning: "Works WITH Apple Intelligence, does MORE"               │
│                                                                              │
│  SCENARIO B: Conversion Rate <2%                                            │
│  ────────────────────────────────────────────────────────────────────────── │
│  Probability: 50%                                                            │
│  Detection: 2 weeks post-launch analytics                                    │
│  Timeline: 30 days to implement changes                                      │
│                                                                              │
│  Response Options (choose based on data):                                    │
│  ├── A) Lower prices: $2.99 Starter, $6.99 Pro                              │
│  ├── B) Extend demo: 10/day for 14 days instead of 10 total                 │
│  ├── C) Freemium: Unlimited text, pay for voice                             │
│  └── D) Pivot to B2B: Therapy tools, coaching platforms                     │
│                                                                              │
│  SCENARIO C: Claude API Costs Double                                        │
│  ────────────────────────────────────────────────────────────────────────── │
│  Probability: 40%                                                            │
│  Detection: Anthropic pricing announcement                                   │
│  Timeline: 1 week to switch providers                                        │
│                                                                              │
│  Response:                                                                   │
│  ├── Automatic: LLMRouter switches to cheaper provider                      │
│  ├── User notification: "We've switched to GPT-4 for better value"          │
│  ├── Pricing adjustment: Pass cost to Pro tier if needed                    │
│  └── Long-term: Invest in local model quality                               │
│                                                                              │
│  SCENARIO D: Sesame/Pi Add Memory Features                                  │
│  ────────────────────────────────────────────────────────────────────────── │
│  Probability: 60%                                                            │
│  Detection: Competitor announcement                                          │
│  Timeline: Ongoing differentiation                                           │
│                                                                              │
│  Response:                                                                   │
│  ├── Accelerate knowledge graph (v1.5 → v1.2)                               │
│  ├── Emphasize: Local-first, BYOK, export capabilities                      │
│  ├── Add integrations: Obsidian, Notion, Roam export                        │
│  └── Positioning: "The open thought companion"                              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Abstraction Layers for Pivoting

```swift
// MARK: - Abstraction for Business Model Pivots

/// Protocol for different revenue models
protocol MonetizationStrategy {
    var availableTiers: [SubscriptionTier] { get }
    func canAccess(feature: Feature, for tier: SubscriptionTier) -> Bool
    func trackUsage(feature: Feature, units: Int)
    var shouldShowPaywall: Bool { get }
}

/// Current: Subscription-based
final class SubscriptionMonetization: MonetizationStrategy {
    var availableTiers: [SubscriptionTier] {
        [.demo, .starter, .pro, .byok]
    }

    func canAccess(feature: Feature, for tier: SubscriptionTier) -> Bool {
        switch (feature, tier) {
        case (_, .pro), (_, .byok):
            return true
        case (.basicConversation, .starter):
            return conversationsThisMonth < 500
        case (.voiceInput, .demo):
            return totalDemoConversations < 10
        default:
            return false
        }
    }

    // ...
}

/// Pivot Option: Usage-based pricing
final class UsageBasedMonetization: MonetizationStrategy {
    var availableTiers: [SubscriptionTier] {
        [.payAsYouGo]
    }

    func canAccess(feature: Feature, for tier: SubscriptionTier) -> Bool {
        return creditBalance > 0
    }

    func trackUsage(feature: Feature, units: Int) {
        let cost = feature.costPerUnit * Decimal(units)
        creditBalance -= cost
    }

    // ...
}

/// Pivot Option: Freemium with ads
final class FreemiumMonetization: MonetizationStrategy {
    var availableTiers: [SubscriptionTier] {
        [.free, .premium]
    }

    func canAccess(feature: Feature, for tier: SubscriptionTier) -> Bool {
        switch tier {
        case .premium: return true
        case .free: return feature.isFreeTier
        default: return false
        }
    }

    var shouldShowPaywall: Bool {
        tier == .free && shouldShowAd
    }

    // ...
}
```

### 4.3 Feature Flags for Gradual Rollout

```swift
// MARK: - Feature Flags

enum FeatureFlag: String, CaseIterable {
    // Input methods
    case textInput = "text_input"
    case voiceInput = "voice_input"
    case photoInput = "photo_input"

    // AI providers
    case multiModelSupport = "multi_model"
    case localModelSupport = "local_model"
    case appleIntelligence = "apple_intelligence"

    // Features
    case knowledgeGraph = "knowledge_graph"
    case cloudSync = "cloud_sync"
    case proactiveNotifications = "proactive_notifications"

    // Monetization experiments
    case showLifetimeOption = "show_lifetime"
    case extendedDemo = "extended_demo"
    case usageBasedPricing = "usage_based"

    // Positioning experiments
    case voiceFirstUI = "voice_first_ui"
    case textFirstUI = "text_first_ui"
}

@Observable
final class FeatureFlagService {
    private var flags: [FeatureFlag: Bool] = [:]
    private var overrides: [FeatureFlag: Bool] = [:]

    /// Check if feature is enabled
    func isEnabled(_ flag: FeatureFlag) -> Bool {
        // Local override takes precedence (for testing)
        if let override = overrides[flag] {
            return override
        }

        // Remote config (Firebase, etc.)
        return flags[flag] ?? flag.defaultValue
    }

    /// Set local override for testing
    func setOverride(_ flag: FeatureFlag, enabled: Bool) {
        overrides[flag] = enabled
    }

    /// Sync with remote config
    func syncRemoteConfig() async {
        // In v1.0: Just use defaults
        // In v1.5: Add Firebase Remote Config
    }
}

extension FeatureFlag {
    var defaultValue: Bool {
        switch self {
        case .textInput: return true
        case .voiceInput: return true
        case .photoInput: return false
        case .multiModelSupport: return true
        case .localModelSupport: return false
        case .appleIntelligence: return false
        case .knowledgeGraph: return false
        case .cloudSync: return false
        case .proactiveNotifications: return false
        case .showLifetimeOption: return false
        case .extendedDemo: return false
        case .usageBasedPricing: return false
        case .voiceFirstUI: return false
        case .textFirstUI: return true  // Default to text-first
        }
    }
}
```

### 4.4 Data Export for Portability

```swift
// MARK: - Export Architecture (GDPR + Pivot-Ready)

protocol DataExporter {
    var formatName: String { get }
    var fileExtension: String { get }
    func export(thoughts: [ThoughtNode], conversations: [ConversationSession]) async throws -> Data
}

/// JSON export for data portability and API integration
final class JSONExporter: DataExporter {
    let formatName = "JSON"
    let fileExtension = "json"

    func export(thoughts: [ThoughtNode], conversations: [ConversationSession]) async throws -> Data {
        let exportData = MYNDExport(
            version: "1.0",
            exportedAt: Date(),
            thoughts: thoughts.map { ThoughtExport(from: $0) },
            conversations: conversations.map { ConversationExport(from: $0) }
        )

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601

        return try encoder.encode(exportData)
    }
}

/// Markdown export for Obsidian/Notion integration
final class MarkdownExporter: DataExporter {
    let formatName = "Markdown"
    let fileExtension = "md"

    func export(thoughts: [ThoughtNode], conversations: [ConversationSession]) async throws -> Data {
        var markdown = "# MYND Export\n\n"
        markdown += "Exported: \(Date().formatted())\n\n"

        markdown += "## Thoughts\n\n"
        for thought in thoughts.sorted(by: { $0.createdAt > $1.createdAt }) {
            markdown += "### \(thought.createdAt.formatted(date: .abbreviated, time: .shortened))\n\n"
            markdown += thought.content + "\n\n"
            if !thought.tags.isEmpty {
                markdown += "Tags: \(thought.tags.joined(separator: ", "))\n\n"
            }
            markdown += "---\n\n"
        }

        return markdown.data(using: .utf8) ?? Data()
    }
}

/// Obsidian-specific export with internal links
final class ObsidianExporter: DataExporter {
    let formatName = "Obsidian"
    let fileExtension = "zip"

    func export(thoughts: [ThoughtNode], conversations: [ConversationSession]) async throws -> Data {
        // Create a folder structure compatible with Obsidian
        // Each thought becomes a note with [[links]] to related thoughts
        fatalError("Implementation: Create zip with markdown files")
    }
}
```

### 4.5 Companion Mode Architecture (Apple Intelligence Integration)

```swift
// MARK: - Apple Intelligence Companion Mode

/// If Apple releases competing features, MYND becomes a companion layer
protocol AppleIntelligenceCompanion {
    /// Import data from Apple's Journal app
    func importFromJournal() async throws -> [ThoughtNode]

    /// Sync with Apple Reminders
    func syncReminders() async throws

    /// Use Apple Intelligence as a provider
    var appleIntelligenceProvider: LLMProviderProtocol? { get }
}

final class AppleIntelligenceAdapter: AppleIntelligenceCompanion {
    func importFromJournal() async throws -> [ThoughtNode] {
        // Use JournalingSuggestions API if available
        // Fall back to manual import
        fatalError("Implement when Apple APIs are available")
    }

    func syncReminders() async throws {
        // EventKit integration
        let store = EKEventStore()
        try await store.requestFullAccessToReminders()
        // Sync MYND goals <-> Apple Reminders
    }

    var appleIntelligenceProvider: LLMProviderProtocol? {
        // If Apple opens on-device LLM API
        // Return adapter to Apple's API
        nil
    }
}
```

---

## 5. Architecture Diagrams

### 5.1 Complete System Architecture

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                                    MYND v3.0 ARCHITECTURE                             │
└──────────────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────────────┐
│                                    PRESENTATION LAYER                                 │
├──────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                       │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ ConversationView│  │ ThoughtListView │  │  SettingsView   │  │ Lock Screen     │ │
│  │                 │  │                 │  │                 │  │ Widget          │ │
│  │ - Text input    │  │ - Timeline      │  │ - AI providers  │  │                 │ │
│  │ - Voice button  │  │ - Search        │  │ - Subscription  │  │ - Quick capture │ │
│  │ - Message list  │  │ - Filter        │  │ - Export        │  │                 │ │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘ │
│           │                    │                    │                    │          │
└───────────┼────────────────────┼────────────────────┼────────────────────┼──────────┘
            │                    │                    │                    │
            ▼                    ▼                    ▼                    ▼
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                                   VIEW MODEL LAYER                                    │
├──────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                       │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐          │
│  │ ConversationVM      │  │ ThoughtListVM       │  │ SettingsVM          │          │
│  │                     │  │                     │  │                     │          │
│  │ @Observable         │  │ @Observable         │  │ @Observable         │          │
│  │ - state machine     │  │ - filter state      │  │ - provider config   │          │
│  │ - message handling  │  │ - search results    │  │ - subscription      │          │
│  └──────────┬──────────┘  └──────────┬──────────┘  └──────────┬──────────┘          │
│             │                        │                        │                      │
└─────────────┼────────────────────────┼────────────────────────┼──────────────────────┘
              │                        │                        │
              ▼                        ▼                        ▼
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                                    SERVICE LAYER                                      │
├──────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                       │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐     │
│  │ LLMRouter      │  │ VoiceEngine    │  │ TTSController  │  │ Subscription   │     │
│  │                │  │                │  │                │  │ Manager        │     │
│  │ - Provider     │  │ - Apple Speech │  │ - AVSpeech     │  │                │     │
│  │   routing      │  │ - Transcription│  │ - Auto-play    │  │ - StoreKit 2   │     │
│  │ - Fallback     │  │ - Stream       │  │ - Rate control │  │ - Entitlements │     │
│  └───────┬────────┘  └───────┬────────┘  └───────┬────────┘  └───────┬────────┘     │
│          │                   │                   │                   │              │
│          │    ┌──────────────┴───────────────────┴───────────────────┘              │
│          │    │                                                                      │
│          ▼    ▼                                                                      │
│  ┌───────────────────────────────────────────────────────────────────────────────┐  │
│  │                              LLM PROVIDER LAYER                                │  │
│  │                                                                                │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │  │
│  │  │ Claude   │  │ OpenAI   │  │ Gemini   │  │ Local    │  │ Apple    │        │  │
│  │  │ Provider │  │ Provider │  │ Provider │  │ MLX      │  │ Intel.   │        │  │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────┘        │  │
│  │                                                                                │  │
│  └───────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                       │
└──────────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                                  PERSISTENCE LAYER                                    │
├──────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                       │
│  ┌─────────────────────────────────┐  ┌─────────────────────────────────┐           │
│  │         SwiftData Store         │  │         Secure Storage          │           │
│  │                                 │  │                                 │           │
│  │  - ThoughtNode                  │  │  - API Keys (Keychain)          │           │
│  │  - ConversationSession          │  │  - User preferences             │           │
│  │  - Message                      │  │                                 │           │
│  │  - UserProfile                  │  │                                 │           │
│  └─────────────────────────────────┘  └─────────────────────────────────┘           │
│                                                                                       │
│  ┌─────────────────────────────────┐                                                 │
│  │     Export Services             │                                                 │
│  │                                 │                                                 │
│  │  - JSON (portability)           │                                                 │
│  │  - Markdown (Obsidian)          │                                                 │
│  │  - GDPR data request            │                                                 │
│  └─────────────────────────────────┘                                                 │
│                                                                                       │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Data Flow Diagram

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                               USER INPUT FLOW                                         │
└──────────────────────────────────────────────────────────────────────────────────────┘

                    ┌───────────────────────────────────┐
                    │           USER INPUT              │
                    │                                   │
                    │   ┌─────────┐    ┌─────────┐     │
                    │   │  Text   │    │  Voice  │     │
                    │   │ (Swift  │    │ (Tap    │     │
                    │   │  UI)    │    │  mic)   │     │
                    │   └────┬────┘    └────┬────┘     │
                    │        │              │          │
                    └────────┼──────────────┼──────────┘
                             │              │
                             │              ▼
                             │    ┌─────────────────┐
                             │    │  VoiceEngine    │
                             │    │                 │
                             │    │ Apple Speech    │
                             │    │ transcription   │
                             │    └────────┬────────┘
                             │             │
                             ▼             ▼
                    ┌─────────────────────────────────┐
                    │     ConversationViewModel       │
                    │                                 │
                    │  1. Add user message to list    │
                    │  2. Build context               │
                    │  3. Route to LLM                │
                    └────────────────┬────────────────┘
                                     │
                                     ▼
                    ┌─────────────────────────────────┐
                    │          LLMRouter              │
                    │                                 │
                    │  1. Select provider (priority)  │
                    │  2. Send request                │
                    │  3. Handle fallback on error    │
                    └────────────────┬────────────────┘
                                     │
        ┌────────────────────────────┼────────────────────────────┐
        │                            │                            │
        ▼                            ▼                            ▼
┌───────────────┐          ┌───────────────┐          ┌───────────────┐
│    Claude     │          │    OpenAI     │          │    Local      │
│    API        │          │    API        │          │    MLX        │
│               │          │               │          │               │
│  Streaming    │          │  Streaming    │          │  On-device    │
│  SSE          │          │  SSE          │          │  inference    │
└───────┬───────┘          └───────┬───────┘          └───────┬───────┘
        │                          │                          │
        └────────────────────────────────────────────────────┘
                                     │
                                     ▼
                    ┌─────────────────────────────────┐
                    │     RESPONSE STREAMING          │
                    │                                 │
                    │  For each token:                │
                    │  1. Update UI (message bubble)  │
                    │  2. Track usage/cost            │
                    │                                 │
                    └────────────────┬────────────────┘
                                     │
                                     ▼
                    ┌─────────────────────────────────┐
                    │      OPTIONAL: TTS              │
                    │                                 │
                    │  If user preference OR tap:     │
                    │  1. AVSpeech synthesis          │
                    │  2. Play audio                  │
                    └────────────────┬────────────────┘
                                     │
                                     ▼
                    ┌─────────────────────────────────┐
                    │      PERSISTENCE                │
                    │                                 │
                    │  1. Save message to SwiftData   │
                    │  2. Extract thoughts (future)   │
                    │  3. Update conversation         │
                    └─────────────────────────────────┘
```

---

## 6. Summary: Key Architectural Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **LLM Provider** | Multi-provider with fallback | Not locked to Anthropic pricing |
| **Primary Input** | Text-first, voice-secondary | Voice latency is liability |
| **Persistence** | SwiftData local-only for MVP | CloudKit too risky, defer to v1.5 |
| **TTS** | AVSpeech with optional playback | User controls when to listen |
| **Monetization** | Subscription + BYOK | Multiple fallback options |
| **Export** | JSON + Markdown from day 1 | Data portability differentiator |
| **Feature Flags** | Built-in from start | Enables rapid pivots |
| **Apple Strategy** | Companion, not competitor | Integrate if Apple enters market |

### 6.1 What This Architecture Enables

1. **Launch before WWDC 2026** - 20-week timeline is achievable
2. **Survive Claude price increase** - Switch providers automatically
3. **Survive Apple competition** - Become companion, not competitor
4. **Survive low conversion** - Adjust pricing via feature flags
5. **Survive Sesame adding memory** - Differentiate on privacy and export

### 6.2 What This Architecture Prevents

1. Provider lock-in
2. Feature scope creep
3. Latency as primary complaint
4. Data portability complaints
5. Single business model dependency

---

*Document Status: STRATEGIC ARCHITECTURE v3.0*
*Purpose: Resilient, pivot-ready, accelerated implementation*
*Next Step: Begin Phase 0 (Foundation)*
*Timeline: 20 weeks to launch (24 with buffer)*
