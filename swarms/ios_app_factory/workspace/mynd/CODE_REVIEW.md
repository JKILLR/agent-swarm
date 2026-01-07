# MYND iOS App - Code Review

**Review Date:** 2026-01-04
**Reviewer:** Claude Code Agent
**Files Reviewed:** 10 Swift files in `MYND/`
**Target:** iOS 17+, Swift 5.9+

---

## Summary

| Severity | Count |
|----------|-------|
| Critical | 4 |
| High | 5 |
| Medium | 7 |
| Low | 6 |
| **Total** | **22** |

**Overall Assessment:** The codebase demonstrates solid architecture with clean separation of concerns (Views, ViewModels, Services, Graph layers). However, there are critical memory management and thread safety issues that must be addressed before production release.

---

## Critical Issues

### 1. Memory Leak - Gesture Recognizers Never Removed

**File:** `Graph/MindMapScene.swift:63-72`
**Category:** Memory Management

```swift
private func setupGestures(for view: SKView) {
    let panGesture = UIPanGestureRecognizer(target: self, action: #selector(handlePan(_:)))
    panGesture.minimumNumberOfTouches = 2
    view.addGestureRecognizer(panGesture)

    let pinchGesture = UIPinchGestureRecognizer(target: self, action: #selector(handlePinch(_:)))
    view.addGestureRecognizer(pinchGesture)
}
```

**Problem:** Gesture recognizers are added with `target: self` but are never removed when the scene is deallocated. The SKView holds strong references to these gesture recognizers, which hold strong references to the scene (`target: self`), creating a retain cycle.

**Fix:** Override `willMove(from:)` to remove gesture recognizers:

```swift
private var panGesture: UIPanGestureRecognizer?
private var pinchGesture: UIPinchGestureRecognizer?

override func willMove(from view: SKView) {
    super.willMove(from: view)
    if let pan = panGesture { view.removeGestureRecognizer(pan) }
    if let pinch = pinchGesture { view.removeGestureRecognizer(pinch) }
}
```

---

### 2. Force Unwrap Crash Risk

**File:** `Graph/ThoughtNodeSprite.swift:174`
**Category:** Error Handling

```swift
addChild(selectionRing!)
```

**Problem:** Force unwrapping `selectionRing!` immediately after optional initialization can crash if `SKShapeNode` initialization fails (unlikely but possible with extreme radius values).

**Fix:** Use safe unwrapping pattern:

```swift
private func addSelectionRing() {
    guard selectionRing == nil else { return }

    let ringRadius = (backgroundNode.frame.width / 2) + 4
    let ring = SKShapeNode(circleOfRadius: ringRadius)
    ring.strokeColor = .white
    ring.lineWidth = 3
    ring.fillColor = .clear
    ring.zPosition = -1

    addChild(ring)
    selectionRing = ring
}
```

---

### 3. Fatal Error on App Initialization

**File:** `App/MYNDApp.swift:40`
**Category:** Error Handling

```swift
} catch {
    fatalError("Could not create ModelContainer: \(error)")
}
```

**Problem:** Using `fatalError` for ModelContainer failure provides no recovery path and no user feedback. The app crashes silently without logging or analytics.

**Fix:** Implement graceful degradation:

```swift
var sharedModelContainer: ModelContainer = {
    let schema = Schema([])
    let modelConfiguration = ModelConfiguration(
        schema: schema,
        isStoredInMemoryOnly: false,
        allowsSave: true
    )

    do {
        return try ModelContainer(for: schema, configurations: [modelConfiguration])
    } catch {
        // Log error for debugging
        print("❌ ModelContainer initialization failed: \(error)")

        // Attempt fallback to in-memory storage
        do {
            let fallbackConfig = ModelConfiguration(
                schema: schema,
                isStoredInMemoryOnly: true
            )
            return try ModelContainer(for: schema, configurations: [fallbackConfig])
        } catch {
            // If even in-memory fails, this is truly unrecoverable
            fatalError("Could not create ModelContainer: \(error)")
        }
    }
}()
```

---

### 4. Unsafe Required Initializer

**File:** `Graph/ThoughtNodeSprite.swift:61-63`
**Category:** Swift Best Practices

```swift
required init?(coder aDecoder: NSCoder) {
    fatalError("init(coder:) has not been implemented")
}
```

**Problem:** If the node is ever used in a storyboard/XIB or serialized, this will crash. While unlikely for programmatic use, it's a latent risk.

**Fix:** Either implement proper deserialization or document the limitation:

```swift
/// This class does not support storyboard/XIB instantiation.
/// Always use `init(thought:)` instead.
@available(*, unavailable, message: "Use init(thought:) instead")
required init?(coder aDecoder: NSCoder) {
    fatalError("ThoughtNodeSprite does not support NSCoding. Use init(thought:) instead.")
}
```

---

## High Severity Issues

### 5. Thread Safety - SpriteKit to SwiftUI ViewModel

**File:** `Graph/MindMapScene.swift:185-188`
**Category:** Thread Safety

```swift
override func touchesMoved(_ touches: Set<UITouch>, with event: UIEvent?) {
    // ...
    if let thought = viewModel?.thoughts.first(where: { $0.id == node.thoughtId }) {
        thought.position = location  // ⚠️ Cross-thread mutation
    }
}
```

**Problem:** SpriteKit touch handlers execute on the rendering thread, but `viewModel` is marked `@MainActor`. This cross-thread mutation can cause data races and undefined behavior.

**Fix:** Dispatch to main actor:

```swift
override func touchesMoved(_ touches: Set<UITouch>, with event: UIEvent?) {
    guard let touch = touches.first, let node = draggedNode else { return }

    let location = touch.location(in: self)
    node.position = location

    // Dispatch ViewModel update to main actor
    let thoughtId = node.thoughtId
    Task { @MainActor in
        viewModel?.updatePosition(for: thoughtId, position: location)
    }
}
```

---

### 6. Thread Safety - updateUIView Mutation

**File:** `Views/MindMap/MindMapView.swift:59-65`
**Category:** Thread Safety

```swift
func updateUIView(_ skView: SKView, context: Context) {
    if let scene = context.coordinator.scene {
        scene.size = size          // ⚠️ Mutating SpriteKit from SwiftUI
        scene.syncWithViewModel()  // ⚠️ May conflict with render loop
    }
}
```

**Problem:** SwiftUI's `updateUIView` runs on the main thread, but SpriteKit's render loop may be accessing `scene.size` concurrently. Additionally, `syncWithViewModel()` modifies scene children during potential frame rendering.

**Fix:** Use SpriteKit's update loop for synchronization:

```swift
func updateUIView(_ skView: SKView, context: Context) {
    if let scene = context.coordinator.scene {
        // Mark scene as needing sync (thread-safe flag)
        scene.pendingSize = size
        scene.needsSync = true
    }
}

// In MindMapScene:
private var pendingSize: CGSize?
private var needsSync = false

override func update(_ currentTime: TimeInterval) {
    if let newSize = pendingSize {
        size = newSize
        pendingSize = nil
    }
    if needsSync {
        syncWithViewModel()
        needsSync = false
    }
}
```

---

### 7. Missing Hashable Conformance

**File:** `ViewModels/MindMapViewModel.swift:131-151`
**Category:** Swift Best Practices

```swift
@Observable
final class ThoughtData: Identifiable {
    let id: UUID
    // ...
}
```

**Problem:** `ThoughtData` conforms to `Identifiable` but not `Hashable`. SwiftUI's `ForEach` and diffing algorithms work best with `Hashable` types. Without it, SwiftUI may perform unnecessary view updates.

**Fix:** Add `Hashable` conformance:

```swift
@Observable
final class ThoughtData: Identifiable, Hashable {
    let id: UUID
    var content: String
    var category: String?
    var position: CGPoint?
    let createdAt: Date

    static func == (lhs: ThoughtData, rhs: ThoughtData) -> Bool {
        lhs.id == rhs.id
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(id)
    }
}
```

---

### 8. Singleton with MainActor May Have Initialization Issues

**File:** `Services/DataService.swift:14-19`
**Category:** Swift Concurrency

```swift
@MainActor
final class DataService {
    static let shared = DataService()
    private init() {}
}
```

**Problem:** Static singleton initialization combined with `@MainActor` can cause subtle initialization timing issues. If accessed from a background thread before the main actor is available, it may deadlock or behave unexpectedly.

**Fix:** Consider dependency injection or lazy initialization with explicit actor isolation:

```swift
@MainActor
final class DataService {
    private static var _shared: DataService?

    static var shared: DataService {
        if _shared == nil {
            _shared = DataService()
        }
        return _shared!
    }

    private init() {}
}

// Or better: Use dependency injection via environment
@MainActor
final class DataService: Observable {
    // ...
}

// In App:
.environment(DataService())
```

---

### 9. No Validation of Empty ModelContainer Schema

**File:** `App/MYNDApp.swift:21-26`
**Category:** Error Handling

```swift
let schema = Schema([
    // Placeholder: Models will be added here
])
```

**Problem:** The app initializes with an empty schema, which works but provides no persistence. There's no runtime check or warning that data won't be saved.

**Fix:** Add a debug assertion or logging:

```swift
let schema = Schema([])

#if DEBUG
if schema.models.isEmpty {
    print("⚠️ Warning: ModelContainer initialized with empty schema. Data will not persist.")
}
#endif
```

---

## Medium Severity Issues

### 10. Camera Node Implicitly Unwrapped

**File:** `Graph/MindMapScene.swift:31`
**Category:** Swift Best Practices

```swift
private var cameraNode: SKCameraNode!
```

**Problem:** Implicitly unwrapped optional (IUO) creates crash risk if accessed before `didMove(to:)`.

**Fix:** Use lazy initialization or optional:

```swift
private lazy var cameraNode: SKCameraNode = {
    let camera = SKCameraNode()
    addChild(camera)
    return camera
}()
```

---

### 11. Background/Label Nodes Implicitly Unwrapped

**File:** `Graph/ThoughtNodeSprite.swift:22-25`
**Category:** Swift Best Practices

```swift
private var backgroundNode: SKShapeNode!
private var labelNode: SKLabelNode!
```

**Problem:** Same IUO risk as above.

**Fix:** Initialize inline or make lazy.

---

### 12. Missing Error Propagation in DataService

**File:** `Services/DataService.swift:44-51`
**Category:** Error Handling

```swift
func fetchAllThoughts() throws -> [Any] {
    // TODO: Implement when Thought model is available
    return []
}
```

**Problem:** Methods declare `throws` but implementation returns empty arrays. When implemented, callers won't be prepared for actual errors.

**Fix:** When implementing, ensure callers handle errors:

```swift
// Current placeholder is fine, but add documentation
/// - Note: Currently returns empty array. Will throw when persistence is implemented.
func fetchAllThoughts() throws -> [Any] {
    return []
}
```

---

### 13. Position Update Doesn't Notify View

**File:** `ViewModels/MindMapViewModel.swift:81-87`
**Category:** SwiftUI Integration

```swift
func updatePosition(for thought: ThoughtData, position: CGPoint) {
    if let index = thoughts.firstIndex(where: { $0.id == thought.id }) {
        thoughts[index].position = position
    }
}
```

**Problem:** Modifying a property of an element in the array doesn't trigger `@Observable` change notification on the array itself. The view may not update.

**Fix:** Since `ThoughtData` is also `@Observable`, property changes should propagate. However, verify this works in testing, or use:

```swift
func updatePosition(for thought: ThoughtData, position: CGPoint) {
    thought.position = position  // Direct mutation on @Observable object
}
```

---

### 14. No Bounds Checking on Spiral Layout

**File:** `Graph/MindMapScene.swift:141-154`
**Category:** Robustness

```swift
private func calculateNewNodePosition() -> CGPoint {
    let nodeCount = thoughtSprites.count
    let angle = CGFloat(nodeCount) * 0.5
    let radius = nodeSpacing + CGFloat(nodeCount) * 20
    // ...
}
```

**Problem:** With many nodes (500+), the radius grows unboundedly and positions may overflow `CGFloat` or extend far beyond visible area.

**Fix:** Add bounds or use modular/wrapping layout:

```swift
private func calculateNewNodePosition() -> CGPoint {
    let nodeCount = thoughtSprites.count
    let angle = CGFloat(nodeCount) * 0.5
    let maxRadius = min(size.width, size.height) * 2
    let radius = min(nodeSpacing + CGFloat(nodeCount) * 20, maxRadius)
    // ...
}
```

---

### 15. Physics Body Setup Uses Potentially Invalid Frame

**File:** `Graph/ThoughtNodeSprite.swift:99-110`
**Category:** SpriteKit

```swift
private func setupPhysicsBody() {
    let radius = backgroundNode.frame.width / 2
    // ...
}
```

**Problem:** `backgroundNode.frame` may not be accurate immediately after initialization before the node is added to a scene and rendered.

**Fix:** Use the known radius value directly:

```swift
private func setupPhysicsBody(radius: CGFloat) {
    physicsBody = SKPhysicsBody(circleOfRadius: radius)
    // ...
}

// In setupBackground:
private func setupBackground(for thought: ThoughtData) -> CGFloat {
    let nodeRadius = minRadius + (maxRadius - minRadius) * sizeMultiplier
    backgroundNode = SKShapeNode(circleOfRadius: nodeRadius)
    // ...
    return nodeRadius
}
```

---

### 16. Missing Accessibility Labels

**File:** `Views/Input/ThoughtInputView.swift:37-42`
**Category:** Accessibility

```swift
Button(action: submitThought) {
    Image(systemName: "arrow.up.circle.fill")
        .font(.system(size: 32))
        .foregroundStyle(thoughtText.isEmpty ? .gray : .blue)
}
```

**Problem:** Submit button has no accessibility label for VoiceOver users.

**Fix:**

```swift
Button(action: submitThought) {
    Image(systemName: "arrow.up.circle.fill")
        .font(.system(size: 32))
        .foregroundStyle(thoughtText.isEmpty ? .gray : .blue)
}
.accessibilityLabel("Submit thought")
.accessibilityHint("Double tap to add your thought")
```

---

## Low Severity Issues

### 17. Magic Numbers in Layout

**File:** `Graph/MindMapScene.swift:35-37`
**Category:** Code Quality

```swift
private let nodeSpacing: CGFloat = 150
private let minZoom: CGFloat = 0.5
private let maxZoom: CGFloat = 3.0
```

**Problem:** These constants duplicate values in `AppConfig`.

**Fix:** Use `AppConfig` values:

```swift
private let nodeSpacing = AppConfig.defaultNodeSpacing
private let minZoom = AppConfig.minZoomScale
private let maxZoom = AppConfig.maxZoomScale
```

---

### 18. Duplicate Constants in ThoughtNodeSprite

**File:** `Graph/ThoughtNodeSprite.swift:35-39`
**Category:** Code Quality

```swift
private let minRadius: CGFloat = 40
private let maxRadius: CGFloat = 80
private let padding: CGFloat = 16
private let fontSize: CGFloat = 14
private let maxLabelWidth: CGFloat = 120
```

**Fix:** Consider moving to `AppConfig` for consistency.

---

### 19. Unused Parameter in updateThought

**File:** `Services/DataService.swift:72-90`
**Category:** Code Quality

```swift
func updateThought(
    _ thought: Any,
    content: String? = nil,
    category: String? = nil,
    position: CGPoint? = nil
) throws {
    // TODO: Implement
}
```

**Problem:** Parameters are unused in placeholder implementation.

**Fix:** Add `_ = thought` or mark with `@_silenceWarnings` when implementing to acknowledge intentional placeholders.

---

### 20. Tab Enum Could Be File-Private

**File:** `App/ContentView.swift:20-24`
**Category:** Swift Best Practices

```swift
enum Tab: Hashable {
    case mindMap
    case list
}
```

**Problem:** `Tab` is only used within `ContentView` but is internal by default.

**Fix:**

```swift
private enum Tab: Hashable {
    case mindMap
    case list
}
```

---

### 21. Preview Uses Default Initialization

**File:** Multiple files
**Category:** Testing

```swift
#Preview {
    MindMapView(viewModel: MindMapViewModel())
}
```

**Problem:** Previews show empty state. Consider adding sample data for better preview experience.

**Fix:**

```swift
#Preview("With Sample Data") {
    let viewModel = MindMapViewModel()
    viewModel.addThought(content: "First idea")
    viewModel.addThought(content: "Second thought")
    return MindMapView(viewModel: viewModel)
}
```

---

### 22. Missing Documentation on Public Methods

**File:** `ViewModels/MindMapViewModel.swift`
**Category:** Documentation

Some public methods lack documentation comments explaining parameters and behavior.

---

## Positive Observations

1. **Clean Architecture:** Excellent separation of concerns with Views, ViewModels, Services, and Graph layers
2. **Modern Swift:** Proper use of Swift 5.9 `@Observable` macro instead of older Combine patterns
3. **Correct Memory Management:** `weak var viewModel` in MindMapScene prevents the most common retain cycle
4. **Feature Flags:** Well-organized AppConfig with phase-based feature flags
5. **ADHD-Friendly UX:** Keyboard stays open for rapid entry - thoughtful accessibility consideration
6. **Good Organization:** MARK comments, consistent file structure, all views include `#Preview` macros
7. **Proper Coordinator Pattern:** UIViewRepresentable uses coordinator correctly for scene reference
8. **Debug Settings Gated:** `#if DEBUG` used appropriately for development-only features
9. **Clean TabView:** Shared ViewModel state across tabs implemented correctly
10. **Animations:** Smooth appearance/disappearance animations for nodes

---

## Recommended Priority

### Immediate (Pre-Release)
1. Fix gesture recognizer memory leak (Critical #1)
2. Fix force unwrap crash risk (Critical #2)
3. Fix thread safety issues (High #5, #6)
4. Add Hashable conformance (High #7)

### Short-Term
5. Improve error handling in MYNDApp (Critical #3)
6. Address implicitly unwrapped optionals (Medium #10, #11)
7. Add accessibility labels (Medium #16)

### Long-Term
8. Consolidate constants to AppConfig (Low #17, #18)
9. Add sample data to previews (Low #21)
10. Complete documentation (Low #22)

---

## Testing Recommendations

1. **Memory Leak Testing:** Use Instruments to verify no leaks when navigating between tabs
2. **Thread Safety Testing:** Use Thread Sanitizer during development
3. **Stress Testing:** Test with 500+ nodes to verify performance and layout bounds
4. **Accessibility Audit:** Run Accessibility Inspector on all views
5. **Error Recovery Testing:** Simulate ModelContainer failures to verify fallback behavior
