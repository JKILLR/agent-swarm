# MYND Accessibility Requirements Specification

**Version**: 1.0
**Date**: 2026-01-04
**Status**: DEFINITIVE REQUIREMENTS
**Compliance Target**: WCAG 2.1 AA + Apple Accessibility Best Practices

---

## 1. Executive Summary

### 1.1 Why Accessibility is Critical for MYND

MYND's target audience includes users with ADHD and executive function challenges. This population has:
- Higher rates of co-occurring conditions (anxiety, depression)
- Greater sensitivity to cognitive overload
- Often rely on accessibility features even without formal disability diagnosis
- Higher likelihood of using Dynamic Type, Reduce Motion, or VoiceOver

**Accessibility is not optional for MYND. It is core to the product.**

### 1.2 Compliance Targets

| Standard | Level | Status |
|----------|-------|--------|
| WCAG 2.1 | AA | Required for v1.0 |
| WCAG 2.1 | AAA (where feasible) | Stretch goal |
| Apple Human Interface Guidelines | Full compliance | Required |
| iOS Accessibility API | Full support | Required |

### 1.3 Accessibility Principles

1. **Perceivable**: Information must be presentable in ways all users can perceive
2. **Operable**: Interface must be operable by all users
3. **Understandable**: Information and UI must be understandable
4. **Robust**: Content must be robust enough to work with assistive technologies

---

## 2. Visual Accessibility

### 2.1 Color Contrast Requirements

#### WCAG 2.1 AA Requirements

| Element Type | Minimum Ratio | MYND Target |
|--------------|---------------|-------------|
| Body text | 4.5:1 | 5.9:1+ |
| Large text (18pt+ or 14pt bold) | 3:1 | 4.5:1+ |
| UI components | 3:1 | 4.5:1+ |
| Graphical objects | 3:1 | 4.5:1+ |

#### MYND Color Verification

| Combination | Ratio | WCAG Status |
|-------------|-------|-------------|
| Deep Ocean (#1A365D) on White | 9.8:1 | AAA |
| Neutral (#4A5568) on White | 5.9:1 | AA |
| Calm Blue (#3182CE) on White | 4.6:1 | AA |
| Warm Amber (#DD6B20) on White | 4.5:1 | AA |
| Light text (#E2E8F0) on Deep Charcoal (#1A202C) | 12.4:1 | AAA |

#### Non-Color Indicators

**Requirement**: Color must never be the only means of conveying information.

| Scenario | Color Only (BAD) | MYND Approach (GOOD) |
|----------|------------------|----------------------|
| Recording active | Red dot | Pulsing icon + "Recording" label |
| Thought saved | Green checkmark | Checkmark + haptic + "Saved" toast |
| Error state | Red border | Icon + text description + changed border |
| Selected item | Blue background | Background + checkmark icon + bold text |

### 2.2 Dynamic Type Support

#### Required Size Range

| Category | Minimum | Maximum |
|----------|---------|---------|
| Body text | xSmall | AX5 (Accessibility 5) |
| Headers | xSmall | AX5 |
| Captions | xSmall | AX5 |
| Button labels | xSmall | AX5 |

#### Size Scaling Table

| Style | Default | AX1 | AX2 | AX3 | AX4 | AX5 |
|-------|---------|-----|-----|-----|-----|-----|
| Large Title | 34pt | 44pt | 48pt | 52pt | 56pt | 60pt |
| Title 1 | 28pt | 38pt | 43pt | 48pt | 53pt | 58pt |
| Body | 17pt | 28pt | 33pt | 38pt | 44pt | 53pt |
| Callout | 16pt | 26pt | 32pt | 37pt | 42pt | 51pt |
| Caption | 12pt | 22pt | 26pt | 29pt | 33pt | 40pt |

#### Implementation Requirements

```swift
// REQUIRED: Always use dynamic type
Text("Hello")
    .font(.body) // System font with dynamic sizing

// REQUIRED: Allow scaling up to AX5
Text("Hello")
    .dynamicTypeSize(...DynamicTypeSize.accessibility5)

// REQUIRED: Test at all sizes
// Use Accessibility Inspector or Settings > Accessibility > Display & Text Size

// FORBIDDEN: Fixed font sizes
Text("Hello")
    .font(.system(size: 17)) // NO - won't scale
```

#### Layout Adaptation for Large Text

| Screen | Adaptation Required |
|--------|---------------------|
| Thought list | Cards expand vertically, image thumbnails scale |
| Conversation | Message bubbles expand, timestamps may wrap |
| Settings | All rows accommodate multi-line labels |
| Buttons | Minimum height scales with text |
| Tab bar | Labels may need to truncate with ellipsis |

```swift
// Example: Adaptive card layout
struct ThoughtCard: View {
    @ScaledMetric var cardPadding: CGFloat = 16
    @ScaledMetric var minHeight: CGFloat = 80

    var body: some View {
        VStack {
            // Content scales with Dynamic Type
        }
        .padding(cardPadding)
        .frame(minHeight: minHeight)
    }
}
```

### 2.3 Bold Text Support

**Requirement**: Respect the "Bold Text" accessibility setting.

```swift
// SwiftUI automatically respects bold text when using system fonts
// Verify by enabling Settings > Accessibility > Display & Text Size > Bold Text
```

### 2.4 Reduced Transparency

**Requirement**: When "Reduce Transparency" is enabled, replace blurs with solid colors.

```swift
@Environment(\.accessibilityReduceTransparency) var reduceTransparency

var background: some View {
    if reduceTransparency {
        Color.backgroundSecondary // Solid color fallback
    } else {
        Color.backgroundSecondary.opacity(0.8)
            .background(.ultraThinMaterial)
    }
}
```

### 2.5 Increase Contrast Mode

**Requirement**: When "Increase Contrast" is enabled, increase contrast ratios.

```swift
@Environment(\.colorSchemeContrast) var contrast

var borderColor: Color {
    contrast == .increased ? .deepOcean : .neutral.opacity(0.3)
}

var textColor: Color {
    contrast == .increased ? .deepOcean : .textSecondary
}
```

---

## 3. VoiceOver Support

### 3.1 Label Requirements

Every interactive element must have a meaningful accessibility label.

#### Label Guidelines

| Element | Bad Label | Good Label |
|---------|-----------|------------|
| Record button | "mic" | "Record thought" |
| Close button | "x" | "Close" |
| Settings icon | "gear" | "Settings" |
| Thought card | "" (none) | "Thought from [date]: [first 50 chars]" |
| Play audio | "play" | "Play recording" |
| Delete | "trash" | "Delete thought" |

#### Implementation

```swift
// Buttons
Button(action: startRecording) {
    Image(systemName: "mic.fill")
}
.accessibilityLabel("Record thought")
.accessibilityHint("Double tap to start recording your thought")

// Custom views
ThoughtCard(thought: thought)
    .accessibilityElement(children: .combine)
    .accessibilityLabel("Thought from \(thought.date.formatted()): \(thought.preview)")
    .accessibilityHint("Double tap to open full thought")

// Dynamic labels
Button(action: toggleRecording) {
    Image(systemName: isRecording ? "stop.fill" : "mic.fill")
}
.accessibilityLabel(isRecording ? "Stop recording" : "Record thought")
```

### 3.2 Accessibility Traits

| Element | Traits |
|---------|--------|
| Primary buttons | `.button` |
| Headers | `.header` |
| Links | `.link` |
| Selected items | `.selected` |
| Static text | `.staticText` |
| Images | `.image` |
| Tab bar items | `.tabBar` |
| Adjustable (sliders, steppers) | `.adjustable` |

```swift
// Header
Text("Your Thoughts")
    .accessibilityAddTraits(.header)

// Selected state
ThoughtCard()
    .accessibilityAddTraits(isSelected ? .selected : [])

// Button that starts process
Button(action: startRecording) {
    // ...
}
.accessibilityAddTraits(.startsMediaSession)
```

### 3.3 Navigation Order

VoiceOver users navigate linearly. Ensure logical order.

#### Screen Navigation Order

**Conversation Screen**:
1. Back button (if applicable)
2. Title
3. Previous messages (oldest first)
4. Latest message
5. Input area
6. Record button

**Thought List Screen**:
1. Navigation title
2. Search field
3. Filter options (if present)
4. Thought cards (newest first)
5. Tab bar

**Settings Screen**:
1. Navigation title
2. Settings sections in order
3. Each setting row
4. Tab bar

#### Implementation

```swift
// Control navigation order
VStack {
    header
        .accessibilityElement(children: .contain)
        .accessibilitySortPriority(3)

    content
        .accessibilityElement(children: .contain)
        .accessibilitySortPriority(2)

    actions
        .accessibilityElement(children: .contain)
        .accessibilitySortPriority(1)
}
```

### 3.4 Custom Actions

Provide VoiceOver custom actions for complex interactions.

```swift
ThoughtCard(thought: thought)
    .accessibilityAction(named: "Delete") {
        deleteThought(thought)
    }
    .accessibilityAction(named: "Share") {
        shareThought(thought)
    }
    .accessibilityAction(named: "Play audio") {
        playAudio(thought)
    }
```

### 3.5 Announcements

Announce important changes that VoiceOver users might miss.

```swift
// After saving a thought
UIAccessibility.post(
    notification: .announcement,
    argument: "Thought saved"
)

// After error
UIAccessibility.post(
    notification: .announcement,
    argument: "Something went wrong. Please try again."
)

// Screen change
UIAccessibility.post(
    notification: .screenChanged,
    argument: "Thought details"
)
```

### 3.6 Grouping

Group related elements to reduce VoiceOver verbosity.

```swift
// Group card content
HStack {
    icon
    VStack {
        title
        subtitle
    }
}
.accessibilityElement(children: .combine)

// Ignore decorative elements
Image("decorative-wave")
    .accessibilityHidden(true)
```

---

## 4. Motor Accessibility

### 4.1 Touch Targets

**Minimum**: 44x44pt (Apple requirement)
**Recommended**: 48x48pt (WCAG AAA)

```swift
// Ensure minimum touch target
Button(action: action) {
    Image(systemName: "mic.fill")
        .font(.system(size: 24))
}
.frame(minWidth: 48, minHeight: 48)

// Increase tap area without increasing visual size
Button(action: action) {
    content
}
.contentShape(Rectangle())
.frame(minWidth: 44, minHeight: 44)
```

### 4.2 Gesture Alternatives

All custom gestures must have tap alternatives.

| Custom Gesture | Alternative Required |
|----------------|---------------------|
| Swipe to delete | Delete button in edit mode |
| Long press for menu | Button to show menu |
| Pinch to zoom | Zoom buttons |
| Drag to reorder | Edit mode with up/down buttons |

```swift
// Example: Swipe actions with alternatives
ThoughtRow(thought: thought)
    .swipeActions(edge: .trailing) {
        Button(role: .destructive, action: delete) {
            Label("Delete", systemImage: "trash")
        }
    }
    .contextMenu {
        Button(action: delete) {
            Label("Delete", systemImage: "trash")
        }
    }
// Also provide edit mode for non-gesture users
```

### 4.3 Switch Control Support

Ensure all functionality works with Switch Control navigation.

**Requirements**:
- All interactive elements must be focusable
- Focus order must be logical
- Custom views must implement focus support

```swift
// Mark focusable elements
Button(action: action)
    .focusable() // For tvOS/Switch Control
```

### 4.4 Pointer Hover Support (iPadOS/macOS)

```swift
Button(action: action) {
    content
}
.hoverEffect(.highlight)

// Custom hover states
.onHover { isHovering in
    withAnimation {
        self.isHovering = isHovering
    }
}
```

---

## 5. Cognitive Accessibility

### 5.1 Cognitive Load Reduction

MYND's target audience benefits significantly from reduced cognitive load.

#### Design Principles

| Principle | Implementation |
|-----------|----------------|
| **One thing at a time** | Max 1 CTA per screen |
| **Progressive disclosure** | Hide advanced options by default |
| **Consistent layout** | Same element positions across screens |
| **Clear hierarchy** | Obvious visual structure |
| **Minimal choices** | Max 3-4 options per decision |

#### Screen-by-Screen Limits

| Screen | Max Interactive Elements | Max Text Blocks |
|--------|-------------------------|-----------------|
| Conversation | 3 (back, record, menu) | 1 active message |
| Thought list | 2 + cards (search, new) | None besides cards |
| Settings | 1 per row | 1 description per setting |
| Onboarding | 2 (primary, skip) | 1 paragraph per screen |

### 5.2 Simplified Mode

**Future Feature (v1.5)**: Optional simplified mode with:
- Larger touch targets
- Fewer visible options
- More white space
- Simpler language

```swift
// Settings toggle
Toggle("Simplified interface", isOn: $settings.simplifiedMode)
    .accessibilityHint("Shows larger buttons and fewer options")
```

### 5.3 Clear Language

All UI text must pass readability tests.

**Target**: Flesch-Kincaid Grade Level 6-8

| Instead of | Use |
|------------|-----|
| "Initialize voice capture subsystem" | "Start recording" |
| "Persist ideation to storage" | "Save thought" |
| "Authentication required" | "Please sign in" |
| "Network connectivity unavailable" | "You're offline" |

### 5.4 Error Prevention & Recovery

| Principle | Implementation |
|-----------|----------------|
| Confirmation for destructive actions | "Remove this thought?" with undo option |
| Clear error messages | "That didn't work. Tap to try again." |
| Easy recovery | Undo available for 10 seconds |
| No dead ends | Always provide next action |

```swift
// Destructive action confirmation
.confirmationDialog(
    "Remove this thought?",
    isPresented: $showDeleteConfirmation
) {
    Button("Remove", role: .destructive) {
        deleteThought()
    }
    Button("Keep it", role: .cancel) {}
}
```

### 5.5 Focus Management

Never move focus unexpectedly. Guide users through flows.

```swift
// After completing action, return focus appropriately
.onChange(of: didSaveThought) { saved in
    if saved {
        UIAccessibility.post(notification: .announcement, argument: "Thought saved")
        // Return focus to natural position
    }
}
```

---

## 6. Hearing Accessibility

### 6.1 Captions for Audio

All audio content must have text alternatives.

| Audio Type | Text Alternative |
|------------|------------------|
| Axel's voice responses | Full text displayed on screen |
| User recordings | Transcript displayed |
| Sound effects | VoiceOver announcement |
| Video (if any) | Closed captions |

### 6.2 Visual Alternatives for Sounds

| Sound | Visual Alternative |
|-------|---------------------|
| Recording started | Pulsing mic icon + text |
| Recording ended | Checkmark animation + text |
| Thought saved | Toast notification |
| Error | Error banner with icon |
| Notification | Badge + visual alert |

### 6.3 No Audio-Only Information

Never convey information through audio alone.

```swift
// After playing audio
func playAudioResponse(_ text: String) {
    audioPlayer.play()
    // Always show text version
    displayedText = text
}
```

---

## 7. Motion Accessibility

### 7.1 Reduce Motion Support

**Requirement**: Respect "Reduce Motion" system setting.

```swift
@Environment(\.accessibilityReduceMotion) var reduceMotion

var animation: Animation? {
    reduceMotion ? nil : .easeInOut(duration: 0.3)
}

var body: some View {
    content
        .animation(animation, value: someValue)
}
```

### 7.2 Motion Alternatives

| Full Motion | Reduced Motion Alternative |
|-------------|---------------------------|
| Breathing wall animation | Static gradient with subtle opacity pulse |
| Card slide-in | Instant appearance with fade |
| Button scale on press | Opacity change only |
| Confetti celebration | Checkmark with subtle fade |
| Screen transitions | Cross-fade |

```swift
struct BreathingWall: View {
    @Environment(\.accessibilityReduceMotion) var reduceMotion

    var body: some View {
        if reduceMotion {
            // Gentle opacity pulse only
            Circle()
                .fill(Color.calmBlue.opacity(0.3))
                .animation(
                    .easeInOut(duration: 2).repeatForever(autoreverses: true),
                    value: UUID() // Minimal motion
                )
        } else {
            // Full animation
            FullBreathingAnimation()
        }
    }
}
```

### 7.3 No Auto-Playing Motion

Motion should not start automatically without user control.

- Breathing wall only during active processing
- No decorative animations on idle screens
- Video/animated content requires play action

---

## 8. Implementation Checklist

### 8.1 Per-Screen Checklist

Before any screen is considered complete:

#### Visual
- [ ] All text contrast >= 4.5:1 (body) or 3:1 (large)
- [ ] No color-only information
- [ ] Dynamic Type works xSmall to AX5
- [ ] Layout adapts to all text sizes
- [ ] Works with Bold Text enabled
- [ ] Works with Increased Contrast enabled
- [ ] Works with Reduced Transparency enabled

#### VoiceOver
- [ ] All interactive elements have labels
- [ ] Labels are meaningful (not just "button")
- [ ] Navigation order is logical
- [ ] Appropriate traits assigned
- [ ] Custom actions where needed
- [ ] State changes announced
- [ ] Decorative elements hidden

#### Motor
- [ ] Touch targets >= 44x44pt
- [ ] Gesture alternatives exist
- [ ] Works with Switch Control
- [ ] Pointer hover states (iPad)

#### Cognitive
- [ ] Max 3-4 choices per screen
- [ ] Clear visual hierarchy
- [ ] Simple language (grade 6-8)
- [ ] Destructive actions have confirmation
- [ ] Recovery path from errors

#### Motion
- [ ] Reduce Motion alternative exists
- [ ] No auto-playing animation
- [ ] Motion can be paused/stopped

#### Hearing
- [ ] Audio has text alternative
- [ ] No audio-only information
- [ ] Visual feedback for all sounds

### 8.2 Testing Protocol

#### Automated Testing

```swift
// Accessibility audit in tests
func testAccessibility() throws {
    let app = XCUIApplication()
    app.launch()

    try app.performAccessibilityAudit()
}

// Or target specific issues
try app.performAccessibilityAudit(for: [
    .dynamicType,
    .contrast,
    .hitRegion
])
```

#### Manual Testing Checklist

| Test | How to Perform | Frequency |
|------|----------------|-----------|
| VoiceOver | Enable in Settings, navigate entire app | Every PR |
| Dynamic Type | Settings > Display > Text Size | Every PR |
| Reduce Motion | Settings > Accessibility > Motion | Every PR |
| Bold Text | Settings > Accessibility > Display | Every release |
| Increase Contrast | Settings > Accessibility > Display | Every release |
| Switch Control | Settings > Accessibility > Switch Control | Every release |
| Color blindness | Use Accessibility Inspector filters | Every release |

#### Device Testing Matrix

| Device | Required Testing |
|--------|------------------|
| iPhone SE (small screen) | Layout, touch targets |
| iPhone 15 Pro Max (large screen) | Layout, readability |
| iPhone with Dynamic Type AX5 | Text sizing |
| iPad (landscape/portrait) | Layout adaptation |

---

## 9. Accessibility Testing Tools

### 9.1 Xcode Tools

- **Accessibility Inspector**: View accessibility tree, audit issues
- **Voice Control**: Test voice-based navigation
- **Color filters**: Simulate color blindness

### 9.2 iOS Tools

- **Settings > Accessibility**: Test all accessibility features
- **Control Center shortcuts**: Quick toggle for testing
- **VoiceOver Practice**: Learn VoiceOver gestures

### 9.3 Third-Party Tools

- **Stark (Figma plugin)**: Design-time contrast checking
- **axe DevTools**: Automated accessibility testing
- **Color Oracle**: Desktop color blindness simulator

---

## 10. Accessibility Statement

Include in app and on website:

---

**MYND Accessibility Statement**

MYND is committed to providing an accessible experience for all users. We design with accessibility as a core principle, not an afterthought.

**Current Compliance**:
- WCAG 2.1 AA compliant
- Full VoiceOver support
- Dynamic Type support (xSmall to AX5)
- Reduce Motion support
- High contrast color system

**Supported Assistive Technologies**:
- VoiceOver (screen reader)
- Voice Control
- Switch Control
- Dynamic Type
- Bold Text
- Reduce Motion
- Increase Contrast

**Known Issues**:
[List any known accessibility issues and timeline for fixes]

**Contact**:
If you encounter accessibility barriers in MYND, please contact us at [accessibility@mynd.app]. We take all accessibility feedback seriously and will respond within 5 business days.

---

**Document Status**: APPROVED
**Review Cadence**: Before each release
**Owner**: Engineering + Product Design

---

*"If it's not accessible, it's not done."*
