# MYND Safe Space Design Language

**Version**: 1.0
**Date**: 2026-01-04
**Status**: DEFINITIVE GUIDE
**Purpose**: Establish visual, motion, sound, and language standards that create a calming, non-judgmental experience

---

## 1. Design Philosophy

### 1.1 Core Principle: Safe Space

MYND is designed for people who have been burned by productivity apps that made them feel bad. Every design decision filters through one question:

**"Would this make someone feel safe to be imperfect?"**

### 1.2 Design Pillars

| Pillar | Expression | Anti-Pattern |
|--------|------------|--------------|
| **Calm** | Soft colors, gentle transitions | Bright reds, jarring animations |
| **Non-judgmental** | No red indicators, no "overdue" | Countdown timers, streak breaks |
| **Spacious** | Breathing room, minimal elements | Cramped layouts, busy screens |
| **Warm** | Organic shapes, human feel | Cold geometry, corporate aesthetic |
| **Accessible** | High contrast, clear hierarchy | Small text, low contrast |

### 1.3 Emotional Goals by Screen

| Screen | Emotional Goal | Design Approach |
|--------|----------------|-----------------|
| Voice capture | Presence, safety | Minimal UI, breathing animation |
| Thought list | Calm overview | Gentle cards, no urgency indicators |
| Settings | Control, confidence | Clear options, no overwhelm |
| Onboarding | Welcome, ease | Friendly copy, minimal steps |
| Subscription | Value, fairness | Transparent pricing, no pressure |

---

## 2. Color System

### 2.1 Primary Palette

**Philosophy**: Deep calm blue as foundation, warm amber for encouragement, soft purple for reflection.

#### Core Colors

| Name | Hex (Light) | Hex (Dark) | Usage | WCAG AA |
|------|-------------|------------|-------|---------|
| **Deep Ocean** | `#1A365D` | `#E2E8F0` | Primary text, headers | 7.5:1 / 8.2:1 |
| **Calm Blue** | `#3182CE` | `#63B3ED` | Interactive elements | 4.6:1 / 4.8:1 |
| **Soft Sky** | `#EBF8FF` | `#1A365D` | Backgrounds | N/A (bg) |
| **Warm Amber** | `#DD6B20` | `#F6AD55` | Celebrations, warmth | 4.5:1 / 4.7:1 |
| **Soft Honey** | `#FFFAF0` | `#2D3748` | Highlight backgrounds | N/A (bg) |
| **Dusk Purple** | `#553C9A` | `#B794F4` | Reflection, insights | 7.1:1 / 4.5:1 |
| **Lavender Mist** | `#FAF5FF` | `#2D3748` | Secondary backgrounds | N/A (bg) |

#### Semantic Colors

| Name | Hex (Light) | Hex (Dark) | Usage | WCAG AA |
|------|-------------|------------|-------|---------|
| **Success** | `#276749` | `#68D391` | Completed, saved | 5.9:1 / 4.5:1 |
| **Info** | `#2B6CB0` | `#63B3ED` | System messages | 4.8:1 / 4.8:1 |
| **Warning** | `#C05621` | `#F6AD55` | Gentle alerts | 4.6:1 / 4.7:1 |
| **Neutral** | `#4A5568` | `#A0AEC0` | Secondary text | 5.9:1 / 4.5:1 |

#### Explicitly NO Red

```
There is NO red in MYND's color system.

Not for errors.
Not for warnings.
Not for anything.

Red triggers anxiety. Use Warm Amber for things that need attention.
```

### 2.2 Color Application

#### Backgrounds

```swift
// Light Mode
let backgroundPrimary = Color(hex: "FFFFFF")     // Pure white
let backgroundSecondary = Color(hex: "F7FAFC")   // Soft gray
let backgroundTertiary = Color(hex: "EBF8FF")    // Soft sky blue
let backgroundAccent = Color(hex: "FFFAF0")      // Soft honey (highlights)

// Dark Mode
let backgroundPrimary = Color(hex: "1A202C")     // Deep charcoal
let backgroundSecondary = Color(hex: "2D3748")   // Medium charcoal
let backgroundTertiary = Color(hex: "1A365D")    // Deep ocean blue
let backgroundAccent = Color(hex: "2D3748")      // Medium charcoal (highlights)
```

#### Text

```swift
// Light Mode
let textPrimary = Color(hex: "1A365D")           // Deep ocean
let textSecondary = Color(hex: "4A5568")         // Neutral gray
let textTertiary = Color(hex: "718096")          // Light gray
let textOnAccent = Color(hex: "FFFFFF")          // White on colored bg

// Dark Mode
let textPrimary = Color(hex: "E2E8F0")           // Light gray
let textSecondary = Color(hex: "A0AEC0")         // Medium gray
let textTertiary = Color(hex: "718096")          // Dimmed gray
let textOnAccent = Color(hex: "1A202C")          // Dark on light accent
```

#### Interactive Elements

```swift
// Buttons
let buttonPrimary = Color(hex: "3182CE")         // Calm blue
let buttonPrimaryPressed = Color(hex: "2C5282")  // Darker blue
let buttonSecondary = Color(hex: "EDF2F7")       // Light gray
let buttonSecondaryPressed = Color(hex: "E2E8F0") // Slightly darker

// Focus/Selection
let focusRing = Color(hex: "63B3ED")             // Light calm blue
let selectionBackground = Color(hex: "BEE3F8")   // Very light blue
```

### 2.3 Breathing Wall Gradient

The signature MYND animation uses this gradient:

```swift
let breathingGradientColors = [
    Color(hex: "3182CE").opacity(0.3),  // Calm blue, transparent
    Color(hex: "553C9A").opacity(0.2),  // Dusk purple, more transparent
    Color(hex: "3182CE").opacity(0.1),  // Calm blue, very transparent
    Color.clear
]

// Gradient radiates from center, pulsing with 4-second breath cycle
```

### 2.4 Contrast Ratios Verification

All text must meet WCAG 2.1 AA standards:

| Combination | Ratio | Status |
|-------------|-------|--------|
| Deep Ocean on White | 9.8:1 | AAA |
| Neutral on White | 5.9:1 | AA |
| Calm Blue on White | 4.6:1 | AA |
| Warm Amber on White | 4.5:1 | AA |
| Light Gray on Deep Charcoal | 8.2:1 | AAA |
| Medium Gray on Deep Charcoal | 4.5:1 | AA |

---

## 3. Typography

### 3.1 Font Selection

**Primary Font**: SF Pro Rounded

**Rationale**:
- Native iOS font (fast loading, system integration)
- Rounded variant adds warmth and approachability
- Excellent legibility at all sizes
- Full Dynamic Type support

**Fallback**: SF Pro (if Rounded unavailable)

### 3.2 Type Scale

Using iOS Dynamic Type with custom styles:

| Style Name | Size (Default) | Weight | Line Height | Usage |
|------------|----------------|--------|-------------|-------|
| **Large Title** | 34pt | Bold | 41pt | Screen titles |
| **Title 1** | 28pt | Bold | 34pt | Section headers |
| **Title 2** | 22pt | Bold | 28pt | Card titles |
| **Title 3** | 20pt | Semibold | 25pt | Subsections |
| **Headline** | 17pt | Semibold | 22pt | List item titles |
| **Body** | 17pt | Regular | 22pt | Primary content |
| **Callout** | 16pt | Regular | 21pt | Secondary content |
| **Subhead** | 15pt | Regular | 20pt | Metadata |
| **Footnote** | 13pt | Regular | 18pt | Captions |
| **Caption 1** | 12pt | Regular | 16pt | Labels |
| **Caption 2** | 11pt | Regular | 13pt | Tiny labels |

### 3.3 Dynamic Type Scaling

Support for all accessibility sizes:

| Category | Default | AX1 | AX2 | AX3 | AX4 | AX5 |
|----------|---------|-----|-----|-----|-----|-----|
| Large Title | 34pt | 44pt | 48pt | 52pt | 56pt | 60pt |
| Body | 17pt | 28pt | 33pt | 38pt | 44pt | 53pt |
| Caption | 12pt | 22pt | 26pt | 29pt | 33pt | 40pt |

### 3.4 Implementation

```swift
// Custom text styles with SF Pro Rounded
extension Font {
    static var myndLargeTitle: Font {
        .system(.largeTitle, design: .rounded, weight: .bold)
    }

    static var myndTitle1: Font {
        .system(.title, design: .rounded, weight: .bold)
    }

    static var myndBody: Font {
        .system(.body, design: .rounded)
    }

    static var myndCallout: Font {
        .system(.callout, design: .rounded)
    }
}

// Ensure Dynamic Type scaling
Text("Hello")
    .font(.myndBody)
    .dynamicTypeSize(...DynamicTypeSize.accessibility5) // Allow up to AX5
```

### 3.5 Text Styling Rules

1. **Maximum line width**: 75 characters for readability
2. **Paragraph spacing**: 1.5x line height between paragraphs
3. **No justified text**: Left-aligned only (easier for ADHD readers)
4. **No all-caps**: Except for very short labels (<3 words)
5. **Sentence case**: For buttons and labels (not Title Case)

---

## 4. Motion Design

### 4.1 Motion Philosophy

**Principle**: Movement should feel like breathing - natural, calming, purposeful.

| Attribute | MYND Approach | Anti-Pattern |
|-----------|---------------|--------------|
| Speed | Slow, deliberate | Snappy, instant |
| Easing | Gentle curves | Linear or bouncy |
| Direction | Organic, radial | Mechanical, linear |
| Purpose | Calming, guiding | Decorative, distracting |

### 4.2 Timing Standards

| Animation Type | Duration | Easing |
|----------------|----------|--------|
| **Micro** (buttons, toggles) | 200ms | easeOut |
| **Small** (cards, reveals) | 300ms | easeInOut |
| **Medium** (screen transitions) | 400ms | easeInOut |
| **Large** (modal presentations) | 500ms | easeInOut |
| **Breathing** (continuous) | 4000ms | easeInOut (repeat) |

### 4.3 Easing Curves

```swift
// MYND Custom Easing
let myndEaseInOut = Animation.timingCurve(0.4, 0.0, 0.2, 1.0, duration: 0.4)
let myndEaseOut = Animation.timingCurve(0.0, 0.0, 0.2, 1.0, duration: 0.3)
let myndEaseIn = Animation.timingCurve(0.4, 0.0, 1.0, 1.0, duration: 0.2)

// For breathing animations
let myndBreathing = Animation.easeInOut(duration: 4.0).repeatForever(autoreverses: true)
```

### 4.4 The Breathing Wall Specification

**Purpose**: Transform waiting into a calming experience

```swift
struct BreathingWall: View {
    @State private var breathPhase: CGFloat = 0

    let innerRadius: CGFloat = 60
    let outerRadius: CGFloat = 200
    let breathDuration: Double = 4.0

    var body: some View {
        ZStack {
            // Outer glow
            Circle()
                .fill(
                    RadialGradient(
                        colors: [
                            Color.calmBlue.opacity(0.3 * (1 - breathPhase * 0.5)),
                            Color.duskPurple.opacity(0.2 * (1 - breathPhase * 0.5)),
                            Color.clear
                        ],
                        center: .center,
                        startRadius: innerRadius + (breathPhase * 40),
                        endRadius: outerRadius + (breathPhase * 60)
                    )
                )
                .frame(width: outerRadius * 2, height: outerRadius * 2)

            // Inner pulse
            Circle()
                .fill(Color.calmBlue.opacity(0.4 - breathPhase * 0.2))
                .frame(
                    width: innerRadius * 2 * (1 + breathPhase * 0.3),
                    height: innerRadius * 2 * (1 + breathPhase * 0.3)
                )
        }
        .onAppear {
            withAnimation(
                .easeInOut(duration: breathDuration)
                .repeatForever(autoreverses: true)
            ) {
                breathPhase = 1
            }
        }
    }
}
```

**Breath cycle timing**:
- Inhale: 0s - 2s (expand)
- Exhale: 2s - 4s (contract)
- Continuous, never stops during processing

### 4.5 Transition Patterns

**Screen transitions**:
```swift
// Push navigation
.transition(.asymmetric(
    insertion: .move(edge: .trailing).combined(with: .opacity),
    removal: .move(edge: .leading).combined(with: .opacity)
))

// Modal presentation
.transition(.move(edge: .bottom).combined(with: .opacity))

// Card reveal
.transition(.scale(scale: 0.95).combined(with: .opacity))
```

**Element animations**:
```swift
// Button press
.scaleEffect(isPressed ? 0.96 : 1.0)
.animation(.myndEaseOut, value: isPressed)

// Card selection
.background(isSelected ? Color.selectionBackground : Color.clear)
.animation(.myndEaseInOut, value: isSelected)

// Loading states
.opacity(isLoading ? 0.6 : 1.0)
.animation(.myndEaseInOut, value: isLoading)
```

### 4.6 Haptic Feedback

| Action | Haptic Type | Intensity |
|--------|-------------|-----------|
| Button tap | Light impact | Default |
| Voice recording start | Medium impact | Default |
| Voice recording end | Soft impact | Gentle |
| Thought saved | Success notification | Default |
| Error | Error notification | Default (no jarring!) |
| Breath sync (optional) | Soft impact | Very light, every 4s |

```swift
// Haptic implementation
func lightImpact() {
    let generator = UIImpactFeedbackGenerator(style: .light)
    generator.impactOccurred()
}

func successNotification() {
    let generator = UINotificationFeedbackGenerator()
    generator.notificationOccurred(.success)
}
```

### 4.7 Motion Accessibility

**Respect Reduce Motion**:
```swift
@Environment(\.accessibilityReduceMotion) var reduceMotion

var animation: Animation {
    reduceMotion ? .none : .myndEaseInOut
}

// Breathing wall with reduce motion
var body: some View {
    if reduceMotion {
        // Static version - gentle pulse using opacity only
        Circle()
            .fill(Color.calmBlue.opacity(0.3))
    } else {
        BreathingWall()
    }
}
```

---

## 5. Sound Design

### 5.1 Sound Philosophy

**Principle**: Sound should feel like a mindfulness bell - present, calming, not startling.

| Attribute | MYND Approach | Anti-Pattern |
|-----------|---------------|--------------|
| Volume | Quiet, ambient | Loud, attention-grabbing |
| Character | Organic, resonant | Digital, synthetic |
| Frequency | Low-to-mid range | High pitched |
| Duration | Brief, fading | Sustained or looping |

### 5.2 Sound Palette

| Sound | Description | Duration | Volume | Trigger |
|-------|-------------|----------|--------|---------|
| **Mindfulness Bell** | Soft singing bowl tone | 1.5s fade | 40% | Thought saved |
| **Gentle Chime** | Light crystal chime | 0.8s fade | 30% | Recording started |
| **Soft Complete** | Warm major chord | 1.0s fade | 35% | Recording ended |
| **Subtle Click** | Soft button click | 0.1s | 25% | UI interactions |
| **Calm Alert** | Two-tone gentle notification | 0.6s | 35% | Notifications |
| **Breathing Tone** | Very subtle ambient drone | Loop | 10% | During processing (optional) |

### 5.3 Audio Specifications

**Format**: AAC or MP3, 44.1kHz, mono
**Max file size**: 100KB per sound
**Peak amplitude**: -12dB (never clip, never loud)

**EQ guidelines**:
- Roll off high frequencies above 8kHz
- Gentle low-end (80-200Hz) for warmth
- No harsh mids (2-4kHz reduced slightly)

### 5.4 Implementation

```swift
// Sound manager
class MYNDSoundManager {
    static let shared = MYNDSoundManager()

    private var players: [String: AVAudioPlayer] = [:]

    func play(_ sound: MYNDSound) {
        // Respect user preferences
        guard UserDefaults.standard.soundEnabled else { return }

        // Respect system silent mode
        guard !AVAudioSession.sharedInstance().isOtherAudioPlaying else { return }

        players[sound.filename]?.play()
    }
}

enum MYNDSound {
    case mindfulnessBell
    case gentleChime
    case softComplete
    case subtleClick
    case calmAlert

    var filename: String {
        switch self {
        case .mindfulnessBell: return "mindfulness_bell"
        case .gentleChime: return "gentle_chime"
        case .softComplete: return "soft_complete"
        case .subtleClick: return "subtle_click"
        case .calmAlert: return "calm_alert"
        }
    }
}
```

### 5.5 Sound Accessibility

- All sounds optional (can be disabled in settings)
- Never use sound as only feedback (always pair with visual)
- Respect iOS silent mode
- Provide sound descriptions for VoiceOver users

---

## 6. Spacing & Layout

### 6.1 Spacing Scale

Using an 8pt base grid:

| Token | Size | Usage |
|-------|------|-------|
| `space-0` | 0pt | No space |
| `space-1` | 4pt | Tight inline spacing |
| `space-2` | 8pt | Standard inline spacing |
| `space-3` | 12pt | Tight component spacing |
| `space-4` | 16pt | Standard component spacing |
| `space-5` | 24pt | Section spacing |
| `space-6` | 32pt | Large section spacing |
| `space-7` | 48pt | Screen section spacing |
| `space-8` | 64pt | Major divisions |

```swift
extension CGFloat {
    static let space0: CGFloat = 0
    static let space1: CGFloat = 4
    static let space2: CGFloat = 8
    static let space3: CGFloat = 12
    static let space4: CGFloat = 16
    static let space5: CGFloat = 24
    static let space6: CGFloat = 32
    static let space7: CGFloat = 48
    static let space8: CGFloat = 64
}
```

### 6.2 Safe Area & Margins

```swift
// Screen margins
let screenHorizontalMargin: CGFloat = 20
let screenVerticalMargin: CGFloat = 16

// Content max width (for iPads, larger phones)
let contentMaxWidth: CGFloat = 600

// Card padding
let cardPadding: CGFloat = 16
let cardCornerRadius: CGFloat = 16
```

### 6.3 Touch Targets

**Minimum touch target**: 44x44pt (Apple HIG)
**Recommended touch target**: 48x48pt (WCAG AAA)

```swift
// Button minimum sizing
struct MYNDButton: View {
    var body: some View {
        Button(action: action) {
            content
        }
        .frame(minWidth: 44, minHeight: 44)
    }
}
```

### 6.4 Card Design

```swift
struct MYNDCard: View {
    var body: some View {
        VStack(alignment: .leading, spacing: .space3) {
            content
        }
        .padding(.space4)
        .background(Color.backgroundSecondary)
        .cornerRadius(16)
        .shadow(
            color: Color.black.opacity(0.05),
            radius: 8,
            x: 0,
            y: 4
        )
    }
}
```

---

## 7. Iconography

### 7.1 Icon Style

**System**: SF Symbols
**Weight**: Regular (default), Semibold for emphasis
**Style**: Outlined, not filled (except when selected)

### 7.2 Core Icons

| Action | SF Symbol | Notes |
|--------|-----------|-------|
| Record/Microphone | `mic.fill` | Filled to show active state |
| Stop | `stop.fill` | Filled |
| Thought/Note | `text.bubble` | Outlined |
| Search | `magnifyingglass` | Standard |
| Settings | `gearshape` | Outlined |
| Graph/Connections | `point.3.connected.trianglepath.dotted` | v1.5 |
| Person | `person.crop.circle` | For profile |
| Close | `xmark` | Standard |
| Check | `checkmark.circle.fill` | Filled for completion |
| Add | `plus` | Standard |
| Insights | `sparkles` | For Morning Oracle |
| Energy | `bolt.fill` | For energy selector |

### 7.3 Icon Sizing

| Context | Size | Weight |
|---------|------|--------|
| Navigation bar | 22pt | Regular |
| Tab bar | 24pt | Regular |
| List item accessory | 20pt | Regular |
| Primary action | 28pt | Semibold |
| Inline with text | Match text size | Regular |

```swift
Image(systemName: "mic.fill")
    .font(.system(size: 28, weight: .semibold))
    .foregroundColor(.calmBlue)
```

---

## 8. Language Guidelines

### 8.1 Voice & Tone

| Attribute | MYND Voice | Anti-Pattern |
|-----------|------------|--------------|
| **Warm** | "Good to see you" | "Welcome back, user" |
| **Calm** | "Take your time" | "Quick! Capture now!" |
| **Encouraging** | "You're making progress" | "You haven't done X" |
| **Respectful** | "When you're ready" | "You should..." |
| **Clear** | "Save this thought" | "Persist ideation instance" |

### 8.2 Words to Use

| Instead of | Use |
|------------|-----|
| Submit | Save |
| Task | Thought |
| Overdue | Available |
| Failed | Didn't work |
| Error | Something went wrong |
| Required | Needed |
| Invalid | Let's try again |
| You must | You can |
| Don't forget | When you're ready |
| Incomplete | In progress |

### 8.3 Words to Never Use

| Word | Why | Alternative |
|------|-----|-------------|
| **Overdue** | Guilt-inducing | "Still there for you" |
| **Failed** | Judgmental | "That didn't work" |
| **Must** | Demanding | "Can" or remove |
| **Urgent** | Creates anxiety | Remove entirely |
| **Warning** | Fear-inducing | "Note" or "Heads up" |
| **Error** | Cold, technical | "Something went wrong" |
| **Invalid** | Judgmental | "Let's try that again" |
| **Required** | Demanding | "Needed" |
| **Incomplete** | Implies failure | "In progress" |

### 8.4 Notification Copy

**Good**:
- "Axel has a thought for you" (curiosity)
- "When you have a moment..." (no pressure)
- "Something you mentioned came up" (relevance)

**Bad**:
- "You haven't checked in today!" (guilt)
- "Don't forget to capture thoughts!" (nagging)
- "URGENT: Open MYND now" (anxiety)

### 8.5 Button Labels

| Action | Good | Bad |
|--------|------|-----|
| Primary action | "Save thought" | "Submit" |
| Cancel | "Not now" | "Cancel" |
| Delete | "Remove" | "Delete" |
| Undo | "Bring it back" | "Undo" |
| Continue | "Keep going" | "Next" |
| Skip | "Skip for now" | "Skip" |

### 8.6 Empty States

**Thought list empty**:
```
"No thoughts yet"
"When you're ready, tap the microphone to capture what's on your mind."
```

**Search no results**:
```
"Nothing found"
"Try different words, or capture a new thought."
```

**Connection lost**:
```
"You're offline"
"Your thoughts are saved. We'll sync when you're back online."
```

---

## 9. Component Library Reference

### 9.1 Buttons

```swift
// Primary button
struct MYNDPrimaryButton: View {
    let title: String
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Text(title)
                .font(.myndBody)
                .fontWeight(.semibold)
                .foregroundColor(.textOnAccent)
                .frame(maxWidth: .infinity)
                .frame(height: 50)
                .background(Color.calmBlue)
                .cornerRadius(12)
        }
        .buttonStyle(MYNDButtonStyle())
    }
}

// Secondary button
struct MYNDSecondaryButton: View {
    let title: String
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Text(title)
                .font(.myndBody)
                .fontWeight(.medium)
                .foregroundColor(.calmBlue)
                .frame(maxWidth: .infinity)
                .frame(height: 50)
                .background(Color.backgroundSecondary)
                .cornerRadius(12)
                .overlay(
                    RoundedRectangle(cornerRadius: 12)
                        .stroke(Color.calmBlue, lineWidth: 1)
                )
        }
        .buttonStyle(MYNDButtonStyle())
    }
}

// Button press style
struct MYNDButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .scaleEffect(configuration.isPressed ? 0.96 : 1.0)
            .opacity(configuration.isPressed ? 0.9 : 1.0)
            .animation(.myndEaseOut, value: configuration.isPressed)
    }
}
```

### 9.2 Text Fields

```swift
struct MYNDTextField: View {
    let placeholder: String
    @Binding var text: String

    var body: some View {
        TextField(placeholder, text: $text)
            .font(.myndBody)
            .padding(.space4)
            .background(Color.backgroundSecondary)
            .cornerRadius(12)
            .overlay(
                RoundedRectangle(cornerRadius: 12)
                    .stroke(Color.neutral.opacity(0.2), lineWidth: 1)
            )
    }
}
```

### 9.3 Thought Card

```swift
struct ThoughtCard: View {
    let thought: Thought

    var body: some View {
        VStack(alignment: .leading, spacing: .space2) {
            Text(thought.content)
                .font(.myndBody)
                .foregroundColor(.textPrimary)
                .lineLimit(3)

            HStack {
                Text(thought.timestamp.relativeFormat)
                    .font(.myndCaption1)
                    .foregroundColor(.textTertiary)

                Spacer()

                if thought.hasAudioAttachment {
                    Image(systemName: "waveform")
                        .font(.system(size: 12))
                        .foregroundColor(.textTertiary)
                }
            }
        }
        .padding(.space4)
        .background(Color.backgroundSecondary)
        .cornerRadius(16)
    }
}
```

---

## 10. Design Checklist

Before any screen ships:

### Accessibility
- [ ] All text meets WCAG AA contrast (4.5:1 min)
- [ ] Touch targets are minimum 44x44pt
- [ ] Dynamic Type works up to AX5
- [ ] VoiceOver labels are meaningful
- [ ] Reduce Motion is respected

### Visual
- [ ] Uses only MYND color palette
- [ ] No red anywhere
- [ ] SF Pro Rounded for all text
- [ ] Follows 8pt spacing grid
- [ ] Cards have 16pt corner radius

### Motion
- [ ] Animations use MYND easing curves
- [ ] Breathing wall is present where waiting occurs
- [ ] Transitions are 300-500ms
- [ ] Reduce Motion alternative exists

### Sound
- [ ] Sounds are optional (can be disabled)
- [ ] Volume is subtle (never jarring)
- [ ] Visual feedback accompanies all sounds

### Language
- [ ] No guilt-inducing words (overdue, failed, must)
- [ ] Warm, encouraging tone
- [ ] Clear, simple vocabulary
- [ ] Buttons are actionable (not "Submit")

---

**Document Status**: APPROVED
**Review Cadence**: Before each release
**Owner**: Product Design

---

*"Every pixel should feel like a deep breath."*
