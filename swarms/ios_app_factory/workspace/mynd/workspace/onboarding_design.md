# MYND Onboarding Flow Design

**Version**: 1.0
**Date**: 2026-01-04
**Status**: DEFINITIVE GUIDE
**Goal**: Value in 30 seconds, subscription decision in 5 minutes

---

## 1. Onboarding Philosophy

### 1.1 Core Principle: Value Before Commitment

Traditional onboarding: Sign up → Learn features → Maybe use app

**MYND onboarding**: Use app → Experience value → Then decide to commit

### 1.2 Design Goals

| Goal | Metric | Target |
|------|--------|--------|
| Time to first thought captured | Seconds | < 30 |
| Time to experience Axel response | Seconds | < 60 |
| Demo to trial conversion | Percentage | > 40% |
| Onboarding completion | Percentage | > 80% |
| Permission grant rate | Percentage | > 90% |

### 1.3 Anti-Patterns to Avoid

| Anti-Pattern | Why It Fails for ADHD Users | MYND Alternative |
|--------------|----------------------------|------------------|
| Multiple onboarding screens | Loses attention, feels like work | Single screen, immediate action |
| Feature tours | Information overload | Learn by doing |
| Account creation first | Friction before value | Value first, account later |
| Complex permission dialogs | Decision paralysis | Contextual, just-in-time |
| Streak/gamification pitch | Creates future guilt | Focus on present value |

---

## 2. The First 30 Seconds (Critical Window)

### 2.1 App Launch Experience

#### State: Cold Launch (Brand New User)

**Screen 1: Instant Value (No Splash Screen)**

```
┌─────────────────────────────────────────┐
│                                         │
│                                         │
│                                         │
│                                         │
│                                         │
│            ╭───────────────╮            │
│            │               │            │
│            │    [  mic  ]  │            │
│            │               │            │
│            ╰───────────────╯            │
│                                         │
│        "What's on your mind?"           │
│                                         │
│      ──────────────────────────         │
│      tap the microphone to start        │
│                                         │
│                                         │
│                                         │
└─────────────────────────────────────────┘
```

**Elements**:
- Large, centered microphone button (80pt diameter)
- Single question: "What's on your mind?"
- Subtle helper text: "tap the microphone to start"
- No navigation, no settings, no back button
- No logo animation or splash delay

**Copy**:
- Headline: "What's on your mind?" (17pt, Deep Ocean)
- Helper: "tap the microphone to start" (13pt, Neutral, lowercase)

**Color**:
- Background: Pure white (light) / Deep charcoal (dark)
- Microphone button: Calm Blue (#3182CE)
- Text: Deep Ocean (#1A365D)

### 2.2 First Interaction Flow

**User Action**: Taps microphone

**System Response**:
1. Microphone button scales slightly (1.05x) and pulses
2. iOS microphone permission dialog appears

```
┌─────────────────────────────────────────┐
│                                         │
│  ┌─────────────────────────────────┐    │
│  │                                 │    │
│  │      "MYND" Would Like to       │    │
│  │      Access the Microphone      │    │
│  │                                 │    │
│  │  MYND uses your microphone      │    │
│  │  to capture your thoughts       │    │
│  │  through voice.                 │    │
│  │                                 │    │
│  │  ┌─────────┐  ┌─────────────┐   │    │
│  │  │  Don't  │  │    Allow    │   │    │
│  │  │  Allow  │  │             │   │    │
│  │  └─────────┘  └─────────────┘   │    │
│  │                                 │    │
│  └─────────────────────────────────┘    │
│                                         │
└─────────────────────────────────────────┘
```

**Permission Request Copy** (in Info.plist):
```
"MYND uses your microphone to capture your thoughts through voice."
```

**If permission granted**: Recording begins immediately
**If permission denied**: Show gentle recovery screen (see Section 5)

### 2.3 During Recording

```
┌─────────────────────────────────────────┐
│                                         │
│                                         │
│    "I keep forgetting to call my        │
│     mom back..."                        │
│                        ← live transcript│
│                                         │
│                                         │
│            ╭───────────────╮            │
│            │               │            │
│            │   [  stop  ]  │← pulsing   │
│            │               │            │
│            ╰───────────────╯            │
│                                         │
│            listening...                 │
│                                         │
│                                         │
└─────────────────────────────────────────┘
```

**Elements**:
- Live transcript appears as user speaks (Apple Speech)
- Microphone becomes stop button (pulsing gently)
- "listening..." status indicator
- Breathing wall animation begins subtly in background

**Animation**:
- Mic icon transitions to stop icon (0.3s)
- Gentle pulse on recording button (1.5s cycle)
- Transcript text fades in as it's recognized

### 2.4 Processing State (Axel Thinking)

**User Action**: Releases/taps stop

```
┌─────────────────────────────────────────┐
│                                         │
│    ╭─────────────────────────────╮      │
│    │  "I keep forgetting to      │      │
│    │   call my mom back"         │      │
│    ╰─────────────────────────────╯      │
│                                         │
│                                         │
│         ╭─────────────────╮             │
│         │                 │             │
│         │   (breathing    │  ← Axel    │
│         │    animation)   │    thinking │
│         │                 │             │
│         ╰─────────────────╯             │
│                                         │
│     "I hear you, let me think..."       │
│                                         │
│                                         │
└─────────────────────────────────────────┘
```

**Sequence**:
1. Transcript finalized and displayed in "bubble"
2. Quick acknowledgment plays (TTS or pre-recorded): "I hear you, let me think..."
3. Breathing wall animation appears
4. Claude API call initiated

**Timing**:
- Acknowledgment: 0-200ms after recording ends
- Breathing animation: immediate, continuous
- Claude response: 500-3000ms

### 2.5 Axel's First Response

```
┌─────────────────────────────────────────┐
│                                         │
│    ╭─────────────────────────────╮      │
│    │  "I keep forgetting to      │      │
│    │   call my mom back"         │      │
│    ╰─────────────────────────────╯      │
│                                         │
│         ╭─────────────────────────╮     │
│         │                         │     │
│         │  "That sounds like      │     │
│         │   it's been weighing    │     │
│         │   on you. Family        │ ← streaming
│         │   connections matter.   │     │
│         │   What usually gets     │     │
│         │   in the way?"          │     │
│         │                         │     │
│         ╰─────────────────────────╯     │
│                                         │
│            [  mic  ]   [  done  ]       │
│                                         │
└─────────────────────────────────────────┘
```

**Elements**:
- User's thought in top bubble
- Axel's response streams in (text appears progressively)
- TTS speaks response as text completes sentences
- Two actions: Continue conversation (mic) or Done

**Copy for First Response** (Demo mode example):
"That sounds like it's been weighing on you. Family connections matter. What usually gets in the way?"

**Timing**:
- Text streams over 2-5 seconds
- TTS begins speaking after first complete sentence
- Buttons appear after response completes

---

## 3. Demo Mode Experience

### 3.1 Demo Mode Overview

**What it is**: 10 free conversations with Axel, no account required

**How it works**:
- On-device transcription only (no API key)
- Pre-computed Claude responses for common patterns
- Full UX experience (breathing wall, acknowledgments)
- Thoughts saved locally only

**Why 10**: Research shows 5-7 interactions create habit; 10 gives buffer for exploration

### 3.2 Demo Response Patterns

The demo mode uses pattern matching to generate relevant responses without API calls.

| Pattern | Trigger Keywords | Demo Response |
|---------|------------------|---------------|
| Task/Reminder | "need to", "should", "forgot", "remember" | "That sounds like it's been on your mind. What's one small step you could take on this when you're ready?" |
| Emotional | "feel", "stressed", "anxious", "overwhelmed", "frustrated" | "It sounds like there's a lot going on for you. What feels most important to talk through right now?" |
| Creative/Ideas | "idea", "what if", "thinking about", "might", "wondering" | "That's an interesting direction. What draws you to this idea? What would it look like to explore it?" |
| Relationship | person names, "friend", "family", "partner", "boss", "mom", "dad" | "Relationships hold a lot of weight. What's the core of what you're noticing here?" |
| Work | "project", "deadline", "meeting", "work" | "Work stuff can take up a lot of mental space. What's the piece of this that's taking up the most energy?" |
| Default | (no pattern matched) | "I hear you. Tell me more about what's on your mind." |

### 3.3 Demo Conversation Counting

**Counter display**: Never shown explicitly (no anxiety)

**After each conversation** (conversations 1-9):
- Thought saved automatically
- No indication of remaining count
- Seamless experience

**After conversation 10**:

```
┌─────────────────────────────────────────┐
│                                         │
│                                         │
│        You've captured 10 thoughts      │
│              with Axel                  │
│                                         │
│        ─────────────────────────        │
│                                         │
│    To keep your thoughts and continue   │
│    talking with Axel, start your free   │
│    trial.                               │
│                                         │
│                                         │
│    ┌─────────────────────────────┐      │
│    │   Start free trial           │      │
│    │   7 days free, then $4.99/mo │      │
│    └─────────────────────────────┘      │
│                                         │
│         Not now - I'll lose my          │
│              thoughts                   │
│                                         │
└─────────────────────────────────────────┘
```

**Copy**:
- Headline: "You've captured 10 thoughts with Axel"
- Body: "To keep your thoughts and continue talking with Axel, start your free trial."
- Primary button: "Start free trial - 7 days free, then $4.99/mo"
- Secondary link: "Not now - I'll lose my thoughts"

**What happens if they decline**:
- Return to app (can still view thoughts)
- Cannot capture new thoughts
- Reminder after 24 hours: "Your thoughts are still here. Ready to continue?"

### 3.4 Demo Mode Technical Implementation

```swift
class DemoModeManager {
    private let maxDemoConversations = 10

    @AppStorage("demoConversationsUsed")
    private var conversationsUsed: Int = 0

    var canStartConversation: Bool {
        conversationsUsed < maxDemoConversations
    }

    func completeConversation() {
        conversationsUsed += 1

        if conversationsUsed >= maxDemoConversations {
            showUpgradePrompt()
        }
    }

    func getDemoResponse(for transcript: String) -> String {
        // Pattern matching logic
        let lowercased = transcript.lowercased()

        if containsTaskKeywords(lowercased) {
            return DemoResponses.task
        } else if containsEmotionalKeywords(lowercased) {
            return DemoResponses.emotional
        }
        // ... etc

        return DemoResponses.default
    }
}
```

---

## 4. Subscription Flow

### 4.1 Pricing Tiers

| Tier | Price | Audience | Features |
|------|-------|----------|----------|
| **Starter** | $4.99/mo | Light users | 500 messages/mo, basic features |
| **Pro** | $9.99/mo | Active users | Unlimited, graph viz, premium voice |
| **Unlimited** | $4.99/mo + own API | Power users | BYOK, all Pro features |
| **Lifetime** | $149 | Early adopters | Pro forever (first 1000) |

### 4.2 Free Trial Flow

**Trial terms**: 7 days free, then Starter ($4.99/mo)

**Subscription screen** (after demo ends or tapping upgrade):

```
┌─────────────────────────────────────────┐
│                     ×                   │
│                                         │
│       Continue your journey with        │
│               Axel                       │
│                                         │
│                                         │
│   ┌─────────────────────────────────┐   │
│   │  ○ Starter           $4.99/mo   │   │
│   │    500 conversations/month      │   │
│   │    Basic search                 │   │
│   └─────────────────────────────────┘   │
│                                         │
│   ┌─────────────────────────────────┐   │
│   │  ● Pro (Popular)     $9.99/mo   │   │
│   │    Unlimited conversations      │   │
│   │    Knowledge graph              │   │
│   │    Premium voice                │   │
│   │    Morning insights             │   │
│   └─────────────────────────────────┘   │
│                                         │
│   ┌─────────────────────────────────┐   │
│   │     Start 7-day free trial      │   │
│   └─────────────────────────────────┘   │
│                                         │
│        Restore purchases                │
│                                         │
└─────────────────────────────────────────┘
```

**Copy principles**:
- No pressure language ("limited time", "act now")
- Clear pricing (no hidden fees)
- Trial terms explicit
- Easy to dismiss

### 4.3 Payment Confirmation

```
┌─────────────────────────────────────────┐
│                                         │
│                                         │
│              ✓                          │
│                                         │
│        Welcome to MYND Pro              │
│                                         │
│   Your 7-day free trial has started.    │
│   You won't be charged until            │
│   January 11, 2026.                     │
│                                         │
│   You can cancel anytime in Settings.   │
│                                         │
│                                         │
│   ┌─────────────────────────────────┐   │
│   │        Start capturing          │   │
│   └─────────────────────────────────┘   │
│                                         │
│                                         │
└─────────────────────────────────────────┘
```

**Elements**:
- Clear confirmation of what they subscribed to
- Explicit trial end date
- How to cancel
- Single action: get started

### 4.4 BYOK Option (Settings Only)

BYOK is available but not prominently featured during onboarding.

**Location**: Settings > Account > Use your own API key

**Flow**:
1. User navigates to Settings > Account
2. Sees "Use your own API key" option
3. Taps to expand
4. Shows instructions and API key field

```
┌─────────────────────────────────────────┐
│ ← Settings                              │
│─────────────────────────────────────────│
│                                         │
│  USE YOUR OWN API KEY                   │
│                                         │
│  For unlimited usage at your own        │
│  cost. Requires Anthropic account.      │
│                                         │
│  ┌─────────────────────────────────┐    │
│  │ sk-ant-api03-...                │    │
│  └─────────────────────────────────┘    │
│                                         │
│  1. Go to console.anthropic.com         │
│  2. Create an API key                   │
│  3. Paste it above                      │
│                                         │
│  Your key is stored securely in         │
│  your device's keychain.                │
│                                         │
│  ┌─────────────────────────────────┐    │
│  │     Validate & save              │    │
│  └─────────────────────────────────┘    │
│                                         │
└─────────────────────────────────────────┘
```

---

## 5. Permission Requests

### 5.1 Permission Philosophy

**Principle**: Ask in context, explain why, accept "no" gracefully

| Permission | When to Ask | Why Needed |
|------------|-------------|------------|
| Microphone | When user taps record | Core functionality |
| Notifications | After 3rd successful thought | Optional value-add |
| Speech Recognition | With microphone (bundled) | Transcription |

### 5.2 Microphone Permission

**Timing**: When user first taps the microphone button

**Pre-prompt** (optional, can skip in v1): None - context is clear

**System dialog copy** (Info.plist):
```
"MYND uses your microphone to capture your thoughts through voice."
```

**If denied**:

```
┌─────────────────────────────────────────┐
│                                         │
│                                         │
│          Microphone access              │
│              needed                     │
│                                         │
│    MYND captures your thoughts          │
│    through voice. Without microphone    │
│    access, you won't be able to         │
│    talk to Axel.                        │
│                                         │
│                                         │
│   ┌─────────────────────────────────┐   │
│   │      Open settings              │   │
│   └─────────────────────────────────┘   │
│                                         │
│         Maybe later                     │
│                                         │
└─────────────────────────────────────────┘
```

**Copy**:
- Headline: "Microphone access needed"
- Body: "MYND captures your thoughts through voice. Without microphone access, you won't be able to talk to Axel."
- Primary: "Open settings" (deep links to Settings)
- Secondary: "Maybe later"

### 5.3 Notification Permission

**Timing**: After user's 3rd successful conversation

**Trigger**: Success screen after 3rd thought saved

```
┌─────────────────────────────────────────┐
│                                         │
│                                         │
│             Thought saved               │
│                                         │
│    ─────────────────────────────────    │
│                                         │
│     Want Axel to check in with you?     │
│                                         │
│    Occasional gentle reminders when     │
│    Axel has insights about your         │
│    thoughts.                            │
│                                         │
│                                         │
│   ┌─────────────────────────────────┐   │
│   │      Sure, enable                │   │
│   └─────────────────────────────────┘   │
│                                         │
│          Not right now                  │
│                                         │
└─────────────────────────────────────────┘
```

**Copy**:
- Lead-in: "Thought saved"
- Question: "Want Axel to check in with you?"
- Explanation: "Occasional gentle reminders when Axel has insights about your thoughts."
- Primary: "Sure, enable"
- Secondary: "Not right now"

**System dialog copy** (Info.plist):
```
"Axel sends occasional gentle reminders when there's something worth revisiting."
```

**If user taps "Sure, enable"**: Show system permission dialog
**If user taps "Not right now"**: Dismiss, ask again after 10th thought (max 2 asks total)

### 5.4 Permission Request Frequency

| Permission | Max Asks | Timing |
|------------|----------|--------|
| Microphone | 1 (required for core function) | First record tap |
| Notifications | 2 | After 3rd and 10th thought |
| Speech Recognition | 1 (with microphone) | First record tap |

---

## 6. Returning User Flows

### 6.1 Returning User (Same Day)

**State**: User has used app today, returns within hours

**Experience**:
- App opens to thought list (if they left mid-session)
- Or conversation view (if they were in a conversation)
- No re-onboarding

### 6.2 Returning User (New Day)

**State**: User hasn't opened app since yesterday

**Experience**:
- App opens to conversation view (fresh start)
- Axel may reference yesterday if relevant:
  "Hey. You mentioned [X] yesterday. How's that going?"
- No forced check-in or recap

### 6.3 Returning User (After Long Gap)

**State**: User hasn't opened app in 7+ days

**Experience**:
- App opens normally (no shame messaging)
- Axel greets warmly: "Hey. It's nice to hear from you again. What's on your mind?"
- No mention of gap unless user brings it up

**What NOT to do**:
- "We missed you!"
- "You've been away for 12 days"
- "Your streak was broken"
- Any guilt-inducing language

### 6.4 Returning User (Trial Expired)

**State**: User's trial ended, they didn't subscribe

**Experience**:

```
┌─────────────────────────────────────────┐
│                                         │
│                                         │
│         Your trial has ended            │
│                                         │
│    Your thoughts are still here.        │
│    Subscribe to continue talking        │
│    with Axel and capturing new          │
│    thoughts.                            │
│                                         │
│                                         │
│   ┌─────────────────────────────────┐   │
│   │        Subscribe                 │   │
│   └─────────────────────────────────┘   │
│                                         │
│         View my thoughts                │
│                                         │
└─────────────────────────────────────────┘
```

**Elements**:
- No pressure, just facts
- Thoughts preserved and viewable
- Option to view (read-only) without subscribing
- Clear path to subscribe

---

## 7. Edge Cases & Error States

### 7.1 Network Unavailable (First Launch)

```
┌─────────────────────────────────────────┐
│                                         │
│                                         │
│           You're offline                │
│                                         │
│    MYND needs an internet connection    │
│    to talk with Axel. But you can       │
│    still capture voice notes - we'll    │
│    process them when you're back        │
│    online.                              │
│                                         │
│                                         │
│            [  mic  ]                    │
│                                         │
│        "Capture for later"              │
│                                         │
└─────────────────────────────────────────┘
```

**Behavior**: Allow recording, save locally, process when online

### 7.2 Speech Recognition Fails

```
┌─────────────────────────────────────────┐
│                                         │
│                                         │
│       I didn't catch that               │
│                                         │
│    The audio wasn't clear enough        │
│    to understand. Would you like        │
│    to try again?                        │
│                                         │
│                                         │
│   ┌─────────────────────────────────┐   │
│   │        Try again                 │   │
│   └─────────────────────────────────┘   │
│                                         │
│         Type instead                    │
│                                         │
└─────────────────────────────────────────┘
```

### 7.3 Claude API Error

```
┌─────────────────────────────────────────┐
│                                         │
│    ╭─────────────────────────────╮      │
│    │  "I keep forgetting to      │      │
│    │   call my mom back"         │      │
│    ╰─────────────────────────────╯      │
│                                         │
│                                         │
│       Something went wrong on           │
│       Axel's end. Your thought          │
│       is saved.                         │
│                                         │
│                                         │
│   ┌─────────────────────────────────┐   │
│   │        Try again                 │   │
│   └─────────────────────────────────┘   │
│                                         │
│         Save without response           │
│                                         │
└─────────────────────────────────────────┘
```

**Key**: Thought is never lost due to API error

### 7.4 App Killed Mid-Recording

**Behavior**:
- Partial recording is saved locally
- On next launch: "We saved your last thought. Would you like to continue or start fresh?"

---

## 8. Onboarding Metrics

### 8.1 Funnel Metrics

| Step | Metric | Target |
|------|--------|--------|
| App Open → Tap Mic | First Action Rate | > 80% |
| Tap Mic → Grant Permission | Permission Grant | > 90% |
| Grant → Complete Recording | Recording Completion | > 95% |
| Recording → See Response | Response Success | > 98% |
| Response → 2nd Thought | Retention | > 60% |
| Demo → Trial | Conversion | > 40% |
| Trial → Paid | Subscription | > 50% |

### 8.2 Timing Metrics

| Metric | Target |
|--------|--------|
| Time to first tap | < 5 seconds |
| Time to first thought saved | < 30 seconds |
| Time to first Axel response | < 60 seconds |
| Time to subscription decision | < 5 minutes |

### 8.3 Analytics Events

```swift
// Track key onboarding events
Analytics.track(.onboardingStart)
Analytics.track(.microphoneTapped)
Analytics.track(.microphonePermissionGranted)
Analytics.track(.microphonePermissionDenied)
Analytics.track(.firstRecordingStarted)
Analytics.track(.firstRecordingCompleted, duration: seconds)
Analytics.track(.firstResponseReceived, latency: ms)
Analytics.track(.demoConversationCompleted, number: n)
Analytics.track(.upgradePromptShown)
Analytics.track(.subscriptionStarted, tier: tier)
Analytics.track(.subscriptionDeclined)
```

---

## 9. Screen Specifications

### 9.1 First Launch Screen

| Element | Specification |
|---------|---------------|
| Microphone button | 80pt diameter, Calm Blue fill, centered |
| Button icon | SF Symbol `mic.fill`, 32pt, white |
| Headline | "What's on your mind?", 22pt, Semibold, Deep Ocean |
| Helper text | "tap the microphone to start", 13pt, Regular, Neutral |
| Background | Pure white (light) / Deep charcoal (dark) |
| Safe area | Respect all safe areas |

### 9.2 Recording Screen

| Element | Specification |
|---------|---------------|
| Stop button | 80pt diameter, Warm Amber fill, pulsing (1.5s cycle) |
| Button icon | SF Symbol `stop.fill`, 32pt, white |
| Transcript | 17pt, Regular, Deep Ocean, max 3 lines visible |
| Status | "listening...", 13pt, Regular, Neutral |
| Breathing wall | Background, 30% opacity, 4s cycle |

### 9.3 Response Screen

| Element | Specification |
|---------|---------------|
| User bubble | Background Secondary, 16pt corner radius, right-aligned |
| Axel bubble | Background Tertiary, 16pt corner radius, left-aligned |
| Bubble text | 17pt, Regular, Deep Ocean |
| Action buttons | 48pt height, full width minus margins |
| Mic button | Calm Blue, "Keep talking" label |
| Done button | Secondary style, "Done" label |

### 9.4 Subscription Screen

| Element | Specification |
|---------|---------------|
| Close button | Top right, 44pt touch target, SF Symbol `xmark` |
| Headline | "Continue your journey with Axel", 22pt, Semibold |
| Tier cards | Full width minus 32pt margins, 16pt corner radius |
| Selected indicator | Radio button style, Calm Blue fill |
| Price | 17pt, Semibold, right-aligned |
| Features | 15pt, Regular, Neutral |
| CTA button | Primary style, 50pt height |
| Restore link | 15pt, Regular, Calm Blue, centered |

---

## 10. Implementation Checklist

### Pre-Launch
- [ ] First launch screen implemented
- [ ] Recording UI implemented with breathing wall
- [ ] Response UI with streaming text
- [ ] Demo mode pattern matching
- [ ] Demo conversation counter (invisible to user)
- [ ] Upgrade prompt at conversation 10
- [ ] Subscription screen with StoreKit 2
- [ ] All permission flows
- [ ] All error states
- [ ] Analytics events firing

### Testing
- [ ] Full flow: cold launch to first thought
- [ ] Demo mode limit reached
- [ ] Subscription purchase flow
- [ ] Permission denied recovery
- [ ] Network offline handling
- [ ] Returning user after trial expired
- [ ] VoiceOver through entire onboarding
- [ ] Dynamic Type at all sizes

### Metrics Validation
- [ ] Funnel tracking working
- [ ] Timing metrics accurate
- [ ] Conversion tracking to revenue

---

**Document Status**: APPROVED
**Review Cadence**: Before each release
**Owner**: Product Design + Engineering

---

*"The best onboarding feels like there was no onboarding."*
