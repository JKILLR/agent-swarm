# MYND - Refined Implementation Plan

**Version**: 1.0
**Date**: 2026-01-04
**Status**: DEFINITIVE IMPLEMENTATION GUIDE
**Synthesized From**: Architecture v2.0, Market Analysis, Critical Review, Creative Brainstorm

---

## 1. Executive Summary

### 1.1 Refined Vision

MYND is a **voice-first thought capture companion** for iOS that helps users with executive function challenges externalize and organize their thinking. The AI assistant "Axel" provides a **thoughtful, empathetic listening experience** rather than attempting to compete with real-time conversational AI.

**Core Insight from Synthesis**: The original vision tried to be everything - voice companion like Sesame, knowledge graph like Roam, ADHD tool like Tiimo, and AI friend like Pi. The refined vision focuses on **one core promise**: *Capture your scattered thoughts through voice, and Axel will help you make sense of them.*

### 1.2 Key Pivots from Original Plan

| Original Assumption | Pivot Decision | Rationale |
|---------------------|----------------|-----------|
| "Natural conversation" with <500ms latency | "Thoughtful companion" with intentional 1-3s pauses | iOS + Claude API makes sub-second latency impossible without $200M+ infrastructure |
| "Hey Axel" wake word activation | Push-to-talk + Siri Shortcuts proxy | Wake word is technically prohibited on iOS for third-party apps |
| SwiftData for knowledge graph at scale | SwiftData for storage + in-memory graph for queries | SwiftData fails at ~2000 nodes for graph operations |
| BYOK-only model | Tiered: Demo → Managed subscription → BYOK option | BYOK-only limits market to 2.5% of potential users |
| 20-week timeline | 32-week timeline with testing | Original had 0% testing allocation; realistic is 25%+ |
| Full knowledge graph in MVP | Basic thought capture in MVP, graph in v1.5 | Ship something excellent rather than many things broken |

### 1.3 Success Metrics

**Launch Criteria (v1.0)**:
- App Store approval achieved
- 100 beta testers complete 7-day usage with >60% retention
- NPS score of 40+ from beta cohort
- <5% crash rate
- Voice recognition accuracy >90% in quiet environments

**6-Month Targets**:
| Metric | Target | Measurement |
|--------|--------|-------------|
| Downloads | 10,000 | App Store Connect |
| DAU | 2,000 | Analytics |
| Free → Paid Conversion | 4% | Revenue tracking |
| Monthly Churn | <5% | Subscription data |
| App Store Rating | 4.5+ | Reviews |
| Voice Capture Success Rate | 85%+ | In-app analytics |

**12-Month Targets**:
| Metric | Target | Measurement |
|--------|--------|-------------|
| Downloads | 50,000 | App Store Connect |
| ARR | $75,000+ | Revenue |
| DAU | 10,000 | Analytics |
| Knowledge Graph Active Users | 5,000+ | Feature adoption |

---

## 2. Showstopper Resolutions

### 2.1 Voice Latency (SHOWSTOPPER #1)

**Original Problem**:
Claude API cold start = 500-3000ms. Full round-trip = 1-20 seconds. Natural conversation requires <500ms. MYND's architecture guarantees 10x worse latency than competitors like Sesame ($200M+ investment in custom speech models).

**Chosen Solution**: The "Thoughtful Companion" Reframe

Instead of competing on speed, reframe Axel as a *wise friend who pauses before responding*. Combine immediate acknowledgments with streaming responses.

**Implementation Approach**:

```
User speaks → (50-200ms)
   ↓
On-device transcription complete → (immediate)
   ↓
Pre-generated acknowledgment plays: "I hear you, let me think about that..." → (instant)
   ↓
Breathing animation displays → (calming visual during wait)
   ↓
Claude response begins streaming → (500-3000ms after user finishes)
   ↓
TTS speaks sentences as they complete → (progressive)
```

**Technical Components**:
1. **Immediate Acknowledgments**: Library of 10-15 pre-recorded phrases in Axel's voice
2. **Breathing Animation**: 4-second breath cycle visual during processing
3. **Streaming TTS**: Speak complete sentences as they stream from Claude
4. **Optimistic UI**: Show transcript immediately, thinking state clearly indicated

**Tradeoffs Accepted**:
- Not true "conversation" - more like thoughtful correspondence
- Requires user expectation management in onboarding
- Some users expecting Sesame-like experience will be disappointed
- Marketing must avoid "conversational AI" claims

**Why This Works**:
- Therapy/coaching sessions have pauses - they feel *safe*, not slow
- ADHD users often appreciate space to continue their thought
- "Axel thinks before speaking" is a feature, not a bug

---

### 2.2 Wake Word Impossible (SHOWSTOPPER #2)

**Original Problem**:
The feature brainstorm listed "Hey Axel" wake word as MVP+. Reality: Apple does NOT provide VoiceTrigger API to third-party developers. Continuous audio monitoring requires background audio mode (battery killer) and iOS will terminate apps + App Store will reject.

**Chosen Solution**: Multi-Modal Quick Access

Eliminate wake word entirely. Provide multiple low-friction alternatives.

**Implementation Approach**:

| Access Method | Taps | Availability | Implementation |
|---------------|------|--------------|----------------|
| Lock Screen Widget | 1 | iOS 16+ | WidgetKit |
| Siri Shortcut | 0 (voice) | iOS 15+ | "Hey Siri, capture thought in MYND" |
| Home Screen Widget | 1 | iOS 14+ | WidgetKit |
| Action Button (iPhone 15 Pro+) | 1 | iOS 17+ | Settings integration |
| App Complication (Apple Watch) | 1 | watchOS 9+ | WatchKit (v2.0) |
| Control Center Toggle | 1 | iOS 18+ | ControlWidget (v1.5) |

**Primary Path (MVP)**:
1. Lock Screen widget with microphone icon
2. Siri Shortcut: "Hey Siri, tell MYND [thought]"
3. App icon deep link

**Tradeoffs Accepted**:
- Users cannot activate hands-free while phone is in pocket
- Requires one tap minimum in most scenarios
- Siri proxy adds extra step ("Hey Siri, MYND" vs "Hey Axel")

**Why This Works**:
- Lock Screen widget achieves 1-tap access (same as wake word would)
- Siri provides true hands-free for users who want it
- Apple Watch (v2.0) enables capture during walks/drives

---

### 2.3 SwiftData Graph Scaling (SHOWSTOPPER #3)

**Original Problem**:
SwiftData/Core Data cannot handle graph operations efficiently. O(N × edges²) for depth-2 traversals. At 500 nodes: noticeable lag. At 2000 nodes: unusable. At 10000 nodes: memory crashes.

**Chosen Solution**: Hybrid Architecture with Migration Path

SwiftData for persistence and sync. In-memory graph rebuilt on launch for fast queries.

**Implementation Approach**:

```
┌─────────────────────────────────────────────────────────────────┐
│                        QUERY ROUTER                              │
└─────────────────────────────────────────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         ▼                    ▼                    ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│   SwiftData     │ │   In-Memory     │ │   SQLite+FTS5   │
│   (CRUD, Sync)  │ │   (Graph Ops)   │ │   (Full-Text)   │
└─────────────────┘ └─────────────────┘ └─────────────────┘
         │                    │                    │
         │        MVP         │      MVP (v1.0)    │    v1.5+
         │        v1.0        │                    │
```

**Phase 1 (MVP v1.0)**:
- SwiftData: Source of truth, CloudKit sync, basic CRUD
- In-Memory Graph: Adjacency lists, O(1) neighbor lookup, BFS/DFS, rebuilt on launch
- No full-text search beyond SwiftData predicates

**Phase 2 (v1.5)**:
- SQLite + FTS5 for full-text search
- Graph cache persisted to disk (not rebuilt from scratch)
- Lazy loading for large graphs

**Phase 3 (v2.0+)**:
- Consider mynd-brain integration for advanced ML
- Potential move to SQLite as primary store if needed

**GraphStore Protocol Abstraction**:
```swift
protocol GraphStoreProtocol {
    func neighbors(of nodeId: UUID) async -> [Node]
    func findPath(from: UUID, to: UUID) async -> [Node]?
    func search(query: String, limit: Int) async -> [Node]
    func clusters() async -> [[Node]]
}
```

**Tradeoffs Accepted**:
- Cold launch requires graph rebuild (500-1000ms for 1000 nodes)
- Memory usage higher than pure SwiftData
- Two data structures to maintain in sync
- Graph visualization limited to ~1000 nodes in MVP

**Why This Works**:
- Provides O(1) neighbor lookups (vs O(N) with pure SwiftData)
- SwiftData handles sync complexity with CloudKit
- Migration path clear when limits hit

---

### 2.4 Onboarding Kills Conversion (SHOWSTOPPER #4)

**Original Problem**:
BYOK flow requires 9 steps: create Anthropic account, add payment, generate API key, copy 51-character key, paste correctly. Industry benchmark: 77% abandon after 1 minute; 9-step flow has <5% completion rate. Target audience (ADHD) especially vulnerable to multi-step friction.

**Chosen Solution**: Tiered Onboarding with Demo-First Experience

Let users experience value BEFORE any setup. Multiple tiers for different user types.

**Implementation Approach**:

**First 30 Seconds Flow**:
```
1. App opens → No onboarding screens, no permissions yet
2. Minimal UI: Microphone icon + "Tell me what's on your mind"
3. User taps, speaks freely (on-device transcription, no API key)
4. Demo response: "I hear you. [Summary]. Want me to remember this?"
5. User says yes → Thought appears in timeline
6. THEN: "To keep talking to me, let's set you up..."
```

**Subscription Tiers**:

| Tier | Price | Target User | Experience |
|------|-------|-------------|------------|
| **Demo** | Free | Everyone | 10 total conversations, on-device only, no account |
| **Starter** | $4.99/mo | Light users | 500 messages/mo, managed Claude API, basic features |
| **Pro** | $9.99/mo | Active users | Unlimited, graph viz, insights, premium voice |
| **Unlimited** | $4.99/mo + own API | Power users | BYOK, all Pro features, no message limits |
| **Lifetime** | $149 one-time | Early adopters | Pro features forever (first 1000 only) |

**Demo Mode Technical Approach**:
- On-device transcription only (Apple Speech)
- Pre-computed responses for common patterns
- No API calls (developer cost = $0)
- After 10 conversations: "You've used your demo. Continue with a free trial?"

**Tradeoffs Accepted**:
- Developer subsidizes infrastructure for managed tiers
- BYOK power users are smaller percentage of revenue
- Demo mode has limited AI capability
- Pricing requires ongoing API cost management

**Why This Works**:
- Users experience core value in <30 seconds
- One-tap subscription for 95% of users
- BYOK option preserved for power users who want control
- Demo creates habit before commitment

---

## 3. Revised Architecture

### 3.1 What Changes from v2.0

| Component | v2.0 Architecture | Revised | Rationale |
|-----------|-------------------|---------|-----------|
| Knowledge Graph MVP | Full graph + visualization | Basic thought list | Ship working MVP first |
| Voice Quality | AVSpeech only | AVSpeech + demo acknowledgments | Immediate feedback critical |
| Onboarding | Settings page API key entry | Demo mode + IAP flow | Conversion is critical |
| Proactive Features | Background processing | Scheduled notifications only | iOS background limits |
| Graph Storage | SwiftData relationships | SwiftData + in-memory | Scalability (per showstopper) |
| CloudKit Sync | Automatic conflict resolution | Explicit merge strategies | Data integrity (per critique) |

### 3.2 What Stays the Same

| Component | Rationale for Keeping |
|-----------|----------------------|
| Push-to-talk voice model | Correct from v2.0, addresses wake word issue |
| MVVM + SwiftUI architecture | Modern, testable, appropriate |
| Repository pattern | Enables storage migration |
| Protocol-based DI | Critical for testing |
| Apple Speech for transcription | On-device, free, good enough |
| Claude API for AI | Best quality, BYOK-compatible |
| Streaming responses | Latency mitigation, essential |

### 3.3 Module Structure (Revised for Phases)

```
MYND/
├── App/                          # MVP v1.0
├── Features/
│   ├── Conversation/             # MVP v1.0 - Core feature
│   ├── ThoughtList/              # MVP v1.0 - Replaces full graph
│   ├── KnowledgeGraph/           # v1.5 - Deferred
│   ├── Proactive/                # v1.5 - Deferred (partial in v1.0)
│   ├── Settings/                 # MVP v1.0
│   └── Onboarding/               # MVP v1.0 - NEW (demo + IAP)
├── Core/
│   ├── Voice/                    # MVP v1.0
│   ├── AI/                       # MVP v1.0
│   ├── Memory/                   # MVP v1.0 (basic) → v1.5 (full)
│   └── Graph/                    # v1.5 - Deferred
├── Data/                         # MVP v1.0
├── Sync/                         # MVP v1.0 (basic) → v1.5 (robust)
├── Security/                     # MVP v1.0
├── Widgets/                      # MVP v1.0 (Lock Screen only)
└── Tests/                        # ALL PHASES - 25% of effort
```

### 3.4 CloudKit Sync Strategy (Revised)

**Problem Identified in Critique**: CloudKit "automatic conflict resolution" is naive for graph data. Last-write-wins causes edge loss. Partial sync states cause crashes.

**Revised Strategy**:

1. **Conflict Resolution Protocol**:
```swift
protocol ConflictResolvable {
    var lastModifiedAt: Date { get }
    var deviceId: String { get }
    func merge(with other: Self) -> Self
}
```

2. **Graph-Aware Merge**:
- Nodes: Merge metadata, keep both if content differs significantly
- Edges: Union of edges from both devices (never delete implicitly)
- Sessions: Ordered by timestamp, interleave if overlapping

3. **Sync State Machine**:
```
SYNCED → MODIFIED → SYNCING → SYNCED
                       ↓
                   CONFLICT → RESOLVING → SYNCED
                       ↓
                   FAILED → RETRY → SYNCING
```

4. **Offline Queue**:
- All writes go to local store immediately
- Background queue syncs when connected
- UI shows sync status badge

### 3.5 Migration Path for Scaling

**Phase 1 → Phase 2 Migration** (SwiftData to SQLite+FTS5):
1. Export SwiftData to SQLite format
2. Run both in parallel for 1 release cycle
3. Validate data integrity
4. Remove SwiftData dependency

**Phase 2 → Phase 3 Migration** (Optional mynd-brain integration):
1. Define API contract (GraphQL or REST)
2. Sync local graph to cloud when online
3. Pull enhanced embeddings/insights from mynd-brain
4. Local remains source of truth

---

## 4. Selected Innovation Features

From the brainstorm document, these 5 innovations provide the best ratio of implementation complexity to user value:

### 4.1 The Breathing Wall (Processing Indicator)

**Why Selected**: Directly addresses the latency showstopper. Transforms waiting into calming experience.

**Implementation Complexity**: Low (2-3 days)
- SwiftUI gradient animation
- 4-second breath cycle timing
- Haptic feedback sync

**User Value**: High
- Reduces perceived wait time by 40%+ (UX research)
- Aligns with "thoughtful companion" positioning
- Differentiates from spinner-based apps

**Technical Spec**:
```swift
struct BreathingWall: View {
    @State private var isInhaling = true

    var body: some View {
        Circle()
            .fill(
                RadialGradient(
                    colors: [.blue.opacity(0.3), .clear],
                    center: .center,
                    startRadius: isInhaling ? 50 : 100,
                    endRadius: isInhaling ? 150 : 200
                )
            )
            .animation(.easeInOut(duration: 4).repeatForever(autoreverses: true), value: isInhaling)
            .onAppear { isInhaling.toggle() }
    }
}
```

### 4.2 Morning Oracle (Pre-Computed Daily Briefing)

**Why Selected**: Delivers proactive value without violating iOS background limits.

**Implementation Complexity**: Medium (1 week)
- Scheduled notification with pre-computed content
- Generate during overnight charging window
- Cache locally for instant display

**User Value**: Very High
- Core differentiator from reactive AI apps
- ADHD users especially value morning structure
- Creates daily habit anchor

**Technical Approach**:
1. Background App Refresh during charging (night)
2. If connected + charging: Call Claude API to generate insight
3. Store locally in SwiftData
4. Schedule Local Notification for user's wake time
5. Notification opens directly to briefing view

**Tradeoff**: Not real-time; based on yesterday's data. Acceptable for morning use case.

### 4.3 Energy-Adaptive Suggestions

**Why Selected**: Directly serves ADHD audience; simple implementation with high impact.

**Implementation Complexity**: Low (3-4 days)
- 3-option energy picker UI
- Context changes system prompt
- Suggestion filtering based on energy level

**User Value**: High
- Meets users where they are
- Prevents overwhelm (low energy → tiny tasks)
- Builds trust (app "gets" them)

**UX Flow**:
```
[How's your energy right now?]
   [Low 🔋]  [Medium ⚡]  [High 🚀]
        ↓
Low: "Here's one tiny thing you could do..."
Medium: "Want to make some progress on [goal]?"
High: "Let's tackle something substantial!"
```

### 4.4 Invisible Progress (Surprise Milestone Reveals)

**Why Selected**: Gamification without shame. Perfect for ADHD audience psychology.

**Implementation Complexity**: Low (2-3 days)
- Track metrics silently
- Trigger celebration at milestones
- Never show gaps or missed days

**User Value**: High
- Dopamine hit without pressure
- No streak anxiety
- Delight moments build emotional connection

**Milestones**:
- 10, 25, 50, 100, 250, 500 thoughts captured
- 1 week, 1 month, 3 months of usage
- First goal completed, 5 goals, 10 goals
- First connection discovered (graph feature)

**Celebration**: Subtle confetti + mindfulness bell sound + "You've captured 50 thoughts with Axel!"

### 4.5 The Safe Space Design Language

**Why Selected**: Foundational to all other features. Differentiates from productivity tools.

**Implementation Complexity**: Medium (integrated throughout)
- Color palette: Deep calm blue, warm amber, soft purple
- Typography: Rounded, friendly sans-serif
- Motion: Slow, purposeful, never startling
- Sound: Mindfulness bell confirmations
- Language: Never judging, always encouraging

**User Value**: Very High
- ADHD users have been burned by shame-inducing apps
- Safe space builds trust and habit
- Emotional design creates loyalty

**Design Principles**:
1. No red indicators ever (even for "overdue")
2. No exclamation marks or urgent styling
3. "When you're ready" vs "overdue"
4. Celebrate completion, ignore gaps
5. Maximum 3 choices per screen
6. Always one clear next action

---

## 5. Revised Timeline

### 5.1 Phase Overview

| Phase | Duration | Focus | Deliverable |
|-------|----------|-------|-------------|
| **Pre-Development** | 2 weeks | Design, setup, validation | Design system, dev environment, user interviews |
| **Phase 1: Core MVP** | 8 weeks | Voice capture + basic AI | Working app with demo mode |
| **Phase 2: MVP Polish** | 4 weeks | Testing, onboarding, IAP | App Store ready v1.0 |
| **Phase 3: Beta** | 4 weeks | Real user testing | Beta with 100 users |
| **Phase 4: Launch** | 2 weeks | Bug fixes, marketing | Public v1.0 |
| **Phase 5: Graph Features** | 6 weeks | Knowledge graph v1.5 | Graph visualization |
| **Phase 6: Proactive** | 6 weeks | Morning oracle, insights | Proactive features v1.5 |
| **Total** | **32 weeks** | | |

### 5.2 Pre-Development (Weeks 1-2)

**Goals**:
- Validate "thoughtful companion" hypothesis with 10 user interviews
- Create design system (colors, typography, components)
- Set up development environment (Xcode, CI/CD, analytics)
- Write Axel personality guidelines
- Define accessibility requirements

**Deliverables**:
- [ ] User interview summary (10 interviews)
- [ ] Figma design system
- [ ] Xcode project with CI (GitHub Actions)
- [ ] Axel personality document (3-5 pages)
- [ ] Accessibility checklist

### 5.3 Phase 1: Core MVP (Weeks 3-10)

**Week 3-4: Voice Engine**
- Apple Speech transcription
- AVSpeech synthesis
- Audio session management
- Breathing wall animation
- Unit tests for voice engine

**Week 5-6: AI Integration**
- Claude API client with streaming
- Context builder (basic)
- Demo mode (no API, canned responses)
- Quick acknowledgment system
- Unit tests for AI client

**Week 7-8: Thought Capture**
- ThoughtNode SwiftData model
- Basic list view (not graph)
- Create/read/delete thoughts
- Simple search
- CloudKit sync (basic)

**Week 9-10: Core UI**
- Conversation view with streaming
- Thought list view
- Settings skeleton
- Lock Screen widget
- Integration tests

### 5.4 Phase 2: MVP Polish (Weeks 11-14)

**Week 11-12: Onboarding & Monetization**
- Demo mode flow (10 conversations)
- StoreKit 2 integration
- Subscription tiers (Starter, Pro)
- BYOK settings (for Unlimited tier)
- Payment flow testing

**Week 13-14: Testing & Polish**
- UI testing (all critical flows)
- Accessibility audit + fixes
- Performance profiling
- Crash analytics integration
- App Store assets preparation

### 5.5 Phase 3: Beta (Weeks 15-18)

**Week 15-16: Private Beta**
- TestFlight distribution
- 50 beta testers (ADHD community, productivity enthusiasts)
- Daily feedback collection
- Bug fixes and iterations

**Week 17-18: Expanded Beta**
- Expand to 100 testers
- A/B test onboarding variations
- Pricing validation surveys
- Final bug fixes

### 5.6 Phase 4: Launch (Weeks 19-20)

**Week 19: App Store Submission**
- Final QA pass
- App Store review submission
- Marketing materials finalized
- Press/influencer outreach

**Week 20: Public Launch**
- ProductHunt launch
- Social media campaign
- Monitor crash reports and reviews
- Rapid hotfix capability

### 5.7 Phase 5: Knowledge Graph (Weeks 21-26)

**Deferred Features Now Implemented**:
- In-memory graph with SwiftData backing
- Graph visualization (force-directed, max 500 nodes)
- Entity extraction (NaturalLanguage)
- Relationship detection (basic)
- Graph search and traversal
- v1.5 release

### 5.8 Phase 6: Proactive Features (Weeks 27-32)

**Deferred Features Now Implemented**:
- Morning Oracle briefings
- Goal tracking and stale goal detection
- Energy-adaptive suggestions
- Weekly insight summaries
- Notification preferences
- v1.5 feature complete

---

## 6. Risk Mitigation

### 6.1 Top 5 Risks After Mitigations

| Risk | Probability | Impact | Mitigation | Contingency |
|------|-------------|--------|------------|-------------|
| **Apple Intelligence competes directly** | 70% | Critical | Differentiate on BYOK, cross-platform future, export, customization | Pivot to Obsidian/Notion companion tool |
| **Voice experience feels laggy despite reframe** | 40% | High | User testing in beta, iterate on acknowledgments | Add text-first mode as equal option |
| **Subscription conversion <2%** | 35% | High | Demo-first experience, pricing experiments in beta | Increase demo limit, lower Starter price |
| **CloudKit sync causes data loss** | 25% | Critical | Comprehensive conflict resolution, local backup | Manual export feature, support escalation path |
| **Claude API costs exceed revenue** | 30% | Medium | BYOK tier, usage limits on Starter, prompt optimization | Renegotiate Anthropic pricing, add OpenAI fallback |

### 6.2 Apple Intelligence Defense Strategy

**When Apple Announces Competing Features** (likely WWDC 2026):

1. **Immediate Actions**:
   - Messaging pivot: "MYND works with YOUR AI" (BYOK)
   - Emphasize: Data portability, no lock-in
   - Highlight: Cross-platform future (Android roadmap)

2. **Differentiation Moat**:
   - BYOK transparency (Apple won't offer)
   - Export to Markdown/JSON (Apple locks data)
   - Community templates (Apple won't do social)
   - Axel personality customization (Apple standardizes)

3. **Worst Case Pivot**:
   - MYND becomes "thought export layer"
   - Integrate with Obsidian, Notion, Roam
   - Sell to knowledge management company

### 6.3 Technical Risk Mitigations

| Technical Risk | Mitigation |
|----------------|------------|
| SwiftData bugs in iOS 17+ | Test on multiple iOS versions; fallback queries |
| CloudKit rate limits | Batch sync, exponential backoff, offline-first |
| Claude API outages | Local cache, graceful degradation, user notification |
| Speech recognition failures | Retry logic, text input fallback, quality indicators |
| Memory pressure on older devices | Lazy loading, graph size limits, test on iPhone 11 |

### 6.4 Market Risk Mitigations

| Market Risk | Mitigation |
|-------------|------------|
| ADHD community skepticism | Partner with community members, transparent development |
| App Store rejection (health claims) | Avoid "treatment" language, position as "productivity" |
| Negative reviews from latency | Clear onboarding expectations, "thoughtful companion" framing |
| Subscription fatigue | Lifetime option, BYOK option, clear value demonstration |

---

## 7. MVP Definition

### 7.1 v1.0 Feature Set (Exact Scope)

**Included**:

| Feature | Description | Priority |
|---------|-------------|----------|
| **Push-to-talk voice capture** | Tap microphone, speak, release | P0 |
| **On-device transcription** | Apple Speech, real-time transcript display | P0 |
| **Claude AI responses** | Streaming text, TTS playback | P0 |
| **Demo mode** | 10 free conversations, no account needed | P0 |
| **Managed subscription** | Starter ($4.99), Pro ($9.99) via StoreKit 2 | P0 |
| **Breathing wall** | Calming animation during processing | P0 |
| **Quick acknowledgments** | Pre-recorded "I hear you..." phrases | P0 |
| **Thought list** | Chronological list of captured thoughts | P0 |
| **Basic search** | Text search in thought content | P0 |
| **Lock Screen widget** | 1-tap voice capture | P0 |
| **Settings** | Voice speed, API key (BYOK), account | P0 |
| **CloudKit sync** | Sync thoughts across devices | P1 |
| **Axel personality** | Consistent, warm, non-judgmental tone | P1 |
| **Accessibility** | VoiceOver, Dynamic Type, high contrast | P1 |

**Architecture in v1.0**:
- SwiftData for storage
- CloudKit for sync (basic conflict resolution)
- No in-memory graph (thoughts are flat list)
- No entity extraction or relationship detection

### 7.2 Explicitly OUT of Scope for v1.0

| Feature | Deferred To | Rationale |
|---------|-------------|-----------|
| Knowledge graph visualization | v1.5 | Too complex for MVP; prove core value first |
| Entity extraction | v1.5 | Requires graph infrastructure |
| Relationship detection | v1.5 | Requires graph infrastructure |
| Morning Oracle | v1.5 | Proactive features need solid foundation |
| Goal tracking | v1.5 | Requires more sophisticated data model |
| Weekly insights | v1.5 | Proactive features deferred |
| Energy-adaptive suggestions | v1.5 | Nice-to-have, not core |
| Apple Watch app | v2.0 | Platform expansion after iOS proven |
| ElevenLabs voice | v1.5 | Premium feature for Pro tier |
| Home Screen widget | v1.5 | Lock Screen sufficient for MVP |
| Siri Shortcuts | v1.5 | Complex integration, not core |
| Family/team features | v2.0+ | Major scope increase |
| Android | v3.0+ | Platform expansion |

### 7.3 v1.0 Launch Criteria (Must Pass)

**Technical**:
- [ ] App runs without crash for 24 hours of active use
- [ ] Voice capture success rate >90% in quiet environment
- [ ] Streaming response begins <3 seconds after user finishes speaking
- [ ] CloudKit sync works reliably (test with 2 devices)
- [ ] StoreKit purchases complete successfully
- [ ] VoiceOver navigation works for all screens

**User Experience**:
- [ ] First-time user can capture thought in <30 seconds
- [ ] Demo → subscription flow <3 taps
- [ ] Breathing wall feels calming (user testing validation)
- [ ] Axel responses feel warm and helpful (user testing validation)

**Quality**:
- [ ] 0 P0 bugs, <5 P1 bugs
- [ ] App size <100MB
- [ ] Cold launch <3 seconds on iPhone 12
- [ ] Memory usage <200MB in normal use

**Business**:
- [ ] App Store submission approved
- [ ] Analytics tracking verified (DAU, retention, conversion)
- [ ] Customer support process documented

### 7.4 v1.5 Scope (6 weeks after v1.0)

| Feature | Week |
|---------|------|
| In-memory graph implementation | Week 1-2 |
| Entity extraction | Week 2 |
| Basic graph visualization | Week 3-4 |
| Morning Oracle | Week 4-5 |
| Energy-adaptive suggestions | Week 5 |
| ElevenLabs integration | Week 6 |
| Home Screen widget | Week 6 |

---

## 8. Appendix: Key Reference Documents

| Document | Purpose | Location |
|----------|---------|----------|
| MYND_ARCHITECTURE.md | Technical architecture v2.0 | workspace/ |
| MYND_CRITIQUE.md | Critical review with showstoppers | workspace/ |
| MYND_BRAINSTORM.md | Creative feature ideas | workspace/ |
| market_analysis.md | Competitive landscape | workspace/research/ |
| This document | Definitive implementation plan | workspace/ |

---

## 9. Decision Log

| Decision | Date | Rationale | Owner |
|----------|------|-----------|-------|
| Reframe "conversational" to "thoughtful companion" | 2026-01-04 | Latency showstopper | Architecture |
| Remove wake word from all plans | 2026-01-04 | iOS platform prohibition | Architecture |
| Tiered subscription with demo mode | 2026-01-04 | Onboarding conversion | Product |
| Defer knowledge graph to v1.5 | 2026-01-04 | Reduce MVP scope | Product |
| 32-week timeline (vs 20 original) | 2026-01-04 | Include testing, realistic estimates | Project |
| Hybrid SwiftData + in-memory graph | 2026-01-04 | Scalability showstopper | Architecture |

---

*Document Status: APPROVED FOR IMPLEMENTATION*
*Next Step: Begin Pre-Development phase*
*Review Cadence: Weekly during development*
