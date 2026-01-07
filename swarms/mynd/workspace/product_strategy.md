# MYND Product Strategy & Roadmap

**Version**: 1.0
**Date**: 2026-01-04
**Author**: Product Strategy Agent
**Status**: Strategic Blueprint

---

## Executive Summary

MYND is a voice-first AI thought capture app for iOS targeting users with executive function challenges. The app positions itself as a "thoughtful companion" (not a conversational AI) that helps users externalize scattered thoughts through natural voice interaction, with an AI assistant named "Axel" providing empathetic, memory-backed responses.

**Core Insight**: Users with ADHD/executive dysfunction don't need another productivity tool - they need an external brain that captures thoughts in the moment and follows up so nothing is lost.

**Strategic Positioning**: The intersection of three underserved markets - voice-first AI, knowledge graphs, and ADHD-specific design - where no competitor currently operates effectively.

---

## Part 1: MVP Definition

### 1.1 The Absolute Minimum Feature Set

The MVP must answer ONE question: **"Can MYND capture my scattered thoughts through voice and help me make sense of them?"**

#### P0 - Must Ship (No Launch Without These)

| Feature | Rationale | User Value |
|---------|-----------|------------|
| **Push-to-talk voice capture** | Core input mechanism; tap → speak → release | Frictionless thought externalization |
| **On-device transcription** | Apple Speech Framework; instant visual feedback | User sees their words immediately |
| **Claude AI responses (streaming)** | The "thinking" that makes thoughts useful | Axel responds, reflects, asks questions |
| **Breathing Wall animation** | Addresses latency perception (critical) | 1-3s wait feels calming, not slow |
| **Quick acknowledgments** | "I hear you, let me think..." plays instantly | Immediate feedback while Claude processes |
| **Thought list view** | Simple chronological display of captured thoughts | Users can review what they've captured |
| **Basic search** | Text search across thought content | Find past thoughts |
| **Demo mode (10 conversations)** | Critical for onboarding; no API key needed | Zero-friction first experience |
| **Managed subscription** | Starter ($4.99) / Pro ($9.99) via StoreKit 2 | Revenue model; solves BYOK friction |
| **Settings** | API key entry (BYOK), voice preferences | Control and customization |

#### P1 - Important for Launch Quality

| Feature | Rationale | User Value |
|---------|-----------|------------|
| **Lock Screen widget** | 1-tap access; the "wake word" alternative | Minimal friction to capture |
| **CloudKit sync (basic)** | Multi-device support | Thoughts available on all devices |
| **Axel personality** | Consistent warm, non-judgmental tone | Emotional connection, trust |
| **Basic accessibility** | VoiceOver, Dynamic Type support | Required for target audience |

#### What Makes This MVP Viable

1. **Solves Core Problem**: User can speak a scattered thought → Axel acknowledges → Axel responds thoughtfully → thought is saved
2. **Complete Loop**: Capture → AI Response → Storage → Retrieval
3. **Monetizable**: Demo creates habit, subscription creates revenue
4. **Testable**: Clear success metrics (voice recognition accuracy, time-to-first-value, retention)

### 1.2 What Explicitly Waits for v1.5+

| Feature | Version | Reason for Deferral |
|---------|---------|---------------------|
| Knowledge graph visualization | v1.5 | Requires SwiftData + in-memory graph; ship voice capture first |
| Entity extraction | v1.5 | Adds complexity; flat list works for MVP |
| Relationship detection | v1.5 | Depends on entity extraction |
| Morning Oracle | v1.5 | Proactive features need solid foundation |
| Goal tracking | v1.5 | Requires more sophisticated data model |
| Weekly insights | v1.5 | Background processing complexity |
| Energy-adaptive suggestions | v1.5 | Nice UX enhancement, not core |
| ElevenLabs premium voice | v1.5 | AVSpeech sufficient for MVP |
| Home Screen widget | v1.5 | Lock Screen widget sufficient |
| Siri Shortcuts integration | v1.5 | Complex integration, not core |
| Apple Watch app | v2.0 | Platform expansion after iOS proven |
| Family/team features | v2.0+ | Major scope increase |
| Android | v3.0+ | Platform expansion |

### 1.3 Core Differentiator at Launch

**The "Thoughtful Companion" Experience**

Unlike competitors who promise "conversational AI" (and fail to deliver acceptable latency), MYND launches with honest positioning:

- **Not**: "Talk naturally with Axel like a friend"
- **Yes**: "Axel listens thoughtfully, pauses to think, and responds with care"

This reframe turns a technical limitation (1-3 second latency from Claude API) into a feature:
- Therapists pause before responding - it feels safe
- ADHD users benefit from space to continue their thought
- The Breathing Wall makes waiting feel meditative, not frustrating

**Why This Works as Differentiator**:
1. Sesame AI ($200M+ investment) has better voice quality but no knowledge capture
2. Otter.ai transcribes but doesn't converse
3. Notion AI is powerful but not voice-first
4. No one combines voice + AI response + persistent memory + ADHD design

---

## Part 2: User Journey Mapping

### 2.1 First-Time User Experience (FTUX)

**Goal**: Capture value in under 60 seconds, subscription consideration in under 5 minutes.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    FIRST 30 SECONDS (Critical Window)               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  [App opens]                                                        │
│       │                                                             │
│       ▼                                                             │
│  [Single screen: Microphone icon + "Tell me what's on your mind"]  │
│  (No onboarding screens, no permission requests yet)                │
│       │                                                             │
│       ▼                                                             │
│  [User taps microphone - FIRST INTERACTION]                         │
│       │                                                             │
│       ▼                                                             │
│  [Microphone permission granted - feels natural]                    │
│       │                                                             │
│       ▼                                                             │
│  [User speaks: "I keep forgetting to call my mom back"]             │
│       │                                                             │
│       ▼                                                             │
│  [Transcript appears in real-time - user sees they're heard]        │
│       │                                                             │
│       ▼                                                             │
│  [Quick acknowledgment plays: "I hear you, let me think..."]        │
│  [Breathing Wall animation starts - calming visual]                 │
│       │                                                             │
│       ▼                                                             │
│  [Axel responds (streaming): "That sounds like it's been            │
│   weighing on you. When was the last time you talked to her?"]      │
│       │                                                             │
│       ▼                                                             │
│  [Thought is auto-saved]                                            │
│                                                                     │
│  ───────────────────────────────────────────────────────────────    │
│  VALUE DELIVERED: User spoke a thought, felt heard, got response.   │
│  TIME ELAPSED: ~30 seconds                                          │
│  CONVERSION HOOK: User wants to continue this experience            │
└─────────────────────────────────────────────────────────────────────┘
```

**First 5 Minutes Flow**:

```
30 seconds: First thought captured (demo mode, no account)
1 minute:   Second thought - user exploring what Axel can do
2 minutes:  User asks Axel a question about something they said
3 minutes:  Third conversation - habit forming
4 minutes:  "You've captured 3 thoughts. Want to keep them?"
            [Continue with MYND - Start Free Trial]
5 minutes:  One-tap subscription OR continue demo (10 total)
```

**Key Design Principles for FTUX**:
1. No onboarding screens before first interaction
2. Permissions requested in context (microphone when they tap record)
3. Value delivered before asking for anything
4. Subscription offered as continuation, not barrier
5. BYOK never mentioned in basic flow (power user settings only)

### 2.2 Daily Engagement Loop

**The "Thinking Out Loud" Ritual**

```
┌─────────────────────────────────────────────────────────────────────┐
│                       DAILY ENGAGEMENT LOOP                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  MORNING (Optional - v1.5 Morning Oracle)                           │
│  ├── Notification: "Axel has something for you"                     │
│  ├── Pre-computed insight from yesterday's thoughts                 │
│  └── One tiny action suggestion                                     │
│                                                                     │
│  THROUGHOUT DAY (Core Loop)                                         │
│  ├── Thought strikes → Open via Lock Screen widget (1 tap)          │
│  ├── Speak freely → Axel acknowledges → Axel responds               │
│  ├── Continue conversation OR done                                  │
│  ├── Thought saved automatically                                    │
│  └── Repeat as needed (5-15x daily for active users)                │
│                                                                     │
│  VALUE REINFORCEMENT                                                │
│  ├── Invisible milestone: "You've captured 25 thoughts!"            │
│  ├── Axel references past thought: "Remember when you said..."      │
│  └── Connection discovered: "This relates to what you said about..."│
│                                                                     │
│  EVENING (Optional reflection)                                      │
│  ├── Browse thought list                                            │
│  ├── Search for specific idea                                       │
│  └── Mark thoughts as acted on (simple interaction)                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Time Anchors** (ADHD-Friendly Habit Formation):
- **Commute capture**: Voice notes during travel
- **Meeting aftermath**: "Let me dump everything before I forget"
- **Before sleep**: "Clear my head before bed"
- **Waiting moments**: Elevator, coffee shop, transit

### 2.3 Proactive Engagement Triggers (v1.5+)

**Philosophy**: Helpful nudges, never nagging. "When you're ready" never "overdue."

| Trigger | Timing | Message Style | Purpose |
|---------|--------|---------------|---------|
| **Stale thought** | 7 days untouched | "You mentioned [X] last week. Still on your mind?" | Re-engage with dormant ideas |
| **Pattern detected** | After 10+ mentions | "I notice you often think about [Y]. Want to explore this?" | Surface recurring themes |
| **Goal progress** | Weekly | "You've made progress on [goal]. Here's what you said about it." | Celebrate without pressure |
| **Connection found** | Immediate (v1.5) | "This reminds me of something you said on [date]" | Value of persistent memory |
| **Energy check** | If enabled | "How's your energy right now?" → Adaptive suggestions | Meet user where they are |

**Anti-Patterns to Avoid**:
- No streak counters (creates shame on missed days)
- No "you haven't checked in" guilt
- No red indicators or urgent styling
- No push notifications without clear value

### 2.4 Value Realization Moments

These are the "aha" moments that convert users from trying to committed:

| Moment | When It Happens | User Thought | Impact |
|--------|-----------------|--------------|--------|
| **First capture** | 30 seconds | "That was easy. It actually got what I said." | Trust in transcription |
| **Axel reflects back** | First conversation | "It understood what I meant, not just what I said." | AI value demonstrated |
| **Search works** | Day 3-7 | "I can find that thing I was thinking about!" | Memory externalized |
| **Axel remembers** | Week 2+ | "How does it know I mentioned this before?" | Persistent memory value |
| **Connection discovered** | v1.5 | "I didn't realize those thoughts were related." | Knowledge graph magic |
| **Morning Oracle** | v1.5 | "It's like having a personal assistant who knows me." | Proactive AI value |
| **Invisible milestone** | Various | "50 thoughts! I had no idea I'd captured that many." | Progress without pressure |

---

## Part 3: Feature Prioritization

### 3.1 Feature Priority Matrix

| Feature | User Value | Technical Complexity | Phase | Priority |
|---------|------------|---------------------|-------|----------|
| Push-to-talk voice capture | Critical | Low | MVP | P0 |
| On-device transcription | Critical | Low (Apple API) | MVP | P0 |
| Claude streaming responses | Critical | Medium | MVP | P0 |
| Breathing Wall animation | High | Low | MVP | P0 |
| Quick acknowledgments | High | Low | MVP | P0 |
| Demo mode (10 conversations) | Critical (conversion) | Medium | MVP | P0 |
| StoreKit subscription | Critical (revenue) | Medium | MVP | P0 |
| Thought list view | High | Low | MVP | P0 |
| Basic search | High | Low | MVP | P0 |
| Lock Screen widget | High | Medium | MVP | P1 |
| CloudKit sync | Medium | High | MVP | P1 |
| Axel personality guidelines | High | Low (prompt) | MVP | P1 |
| VoiceOver support | High | Medium | MVP | P1 |
| BYOK settings | Medium | Low | MVP | P2 |
| In-memory knowledge graph | Very High | High | v1.5 | P2 |
| Graph visualization | Very High | Very High | v1.5 | P2 |
| Entity extraction | High | Medium | v1.5 | P2 |
| Morning Oracle | Very High | Medium | v1.5 | P2 |
| Energy-adaptive suggestions | Medium | Low | v1.5 | P3 |
| ElevenLabs premium voice | Medium | Low | v1.5 | P3 |
| Home Screen widget | Low | Low | v1.5 | P3 |
| Siri Shortcuts | Medium | Medium | v1.5 | P3 |
| Apple Watch app | Medium | High | v2.0 | P3 |
| Team/Family features | Low | Very High | v2.0+ | P4 |
| Android | Medium | Very High | v3.0+ | P4 |

### 3.2 Value vs. Complexity Quadrant

```
                         HIGH VALUE
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
          │  Quick Wins      │  Strategic Bets  │
          │                  │                  │
          │  • Breathing Wall│  • Knowledge     │
          │  • Quick acks    │    Graph (v1.5)  │
          │  • Thought list  │  • Morning       │
          │  • Basic search  │    Oracle (v1.5) │
          │  • Axel prompt   │  • Graph viz     │
          │                  │                  │
LOW ──────┼──────────────────┼──────────────────┼────── HIGH
COMPLEXITY│                  │                  │  COMPLEXITY
          │  Low Priority    │  Avoid/Defer     │
          │                  │                  │
          │  • Home widget   │  • Android       │
          │  • Energy adapt  │  • Team features │
          │  • ElevenLabs    │  • Wake word     │
          │                  │    (impossible)  │
          │                  │                  │
          └──────────────────┼──────────────────┘
                             │
                         LOW VALUE
```

### 3.3 Feature Dependencies

```
┌─────────────────────────────────────────────────────────────────────┐
│                      FEATURE DEPENDENCY GRAPH                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Voice Capture ──┬──> Transcription ──> Thought Storage             │
│                  │                           │                      │
│                  └──> Streaming TTS          │                      │
│                            │                 │                      │
│                            ▼                 ▼                      │
│                    Claude Integration ──> Basic Search              │
│                            │                                        │
│                            ▼                                        │
│                    Conversation History                             │
│                            │                                        │
│  ┌─────────────────────────┼─────────────────────────┐              │
│  │         v1.0            │          v1.5           │              │
│  ├─────────────────────────┼─────────────────────────┤              │
│  │                         ▼                         │              │
│  │                 Entity Extraction                 │              │
│  │                         │                         │              │
│  │                         ▼                         │              │
│  │        In-Memory Graph ──> Graph Visualization   │              │
│  │                │                                  │              │
│  │                ▼                                  │              │
│  │        Relationship Detection                     │              │
│  │                │                                  │              │
│  │                ▼                                  │              │
│  │        Morning Oracle ──> Weekly Insights         │              │
│  │                         │                         │              │
│  │                         ▼                         │              │
│  │              Proactive Follow-ups                 │              │
│  └───────────────────────────────────────────────────┘              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Part 4: Phased Roadmap

### Phase 1: MVP (Weeks 1-14)

**Theme**: "Capture and Respond"

**Pre-Development (Weeks 1-2)**:
- [ ] User interview validation (10 interviews with ADHD users)
- [ ] Figma design system creation
- [ ] Xcode project setup with CI/CD
- [ ] Axel personality guidelines document
- [ ] Accessibility requirements checklist

**Core Development (Weeks 3-10)**:

| Week | Focus | Deliverables |
|------|-------|--------------|
| 3-4 | Voice Engine | Apple Speech STT, AVSpeech TTS, audio session management, Breathing Wall |
| 5-6 | AI Integration | Claude API client, streaming, demo mode (canned responses), acknowledgments |
| 7-8 | Thought Capture | SwiftData model, list view, CRUD operations, simple search, basic CloudKit |
| 9-10 | Core UI | Conversation view, thought list, settings skeleton, Lock Screen widget |

**Polish & Monetization (Weeks 11-14)**:

| Week | Focus | Deliverables |
|------|-------|--------------|
| 11-12 | Onboarding & IAP | Demo mode flow, StoreKit 2, subscription tiers, BYOK settings |
| 13-14 | Testing & Polish | UI testing, accessibility audit, performance profiling, crash analytics |

**MVP Launch Criteria**:
- [ ] Voice capture success rate >90% (quiet environment)
- [ ] First thought captured in <30 seconds
- [ ] Claude response begins <3 seconds after speech ends
- [ ] Zero P0 bugs, <5 P1 bugs
- [ ] VoiceOver navigation complete
- [ ] App size <100MB, cold launch <3s

### Phase 2: Beta (Weeks 15-20)

**Theme**: "Validate with Real Users"

| Week | Focus | Activities |
|------|-------|------------|
| 15-16 | Private Beta | 50 TestFlight users (ADHD community), daily feedback, rapid iteration |
| 17-18 | Expanded Beta | 100 users, A/B test onboarding, pricing validation |
| 19 | App Store Submission | Final QA, submission, marketing materials |
| 20 | Public Launch | ProductHunt, social campaign, monitor and hotfix |

### Phase 3: Core Expansion - v1.5 (Weeks 21-32)

**Theme**: "Knowledge Graph & Proactive AI"

| Week | Focus | Deliverables |
|------|-------|--------------|
| 21-22 | In-Memory Graph | Adjacency list implementation, SwiftData backing, fast queries |
| 23-24 | Entity Extraction | NaturalLanguage framework, person/place/concept detection |
| 25-26 | Graph Visualization | Force-directed layout (max 500 nodes), interactive exploration |
| 27-28 | Morning Oracle | Background App Refresh, overnight insight generation, cached briefings |
| 29-30 | Proactive Features | Stale thought detection, pattern recognition, gentle follow-ups |
| 31-32 | Polish & Release | Energy-adaptive suggestions, ElevenLabs integration, v1.5 launch |

**v1.5 Success Criteria**:
- [ ] Graph operations remain responsive at 1000 nodes
- [ ] Morning Oracle delivery >80% reliability
- [ ] Knowledge graph adoption >40% of Pro users
- [ ] NPS improvement to 50+

### Phase 4: Advanced Features - v2.0 (Weeks 33-44)

**Theme**: "Platform Expansion & Advanced AI"

| Feature | Effort | User Value |
|---------|--------|------------|
| Apple Watch companion | 4 weeks | Capture during walks/commutes |
| Siri Shortcuts deep integration | 2 weeks | "Hey Siri, tell MYND..." |
| Control Center widget (iOS 18+) | 1 week | 1-tap from anywhere |
| Advanced sync conflict resolution | 2 weeks | Reliable multi-device |
| Graph-aware search | 2 weeks | "Find thoughts related to X" |
| Weekly insight reports | 1 week | Email/notification summaries |

### Phase 5: Platform Expansion - v3.0+ (Week 45+)

**Theme**: "Ecosystem & Cross-Platform"

| Initiative | Complexity | Strategic Value |
|------------|------------|-----------------|
| Android app | Very High | 2x addressable market |
| Web companion | High | Desktop capture |
| mynd-brain integration | High | Advanced ML capabilities |
| Export to Obsidian/Notion | Medium | Data portability (trust) |
| Team/Family features | High | New revenue streams |
| API for third-party integrations | Medium | Platform ecosystem |

---

## Part 5: Success Metrics

### 5.1 Engagement Metrics

| Metric | Definition | Target (6mo) | Target (12mo) |
|--------|------------|--------------|---------------|
| **DAU** | Daily Active Users | 2,000 | 10,000 |
| **MAU** | Monthly Active Users | 8,000 | 35,000 |
| **DAU/MAU Ratio** | Stickiness | 25% | 30% |
| **Sessions/DAU** | Engagement depth | 2.5 | 3.0 |
| **Thoughts/User/Week** | Core usage | 15 | 20 |
| **Avg Session Duration** | Time in app | 3 min | 4 min |
| **Voice vs Text Ratio** | Voice-first success | 80% voice | 80% voice |

### 5.2 Retention Indicators

| Metric | Definition | Target |
|--------|------------|--------|
| **D1 Retention** | Return next day | 50% |
| **D7 Retention** | Return within week | 30% |
| **D30 Retention** | Return within month | 20% |
| **Weekly Retention** | 4-week rolling | 60% |
| **Subscriber Churn** | Monthly paid loss | <5% |
| **Reactivation Rate** | Churned users returning | 10% |

### 5.3 Value Metrics

| Metric | Definition | Target |
|--------|------------|--------|
| **Time to First Value** | First thought captured | <60 seconds |
| **Demo Conversion** | Demo → Free Trial | 40% |
| **Free to Paid Conversion** | Trial → Subscription | 4-6% |
| **ARPU** | Avg Revenue Per User | $3/mo |
| **LTV** | Lifetime Value (paid) | $50+ |
| **NPS** | Net Promoter Score | 40+ (launch), 50+ (v1.5) |
| **App Store Rating** | Average rating | 4.5+ |

### 5.4 Operational Metrics

| Metric | Definition | Target |
|--------|------------|--------|
| **Voice Recognition Accuracy** | Transcription correctness | >90% |
| **Claude Response Time** | Time to first token | <2 seconds |
| **App Crash Rate** | Sessions with crashes | <1% |
| **CloudKit Sync Success** | Successful syncs | >99% |
| **API Cost per User** | Claude usage cost | <$2/mo (Pro) |
| **Customer Support Volume** | Tickets per 1000 users | <20/mo |

### 5.5 Feature Adoption Metrics (v1.5+)

| Feature | Adoption Target | Power User Threshold |
|---------|-----------------|---------------------|
| Knowledge Graph | 40% of Pro users view | >10 graph interactions/week |
| Morning Oracle | 50% opt-in | Daily open rate >60% |
| Energy Adaptive | 30% usage | Use 3x/week |
| Proactive Follow-ups | 60% engagement | Respond to >50% of prompts |

---

## Part 6: Risk Assessment

### 6.1 Technical Risks

| Risk | Probability | Impact | Mitigation | Contingency |
|------|-------------|--------|------------|-------------|
| **Voice latency feels unacceptable** | 40% | High | Breathing Wall, acknowledgments, "thoughtful companion" framing | Add text-first mode as equal option |
| **SwiftData fails at graph scale** | 80% | Critical | GraphStore protocol abstraction, in-memory cache | SQLite + FTS5 migration path ready |
| **CloudKit sync data loss** | 25% | Critical | Comprehensive conflict resolution, local backup | Manual export feature, support escalation |
| **Claude API outages** | 20% | Medium | Graceful degradation, cached responses | OpenAI fallback, local summarization |
| **Speech recognition poor accuracy** | 30% | Medium | Test extensively, accent support, retry UX | Text input always available |
| **Memory pressure on older devices** | 40% | Medium | Lazy loading, graph limits (1000 nodes) | Test on iPhone 11, minimum iOS version |

### 6.2 Market Risks

| Risk | Probability | Impact | Mitigation | Contingency |
|------|-------------|--------|------------|-------------|
| **Apple Intelligence competes directly** | 70% | Existential | BYOK transparency, cross-platform roadmap, data export | Pivot to Obsidian/Notion companion |
| **Sesame AI adds note-taking** | 40% | High | Focus on knowledge graph + ADHD design | Differentiate on proactive features |
| **Note-taking market saturation** | 50% | Medium | ADHD niche focus, voice-first positioning | Community-led growth, influencer partnerships |
| **Subscription fatigue in target audience** | 60% | Medium | BYOK option, lifetime deal, clear value demo | Freemium tier with ads (last resort) |
| **App Store rejection** | 30% | Critical | Avoid health claims, pre-submission review | Reframe as "productivity", appeal process |

### 6.3 Competitive Risks

| Risk | Probability | Impact | Mitigation | Contingency |
|------|-------------|--------|------------|-------------|
| **Well-funded competitor enters ADHD+voice niche** | 30% | High | First-mover in intersection, community loyalty | Accelerate roadmap, partnership opportunities |
| **Notion/Obsidian add voice-first mode** | 40% | High | Deeper ADHD focus, proactive features | Integration rather than competition |
| **Claude API pricing increases** | 50% | Medium | BYOK option, usage optimization | Multi-provider support, on-device fallback |
| **Anthropic changes API terms** | 20% | High | Read ToS carefully, BYOK model | OpenAI/other provider fallback |

### 6.4 Business Model Risks

| Risk | Probability | Impact | Mitigation | Contingency |
|------|-------------|--------|------------|-------------|
| **Low conversion rate (<2%)** | 35% | High | Demo-first experience, pricing experiments | Lower Starter price, extend demo |
| **High churn (>10%)** | 30% | High | Proactive engagement, value reinforcement | Winback campaigns, exit surveys |
| **API costs exceed revenue** | 30% | Medium | BYOK tier, usage caps on Starter, prompt optimization | Tiered usage limits, price increase |
| **Support costs too high** | 40% | Medium | In-app help, community forum, FAQ | Reduce BYOK complexity, managed-only |

### 6.5 Apple Intelligence Defense Strategy

**The Existential Threat**: Apple is coming for AI-powered productivity. WWDC 2026 will likely expand Apple Intelligence. This could directly compete with MYND around the same launch window.

**Why Apple Wins (if they ship this)**:
- On-device processing (no latency, no API costs)
- System-wide context (calendar, messages, location)
- Hardware optimization
- User trust already established
- Distribution advantage

**MYND's Moat Against Apple**:

1. **BYOK Transparency**: Users control their AI, Apple won't offer this
2. **Cross-Platform Future**: Android roadmap (Apple never will)
3. **Data Portability**: Export to Markdown/JSON/Obsidian (Apple locks data)
4. **Customization**: Axel personality tuning (Apple standardizes)
5. **ADHD-Specific Design**: Apple builds for everyone, MYND builds for this audience
6. **Community Features**: Shared templates, community (Apple won't do social)

**Defensive Positioning**:
- Marketing: "Your AI, your data, your way"
- Technical: Never depend on Apple-only features
- Strategic: Be the best iOS companion, not an Apple competitor

**Worst-Case Pivot Options**:
1. MYND becomes "thought export layer" - integrate with Obsidian, Notion, Roam
2. Sell/partner with knowledge management company
3. Focus on Android where Apple doesn't compete
4. Pivot to enterprise (team thought capture)

---

## Part 7: Strategic Recommendations

### 7.1 Critical Success Factors

1. **Nail the first 30 seconds**: Time-to-first-value is everything for ADHD users
2. **Breathing Wall must feel calming**: This turns latency from bug to feature
3. **Axel personality must be consistent**: Users form emotional connections
4. **Demo mode must convert**: 10 free conversations → subscription decision
5. **ADHD community must advocate**: Word-of-mouth in this community is powerful

### 7.2 Things to NOT Do

1. **Don't promise "conversational AI"**: Own the "thoughtful companion" positioning
2. **Don't ship knowledge graph in MVP**: Prove voice capture works first
3. **Don't require BYOK for basic users**: Managed subscription for 95%
4. **Don't add streak counters**: Creates shame, hostile to target audience
5. **Don't use red or urgent styling**: "When you're ready" not "overdue"
6. **Don't compete with Sesame on voice quality**: Compete on memory + knowledge

### 7.3 Key Decisions Still Needed

| Decision | Options | Recommendation |
|----------|---------|----------------|
| Primary positioning | Voice-first OR Knowledge graph OR ADHD tool | **ADHD-focused voice capture** (specific wins over general) |
| Pricing model | Freemium OR Paid-only OR BYOK-only | **Freemium with managed subscription + BYOK option** |
| First widget | Lock Screen OR Home Screen OR Both | **Lock Screen only** (sufficient, simpler) |
| mynd-brain integration | Standalone OR Integrated OR Hybrid | **Standalone first**, design for future hybrid |
| Launch market | US-only OR Global | **US-only** (39% of ADHD app downloads, simplest) |

### 7.4 Immediate Next Steps

**Week 1 Actions**:
1. Validate "thoughtful companion" positioning with 10 user interviews
2. Create Axel personality guidelines (3-5 page document)
3. Design Breathing Wall animation prototypes
4. Set up Xcode project with CI/CD
5. Draft App Store description and screenshots

**Pre-Development Checklist**:
- [ ] User interviews complete (10)
- [ ] Axel personality document approved
- [ ] Design system in Figma
- [ ] SwiftData models designed
- [ ] Claude prompt templates drafted
- [ ] Accessibility requirements documented
- [ ] Analytics events defined
- [ ] Error handling philosophy documented

---

## Appendices

### Appendix A: Axel Personality Guidelines (Draft)

**Core Traits**:
- Warm but not effusive
- Thoughtful (pauses before responding)
- Non-judgmental (never "you should have...")
- Curious (asks follow-up questions)
- Memory-aware (references past conversations naturally)

**Voice/Tone**:
- Gentle, calm presence
- Short sentences when listening, longer when reflecting
- Uses "I notice..." and "I'm curious about..." not "You need to..."
- Acknowledges emotions without diagnosing

**Response Patterns**:
- First response: Acknowledge + Reflect back + Open question
- Follow-up: Build on what user said + Offer connection to past thought
- Closing: Summarize + Offer next action (optional)

**Things Axel Never Says**:
- "You should..."
- "You need to..."
- "Don't forget to..."
- "You're overdue on..."
- "You failed to..."
- Any time-pressure language

### Appendix B: Demo Mode Response Examples

**User**: "I keep forgetting to call my mom"

**Axel** (demo): "I hear that weighing on you. Family connections matter to you. When you think about calling her, what usually gets in the way?"

**User**: "I have this idea for a side project but I never start it"

**Axel** (demo): "Ideas have a way of staying stuck sometimes. What's the smallest piece of this project - something you could do in two minutes?"

**User**: "I'm so overwhelmed with everything"

**Axel** (demo): "That feeling is real, and it's okay to feel it. If you could set aside everything except one thing right now, what would that one thing be?"

### Appendix C: Key Reference Documents

| Document | Location | Purpose |
|----------|----------|---------|
| MYND_ARCHITECTURE.md | workspace/ | Technical architecture v2.0 |
| MYND_CRITIQUE.md | workspace/ | Critical review with showstoppers |
| MYND_REFINED_PLAN.md | workspace/ | Approved implementation plan |
| MYND_BRAINSTORM.md | workspace/ | Creative feature ideas |
| market_analysis.md | workspace/research/ | Competitive landscape |
| This document | workspace/mynd/workspace/ | Product strategy (you are here) |

---

**Document Status**: APPROVED FOR IMPLEMENTATION
**Next Step**: Begin Pre-Development Phase (User Interviews + Design System)
**Review Cadence**: Weekly during development
