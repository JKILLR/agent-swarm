# MYND Master Plan

**Version**: 1.0 (Definitive)
**Date**: 2026-01-04
**Status**: SINGLE SOURCE OF TRUTH
**Document Purpose**: Synthesis of all research, critiques, and iterations into definitive product specification

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Core Strategy](#2-core-strategy)
3. [MVP Scope](#3-mvp-scope)
4. [Technical Architecture](#4-technical-architecture)
5. [Business Model](#5-business-model)
6. [Risk Matrix](#6-risk-matrix)
7. [Kill Criteria](#7-kill-criteria)
8. [Next Steps](#8-next-steps)

---

## 1. Executive Summary

### What is MYND?

MYND is an **AI-powered thought capture app** for iOS that helps users with executive function challenges (particularly ADHD) capture, organize, and retrieve their thoughts through conversational AI.

### The Problem MYND Solves

People with ADHD and executive function challenges struggle with:
- **Capturing fleeting thoughts** before they're forgotten
- **Organizing scattered ideas** into coherent structures
- **Retrieving past insights** when they're relevant
- **Maintaining context** across conversations over time

Current solutions fail because:
- Traditional note apps require structure that ADHD brains resist
- Voice assistants lack persistent memory
- AI companions (Pi, Replika) reset context or lack knowledge organization
- Productivity tools (Notion, Obsidian) have high cognitive overhead

### Key Decisions Made

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Positioning** | Multi-modal (not voice-first) | Voice latency is liability; text enables broader contexts |
| **Primary Input** | Text-first, voice-secondary | ADHD delay aversion makes 1-3s voice latency problematic |
| **AI Provider** | Multi-model architecture | Avoid Anthropic lock-in; enable cost optimization |
| **Persistence** | Local-only for MVP | CloudKit sync too risky; defer to v1.5 |
| **Pricing** | Hard limits, not unlimited | "Unlimited" at $9.99 is economically unsustainable |
| **Trial Model** | 7-day unlimited trial | Builds habit better than conversation count |
| **Knowledge Graph** | Deferred to v1.5 | Core capture must work first |
| **macOS** | Deferred to v1.5 | iOS-only for MVP reduces complexity |

### Success Criteria

| Metric | 90-Day Target | Stretch |
|--------|---------------|---------|
| App Store Rating | 4.0+ | 4.5+ |
| DAU/MAU Ratio | >30% | >40% |
| Trial-to-Paid Conversion | 2% | 4% |
| 30-Day Retention | 30% | 40% |
| Paying Subscribers | 500+ | 1,000+ |
| NPS | +20 | +40 |

---

## 2. Core Strategy

### 2.1 Positioning: Multi-Modal Thought Capture

**What MYND Is:**
- A conversational AI companion for capturing thoughts naturally
- Works with both voice AND text equally
- Remembers context across sessions (persistent memory)
- Designed for scattered/ADHD minds

**What MYND Is NOT:**
- A voice-first app (voice is an option, not the default)
- A therapist or medical tool
- A replacement for journaling apps
- A general-purpose AI assistant

### 2.2 Differentiation: The Moat

MYND's differentiation is structural - competitors cannot easily replicate:

| Differentiator | Why Competitors Can't Copy |
|----------------|---------------------------|
| **BYOK (Bring Your Own Key)** | Apple, Pi, Sesame subsidize AI; BYOK reveals their costs |
| **Full Data Export** | Competitors lock data for retention; MYND enables portability |
| **Local-First Privacy** | Cloud-first competitors can't retroactively add local-first |
| **AI Model Choice** | Competitors are locked to their AI provider |
| **ADHD-First Design** | Requires deep understanding, not feature bolt-on |

### 2.3 Competitive Landscape

| Competitor | Strength | MYND Advantage |
|------------|----------|----------------|
| **Apple Intelligence** | Free, pre-installed, hardware-optimized | BYOK, export, model choice, ADHD design |
| **Sesame AI** | Best voice quality (<200ms) | Persistent memory, knowledge capture |
| **Pi AI** | Warm personality, free | Memory doesn't reset, organization features |
| **Notion AI** | Full workspace | Simplicity, voice-first capture, ADHD focus |

### 2.4 Target User

**Primary:** Adults with ADHD (18-45) who:
- Have tried and abandoned multiple productivity systems
- Value their privacy
- Are willing to pay for tools that work
- Are comfortable with technology
- Prefer conversation over structured input

**Secondary:** Anyone with:
- Executive function challenges
- Information overload
- Desire for AI companion with memory

---

## 3. MVP Scope

### 3.1 Timeline: 20 Weeks (+ 4-Week Buffer)

```
Week 0-2:   Foundation (design, LLM abstraction, legal)
Week 3-6:   Core Capture (text/voice input, AI response, persistence)
Week 7-9:   Polish & Monetization (subscription, BYOK, widget)
Week 10-11: Testing (unit, UI, performance, accessibility)
Week 12-15: Private Beta (TestFlight, bug fixes, iteration)
Week 16:    Launch (App Store submission, marketing)
Week 17-20: Buffer (rejections, hotfixes, unexpected issues)
```

### 3.2 What's IN (MVP v1.0)

| Feature | Priority | Notes |
|---------|----------|-------|
| **Text input** | P0 | Primary capture method |
| **Voice input (push-to-talk)** | P0 | Apple Speech Framework |
| **AI conversation (streaming)** | P0 | Multi-model (Claude, GPT, Gemini) |
| **Thought timeline** | P0 | Chronological list view |
| **Basic search** | P0 | Text matching |
| **Local persistence** | P0 | SwiftData |
| **TTS playback** | P1 | AVSpeechSynthesizer |
| **Lock Screen widget** | P1 | Quick capture |
| **Subscription tiers** | P0 | Starter, Pro with hard limits |
| **BYOK option** | P0 | Power user monetization |
| **7-day trial** | P0 | Unlimited during trial |
| **JSON export** | P1 | Data portability |
| **Settings** | P1 | AI provider, voice, appearance |
| **Onboarding** | P0 | First-time user experience |

### 3.3 What's OUT (Deferred to v1.5)

| Feature | Reason for Deferral |
|---------|---------------------|
| **Knowledge graph visualization** | Complexity; core capture must work first |
| **CloudKit sync** | High risk of data loss; need more testing |
| **Premium voice (ElevenLabs)** | AVSpeech sufficient for MVP |
| **Morning Oracle (proactive)** | Reactive-only in MVP |
| **Goal tracking** | Scope creep; simple notes sufficient |
| **macOS app** | iOS-only reduces complexity |
| **Home Screen widget** | Lock Screen widget sufficient |
| **Tags/categories** | Timeline view sufficient for MVP |
| **Photo/document input** | Text/voice covers 90% of use cases |

### 3.4 Never Building

| Feature | Reason |
|---------|--------|
| Wake word activation | iOS doesn't allow |
| Always-on listening | Battery, privacy, iOS limits |
| Real-time collaboration | Out of scope |
| Android app | Not until v2.0+ |

---

## 4. Technical Architecture

### 4.1 Core Principles

| Principle | Implementation |
|-----------|----------------|
| **Provider Independence** | All LLM calls through `LLMProviderProtocol` |
| **Graceful Degradation** | Every premium feature has free fallback |
| **Scope Discipline** | Cut features, not quality |
| **Pivot Ready** | Architecture supports multiple business models |

### 4.2 System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     PRESENTATION LAYER                          │
├─────────────────────────────────────────────────────────────────┤
│  ConversationView  │  ThoughtListView  │  SettingsView  │ Widget│
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                     VIEW MODEL LAYER                            │
├─────────────────────────────────────────────────────────────────┤
│  ConversationVM (@Observable)  │  ThoughtListVM  │  SettingsVM  │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      SERVICE LAYER                              │
├─────────────────────────────────────────────────────────────────┤
│  LLMRouter  │  VoiceEngine  │  TTSController  │  Subscription   │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                   LLM PROVIDER LAYER                            │
├─────────────────────────────────────────────────────────────────┤
│  Claude  │  OpenAI  │  Gemini  │  Local MLX  │  (Apple Intel.)  │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                   PERSISTENCE LAYER                             │
├─────────────────────────────────────────────────────────────────┤
│       SwiftData Store         │        Keychain (API keys)      │
└─────────────────────────────────────────────────────────────────┘
```

### 4.3 Multi-Model LLM Architecture

**Why Multi-Model:**

| Risk | Probability | Multi-Model Mitigation |
|------|-------------|------------------------|
| Claude API costs double | 40% | Switch to GPT-4o-mini or Gemini |
| Anthropic rate limits | 60% | Automatic fallback to OpenAI |
| Apple requires on-device | 20% | Local model support ready |
| User demands choice | 30% | BYOK supports any provider |

**Provider Priority:**
1. Claude 3.5 Haiku (primary - best instruction following)
2. GPT-4o-mini (fallback - fast, cheap)
3. Gemini Flash (fallback - cheapest)
4. Local MLX (offline - free, private)

**LLM Protocol:**
```swift
protocol LLMProviderProtocol: Sendable {
    var providerId: String { get }
    var capabilities: LLMCapabilities { get }
    var costEstimate: LLMCostEstimate { get }
    var isAvailable: Bool { get async }

    func complete(messages:, systemPrompt:, tools:, options:) async throws -> LLMResponse
    func streamComplete(...) -> AsyncThrowingStream<LLMStreamChunk, Error>
    func embed(texts:) async throws -> [[Float]]
}
```

### 4.4 Text-First UX Flow

**Input Priority:**
1. **Text (Primary)** - Large text field, always visible
2. **Voice (Secondary)** - Mic button to the right
3. **Voice Continuation** - Optional hands-free mode after TTS

**Response Display:**
1. Immediate text streaming (no waiting)
2. Optional TTS playback (user taps speaker or auto-play setting)
3. Save to thoughts (bookmark action)

### 4.5 Data Model

```swift
@Model
class ThoughtNode {
    var id: UUID
    var content: String
    var createdAt: Date
    var updatedAt: Date
    var source: CaptureSource  // text, voice, import
    var sessionId: UUID?
}

@Model
class ConversationSession {
    var id: UUID
    var messages: [Message]
    var startedAt: Date
    var endedAt: Date?
    var tokenUsage: TokenUsage
}

@Model
class Message {
    var id: UUID
    var role: MessageRole  // user, assistant
    var content: String
    var timestamp: Date
    var provider: String  // claude, openai, etc.
}
```

### 4.6 Security Implementation

| Layer | Implementation |
|-------|----------------|
| API Keys | Keychain with `kSecAttrAccessibleWhenUnlockedThisDeviceOnly` |
| Network | HTTPS only, TLS 1.3 |
| Device | Default iOS encryption (Data Protection) |
| Consent | Explicit consent screen for voice recording |
| Export | User-initiated JSON export (GDPR compliant) |

---

## 5. Business Model

### 5.1 Pricing Tiers (Revised with Hard Limits)

| Tier | Price | Messages/Month | Features |
|------|-------|----------------|----------|
| **Trial** | Free | Unlimited | 7 days, all features |
| **Starter** | $4.99/mo | 500 | Text + voice, basic AI |
| **Pro** | $9.99/mo | 2,000 | All features, priority response |
| **BYOK** | $4.99/mo | Unlimited* | User provides API key |

*BYOK users pay their own API costs directly to provider.

### 5.2 Economics Validation

**Starter Tier ($4.99/mo, 500 messages):**

| User Type | Messages | API Cost | Apple (30%) | Net |
|-----------|----------|----------|-------------|-----|
| Light | 100 | $0.20 | $1.50 | $3.29 |
| Average | 300 | $0.60 | $1.50 | $2.89 |
| Max | 500 | $1.00 | $1.50 | $2.49 |

**Pro Tier ($9.99/mo, 2,000 messages):**

| User Type | Messages | API Cost | Apple (30%) | Net |
|-----------|----------|----------|-------------|-----|
| Light | 400 | $0.80 | $3.00 | $6.19 |
| Average | 1,000 | $2.00 | $3.00 | $4.99 |
| Heavy | 2,000 | $4.00 | $3.00 | $2.99 |

**Conclusion:** Hard limits are essential. "Unlimited" at $9.99 would lose money on heavy users.

### 5.3 Trial Design

| Aspect | Decision | Rationale |
|--------|----------|-----------|
| Duration | 7 days | Builds habit better than message count |
| Limits | Unlimited | Full experience during trial |
| Data | Persists | Migrates to subscription |
| Device ID | DeviceCheck API | Prevents abuse (accepts some reset via factory restore) |

### 5.4 Revenue Projections (Conservative)

**Assumptions:**
- 1,000 downloads/month after launch
- 1% trial-to-paid conversion (conservative)
- 50% Starter, 40% Pro, 10% BYOK distribution
- 5% monthly churn

| Month | Downloads | Conversions | Subscribers | MRR |
|-------|-----------|-------------|-------------|-----|
| 1 | 1,000 | 10 | 10 | $75 |
| 3 | 3,000 | 30 | 57 | $428 |
| 6 | 6,000 | 60 | 171 | $1,283 |
| 12 | 12,000 | 120 | 456 | $3,420 |

**Break-even:** ~$2,000 MRR (covers hosting, analytics, legal, dev tools)

---

## 6. Risk Matrix

### 6.1 Top 5 Risks with Mitigations

#### Risk 1: Apple Intelligence Competition (70% probability)

**Threat:** Apple announces competing features at WWDC 2026 that are free, pre-installed, and instant.

**Mitigations:**
- Position MYND as complement, not competitor: "Works WITH Apple Intelligence, does MORE"
- Emphasize differentiators Apple can't match: BYOK, export, model choice, cross-platform future
- Accelerate features Apple won't have: knowledge graph, ADHD-specific design
- Add Apple Intelligence as a provider option if API becomes available

#### Risk 2: Voice Latency Rejection (40% probability)

**Threat:** Users find 1-3 second voice response latency unacceptable despite ADHD delay aversion.

**Mitigations:**
- Text-first positioning reduces voice dependency
- Immediate visual acknowledgment before full response
- Streaming text display starts instantly
- Pre-cache common response patterns
- Honest marketing: "Axel is thoughtful, not instant"
- Target users who value deliberate responses

#### Risk 3: Claude API Cost Increase (40% probability)

**Threat:** Anthropic raises prices 2-3x, destroying managed tier margins.

**Mitigations:**
- Multi-model architecture from day one
- LLMRouter switches to cheaper provider automatically
- User notification of provider change
- Build pricing buffer into tiers
- Invest in local model quality for simple queries

#### Risk 4: Low Conversion Rate (<2%) (35% probability)

**Threat:** Trial users don't convert, business is unsustainable.

**Mitigations:**
- Price experiments via feature flags ($2.99, $4.99, $7.99)
- Extended trial option (14 days)
- Strong annual discount (40% off)
- Lifetime deal for early adopters
- Referral program

#### Risk 5: CloudKit Data Loss (25% probability in v1.5)

**Threat:** Sync conflicts cause data loss, destroying trust permanently.

**Mitigations:**
- Local-only for MVP (no sync risk)
- Extensive conflict resolution testing before v1.5
- Soft-delete with tombstones (never hard delete)
- Local backup before any sync operation
- Sync status indicator on all screens
- Manual conflict resolution UI

### 6.2 Risk Tracking

| Risk | Probability | Impact | Status | Owner | Monitor |
|------|-------------|--------|--------|-------|---------|
| Apple Intelligence | 70% | Existential | ACTIVE | Founder | WWDC announcements |
| Voice latency rejection | 40% | High | ACTIVE | Founder | Beta NPS, retention |
| API cost increase | 40% | High | MITIGATED | Founder | Anthropic pricing |
| Low conversion | 35% | High | ACTIVE | Founder | Trial conversion rate |
| CloudKit data loss | 25% | Critical | DEFERRED | - | Not applicable for MVP |
| Solo founder burnout | 50% | High | ACTIVE | Founder | Self-monitoring |

---

## 7. Kill Criteria

### Gate 0: Pre-Development Validation (Week 0-2)

| Trigger | Action |
|---------|--------|
| >40% of user interviews say 3+ second latency is unacceptable | PIVOT to text-only |
| Anthropic declines commercial terms | PIVOT to OpenAI-primary |
| Economic modeling shows Starter unprofitable at $4.99 | PIVOT pricing to $6.99+ |

### Gate 1: Post-Phase 1 (Week 10)

| Trigger | Action |
|---------|--------|
| Streaming response latency consistently >10 seconds | STOP and fix before Phase 2 |
| Voice recognition accuracy <85% | STOP and fix |
| Developer health deteriorating | PAUSE for 2-week break |

### Gate 2: Beta Launch (Week 15)

| Trigger | Action |
|---------|--------|
| 7-day retention <30% | RED FLAG - interview churned users |
| NPS <20 | RED FLAG - identify top complaints |
| Crash rate >10% | STOP BETA - fix stability |

### Gate 3: Public Launch (Week 20)

| Trigger | Action |
|---------|--------|
| Week 1 downloads <1,000 | YELLOW FLAG - increase marketing |
| Week 1 App Store rating <3.5 | RED FLAG - respond to every review |
| Week 1 crash reports >100 | STOP - hotfix immediately |

### Gate 4: 30 Days Post-Launch

| Trigger | Action |
|---------|--------|
| Conversion rate <1% | SERIOUS - price experiments |
| DAU declining week-over-week | SERIOUS - no product-market fit |
| Revenue doesn't cover API costs | CRITICAL - add hard limits |

### Gate 5: 90 Days Post-Launch

| Trigger | Action |
|---------|--------|
| <500 paying subscribers | PROJECT REVIEW |
| ARR <$30K (annualized) | PROJECT REVIEW |
| Founder wants to quit | PROJECT REVIEW |

**Project Review Options:**
1. Continue with adjusted strategy
2. Sell IP/project
3. Pivot to different market (B2B, therapy tools)
4. Shut down gracefully

---

## 8. Next Steps

### Week 0 Tasks (Before Development Begins)

#### Business & Legal (Days 1-3)
- [ ] Calculate managed tier economics with 2x and 3x API cost scenarios
- [ ] Contact Anthropic sales to confirm commercial API resale terms (get in writing)
- [ ] Draft privacy policy outline
- [ ] Create GDPR/CCPA compliance checklist
- [ ] Budget for legal review ($3-5K)

#### Technical Foundation (Days 4-7)
- [ ] Set up Xcode project with SwiftUI, SwiftData
- [ ] Implement `LLMProviderProtocol` abstraction
- [ ] Implement Claude provider
- [ ] Implement OpenAI provider (fallback)
- [ ] Set up Keychain service for API key storage

#### Design & UX (Days 8-10)
- [ ] Create design system (colors, typography, spacing)
- [ ] Design text-first input component
- [ ] Design conversation view mockups
- [ ] Create accessibility requirements document
- [ ] Write Axel personality guidelines (style guide)

#### Research & Validation (Days 11-14)
- [ ] Recruit 10 ADHD users for interviews
- [ ] Conduct interviews on latency tolerance
- [ ] Validate voice vs text preference in different contexts
- [ ] Document interview findings

### Pre-Development Checklist Summary

| Category | Item | Status |
|----------|------|--------|
| Legal | Privacy policy draft | [ ] |
| Legal | GDPR checklist | [ ] |
| Legal | Anthropic terms confirmed | [ ] |
| Technical | LLM abstraction layer | [ ] |
| Technical | Multi-model support | [ ] |
| Design | Design system | [ ] |
| Design | Accessibility requirements | [ ] |
| Research | User interviews (10) | [ ] |
| Business | Economic model validated | [ ] |
| Business | Apple Developer account setup | [ ] |

### Key Milestones

| Milestone | Week | Deliverable |
|-----------|------|-------------|
| Foundation Complete | 2 | LLM layer, design system, legal outline |
| Core Capture Working | 6 | Text/voice input, AI response, persistence |
| Monetization Ready | 9 | Subscription, BYOK, trial system |
| Beta Launch | 15 | TestFlight with 50 users |
| Public Launch | 20 | App Store submission |
| v1.5 Release | 32 | CloudKit sync, knowledge graph |

---

## Document Change Log

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01-04 | Initial master plan synthesizing all research and critiques |

---

## Appendices

### Appendix A: Source Documents

1. `STATE.md` - Project state and research notes
2. `mynd/ASSUMPTION_VALIDATION.md` - Research-backed validation of key assumptions
3. `mynd/ARCHITECTURE_V3.md` - Technical architecture specification
4. `mynd/CRITIQUE_V3.md` - Adversarial critique and kill criteria
5. `MYND_FINAL_REVIEW.md` - Go/no-go assessment

### Appendix B: Quick Reference

**Target Launch:** May 2026 (Week 20)
**Pricing:** $4.99 Starter (500 msg), $9.99 Pro (2,000 msg), $4.99 BYOK
**Trial:** 7 days unlimited
**Platform:** iOS 17+ only (MVP)
**AI Providers:** Claude (primary), OpenAI (fallback), Gemini (fallback), Local MLX (offline)

**Key Differentiators:**
- BYOK (Bring Your Own Key)
- Full data export
- Multi-model choice
- ADHD-first design
- Local-first privacy

**Kill Criteria Summary:**
- Pre-dev: >40% reject latency → pivot to text-only
- Beta: <30% retention → investigate
- Launch: <1% conversion → price experiments
- 90 days: <500 subscribers → project review

---

*This document is the SINGLE SOURCE OF TRUTH for MYND development.*
*All decisions should reference this document.*
*Updates require explicit version control and changelog entry.*
