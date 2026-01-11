# MYND Feature Brainstorm

**Date**: 2026-01-04
**Author**: System Architect
**Status**: BRAINSTORM DOCUMENT

---

## Executive Summary

This document explores feature ideas for MYND, a voice-first AI thought capture app with a knowledge graph backend and proactive AI assistant named Axel. Features are organized by category, prioritized by implementation phase, and assessed for complexity and user value.

**Core Vision Alignment**: Every feature should support users with scattered thoughts by making capture effortless, organization automatic, and action achievable.

---

## 1. Voice Experience Innovation

### 1.1 Beyond Tap-to-Talk

| Feature | Description | Priority | Complexity | User Value | Phase |
|---------|-------------|----------|------------|------------|-------|
| **Wake Word Activation** | "Hey Axel" triggers listening mode | Must-Have | Medium | High | MVP+ |
| **Continuous Listening Mode** | Background listening with privacy indicators | Nice-to-Have | High | Medium | Future |
| **Ambient Mode** | Low-power always-listening for quick captures | Nice-to-Have | High | Medium | Future |
| **Whisper Detection** | Recognize when user is speaking quietly (privacy mode) | Nice-to-Have | High | Low | Future |
| **Hands-Free Navigation** | Voice commands for "show my goals", "what's next" | Nice-to-Have | Medium | High | v2 |
| **Voice Interruption** | Say "pause" or "stop" to interrupt Axel mid-speech | Must-Have | Low | High | MVP |

**MVP Implementation**: Start with tap-to-talk + voice interruption. Wake word ("Hey Axel") can use Apple VoiceTrigger API in Phase 1.5.

**Technical Considerations**:
- Wake word requires background audio session (battery impact)
- Continuous listening raises significant privacy concerns
- Apple restricts always-on microphone access for good reason
- Consider ShortcutKit for Siri integration as workaround

---

### 1.2 Voice Personas for Axel

| Feature | Description | Priority | Complexity | User Value | Phase |
|---------|-------------|----------|------------|------------|-------|
| **Voice Selection** | Choose Axel's voice (warm/energetic/calm presets) | Nice-to-Have | Low | Medium | v1.5 |
| **ElevenLabs Premium Voice** | High-quality, natural AI voice | Nice-to-Have | Medium | High | v2 |
| **Custom Voice Training** | User-trained voice model (their therapist, mentor) | Wild Idea | Very High | Medium | Future |
| **Contextual Tone Adaptation** | Calmer voice for evening, energetic for morning | Nice-to-Have | Medium | Medium | v2 |
| **Speaking Speed Preference** | User-adjustable speech rate | Must-Have | Low | High | MVP |
| **Accent/Language Options** | Regional accents, multilingual Axel | Nice-to-Have | Medium | Medium | v2 |

**MVP Implementation**: Use Apple AVSpeechSynthesizer with enhanced voices. Allow speaking rate adjustment in settings.

**Recommendation**: ElevenLabs integration for v2 as a premium feature. The voice quality difference is significant and worth the API cost for paid users.

---

### 1.3 Tone/Emotion Detection

| Feature | Description | Priority | Complexity | User Value | Phase |
|---------|-------------|----------|------------|------------|-------|
| **Stress Detection** | Detect stressed speech patterns, adjust Axel's response | Nice-to-Have | High | High | v2 |
| **Energy Level Detection** | Recognize tired vs energetic speech | Nice-to-Have | High | Medium | v2 |
| **Excitement Detection** | Notice when user is excited about something | Nice-to-Have | High | Medium | v2 |
| **Frustration Recognition** | Detect frustration and offer support | Nice-to-Have | High | High | v2 |
| **Speech Pattern Learning** | Learn user's baseline for accurate detection | Nice-to-Have | Very High | Medium | Future |

**Technical Approach**:
- Apple's Sound Analysis framework can detect speech patterns
- Core ML model for prosody analysis (pitch, tempo, volume)
- Significant R&D investment - defer to v2 minimum

**User Value Assessment**: High for executive function support users. Knowing "you sound stressed, want to just capture this quickly and come back to it?" could be powerful.

---

### 1.4 Multi-Speaker Detection

| Feature | Description | Priority | Complexity | User Value | Phase |
|---------|-------------|----------|------------|------------|-------|
| **Speaker Diarization** | Distinguish between user and others in recording | Nice-to-Have | High | Medium | Future |
| **Meeting Mode** | Capture thoughts during meetings, attribute to speakers | Nice-to-Have | Very High | Medium | Future |
| **Family Mode** | Multiple family members can use shared device | Nice-to-Have | High | Low | Future |

**Assessment**: Low priority for MVP. MYND is personal thought capture, not meeting transcription. Consider if user research indicates demand.

---

## 2. Capture Innovation

### 2.1 Photo & Visual Capture

| Feature | Description | Priority | Complexity | User Value | Phase |
|---------|-------------|----------|------------|------------|-------|
| **Whiteboard Capture** | Photo of whiteboard, OCR + structure extraction | Must-Have | Medium | High | v1.5 |
| **Handwritten Note Scan** | OCR for handwritten notes | Must-Have | Medium | High | v1.5 |
| **Screenshot Understanding** | Paste screenshot, AI extracts meaning | Must-Have | Medium | High | v1.5 |
| **Business Card Scan** | Extract contact info, create Person node | Nice-to-Have | Medium | Medium | v2 |
| **Document Summarization** | Upload PDF, AI summarizes key points | Nice-to-Have | Medium | High | v2 |
| **Mind Map Photo Import** | Photo of hand-drawn mind map, convert to nodes | Nice-to-Have | High | Medium | v2 |
| **Real-time AR Annotation** | Point camera, speak annotation, saves with photo | Wild Idea | Very High | Low | Future |

**MVP Implementation**: Vision framework for OCR is mature. Claude API with vision can handle most extraction. This is high-value, moderate-complexity.

**Technical Stack**:
- VNRecognizeTextRequest for OCR
- Claude 3 vision API for semantic understanding
- PhotoKit for camera/library access

**User Story**: "I jotted ideas on a napkin. Snap a photo, MYND extracts the thoughts and links them to my current project."

---

### 2.2 Share Sheet & Widget Capture

| Feature | Description | Priority | Complexity | User Value | Phase |
|---------|-------------|----------|------------|------------|-------|
| **Share Sheet Extension** | Share URLs, text, images from any app | Must-Have | Low | High | MVP+ |
| **Lock Screen Widget** | Quick capture button on lock screen | Must-Have | Low | High | MVP |
| **Home Screen Widgets** | Multiple sizes: quick capture, recent thoughts, goal progress | Must-Have | Medium | High | MVP |
| **Control Center Toggle** | Start voice capture from Control Center | Nice-to-Have | Low | Medium | v1.5 |
| **Spotlight Integration** | Search thoughts from system Spotlight | Nice-to-Have | Medium | Medium | v2 |
| **Clipboard Monitoring** | Offer to capture copied text | Wild Idea | Medium | Low | Future |

**MVP Must-Have**:
- Lock Screen widget for instant capture (iOS 16+ Lock Screen widgets)
- Small/Medium Home Screen widgets
- Share Sheet extension for text/URLs

**Technical Note**: Share Sheet extensions have their own app lifecycle. Keep light - queue to main app for processing.

---

### 2.3 Audio Context & Environment

| Feature | Description | Priority | Complexity | User Value | Phase |
|---------|-------------|----------|------------|------------|-------|
| **Location Tagging** | Auto-tag thoughts with location (optional) | Nice-to-Have | Low | Medium | v1.5 |
| **Time-of-Day Context** | Morning vs evening influences Axel's approach | Nice-to-Have | Low | High | MVP |
| **Calendar Context Awareness** | "You have a meeting in 10 minutes" | Nice-to-Have | Medium | High | v2 |
| **Background Audio Detection** | Note if user is in car, coffee shop, etc. | Wild Idea | High | Low | Future |
| **Workout Detection** | Know when user is exercising for appropriate prompts | Nice-to-Have | Low | Low | v2 |

**Recommendation**: Start with simple time-of-day context (morning briefing vs evening reflection tone). Calendar integration high-value for v2.

---

## 3. Knowledge Graph Visualization

### 3.1 3D Mind Map

| Feature | Description | Priority | Complexity | User Value | Phase |
|---------|-------------|----------|------------|------------|-------|
| **Interactive 3D Graph** | Rotate, zoom, explore thought connections | Nice-to-Have | High | High | v2 |
| **Force-Directed Layout** | Nodes naturally cluster by relationships | Nice-to-Have | High | High | v2 |
| **AR Graph View** | View your thoughts in physical space | Wild Idea | Very High | Medium | Future |
| **Pinch-to-Navigate Clusters** | Semantic zoom into topic areas | Nice-to-Have | Medium | Medium | v2 |
| **Node Size = Importance** | More connected nodes appear larger | Nice-to-Have | Low | Medium | v1.5 |

**Technical Approach**:
- SceneKit or RealityKit for 3D rendering
- Force-directed layout algorithms (D3.js ports exist)
- Consider SpriteKit for 2D alternative (less complexity)

**Recommendation**: Start with 2D graph in v1.5 (radial layout). 3D adds significant complexity. Test if users actually want 3D vs good 2D.

---

### 3.2 Timeline & History Views

| Feature | Description | Priority | Complexity | User Value | Phase |
|---------|-------------|----------|------------|------------|-------|
| **Thought Timeline** | Chronological view of all captures | Must-Have | Low | High | MVP |
| **Evolution View** | See how a goal/idea developed over time | Nice-to-Have | Medium | High | v2 |
| **Comparison Mode** | Side-by-side: what you thought then vs now | Nice-to-Have | Medium | Medium | v2 |
| **Calendar Heat Map** | GitHub-style: which days had most thought activity | Nice-to-Have | Low | Medium | v1.5 |
| **Replay Mode** | Replay a past conversation session | Nice-to-Have | Low | Medium | v1.5 |

**MVP Must-Have**: Simple timeline/list of recent captures. More advanced views in later phases.

---

### 3.3 Cluster & Topic Views

| Feature | Description | Priority | Complexity | User Value | Phase |
|---------|-------------|----------|------------|------------|-------|
| **Topic Clusters** | Auto-grouped thoughts by topic | Must-Have | Medium | High | v1.5 |
| **Project View** | All nodes related to a project | Must-Have | Low | High | MVP |
| **Person View** | Everything related to a specific person | Nice-to-Have | Low | High | v1.5 |
| **Stale Items View** | Thoughts not touched in X days | Must-Have | Low | High | MVP |
| **Relationship Strength Visualization** | Thicker lines = stronger connections | Nice-to-Have | Low | Medium | v2 |
| **Cluster Discovery** | AI suggests "these thoughts seem related" | Nice-to-Have | Medium | High | v2 |

**MVP Implementation**: Simple list views filtered by nodeType. Graph visualization comes in Phase 2.

---

## 4. Proactive AI Features

### 4.1 Scheduled Check-ins

| Feature | Description | Priority | Complexity | User Value | Phase |
|---------|-------------|----------|------------|------------|-------|
| **Morning Briefing** | "Good morning! Here's what's on your mind today..." | Must-Have | Medium | Very High | v1.5 |
| **End-of-Day Reflection** | "How did today go? Anything to capture?" | Must-Have | Medium | Very High | v1.5 |
| **Weekly Review** | Summary of week's thoughts, patterns, progress | Must-Have | Medium | Very High | v2 |
| **Monthly Insights** | Deeper patterns, goal progress, thinking evolution | Nice-to-Have | Medium | High | v2 |
| **Custom Check-in Schedule** | User-defined reminder times | Must-Have | Low | High | v1.5 |

**Implementation Priority**: Morning briefing and evening reflection are core differentiators. Should be in Phase 4 (Proactive Features).

**UX Consideration**: Notifications must be actionable. "Good morning! You mentioned wanting to finish the proposal. Ready to work on it?" with quick action buttons.

---

### 4.2 Contextual Follow-ups

| Feature | Description | Priority | Complexity | User Value | Phase |
|---------|-------------|----------|------------|------------|-------|
| **"You mentioned X, updates?"** | Follow up on mentioned but not-resolved items | Must-Have | Medium | Very High | v2 |
| **Deadline Approaching** | "Your goal X is due in 3 days, how's it going?" | Must-Have | Medium | High | v2 |
| **Pattern Interruption** | "You usually check in around now but haven't today" | Nice-to-Have | Medium | Medium | v2 |
| **Celebration Triggers** | "You completed 3 actions this week, nice work!" | Nice-to-Have | Low | High | v1.5 |
| **Stale Goal Nudge** | "It's been 2 weeks since you mentioned X. Still relevant?" | Must-Have | Medium | High | v2 |
| **Connection Suggestions** | "This new thought might relate to your project Y" | Nice-to-Have | High | Medium | v2 |

**Core Insight**: Proactive features are MYND's killer differentiator. Most thought capture apps are passive. Axel should gently, helpfully initiate.

---

### 4.3 Insight Generation

| Feature | Description | Priority | Complexity | User Value | Phase |
|---------|-------------|----------|------------|------------|-------|
| **Weekly Thinking Patterns** | "You've been focused on work, less on health this week" | Nice-to-Have | Medium | High | v2 |
| **Goal Velocity Tracking** | "You make faster progress on creative vs admin goals" | Nice-to-Have | High | Medium | v2 |
| **Energy Pattern Detection** | "You capture more ideas in the morning" | Nice-to-Have | Medium | Medium | v2 |
| **Topic Trend Analysis** | "You've mentioned X topic 5x more this month" | Nice-to-Have | Medium | Medium | v2 |
| **Blocker Pattern Recognition** | "Admin tasks tend to block your creative goals" | Nice-to-Have | High | High | Future |
| **Decision Outcome Review** | "Last month you decided X, here's what happened" | Nice-to-Have | Medium | High | v2 |

**LLM Opportunity**: These insights leverage Claude's analytical capabilities well. Generate weekly/monthly insight prompts and deliver via notification or morning briefing.

---

## 5. Executive Function Support

### 5.1 Decision Paralysis Mode

| Feature | Description | Priority | Complexity | User Value | Phase |
|---------|-------------|----------|------------|------------|-------|
| **"Just One Thing" Mode** | Hide everything except the single next action | Must-Have | Low | Very High | MVP |
| **Random Action Selection** | "I'll pick something for you. How about X?" | Nice-to-Have | Low | High | v1.5 |
| **Energy-Based Suggestions** | "Low energy? Here's something small you can do" | Nice-to-Have | Medium | High | v2 |
| **2-Minute Actions View** | Filter to only quick wins | Must-Have | Low | High | MVP |
| **Decision Timer** | "You have 30 seconds to decide, or I'll pick" | Wild Idea | Low | Medium | v2 |

**Core User Need**: Users with executive function challenges often know what to do but can't start. MYND should reduce choice paralysis, not add to it.

**UX Principle**: The default view should be minimal. One thing. Not a overwhelming list.

---

### 5.2 Energy & Focus Check-ins

| Feature | Description | Priority | Complexity | User Value | Phase |
|---------|-------------|----------|------------|------------|-------|
| **Energy Level Check** | "How's your energy? High/Medium/Low" | Nice-to-Have | Low | High | v1.5 |
| **Task Matching** | Suggest high-effort tasks for high-energy times | Nice-to-Have | Medium | High | v2 |
| **Focus Session Tracking** | "You've been focused for 45 min, take a break?" | Nice-to-Have | Medium | Medium | v2 |
| **Mood Logging** | Quick emoji mood capture | Nice-to-Have | Low | Medium | v1.5 |
| **Productivity Patterns** | "You're most productive Tuesday afternoons" | Nice-to-Have | Medium | Medium | v2 |

**Integration Opportunity**: Health app integration for sleep data could inform energy estimates.

---

### 5.3 Goal Breakdown & Progress

| Feature | Description | Priority | Complexity | User Value | Phase |
|---------|-------------|----------|------------|------------|-------|
| **AI Goal Breakdown** | "That's big. Let's break it into 5 steps." | Must-Have | Medium | Very High | v1.5 |
| **Next Action Clarity** | Always show the ONE next physical action | Must-Have | Low | Very High | MVP |
| **Progress Visualization** | Simple progress bar or completion percentage | Must-Have | Low | High | v1.5 |
| **Tiny Steps Mode** | Break actions down to 5-minute chunks | Nice-to-Have | Medium | High | v2 |
| **"Just Start" Prompt** | "Can you do just 2 minutes on this?" | Nice-to-Have | Low | High | v1.5 |
| **Effort Estimation** | Tag tasks with estimated effort | Nice-to-Have | Low | Medium | v1.5 |

**Core Philosophy**: GTD-inspired - make the next action so small and clear it's impossible not to start.

---

### 5.4 Celebration & Positive Reinforcement

| Feature | Description | Priority | Complexity | User Value | Phase |
|---------|-------------|----------|------------|------------|-------|
| **Completion Celebration** | Satisfying UI/sound when action completed | Must-Have | Low | High | MVP |
| **Streak Recognition** | "5 days of capturing thoughts, nice!" | Nice-to-Have | Low | Medium | v1.5 |
| **Milestone Moments** | "You've captured 100 thoughts!" | Nice-to-Have | Low | Medium | v1.5 |
| **Weekly Wins Summary** | "This week you completed: X, Y, Z" | Nice-to-Have | Low | High | v1.5 |
| **Encouraging Language** | Axel celebrates small wins genuinely | Must-Have | Low | High | MVP |
| **Sharing Wins** | Option to share accomplishments | Nice-to-Have | Medium | Low | Future |

**UX Note**: Avoid over-gamification. Target users may find excessive badges/points stressful. Subtle, genuine celebration is better.

---

## 6. Apple Ecosystem Integration

### 6.1 Apple Watch

| Feature | Description | Priority | Complexity | User Value | Phase |
|---------|-------------|----------|------------|------------|-------|
| **Quick Voice Capture** | Tap to capture a thought from wrist | Must-Have | Medium | Very High | v2 |
| **Complication** | Show current goal or quick capture button | Must-Have | Medium | High | v2 |
| **Voice Reply from Watch** | Respond to Axel prompts via Watch | Nice-to-Have | High | Medium | v2 |
| **Haptic Reminders** | Gentle tap for scheduled check-ins | Nice-to-Have | Low | High | v2 |
| **Walking Brainstorm Mode** | Capture ideas while walking | Nice-to-Have | Medium | High | v2 |

**Priority**: Watch app should be in v2. Quick capture is the killer use case - "thought capture in 3 seconds from your wrist."

---

### 6.2 CarPlay

| Feature | Description | Priority | Complexity | User Value | Phase |
|---------|-------------|----------|------------|------------|-------|
| **Driving Mode** | Full voice interaction while driving | Nice-to-Have | High | High | v2 |
| **Commute Capture** | "What's on your mind for today?" during commute | Nice-to-Have | High | High | v2 |
| **End-of-Day Car Prompt** | "Leaving work. Anything to capture before home?" | Nice-to-Have | Medium | Medium | v2 |
| **Audio Playback** | Listen to your captured thoughts | Nice-to-Have | Medium | Medium | v2 |

**Use Case**: Commute time is often thinking time. CarPlay integration could be powerful for capture.

**Technical Note**: CarPlay development requires Apple approval and has specific UI constraints.

---

### 6.3 Mac Integration

| Feature | Description | Priority | Complexity | User Value | Phase |
|---------|-------------|----------|------------|------------|-------|
| **Mac App (Catalyst or native)** | Full MYND experience on Mac | Must-Have | High | High | v2 |
| **Menu Bar Widget** | Quick capture from menu bar | Must-Have | Medium | Very High | v2 |
| **Keyboard Shortcut Capture** | Cmd+Shift+M to capture thought | Must-Have | Low | High | v2 |
| **Selection Capture** | Select text, right-click, add to MYND | Nice-to-Have | Medium | High | v2 |
| **Screen Content Capture** | Capture what you're working on with context | Nice-to-Have | High | Medium | Future |

**Architecture Note**: SwiftUI enables shared code between iOS and macOS. Menu bar widget should be priority for v2.

---

### 6.4 System Integration

| Feature | Description | Priority | Complexity | User Value | Phase |
|---------|-------------|----------|------------|------------|-------|
| **Focus Mode Integration** | Different Axel behavior per Focus mode | Nice-to-Have | Low | Medium | v1.5 |
| **Calendar Integration** | Pull events for context, suggest captures | Nice-to-Have | Medium | High | v2 |
| **Reminders Integration** | Sync actions to Apple Reminders | Nice-to-Have | Medium | Medium | v2 |
| **Health App Integration** | Pull sleep/activity for energy estimates | Nice-to-Have | Medium | Medium | v2 |
| **Siri Shortcuts** | "Hey Siri, capture a thought in MYND" | Must-Have | Low | High | v1.5 |
| **Handoff Support** | Start on iPhone, continue on Mac | Nice-to-Have | Medium | Medium | v2 |

**MVP Must-Have**: Siri Shortcuts is low-effort, high-value. Single shortcut: "Capture thought in MYND."

---

## 7. Gamification (Subtle)

### 7.1 Progress & Streaks

| Feature | Description | Priority | Complexity | User Value | Phase |
|---------|-------------|----------|------------|------------|-------|
| **Capture Streak** | Days in a row of thought capture | Nice-to-Have | Low | Medium | v1.5 |
| **Weekly Goal Progress** | Visual progress toward weekly targets | Nice-to-Have | Low | Medium | v1.5 |
| **Thinking Momentum** | "You're on a roll - 3 thoughts in 10 minutes" | Nice-to-Have | Low | Low | v2 |
| **Streak Recovery** | "You missed yesterday. That's ok, start fresh today" | Nice-to-Have | Low | High | v1.5 |

---

### 7.2 Milestones & Achievements

| Feature | Description | Priority | Complexity | User Value | Phase |
|---------|-------------|----------|------------|------------|-------|
| **Thought Count Milestones** | 10, 50, 100, 500 thoughts captured | Nice-to-Have | Low | Medium | v1.5 |
| **Goal Completion Badges** | Celebrate finished goals | Nice-to-Have | Low | Medium | v2 |
| **Connection Explorer** | "You've made 100 connections between thoughts!" | Nice-to-Have | Low | Low | v2 |
| **Pattern Recognition Unlock** | "Axel learned a new pattern about you" | Nice-to-Have | Low | Medium | v2 |

---

### 7.3 Insights Unlocked

| Feature | Description | Priority | Complexity | User Value | Phase |
|---------|-------------|----------|------------|------------|-------|
| **Weekly Insight Card** | Special insight unlocked each week | Nice-to-Have | Medium | High | v2 |
| **Thinking Evolution** | "One month ago you were focused on X, now Y" | Nice-to-Have | Medium | High | v2 |
| **Hidden Connections** | "AI found a surprising link between X and Y" | Nice-to-Have | High | High | v2 |

**Gamification Philosophy**: Keep subtle. Users with executive function challenges may find aggressive gamification anxiety-inducing. Gentle acknowledgment > dopamine manipulation.

---

## 8. Wild Ideas Section

These are experimental, high-risk, high-potential ideas for future exploration:

### 8.1 Voice & Conversation

| Idea | Description | Why Wild | Potential |
|------|-------------|----------|-----------|
| **Axel Debates You** | Play devil's advocate on decisions | Could frustrate users | Very High for decision quality |
| **Drunk Ideas Mode** | Late-night capture with AI that's also "loose" | Legal/brand risk | Cult feature potential |
| **Voice Clone for Self-Talk** | Hear your past self's advice | Uncanny valley risk | High for reflection |
| **Group Axel** | Shared Axel for couples/teams | Complexity explosion | High if social features wanted |

### 8.2 Capture & Understanding

| Idea | Description | Why Wild | Potential |
|------|-------------|----------|-----------|
| **Dream Capture** | Morning prompt to capture dreams before forgotten | Niche audience | Medium, devoted users |
| **Ambient Life Logging** | Always-on audio with AI summarization | Privacy nightmare | High if trust established |
| **Biometric Capture Trigger** | Heart rate spike triggers "what's happening?" | Creepy factor | High for emotional awareness |
| **Predictive Capture** | "I bet you're thinking about X right now" | Wrong guesses frustrating | High when right |

### 8.3 Visualization & Understanding

| Idea | Description | Why Wild | Potential |
|------|-------------|----------|-----------|
| **VisionOS Thought Space** | Full 3D room of your thoughts in VR | Hardware dependency | Very High long-term |
| **Thought DNA Visualization** | Unique visual fingerprint of your thinking | Unclear utility | Medium, cool factor |
| **Time Travel Mode** | See your knowledge graph at any past date | Complex storage | High for reflection |
| **AI-Generated Thought Art** | Visualize thoughts as AI-generated images | Resource intensive | Medium, shareable |

### 8.4 Proactive & Social

| Idea | Description | Why Wild | Potential |
|------|-------------|----------|-----------|
| **Axel Therapy Mode** | Deeper psychological support | Liability concerns | Very High need, risky |
| **Mentor Matching** | Connect with people who've solved similar goals | Social platform scope creep | High if executed |
| **Thought Marketplace** | Share frameworks/templates with others | Monetization complexity | Medium |
| **Dead Man's Switch** | Loved ones receive your thoughts if inactive | Morbid, complex | Niche but meaningful |

---

## 9. Phase Recommendations

### MVP (Phase 1-4, Weeks 1-16)

Core must-haves for initial App Store launch:

**Voice**:
- Tap-to-talk + voice interruption
- Speaking speed preference
- Time-of-day context for Axel's tone

**Capture**:
- Lock Screen widget
- Home Screen widgets (small/medium)
- Thought timeline view

**Knowledge Graph**:
- Project view (filter by project)
- Stale items view
- Simple list-based navigation

**Proactive**:
- Basic notifications for follow-ups
- Celebration on completion

**Executive Function**:
- "Just One Thing" mode
- 2-minute actions view
- Next action clarity
- Encouraging language from Axel

---

### v1.5 (Weeks 17-24)

Post-launch quick wins:

**Voice**:
- Wake word ("Hey Axel")
- Voice selection (presets)

**Capture**:
- Share Sheet extension
- Photo/whiteboard OCR
- Screenshot understanding

**Knowledge Graph**:
- Calendar heat map
- Topic clusters
- Node importance sizing

**Proactive**:
- Morning briefing
- End-of-day reflection
- Custom check-in schedule
- Celebration triggers

**Executive Function**:
- AI goal breakdown
- Progress visualization
- "Just Start" prompts
- Energy level check

**Gamification**:
- Capture streaks (gentle)
- Milestones

**Apple Ecosystem**:
- Siri Shortcuts
- Focus Mode integration

---

### v2 (Months 4-8)

Major expansion release:

**Voice**:
- ElevenLabs premium voice
- Hands-free navigation
- Contextual tone adaptation
- Emotion/stress detection (research phase)

**Capture**:
- Business card scan
- Document summarization

**Knowledge Graph**:
- 2D interactive graph visualization
- Evolution view (idea over time)
- Relationship strength visualization
- Cluster discovery

**Proactive**:
- "You mentioned X, updates?"
- Deadline approaching notifications
- Weekly review
- Monthly insights
- Stale goal nudge
- Weekly thinking patterns

**Executive Function**:
- Energy-based suggestions
- Tiny steps mode
- Task matching by energy
- Focus session tracking

**Apple Ecosystem**:
- Apple Watch app + complications
- Mac app + menu bar
- CarPlay mode
- Calendar integration
- Handoff support

---

### Future (Year 2+)

Long-term vision:

- Continuous/ambient listening (with privacy framework)
- VisionOS/spatial computing
- Advanced emotion detection
- Team/family shared graphs
- Custom voice training
- AR thought visualization
- Predictive capture

---

## 10. Complexity vs. Value Matrix

### High Value, Low Complexity (Do First)

1. "Just One Thing" mode
2. Lock Screen widget
3. Siri Shortcuts
4. Speaking speed preference
5. Celebration on completion
6. 2-minute actions view
7. Capture streaks

### High Value, Medium Complexity (Core Roadmap)

1. Morning briefing
2. End-of-day reflection
3. AI goal breakdown
4. Wake word activation
5. Photo/OCR capture
6. Share Sheet extension
7. Apple Watch quick capture
8. Menu bar widget (Mac)

### High Value, High Complexity (Invest Carefully)

1. Interactive graph visualization
2. Emotion/stress detection
3. ElevenLabs integration
4. CarPlay mode
5. Weekly thinking patterns

### Lower Priority (Nice-to-Have)

1. AR annotations
2. Multi-speaker detection
3. Ambient listening
4. Custom voice training
5. VisionOS app

---

## 11. Technical Dependencies

### Required for MVP
- Apple Speech Framework (on-device STT)
- AVSpeechSynthesizer (TTS)
- Claude API (conversation)
- SwiftData (persistence)
- WidgetKit (widgets)
- UserNotifications (proactive features)

### Required for v1.5
- Vision Framework (OCR)
- Claude Vision API (image understanding)
- AppIntents (Siri Shortcuts)
- Share Extension framework

### Required for v2
- WatchKit / WatchConnectivity
- Catalyst or SwiftUI for Mac
- EventKit (calendar)
- HealthKit (energy estimates)
- Optional: ElevenLabs API

### Research Needed
- Sound Analysis framework (emotion detection)
- Custom Core ML models (prosody analysis)
- SceneKit/RealityKit (3D visualization)
- CarPlay framework

---

## 12. Summary

MYND's feature set should prioritize:

1. **Capture simplicity** - Reduce friction to zero
2. **Proactive intelligence** - Axel initiates helpfully
3. **Executive function support** - Reduce overwhelm, increase action
4. **Apple ecosystem depth** - Be the best on Apple platforms

The wild ideas section contains seeds for future differentiation, but the core value is in making thought capture and action feel effortless for people who struggle with traditional productivity tools.

**Next Steps**:
1. Validate MVP feature set with user interviews
2. Prototype "Just One Thing" mode and morning briefing
3. Test voice interaction latency and quality
4. Design minimal, calming UI that reduces anxiety

---

*Document generated: 2026-01-04*
*Status: Feature brainstorm for team review*
