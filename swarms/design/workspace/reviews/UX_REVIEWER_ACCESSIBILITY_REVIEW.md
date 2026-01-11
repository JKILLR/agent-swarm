# AI Corp - UX & Accessibility Audit

**Reviewer:** UX Accessibility Specialist
**Date:** 2026-01-05
**Document Version:** 1.0
**WCAG Target:** 2.2 AA Compliance

---

## Executive Summary

This audit evaluates the AI Corp system from a user experience and accessibility perspective. The system presents unique UX challenges due to its hierarchical multi-agent architecture, complex workflow visualization needs, and real-time communication patterns. This document provides comprehensive guidelines for ensuring the interface is accessible, usable, and inclusive.

---

## 1. User Journey Maps

### 1.1 CEO Dashboard Journey

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CEO DASHBOARD USER JOURNEY                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ENTRY POINT                                                                │
│  ────────────                                                               │
│  User arrives at dashboard → System loads org overview                      │
│                                                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │  LOGIN   │───▶│ OVERVIEW │───▶│  DRILL   │───▶│  ACTION  │              │
│  │          │    │          │    │  DOWN    │    │          │              │
│  │ Auth     │    │ KPIs     │    │ Details  │    │ Delegate │              │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘              │
│       │              │               │               │                      │
│       ▼              ▼               ▼               ▼                      │
│  • SSO/Auth     • Active        • Molecule      • Create task              │
│  • 2FA          • Molecules      details       • Assign work               │
│  • Session      • Pipeline      • Agent        • Set priority              │
│    restore        status          status       • Approve gates             │
│                 • Blockers      • History                                   │
│                 • Alerts        • Logs                                      │
│                                                                             │
│  PAIN POINTS TO ADDRESS:                                                    │
│  ─────────────────────────                                                  │
│  • Information overload on initial load                                     │
│  • Complex hierarchy difficult to visualize                                 │
│  • Real-time updates can be disorienting                                   │
│  • Gate approval workflow unclear                                           │
│                                                                             │
│  ACCESSIBILITY CONSIDERATIONS:                                              │
│  ─────────────────────────────                                              │
│  • Screen reader must announce state changes                                │
│  • Focus management during real-time updates                                │
│  • Keyboard shortcuts for power users                                       │
│  • Color-independent status indicators                                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Task Creation Journey

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      TASK CREATION USER JOURNEY                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │  INIT    │───▶│  DEFINE  │───▶│  ASSIGN  │───▶│ CONFIRM  │              │
│  │          │    │          │    │          │    │          │              │
│  │ "+New"   │    │ Details  │    │ RACI     │    │ Review   │              │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘              │
│       │              │               │               │                      │
│       ▼              ▼               ▼               ▼                      │
│  • Click/Kbd    • Title         • Accountable   • Summary view             │
│  • Focus trap   • Description   • Responsible   • Edit option              │
│  • Modal open   • Priority      • Consulted     • Submit                   │
│                 • Dependencies  • Informed      • Success feedback         │
│                                                                             │
│  FORM REQUIREMENTS:                                                         │
│  ──────────────────                                                         │
│  • Clear field labels (visible, not placeholder-only)                       │
│  • Required field indicators (*) with legend                                │
│  • Inline validation with descriptive errors                                │
│  • Auto-save drafts every 30 seconds                                        │
│  • Confirmation before closing with unsaved changes                         │
│                                                                             │
│  STEP INDICATORS:                                                           │
│  ────────────────                                                           │
│  Step 1 of 4: Define Task → Step 2 of 4: Set RACI → ...                    │
│  (Must be announced by screen readers on step change)                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.3 Pipeline Monitoring Journey

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     PIPELINE MONITORING USER JOURNEY                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  KANBAN VIEW                                                                │
│  ───────────                                                                │
│                                                                             │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐           │
│  │ INBOX   │  │RESEARCH │  │ DESIGN  │  │  BUILD  │  │   QA    │           │
│  │         │  │         │  │         │  │         │  │         │           │
│  │ ┌─────┐ │  │ ┌─────┐ │  │ ┌─────┐ │  │ ┌─────┐ │  │ ┌─────┐ │           │
│  │ │Card │ │  │ │Card │ │  │ │Card │ │  │ │Card │ │  │ │Card │ │           │
│  │ └─────┘ │  │ └─────┘ │  │ └─────┘ │  │ └─────┘ │  │ └─────┘ │           │
│  │ ┌─────┐ │  │         │  │ ┌─────┐ │  │ ┌─────┐ │  │         │           │
│  │ │Card │ │  │         │  │ │Card │ │  │ │Card │ │  │         │           │
│  │ └─────┘ │  │         │  │ └─────┘ │  │ └─────┘ │  │         │           │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘           │
│       │            │            │            │            │                 │
│       └────────────┴────────────┴────────────┴────────────┘                 │
│                              │                                              │
│                              ▼                                              │
│                    INTERACTIONS:                                            │
│                    • Drag & drop (mouse)                                    │
│                    • Arrow keys (keyboard)                                  │
│                    • Context menu (actions)                                 │
│                    • Double-click (details)                                 │
│                                                                             │
│  DRAG & DROP ACCESSIBILITY:                                                 │
│  ──────────────────────────                                                 │
│  • MUST have keyboard alternative (Space to pick up, arrows to move,       │
│    Space/Enter to drop)                                                     │
│  • Live region announces: "Card MOL-123 picked up. Use arrows to move,     │
│    Space to drop"                                                           │
│  • Drop zones highlight on focus                                            │
│  • Cancel with Escape                                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.4 Agent Communication Journey

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    AGENT COMMUNICATION USER JOURNEY                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  CHANNEL TYPES:                                                             │
│  ──────────────                                                             │
│  • Upchain (Worker → Director → VP → COO → CEO)                            │
│  • Downchain (CEO → COO → VP → Director → Worker)                          │
│  • Peer-to-peer (Same level communication)                                  │
│  • Broadcast (One to many)                                                  │
│                                                                             │
│  UI REPRESENTATION:                                                         │
│  ──────────────────                                                         │
│                                                                             │
│  ┌────────────────────────────────────────┐                                 │
│  │ MESSAGES                          [+]  │                                 │
│  ├────────────────────────────────────────┤                                 │
│  │ ↑ From: frontend_worker_01             │  ← Upchain indicator            │
│  │   "Completed Header component"         │                                 │
│  │   2 min ago                            │                                 │
│  ├────────────────────────────────────────┤                                 │
│  │ ↓ From: vp_engineering                 │  ← Downchain indicator          │
│  │   "Priority shift: Dashboard first"    │                                 │
│  │   15 min ago                           │                                 │
│  ├────────────────────────────────────────┤                                 │
│  │ ↔ From: design_director                │  ← Peer indicator               │
│  │   "Design specs attached"              │                                 │
│  │   1 hour ago                           │                                 │
│  └────────────────────────────────────────┘                                 │
│                                                                             │
│  NOTIFICATION REQUIREMENTS:                                                 │
│  ──────────────────────────                                                 │
│  • Unread count badge (aria-label="5 unread messages")                      │
│  • Sound optional (user preference)                                         │
│  • Toast notifications with aria-live="polite"                              │
│  • Priority messages use aria-live="assertive"                              │
│  • Do NOT auto-dismiss critical notifications                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.5 Gate Approval Journey

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       GATE APPROVAL USER JOURNEY                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │ PENDING  │───▶│  REVIEW  │───▶│ DECISION │───▶│ COMPLETE │              │
│  │          │    │          │    │          │    │          │              │
│  │ Awaiting │    │ Examine  │    │ Approve/ │    │ Feedback │              │
│  └──────────┘    └──────────┘    │ Reject   │    └──────────┘              │
│       │              │           └──────────┘          │                    │
│       ▼              ▼               │                 ▼                    │
│  • Gate list    • Artifacts      ┌───┴───┐       • Confirmation            │
│  • Priority     • Criteria       │       │       • Next steps              │
│  • Urgency      • History        ▼       ▼       • Audit log               │
│                                Approve  Reject                              │
│                                  │       │                                  │
│                                  ▼       ▼                                  │
│                              Continue  Request                              │
│                              pipeline  changes                              │
│                                                                             │
│  DECISION BUTTON REQUIREMENTS:                                              │
│  ──────────────────────────────                                             │
│  • "Approve" - Primary action (prominent styling)                           │
│  • "Request Changes" - Secondary action                                     │
│  • Confirmation dialog for both actions                                     │
│  • Required comment field for rejections                                    │
│  • Undo available for 30 seconds post-decision                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. WCAG 2.2 AA Accessibility Requirements

### 2.1 Perceivable

| Criterion | Requirement | Implementation |
|-----------|-------------|----------------|
| **1.1.1 Non-text Content** | All images, icons, charts have text alternatives | `alt` text for images; `aria-label` for icon buttons; data tables for charts |
| **1.2.1 Audio-only/Video-only** | Provide alternatives for media | Transcripts for any audio notifications |
| **1.3.1 Info and Relationships** | Structure conveyed programmatically | Semantic HTML; ARIA landmarks; proper heading hierarchy |
| **1.3.2 Meaningful Sequence** | Reading order matches visual order | DOM order = visual order; CSS doesn't break flow |
| **1.3.3 Sensory Characteristics** | Don't rely only on shape/color/location | "Click the green button" → "Click the Approve button (green)" |
| **1.4.1 Use of Color** | Color not sole means of conveying info | Icons + color for status; patterns + color for charts |
| **1.4.3 Contrast (Minimum)** | 4.5:1 for normal text; 3:1 for large text | See Section 5: Color Contrast |
| **1.4.4 Resize Text** | 200% zoom without loss of functionality | Responsive design; no horizontal scroll at 320px |
| **1.4.10 Reflow** | Content reflows at 400% zoom | Single column at narrow widths |
| **1.4.11 Non-text Contrast** | 3:1 for UI components and graphics | Borders, icons, focus indicators |
| **1.4.12 Text Spacing** | Adjustable line height, spacing | No content loss when spacing increased |
| **1.4.13 Content on Hover/Focus** | Hovercards dismissible, hoverable, persistent | Escape to dismiss; can hover tooltip; stays until dismissed |

### 2.2 Operable

| Criterion | Requirement | Implementation |
|-----------|-------------|----------------|
| **2.1.1 Keyboard** | All functionality via keyboard | Tab navigation; Enter/Space activation; Arrow keys in widgets |
| **2.1.2 No Keyboard Trap** | Focus can always move away | Modal focus trap with Escape to close |
| **2.1.4 Character Key Shortcuts** | Single-key shortcuts can be turned off | Settings toggle for keyboard shortcuts |
| **2.2.1 Timing Adjustable** | Time limits can be extended | Auto-logout warning with extend option |
| **2.2.2 Pause, Stop, Hide** | Moving content can be controlled | Pause button for real-time updates |
| **2.3.1 Three Flashes** | No content flashes >3 times/sec | Animations respect prefers-reduced-motion |
| **2.4.1 Bypass Blocks** | Skip to main content link | Skip link as first focusable element |
| **2.4.2 Page Titled** | Descriptive page titles | "Pipeline - AI Corp" not just "AI Corp" |
| **2.4.3 Focus Order** | Logical focus sequence | Left-to-right, top-to-bottom; modal before page |
| **2.4.4 Link Purpose** | Link text describes destination | "View MOL-123 details" not "Click here" |
| **2.4.6 Headings and Labels** | Descriptive headings/labels | "Active Molecules" not "List" |
| **2.4.7 Focus Visible** | Keyboard focus indicator visible | 2px outline; contrasting color; never hidden |
| **2.4.11 Focus Not Obscured** | Focus indicator not hidden by other content | Sticky headers don't cover focused items |
| **2.5.1 Pointer Gestures** | Multipoint gestures have alternatives | Pinch-to-zoom has +/- buttons |
| **2.5.2 Pointer Cancellation** | Down-event doesn't trigger; up-event does | Click on mouseup/touchend |
| **2.5.3 Label in Name** | Visible label in accessible name | Button "Submit" has aria-label="Submit task" |
| **2.5.4 Motion Actuation** | Motion-triggered functions have alternatives | Shake-to-refresh has refresh button |
| **2.5.7 Dragging Movements** | Drag operations have non-drag alternatives | Keyboard card moving; context menu "Move to..." |
| **2.5.8 Target Size** | Touch targets minimum 24x24px | Buttons, links, interactive elements |

### 2.3 Understandable

| Criterion | Requirement | Implementation |
|-----------|-------------|----------------|
| **3.1.1 Language of Page** | Page lang attribute set | `<html lang="en">` |
| **3.1.2 Language of Parts** | Content in other languages marked | `<span lang="ja">日本語</span>` |
| **3.2.1 On Focus** | Focus doesn't trigger context change | No auto-submit on focus |
| **3.2.2 On Input** | Input doesn't auto-change context | Warning before auto-navigation |
| **3.2.6 Consistent Help** | Help mechanism in consistent location | Help icon always in header, same position |
| **3.3.1 Error Identification** | Errors identified in text | "Email field: Invalid email format" |
| **3.3.2 Labels or Instructions** | Form fields have instructions | Labels + helper text where needed |
| **3.3.3 Error Suggestion** | Error messages suggest correction | "Use format: name@domain.com" |
| **3.3.4 Error Prevention** | Reversible/confirmable for important actions | Confirm before gate approval |
| **3.3.7 Redundant Entry** | Don't ask for same info twice | Auto-fill from previous entries |
| **3.3.8 Accessible Authentication** | No cognitive function tests | No CAPTCHAs; support password managers |

### 2.4 Robust

| Criterion | Requirement | Implementation |
|-----------|-------------|----------------|
| **4.1.1 Parsing** | Valid HTML | No duplicate IDs; proper nesting |
| **4.1.2 Name, Role, Value** | Custom components have ARIA | Custom dropdowns use combobox pattern |
| **4.1.3 Status Messages** | Status communicated without focus | `aria-live` for toast notifications |

---

## 3. Keyboard Navigation Plan

### 3.1 Global Shortcuts

| Shortcut | Action | Context |
|----------|--------|---------|
| `?` | Show keyboard shortcuts modal | Global |
| `/` | Focus search | Global |
| `g` then `d` | Go to Dashboard | Global |
| `g` then `p` | Go to Pipeline | Global |
| `g` then `m` | Go to Messages | Global |
| `g` then `s` | Go to Settings | Global |
| `Escape` | Close modal/Cancel action | Any overlay |
| `Tab` | Next focusable element | Global |
| `Shift+Tab` | Previous focusable element | Global |

### 3.2 Pipeline View Shortcuts

| Shortcut | Action | Context |
|----------|--------|---------|
| `←` `→` | Move between columns | Pipeline kanban |
| `↑` `↓` | Move between cards in column | Pipeline kanban |
| `Enter` | Open card details | Card focused |
| `Space` | Pick up card for moving | Card focused |
| `Space` (while holding) | Drop card | During move |
| `m` | Move card (opens column selector) | Card focused |
| `e` | Edit card | Card focused |
| `a` | Approve gate (if applicable) | Gate card focused |
| `r` | Request changes (if applicable) | Gate card focused |

### 3.3 Form Navigation

| Shortcut | Action | Context |
|----------|--------|---------|
| `Tab` | Next field | Form |
| `Shift+Tab` | Previous field | Form |
| `Enter` | Submit form (when on submit button) | Form |
| `Escape` | Cancel/Close form | Form modal |
| `Space` | Toggle checkbox/radio | Checkbox/Radio focused |
| `↑` `↓` | Navigate options | Select/Radio group |

### 3.4 Tree Navigation (Org Chart)

| Shortcut | Action | Context |
|----------|--------|---------|
| `↑` `↓` | Move between siblings | Org tree |
| `←` | Collapse/Move to parent | Expanded node |
| `→` | Expand/Move to first child | Collapsed node |
| `Home` | First node | Org tree |
| `End` | Last visible node | Org tree |
| `Enter` | Select/Activate node | Node focused |
| `*` | Expand all siblings | Org tree |

### 3.5 Focus Management Rules

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        FOCUS MANAGEMENT RULES                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  MODAL DIALOGS:                                                             │
│  ──────────────                                                             │
│  1. On open: Focus moves to first focusable element in modal                │
│  2. Focus trapped within modal (Tab cycles through modal only)              │
│  3. On close: Focus returns to triggering element                           │
│  4. Escape key closes modal                                                 │
│  5. Click outside closes modal (optional, user preference)                  │
│                                                                             │
│  PAGE NAVIGATION:                                                           │
│  ────────────────                                                           │
│  1. On route change: Focus moves to main content area                       │
│  2. Skip link appears on first Tab (hidden until focused)                   │
│  3. Page title announced by screen reader                                   │
│                                                                             │
│  DYNAMIC CONTENT:                                                           │
│  ────────────────                                                           │
│  1. New content doesn't steal focus unless user-initiated                   │
│  2. Deleted content: Focus moves to next logical element                    │
│  3. Error messages: Focus can optionally move to error (user setting)       │
│                                                                             │
│  INFINITE SCROLL:                                                           │
│  ────────────────                                                           │
│  1. Focus remains on current item during load                               │
│  2. Announce "Loading more items" via aria-live                             │
│  3. Announce "X new items loaded" when complete                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Screen Reader Considerations

### 4.1 ARIA Landmarks

```html
<body>
  <header role="banner">
    <!-- Logo, global nav, user menu -->
  </header>

  <nav role="navigation" aria-label="Main navigation">
    <!-- Primary navigation -->
  </nav>

  <main role="main" id="main-content">
    <!-- Page content -->
  </main>

  <aside role="complementary" aria-label="Activity feed">
    <!-- Sidebar content -->
  </aside>

  <footer role="contentinfo">
    <!-- Footer content -->
  </footer>
</body>
```

### 4.2 Heading Hierarchy

```
h1: Page Title (one per page)
  h2: Major Sections
    h3: Subsections
      h4: Sub-subsections (rarely needed)
```

**Example - Pipeline Page:**
```
h1: Pipeline Overview
  h2: Inbox (3 items)
  h2: Research (2 items)
  h2: Design (4 items)
  h2: Build (1 item)
  h2: QA (2 items)
  h2: Security (0 items)
  h2: Deploy (1 item)
```

### 4.3 Live Regions

```html
<!-- For toast notifications -->
<div aria-live="polite" aria-atomic="true" class="sr-only" id="notifications">
  <!-- Dynamically populated -->
</div>

<!-- For urgent alerts -->
<div aria-live="assertive" aria-atomic="true" class="sr-only" id="alerts">
  <!-- Dynamically populated -->
</div>

<!-- For status updates -->
<div role="status" aria-live="polite" id="status">
  <!-- "Saving...", "Saved", etc. -->
</div>

<!-- For progress updates -->
<div role="progressbar" aria-valuenow="70" aria-valuemin="0" aria-valuemax="100">
  70% complete
</div>
```

### 4.4 Component Announcements

| Component | Announcement Pattern |
|-----------|---------------------|
| **Pipeline Card** | "[Priority] [Title]. Status: [Status]. Assigned to: [Agent]. [X] days in stage." |
| **Gate Pending** | "Gate: [Name]. Awaiting approval. [X] items pending review." |
| **Agent Status** | "[Agent name]. Status: [Working/Idle/Blocked]. Current task: [Task or None]." |
| **Message** | "[Direction: Upchain/Downchain/Peer] from [Sender]. [Preview]. [Time]." |
| **Org Node** | "[Role]: [Name]. [X] direct reports. [Expanded/Collapsed]." |

### 4.5 Form Accessibility

```html
<!-- Required field pattern -->
<div class="form-field">
  <label for="task-title">
    Task Title
    <span aria-hidden="true">*</span>
    <span class="sr-only">(required)</span>
  </label>
  <input
    type="text"
    id="task-title"
    name="title"
    required
    aria-required="true"
    aria-describedby="title-hint title-error"
  >
  <span id="title-hint" class="hint">Brief description of the task</span>
  <span id="title-error" class="error" role="alert" aria-live="polite">
    <!-- Error message appears here -->
  </span>
</div>

<!-- Select with custom styling -->
<div class="custom-select">
  <label id="priority-label">Priority</label>
  <div
    role="combobox"
    aria-labelledby="priority-label"
    aria-expanded="false"
    aria-haspopup="listbox"
    aria-controls="priority-options"
    tabindex="0"
  >
    <span>Select priority</span>
  </div>
  <ul
    id="priority-options"
    role="listbox"
    aria-labelledby="priority-label"
    hidden
  >
    <li role="option" id="opt-high">High</li>
    <li role="option" id="opt-medium">Medium</li>
    <li role="option" id="opt-low">Low</li>
  </ul>
</div>
```

### 4.6 Table Accessibility

```html
<!-- Data table for agent status -->
<table>
  <caption>Engineering Department Agent Status</caption>
  <thead>
    <tr>
      <th scope="col">Agent</th>
      <th scope="col">Role</th>
      <th scope="col">Status</th>
      <th scope="col">Current Task</th>
      <th scope="col">Actions</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th scope="row">frontend_worker_01</th>
      <td>Frontend Worker</td>
      <td>
        <span class="status-badge status-working">
          <span aria-hidden="true">●</span>
          Working
        </span>
      </td>
      <td>MOL-123: Build Header</td>
      <td>
        <button aria-label="View details for frontend_worker_01">View</button>
        <button aria-label="Reassign frontend_worker_01">Reassign</button>
      </td>
    </tr>
  </tbody>
</table>
```

---

## 5. Color Contrast Requirements

### 5.1 Minimum Contrast Ratios

| Element Type | Minimum Ratio | Notes |
|--------------|---------------|-------|
| Normal text (<18px) | 4.5:1 | Body text, labels, buttons |
| Large text (≥18px bold or ≥24px) | 3:1 | Headings, large buttons |
| UI Components | 3:1 | Borders, icons, focus indicators |
| Graphical Objects | 3:1 | Charts, diagrams, icons |
| Disabled elements | No requirement | But should be distinguishable |

### 5.2 Color Palette with Contrast

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         COLOR PALETTE - ACCESSIBLE                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  BACKGROUNDS:                                                               │
│  ─────────────                                                              │
│  Primary Background:     #FFFFFF (White)                                    │
│  Secondary Background:   #F5F5F5 (Light Gray)                               │
│  Dark Background:        #1A1A1A (Near Black)                               │
│                                                                             │
│  TEXT COLORS (on white background):                                         │
│  ──────────────────────────────────                                         │
│  Primary Text:           #1A1A1A → Ratio: 16.1:1 ✓                         │
│  Secondary Text:         #595959 → Ratio: 7.0:1 ✓                          │
│  Tertiary Text:          #757575 → Ratio: 4.6:1 ✓                          │
│  Placeholder Text:       #757575 → Ratio: 4.6:1 ✓                          │
│                                                                             │
│  STATUS COLORS (on white background):                                       │
│  ─────────────────────────────────────                                      │
│  Success Green:          #0D6D0D → Ratio: 5.9:1 ✓                          │
│  Warning Orange:         #945F07 → Ratio: 5.2:1 ✓                          │
│  Error Red:              #B3261E → Ratio: 5.5:1 ✓                          │
│  Info Blue:              #0055B3 → Ratio: 6.7:1 ✓                          │
│                                                                             │
│  PIPELINE STAGE COLORS (require text labels + icons):                       │
│  ─────────────────────────────────────────────────────                      │
│  Inbox:                  #E3F2FD (Light Blue)   + 📥 icon                   │
│  Research:               #F3E5F5 (Light Purple) + 🔬 icon                   │
│  Design:                 #FFF3E0 (Light Orange) + 🎨 icon                   │
│  Build:                  #E8F5E9 (Light Green)  + 🔨 icon                   │
│  QA:                     #FFEBEE (Light Red)    + ✓ icon                    │
│  Security:               #FFF8E1 (Light Yellow) + 🔒 icon                   │
│  Deploy:                 #E0F2F1 (Light Teal)   + 🚀 icon                   │
│                                                                             │
│  AGENT STATUS COLORS:                                                       │
│  ────────────────────                                                       │
│  Idle:                   #757575 (Gray)    + "○" symbol                     │
│  Working:                #0D6D0D (Green)   + "●" symbol + pulse animation   │
│  Blocked:                #B3261E (Red)     + "■" symbol                     │
│  Claimed:                #0055B3 (Blue)    + "◐" symbol                     │
│                                                                             │
│  INTERACTIVE ELEMENTS:                                                      │
│  ─────────────────────                                                      │
│  Primary Button BG:      #0055B3 (Blue)                                     │
│  Primary Button Text:    #FFFFFF → Ratio: 6.7:1 ✓                          │
│  Secondary Button BG:    #FFFFFF                                            │
│  Secondary Button Border:#0055B3 → Ratio: 6.7:1 ✓                          │
│  Focus Ring:             #0055B3, 2px solid                                 │
│  Link Color:             #0055B3 → Ratio: 6.7:1 ✓                          │
│  Link Visited:           #6B3FA0 → Ratio: 5.9:1 ✓                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.3 Dark Mode Considerations

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DARK MODE PALETTE                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  BACKGROUNDS:                                                               │
│  Primary Background:     #121212                                            │
│  Surface:                #1E1E1E                                            │
│  Surface Elevated:       #2C2C2C                                            │
│                                                                             │
│  TEXT (on #121212):                                                         │
│  Primary Text:           #FFFFFF → Ratio: 15.8:1 ✓                         │
│  Secondary Text:         #B3B3B3 → Ratio: 8.3:1 ✓                          │
│  Disabled Text:          #666666 → Ratio: 4.1:1 (acceptable for disabled)  │
│                                                                             │
│  COLORS:                                                                    │
│  Adjust saturation down by ~20% to reduce eye strain                        │
│  Success:                #4CAF50                                            │
│  Warning:                #FFA726                                            │
│  Error:                  #EF5350                                            │
│  Info:                   #42A5F5                                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.4 Color Blindness Considerations

| Condition | Affected Colors | Solution |
|-----------|-----------------|----------|
| Protanopia (red-blind) | Red-Green confusion | Use blue/orange instead; add icons |
| Deuteranopia (green-blind) | Red-Green confusion | Use blue/orange instead; add icons |
| Tritanopia (blue-blind) | Blue-Yellow confusion | Rare; use patterns; add text labels |

**Implementation:**
- Never use color alone to convey meaning
- Always pair colors with: icons, patterns, text labels, or shapes
- Test with color blindness simulators (built into Chrome DevTools)

---

## 6. Error State Handling

### 6.1 Error Types

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ERROR CLASSIFICATION                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  VALIDATION ERRORS (User Input)                                             │
│  ──────────────────────────────                                             │
│  • Field-level: Shown inline below field                                    │
│  • Form-level: Summary at top of form                                       │
│  • Severity: Warning (can proceed) vs Error (must fix)                      │
│                                                                             │
│  SYSTEM ERRORS (Application)                                                │
│  ──────────────────────────────                                             │
│  • Connection lost                                                          │
│  • Server error (500)                                                       │
│  • Timeout                                                                  │
│  • Agent crash/failure                                                      │
│                                                                             │
│  BUSINESS ERRORS (Logic)                                                    │
│  ───────────────────────────                                                │
│  • Gate rejection                                                           │
│  • Permission denied                                                        │
│  • Resource conflict                                                        │
│  • Dependency failure                                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Error Message Patterns

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       ERROR MESSAGE STRUCTURE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  FORMULA:                                                                   │
│  [What happened] + [Why it happened (if known)] + [How to fix it]          │
│                                                                             │
│  GOOD EXAMPLES:                                                             │
│  ─────────────                                                              │
│  ✓ "Task title is required. Please enter a title to continue."             │
│  ✓ "Email format invalid. Use format: name@domain.com"                     │
│  ✓ "Connection lost. Retrying in 5 seconds... [Retry Now]"                 │
│  ✓ "Gate approval failed: Missing security review. Complete security       │
│     review before approving this gate."                                     │
│  ✓ "Agent frontend_worker_01 crashed. Work saved at checkpoint.            │
│     [Reassign Task] [View Details]"                                         │
│                                                                             │
│  BAD EXAMPLES:                                                              │
│  ────────────                                                               │
│  ✗ "Error"                                                                  │
│  ✗ "Invalid input"                                                          │
│  ✗ "Something went wrong"                                                   │
│  ✗ "Error code: 0x8007045D"                                                 │
│  ✗ "Null pointer exception in TaskService.java:142"                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.3 Error UI Components

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         INLINE FIELD ERROR                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Task Title *                                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│  ⚠ Task title is required                                                  │
│                                                                             │
│  CSS:                                                                       │
│  • Red border on field (#B3261E)                                           │
│  • Error icon (⚠) + red text below                                         │
│  • aria-invalid="true" on input                                            │
│  • aria-describedby points to error message                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                          FORM ERROR SUMMARY                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │ ⚠ Please fix the following errors:                                    │ │
│  │                                                                       │ │
│  │   • Task title is required                                           │ │
│  │   • Priority must be selected                                        │ │
│  │   • At least one Responsible party is required                       │ │
│  │                                                                       │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  • role="alert" for screen reader announcement                             │
│  • Links to specific fields for easy navigation                            │
│  • Appears at top of form on submit attempt                                │
│  • Focus moves to summary on submit failure                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                           TOAST ERROR                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                              ┌────────────────────────────────┐             │
│                              │ ⚠ Connection lost              │             │
│                              │                                │             │
│                              │ Retrying in 5 seconds...       │             │
│                              │ [Retry Now]                    │             │
│                              └────────────────────────────────┘             │
│                                                                             │
│  • aria-live="assertive" for critical errors                               │
│  • aria-live="polite" for non-critical                                     │
│  • Does NOT auto-dismiss for errors (user must acknowledge)                │
│  • Dismiss button with aria-label="Dismiss error notification"             │
│  • Action buttons when applicable                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                          FULL PAGE ERROR                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                    ┌─────────────────────────────────┐                      │
│                    │                                 │                      │
│                    │      [Illustration/Icon]        │                      │
│                    │                                 │                      │
│                    │   Something went wrong          │                      │
│                    │                                 │                      │
│                    │   We couldn't load the          │                      │
│                    │   pipeline data. This might     │                      │
│                    │   be a temporary issue.         │                      │
│                    │                                 │                      │
│                    │   [Try Again]  [Go to Dashboard]│                      │
│                    │                                 │                      │
│                    │   Error code: PIPE_001          │                      │
│                    │   [Copy details for support]    │                      │
│                    │                                 │                      │
│                    └─────────────────────────────────┘                      │
│                                                                             │
│  • Friendly, non-technical language                                        │
│  • Clear actions user can take                                             │
│  • Technical details available but not prominent                           │
│  • Support contact option                                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.4 Agent-Specific Error Handling

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      AGENT ERROR PATTERNS                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  AGENT CRASH:                                                               │
│  ────────────                                                               │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │ ⚠ Agent Disruption                                                    │ │
│  │                                                                       │ │
│  │ frontend_worker_01 has stopped unexpectedly.                         │ │
│  │                                                                       │ │
│  │ Current task: MOL-123 "Build Header Component"                       │ │
│  │ Progress: Saved at checkpoint (Step 3 of 5)                          │ │
│  │                                                                       │ │
│  │ Options:                                                             │ │
│  │ [Reassign to another worker]  [Retry with same worker]              │ │
│  │ [View task details]                                                  │ │
│  │                                                                       │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  AGENT BLOCKED:                                                             │
│  ──────────────                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │ ⏸ Agent Blocked                                                       │ │
│  │                                                                       │ │
│  │ qa_engineer_01 is waiting for input.                                 │ │
│  │                                                                       │ │
│  │ Blocker: Needs design specs from Design Director                     │ │
│  │ Waiting since: 2 hours ago                                           │ │
│  │                                                                       │ │
│  │ [Message Design Director]  [Reassign dependency]  [Override block]   │ │
│  │                                                                       │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  GATE REJECTION:                                                            │
│  ───────────────                                                            │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │ ✗ Gate Not Passed                                                     │ │
│  │                                                                       │ │
│  │ Security Review gate rejected for MOL-123.                           │ │
│  │                                                                       │ │
│  │ Reviewer: security_director                                          │ │
│  │ Reason: "XSS vulnerability in user input handling"                   │ │
│  │                                                                       │ │
│  │ Required action: Fix security issues and resubmit                    │ │
│  │                                                                       │ │
│  │ [View full feedback]  [Assign to developer]  [Escalate]             │ │
│  │                                                                       │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Loading State Patterns

### 7.1 Loading Types

| Type | Duration | Pattern |
|------|----------|---------|
| Instant | <100ms | No indicator needed |
| Brief | 100ms-1s | Subtle indicator (opacity change) |
| Normal | 1s-5s | Spinner/Progress |
| Extended | 5s-30s | Progress with percentage |
| Long-running | >30s | Background with notification |

### 7.2 Loading UI Components

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SKELETON LOADING                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Use for: Initial page load, content areas                                  │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ ████████████████████                                               │   │
│  │ ████████████████████████████████████████████                       │   │
│  │ ████████████████████████████                                       │   │
│  │                                                                     │   │
│  │ ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                 │   │
│  │ │ ██████████  │  │ ██████████  │  │ ██████████  │                 │   │
│  │ │ ████████    │  │ ████████    │  │ ████████    │                 │   │
│  │ │ ██████████  │  │ ██████████  │  │ ██████████  │                 │   │
│  │ └─────────────┘  └─────────────┘  └─────────────┘                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  • Matches approximate layout of content                                    │
│  • Subtle pulse animation (respects prefers-reduced-motion)                 │
│  • aria-busy="true" on container                                           │
│  • aria-label="Loading content"                                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                           INLINE SPINNER                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Use for: Button actions, small data fetches                                │
│                                                                             │
│  Before: [  Submit  ]                                                       │
│  During: [  ◠  Submitting...  ]                                            │
│  After:  [  ✓  Submitted  ] → returns to [  Submit  ]                      │
│                                                                             │
│  • Button disabled during loading                                           │
│  • aria-disabled="true"                                                     │
│  • aria-label="Submitting task, please wait"                               │
│  • Success/failure state shown briefly (2s)                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                        PROGRESS INDICATOR                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Use for: File uploads, batch operations, known-duration tasks              │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Processing molecules...                                             │   │
│  │                                                                     │   │
│  │ ████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 45%      │   │
│  │                                                                     │   │
│  │ 9 of 20 items processed                                            │   │
│  │ Estimated time remaining: ~2 minutes                               │   │
│  │                                                                     │   │
│  │ [Cancel]                                                           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  • role="progressbar"                                                      │
│  • aria-valuenow="45" aria-valuemin="0" aria-valuemax="100"               │
│  • aria-valuetext="45 percent, 9 of 20 items processed"                   │
│  • Updates announced every 25% or significant milestone                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                      BACKGROUND TASK                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Use for: Long-running operations (>30s)                                    │
│                                                                             │
│  User initiates → Modal:                                                    │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │ This operation will take a few minutes.                              │ │
│  │                                                                       │ │
│  │ You can continue working and we'll notify you when it's complete.    │ │
│  │                                                                       │ │
│  │ [Run in Background]  [Cancel]                                        │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  During (in status bar):                                                    │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │ ◠ Generating report... 45%                              [Cancel]     │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  On completion (notification):                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │ ✓ Report generated successfully                                      │ │
│  │   [View Report]  [Dismiss]                                           │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 7.3 Loading Best Practices

| Practice | Implementation |
|----------|----------------|
| Show loading immediately | No delay before showing indicator |
| Preserve scroll position | Don't jump to top on data refresh |
| Avoid layout shift | Skeleton matches content dimensions |
| Allow cancellation | Long operations have cancel option |
| Maintain interactivity | Don't block entire UI for single operation |
| Announce to screen readers | Use aria-live for status updates |
| Respect reduced motion | Use opacity instead of animation if preferred |

---

## 8. Empty State Designs

### 8.1 Empty State Types

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          EMPTY STATE TYPES                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. FIRST-TIME / NO DATA YET                                               │
│     No items exist because user hasn't created any                          │
│     Tone: Encouraging, instructional                                        │
│                                                                             │
│  2. NO RESULTS                                                              │
│     Search/filter returned nothing                                          │
│     Tone: Helpful, offer alternatives                                       │
│                                                                             │
│  3. COMPLETED / CLEARED                                                     │
│     All items processed (e.g., inbox zero)                                  │
│     Tone: Celebratory, positive                                             │
│                                                                             │
│  4. ERROR / UNAVAILABLE                                                     │
│     Data couldn't be loaded                                                 │
│     Tone: Apologetic, actionable                                            │
│                                                                             │
│  5. PERMISSION DENIED                                                       │
│     User can't access this content                                          │
│     Tone: Clear, redirect to alternatives                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 8.2 Empty State UI Components

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      FIRST-TIME EMPTY STATE                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                    ┌─────────────────────────────────┐                      │
│                    │                                 │                      │
│                    │      [Illustration: Tasks]      │                      │
│                    │                                 │                      │
│                    │   No molecules yet              │                      │
│                    │                                 │                      │
│                    │   Molecules are work items that │                      │
│                    │   flow through your pipeline.   │                      │
│                    │   Create your first one to get  │                      │
│                    │   started.                      │                      │
│                    │                                 │                      │
│                    │   [+ Create Molecule]           │                      │
│                    │                                 │                      │
│                    │   📖 Learn about molecules      │                      │
│                    │                                 │                      │
│                    └─────────────────────────────────┘                      │
│                                                                             │
│  • Primary action prominent                                                │
│  • Secondary help link available                                           │
│  • Illustration adds visual interest (with alt text)                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                       NO RESULTS EMPTY STATE                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                    ┌─────────────────────────────────┐                      │
│                    │                                 │                      │
│                    │      [Illustration: Search]     │                      │
│                    │                                 │                      │
│                    │   No results for "dashbord"     │                      │
│                    │                                 │                      │
│                    │   Suggestions:                  │                      │
│                    │   • Check your spelling         │                      │
│                    │   • Try broader search terms    │                      │
│                    │   • Remove some filters         │                      │
│                    │                                 │                      │
│                    │   Did you mean: "dashboard"?    │                      │
│                    │                                 │                      │
│                    │   [Clear Search]  [Reset Filters]                      │
│                    │                                 │                      │
│                    └─────────────────────────────────┘                      │
│                                                                             │
│  • Show what was searched                                                  │
│  • Offer spelling correction if applicable                                 │
│  • Provide clear actions                                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                     COMPLETED EMPTY STATE                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                    ┌─────────────────────────────────┐                      │
│                    │                                 │                      │
│                    │      [Illustration: Success]    │                      │
│                    │                                 │                      │
│                    │   All caught up!                │                      │
│                    │                                 │                      │
│                    │   No pending approvals.         │                      │
│                    │   Check back later or review    │                      │
│                    │   completed items.              │                      │
│                    │                                 │                      │
│                    │   [View Completed]              │                      │
│                    │                                 │                      │
│                    └─────────────────────────────────┘                      │
│                                                                             │
│  • Positive, celebratory tone                                              │
│  • Suggest next action                                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                      PIPELINE COLUMN EMPTY STATE                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  INBOX        RESEARCH      DESIGN        BUILD         QA                  │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐           │
│  │         │  │ ┌─────┐ │  │         │  │ ┌─────┐ │  │         │           │
│  │  No     │  │ │Card │ │  │  No     │  │ │Card │ │  │  No     │           │
│  │  items  │  │ └─────┘ │  │  items  │  │ └─────┘ │  │  items  │           │
│  │         │  │         │  │         │  │         │  │         │           │
│  │  Drag   │  │         │  │  Items  │  │         │  │  All    │           │
│  │  items  │  │         │  │  move   │  │         │  │  tests  │           │
│  │  here   │  │         │  │  here   │  │         │  │  passed │           │
│  │         │  │         │  │  after  │  │         │  │         │           │
│  │         │  │         │  │  research│ │         │  │         │           │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘  └─────────┘           │
│                                                                             │
│  • Context-specific messaging for each stage                               │
│  • Drop zone highlighted when dragging                                     │
│  • Compact design (no large illustrations)                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 8.3 Empty State Content Guidelines

| Element | Guidelines |
|---------|------------|
| Headline | Short, descriptive (2-4 words) |
| Body text | Explain why empty, what to do (1-2 sentences) |
| Primary action | One clear CTA button |
| Secondary action | Optional text link |
| Illustration | Simple, relevant, has alt text |
| Avoid | Blame ("You haven't created any"), Technical jargon |

---

## 9. Information Density Considerations

### 9.1 Density Levels

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        INFORMATION DENSITY LEVELS                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  COMPACT (Power Users)                                                      │
│  ─────────────────────                                                      │
│  • Reduced padding (4-8px)                                                  │
│  • Smaller text (13-14px)                                                   │
│  • More items visible                                                       │
│  • Minimal whitespace                                                       │
│  • Icons only (tooltips for labels)                                         │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ MOL-001  Dashboard      ●Build   High  2d  JD  [⋮]                │   │
│  │ MOL-002  Auth Flow      ○Design  Med   5d  SK  [⋮]                │   │
│  │ MOL-003  API Refactor   ●QA      Low   1d  MR  [⋮]                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  COMFORTABLE (Default)                                                      │
│  ─────────────────────                                                      │
│  • Standard padding (12-16px)                                               │
│  • Default text (15-16px)                                                   │
│  • Balanced whitespace                                                      │
│  • Icons + text labels                                                      │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ MOL-001                                                            │   │
│  │ Build User Dashboard                                               │   │
│  │                                                                     │   │
│  │ ●Build    High Priority    2 days in stage                        │   │
│  │ Assigned: John Doe                                          [⋮]   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  SPACIOUS (Accessibility/New Users)                                         │
│  ──────────────────────────────────                                         │
│  • Extra padding (20-24px)                                                  │
│  • Larger text (17-18px)                                                    │
│  • More whitespace                                                          │
│  • Full labels, descriptions                                                │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │  MOL-001                                                           │   │
│  │                                                                     │   │
│  │  Build User Dashboard                                              │   │
│  │  Create the main dashboard view for end users                      │   │
│  │                                                                     │   │
│  │  Status: ● Build (In Progress)                                     │   │
│  │  Priority: High                                                    │   │
│  │  Time in stage: 2 days                                             │   │
│  │  Assigned to: John Doe (Frontend Worker)                           │   │
│  │                                                                     │   │
│  │  [View Details]  [Edit]  [More Actions]                           │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 9.2 Density Settings Implementation

```typescript
// User preference stored
interface DensityPreference {
  level: 'compact' | 'comfortable' | 'spacious';
  fontSize: 'small' | 'medium' | 'large';
  sidebarCollapsed: boolean;
}

// CSS Custom Properties
:root {
  /* Compact */
  --density-compact-spacing: 4px;
  --density-compact-padding: 8px;
  --density-compact-font: 13px;
  --density-compact-line-height: 1.3;

  /* Comfortable (default) */
  --density-comfortable-spacing: 8px;
  --density-comfortable-padding: 16px;
  --density-comfortable-font: 15px;
  --density-comfortable-line-height: 1.5;

  /* Spacious */
  --density-spacious-spacing: 12px;
  --density-spacious-padding: 24px;
  --density-spacious-font: 17px;
  --density-spacious-line-height: 1.6;
}

/* Apply based on preference */
[data-density="compact"] {
  --spacing: var(--density-compact-spacing);
  --padding: var(--density-compact-padding);
  --font-size: var(--density-compact-font);
  --line-height: var(--density-compact-line-height);
}
```

### 9.3 Progressive Disclosure

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      PROGRESSIVE DISCLOSURE                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  LEVEL 1: OVERVIEW (Always Visible)                                         │
│  ──────────────────────────────────                                         │
│  • Card title                                                               │
│  • Status indicator                                                         │
│  • Primary metric (e.g., days in stage)                                     │
│                                                                             │
│  LEVEL 2: SUMMARY (On Hover/Focus)                                          │
│  ──────────────────────────────────                                         │
│  • Description preview                                                      │
│  • Assignee                                                                 │
│  • Priority                                                                 │
│  • Quick actions                                                            │
│                                                                             │
│  LEVEL 3: DETAILS (On Click/Enter)                                          │
│  ─────────────────────────────────                                          │
│  • Full description                                                         │
│  • All metadata                                                             │
│  • History/Activity                                                         │
│  • Related items                                                            │
│  • All actions                                                              │
│                                                                             │
│  LEVEL 4: DEEP DIVE (Separate Page)                                         │
│  ──────────────────────────────────                                         │
│  • Complete audit trail                                                     │
│  • All conversations                                                        │
│  • File attachments                                                         │
│  • Edit interface                                                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 9.4 Dashboard Widget Density

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       DASHBOARD LAYOUT                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Desktop (≥1200px):                                                         │
│  ┌────────────────┬────────────────┬────────────────┐                      │
│  │ KPI Widget     │ Pipeline Mini  │ Alerts         │                      │
│  │ (Expandable)   │ (5 stages)     │ (Scrollable)   │                      │
│  └────────────────┴────────────────┴────────────────┘                      │
│  ┌────────────────────────────────┬─────────────────┐                      │
│  │ Org Chart (Collapsible)       │ Activity Feed   │                      │
│  │                               │ (Virtual scroll) │                      │
│  └────────────────────────────────┴─────────────────┘                      │
│                                                                             │
│  Tablet (768-1199px):                                                       │
│  ┌────────────────┬────────────────┐                                       │
│  │ KPI Widget     │ Alerts         │                                       │
│  └────────────────┴────────────────┘                                       │
│  ┌─────────────────────────────────┐                                       │
│  │ Pipeline (Horizontal scroll)   │                                       │
│  └─────────────────────────────────┘                                       │
│  ┌─────────────────────────────────┐                                       │
│  │ Activity Feed                   │                                       │
│  └─────────────────────────────────┘                                       │
│                                                                             │
│  Mobile (≤767px):                                                           │
│  ┌─────────────────────────────────┐                                       │
│  │ KPI Summary (Compact)          │                                       │
│  └─────────────────────────────────┘                                       │
│  ┌─────────────────────────────────┐                                       │
│  │ Tab Navigation:                │                                       │
│  │ [Pipeline] [Activity] [Alerts] │                                       │
│  └─────────────────────────────────┘                                       │
│  ┌─────────────────────────────────┐                                       │
│  │ Selected Tab Content           │                                       │
│  │ (Full width, scrollable)       │                                       │
│  └─────────────────────────────────┘                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 10. Mobile Responsiveness

### 10.1 Breakpoint System

| Breakpoint | Width | Target |
|------------|-------|--------|
| XS (Phone) | 0-575px | Small phones |
| SM (Phone+) | 576-767px | Large phones |
| MD (Tablet) | 768-991px | Tablets portrait |
| LG (Tablet+) | 992-1199px | Tablets landscape |
| XL (Desktop) | 1200-1399px | Laptops |
| XXL (Large) | ≥1400px | Desktops |

### 10.2 Mobile Navigation Pattern

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       MOBILE NAVIGATION                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  HEADER (Fixed):                                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ [☰]  AI Corp  [🔔 3]  [👤]                                         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  HAMBURGER MENU (Slide-in from left):                                       │
│  ┌──────────────────────┬──────────────────────────────────────────────┐   │
│  │                      │                                              │   │
│  │  AI Corp             │                                              │   │
│  │                      │        (Dimmed content)                      │   │
│  │  ─────────────────── │                                              │   │
│  │                      │                                              │   │
│  │  📊 Dashboard        │                                              │   │
│  │  📋 Pipeline         │                                              │   │
│  │  💬 Messages         │                                              │   │
│  │  🏢 Organization     │                                              │   │
│  │  ⚙️ Settings         │                                              │   │
│  │                      │                                              │   │
│  │  ─────────────────── │                                              │   │
│  │                      │                                              │   │
│  │  📖 Help             │                                              │   │
│  │  🚪 Sign Out         │                                              │   │
│  │                      │                                              │   │
│  └──────────────────────┴──────────────────────────────────────────────┘   │
│                                                                             │
│  BOTTOM NAVIGATION (Alternative - Fixed):                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │   📊        📋         💬         🏢         ⋮                     │   │
│  │ Dashboard  Pipeline  Messages    Org       More                    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ACCESSIBILITY NOTES:                                                       │
│  • Hamburger button: aria-label="Open navigation menu"                     │
│  • Menu: role="navigation", aria-label="Main navigation"                   │
│  • Expanded state: aria-expanded="true" on trigger                         │
│  • Focus trap when open                                                    │
│  • Escape closes menu                                                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 10.3 Touch Targets

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        TOUCH TARGET SIZES                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  MINIMUM SIZES (WCAG 2.5.8):                                               │
│  ───────────────────────────                                                │
│  • Touch targets: 44x44px minimum (iOS guideline)                          │
│  • WCAG AA: 24x24px minimum                                                │
│  • Spacing between targets: 8px minimum                                     │
│                                                                             │
│  BUTTON SIZES:                                                              │
│  ─────────────                                                              │
│  • Primary buttons: 48px height, full width on mobile                      │
│  • Secondary buttons: 44px height                                          │
│  • Icon buttons: 44x44px                                                   │
│                                                                             │
│  LIST ITEMS:                                                                │
│  ───────────                                                                │
│  • Minimum row height: 48px                                                │
│  • Tappable area: Full width of row                                        │
│                                                                             │
│  FORM INPUTS:                                                               │
│  ────────────                                                               │
│  • Input height: 48px minimum                                              │
│  • Checkbox/Radio: 24x24px control + 44px tap area                         │
│                                                                             │
│  VISUAL:                                                                    │
│                                                                             │
│  ┌──────┐    Too small        ┌────────────┐    Good                       │
│  │ ✓    │    (24x24)          │     ✓      │    (44x44)                    │
│  └──────┘                     │            │                                │
│                               └────────────┘                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 10.4 Mobile-Specific Patterns

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     MOBILE PIPELINE VIEW                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  OPTION 1: HORIZONTAL SCROLL                                                │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ [Inbox] [Research] [Design] [Build] [QA] →                         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │  [Card 1]    [Card 1]    [Card 1]                                  │   │
│  │  [Card 2]                [Card 2]                                  │   │
│  │                          [Card 3]                                  │   │
│  │                                                     → scroll       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  OPTION 2: STAGE SELECTOR (Recommended for mobile)                          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Stage: [Inbox ▼]                           Filter: [All ▼]         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ ┌─────────────────────────────────────────────────────────────┐   │   │
│  │ │ MOL-001: Build User Dashboard                               │   │   │
│  │ │ High Priority • 2 days • John D.                            │   │   │
│  │ └─────────────────────────────────────────────────────────────┘   │   │
│  │ ┌─────────────────────────────────────────────────────────────┐   │   │
│  │ │ MOL-002: Auth Flow                                          │   │   │
│  │ │ Medium Priority • 5 days • Sarah K.                         │   │   │
│  │ └─────────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │              [+ New Molecule]                                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                    MOBILE ORG CHART                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Tree List View (instead of visual hierarchy):                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 👤 CEO (You)                                              [▼]      │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │   └─ 👤 COO                                               [▼]      │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │      ├─ 👤 VP Engineering                                 [▼]      │   │
│  │      ├─ 👤 VP Research                                    [▶]      │   │
│  │      ├─ 👤 VP Product                                     [▶]      │   │
│  │      ├─ 👤 VP Quality                                     [▶]      │   │
│  │      └─ 👤 VP Operations                                  [▶]      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  • Collapsible tree structure                                              │
│  • Tap to expand/collapse                                                  │
│  • Long press or swipe for actions                                         │
│  • Search/filter at top                                                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                   MOBILE FORMS                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Create Molecule                                            [✕]     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │ Title *                                                            │   │
│  │ ┌───────────────────────────────────────────────────────────────┐ │   │
│  │ │                                                               │ │   │
│  │ └───────────────────────────────────────────────────────────────┘ │   │
│  │                                                                     │   │
│  │ Description                                                        │   │
│  │ ┌───────────────────────────────────────────────────────────────┐ │   │
│  │ │                                                               │ │   │
│  │ │                                                               │ │   │
│  │ │                                                               │ │   │
│  │ └───────────────────────────────────────────────────────────────┘ │   │
│  │                                                                     │   │
│  │ Priority                                                           │   │
│  │ ○ High   ○ Medium   ○ Low                                         │   │
│  │                                                                     │   │
│  │ Accountable (required) *                                           │   │
│  │ ┌───────────────────────────────────────────────────────────────┐ │   │
│  │ │ Select person...                                         [▼] │ │   │
│  │ └───────────────────────────────────────────────────────────────┘ │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     [Create Molecule]                              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  • Full-screen modal on mobile                                             │
│  • Sticky header with close button                                         │
│  • Sticky footer with submit button                                        │
│  • Single column layout                                                    │
│  • Large touch targets for inputs                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 10.5 Responsive Tables

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     RESPONSIVE TABLE STRATEGIES                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  DESKTOP (Full table):                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Agent         │ Role          │ Status  │ Task        │ Actions    │   │
│  ├───────────────┼───────────────┼─────────┼─────────────┼────────────│   │
│  │ worker_01     │ Frontend      │ Working │ MOL-123     │ [V] [R]    │   │
│  │ worker_02     │ Frontend      │ Idle    │ -           │ [V] [R]    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  TABLET (Hide less important columns):                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Agent         │ Status  │ Task        │ ⋮           │               │   │
│  ├───────────────┼─────────┼─────────────┼─────────────│               │   │
│  │ worker_01     │ Working │ MOL-123     │ [⋮]         │               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  MOBILE (Card layout):                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ ┌───────────────────────────────────────────────────────────────┐ │   │
│  │ │ worker_01                                              [⋮]    │ │   │
│  │ │ Frontend Worker                                               │ │   │
│  │ │ ● Working • MOL-123                                          │ │   │
│  │ └───────────────────────────────────────────────────────────────┘ │   │
│  │ ┌───────────────────────────────────────────────────────────────┐ │   │
│  │ │ worker_02                                              [⋮]    │ │   │
│  │ │ Frontend Worker                                               │ │   │
│  │ │ ○ Idle                                                       │ │   │
│  │ └───────────────────────────────────────────────────────────────┘ │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ACCESSIBILITY:                                                             │
│  • Keep table structure for screen readers even when visually card-like    │
│  • Use aria-label on mobile cards: "Agent worker_01, Frontend, Working"    │
│  • Maintain data relationships programmatically                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 11. Testing Checklist

### 11.1 Automated Testing

| Tool | Purpose | Frequency |
|------|---------|-----------|
| axe-core | WCAG violations | Every PR |
| Lighthouse | Performance + a11y score | Daily |
| Pa11y | CI accessibility testing | Every PR |
| jest-axe | Component-level testing | Every PR |
| Storybook a11y addon | Interactive testing | Development |

### 11.2 Manual Testing Checklist

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     MANUAL TESTING CHECKLIST                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  KEYBOARD TESTING:                                                          │
│  ☐ All functionality accessible via keyboard                               │
│  ☐ Focus order is logical                                                  │
│  ☐ Focus indicator always visible                                          │
│  ☐ No keyboard traps                                                       │
│  ☐ Modals trap focus correctly                                             │
│  ☐ Skip link works                                                         │
│  ☐ Custom shortcuts work and can be disabled                               │
│                                                                             │
│  SCREEN READER TESTING (Test with NVDA, VoiceOver, JAWS):                  │
│  ☐ Page title announced on load                                            │
│  ☐ Headings form logical hierarchy                                         │
│  ☐ Landmarks present and labeled                                           │
│  ☐ Images have alt text                                                    │
│  ☐ Forms have proper labels                                                │
│  ☐ Error messages announced                                                │
│  ☐ Dynamic content updates announced                                       │
│  ☐ Tables have proper headers                                              │
│  ☐ Custom widgets have correct ARIA                                        │
│                                                                             │
│  ZOOM/MAGNIFICATION TESTING:                                                │
│  ☐ 200% zoom - no horizontal scroll                                        │
│  ☐ 400% zoom - content still usable                                        │
│  ☐ Text-only zoom - no overlap/truncation                                  │
│  ☐ No loss of functionality at any zoom level                              │
│                                                                             │
│  COLOR/CONTRAST TESTING:                                                    │
│  ☐ Color contrast meets 4.5:1 (text) and 3:1 (UI)                         │
│  ☐ Information not conveyed by color alone                                 │
│  ☐ High contrast mode supported                                            │
│  ☐ Dark mode meets contrast requirements                                   │
│                                                                             │
│  MOTION TESTING:                                                            │
│  ☐ Animations respect prefers-reduced-motion                               │
│  ☐ Auto-playing content can be paused                                      │
│  ☐ No content flashes more than 3 times/second                            │
│                                                                             │
│  MOBILE TESTING:                                                            │
│  ☐ Touch targets minimum 44x44px                                           │
│  ☐ Orientation change doesn't break layout                                 │
│  ☐ Pinch-to-zoom not disabled                                              │
│  ☐ Forms work with mobile keyboards                                        │
│  ☐ No horizontal scroll at 320px width                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 11.3 User Testing Recommendations

| User Group | Testing Focus |
|------------|---------------|
| Screen reader users | Navigation, forms, dynamic content |
| Keyboard-only users | All workflows without mouse |
| Low vision users | Zoom, color contrast, text size |
| Motor impairments | Touch targets, timing, error recovery |
| Cognitive disabilities | Clear language, consistent UI, error prevention |
| New users | Onboarding, empty states, help content |
| Power users | Keyboard shortcuts, density options, efficiency |

---

## 12. Implementation Priority

### 12.1 Phase 1: Critical (Must Have for Launch)

| Item | WCAG Criterion | Impact |
|------|----------------|--------|
| Keyboard navigation | 2.1.1 | Blockers for many users |
| Focus management | 2.4.7 | Unusable without |
| Color contrast | 1.4.3, 1.4.11 | Legal compliance |
| Form labels | 1.3.1, 3.3.2 | Form unusable without |
| Error messages | 3.3.1, 3.3.3 | Critical for task completion |
| Alt text | 1.1.1 | Screen reader blocking |
| Skip link | 2.4.1 | Navigation blocker |
| Page titles | 2.4.2 | Screen reader orientation |

### 12.2 Phase 2: Important (First Update)

| Item | WCAG Criterion | Impact |
|------|----------------|--------|
| ARIA landmarks | 1.3.1 | Navigation efficiency |
| Live regions | 4.1.3 | Dynamic content awareness |
| Heading hierarchy | 1.3.1, 2.4.6 | Content structure |
| Touch targets | 2.5.8 | Mobile usability |
| Loading states | - | User confidence |
| Empty states | - | User guidance |

### 12.3 Phase 3: Enhancement (Subsequent Updates)

| Item | WCAG Criterion | Impact |
|------|----------------|--------|
| Density settings | - | Power user efficiency |
| Keyboard shortcuts | 2.1.4 | Power user efficiency |
| Dark mode | - | User preference |
| Reduced motion | 2.3.1 | Comfort/safety |
| High contrast mode | - | Low vision support |
| Mobile optimization | 1.4.10 | Mobile experience |

---

## 13. Appendix: ARIA Patterns Reference

### 13.1 Common Patterns Used in AI Corp

| Pattern | Usage | Reference |
|---------|-------|-----------|
| Tabs | Pipeline stage navigation | APG Tabs Pattern |
| Tree | Org chart, folder structure | APG Tree Pattern |
| Dialog | Modals, confirmations | APG Dialog Pattern |
| Combobox | Searchable selects | APG Combobox Pattern |
| Listbox | Multi-select | APG Listbox Pattern |
| Alert | Error messages | APG Alert Pattern |
| Feed | Activity stream | APG Feed Pattern |
| Menu | Context menus | APG Menu Pattern |
| Disclosure | Expandable sections | APG Disclosure Pattern |

### 13.2 Resources

- [WCAG 2.2 Guidelines](https://www.w3.org/WAI/WCAG22/quickref/)
- [ARIA Authoring Practices Guide](https://www.w3.org/WAI/ARIA/apg/)
- [Inclusive Components](https://inclusive-components.design/)
- [A11y Project Checklist](https://www.a11yproject.com/checklist/)
- [Deque University](https://dequeuniversity.com/)

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-01-05 | UX Accessibility Specialist | Initial comprehensive audit |
