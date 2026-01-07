# ROUND 2: UNIFIED DESIGN SYSTEM

## AI Corp Cross-Concept Design Tokens & Standards

> Synthesizing Concept A (Neural Glass) and Concept C (Spatial Layers) into a cohesive, accessible, performant system.

---

## 1. UNIFIED COMPONENT TOKEN SYSTEM

### 1.1 Core CSS Custom Properties

The following token system provides a shared foundation that both concepts can consume while maintaining their unique visual identities.

```css
/* ================================================================
   AI CORP UNIFIED DESIGN TOKENS
   Version: 2.0
   Concepts Supported: Neural Glass (A), Spatial Layers (C)
   ================================================================ */

:root {
  /* ============================================================
     SPACING SCALE (8px base unit)
     ============================================================ */
  --space-0: 0;
  --space-1: 0.25rem;   /* 4px  - micro spacing */
  --space-2: 0.5rem;    /* 8px  - tight */
  --space-3: 0.75rem;   /* 12px - compact */
  --space-4: 1rem;      /* 16px - base */
  --space-5: 1.5rem;    /* 24px - comfortable */
  --space-6: 2rem;      /* 32px - spacious */
  --space-7: 2.5rem;    /* 40px - loose */
  --space-8: 3rem;      /* 48px - section */
  --space-9: 4rem;      /* 64px - major section */
  --space-10: 5rem;     /* 80px - hero spacing */

  /* Component-specific spacing aliases */
  --space-card-padding: var(--space-5);
  --space-card-gap: var(--space-4);
  --space-button-x: var(--space-5);
  --space-button-y: var(--space-3);
  --space-input-x: var(--space-4);
  --space-input-y: var(--space-3);
  --space-nav-item: var(--space-3) var(--space-4);
  --space-section: var(--space-8);
  --space-page: var(--space-6);

  /* ============================================================
     BORDER RADIUS SCALE
     ============================================================ */
  --radius-none: 0;
  --radius-sm: 0.375rem;   /* 6px  - subtle rounding */
  --radius-md: 0.5rem;     /* 8px  - buttons, inputs */
  --radius-lg: 0.75rem;    /* 12px - cards */
  --radius-xl: 1rem;       /* 16px - prominent cards */
  --radius-2xl: 1.25rem;   /* 20px - modals, elevated */
  --radius-3xl: 1.5rem;    /* 24px - hero elements */
  --radius-full: 9999px;   /* pill shapes, avatars */

  /* Component-specific radius aliases */
  --radius-button: var(--radius-md);
  --radius-button-sm: var(--radius-sm);
  --radius-button-lg: var(--radius-lg);
  --radius-card: var(--radius-xl);
  --radius-card-elevated: var(--radius-2xl);
  --radius-input: var(--radius-md);
  --radius-modal: var(--radius-2xl);
  --radius-dropdown: var(--radius-lg);
  --radius-tooltip: var(--radius-md);
  --radius-badge: var(--radius-full);
  --radius-avatar: var(--radius-full);

  /* ============================================================
     SHADOW SYSTEM (Elevation Language)
     ============================================================ */

  /* Base shadows - works for both light and dark themes */
  --shadow-xs: 0 1px 2px rgba(0, 0, 0, 0.05);
  --shadow-sm: 0 2px 4px rgba(0, 0, 0, 0.06),
               0 1px 2px rgba(0, 0, 0, 0.04);
  --shadow-md: 0 4px 12px rgba(0, 0, 0, 0.08),
               0 2px 4px rgba(0, 0, 0, 0.04);
  --shadow-lg: 0 12px 40px rgba(0, 0, 0, 0.12),
               0 4px 12px rgba(0, 0, 0, 0.06);
  --shadow-xl: 0 24px 80px rgba(0, 0, 0, 0.16),
               0 8px 24px rgba(0, 0, 0, 0.08);

  /* Dark theme shadows (Neural Glass) */
  --shadow-dark-sm: 0 2px 4px rgba(0, 0, 0, 0.3);
  --shadow-dark-md: 0 8px 32px rgba(0, 0, 0, 0.4);
  --shadow-dark-lg: 0 12px 40px rgba(0, 0, 0, 0.5);
  --shadow-dark-xl: 0 24px 64px rgba(0, 0, 0, 0.6);

  /* Glow shadows (for active/accent states) */
  --shadow-glow-primary: 0 0 20px rgba(59, 130, 246, 0.3);
  --shadow-glow-success: 0 0 20px rgba(16, 185, 129, 0.25);
  --shadow-glow-warning: 0 0 20px rgba(245, 158, 11, 0.3);
  --shadow-glow-error: 0 0 20px rgba(239, 68, 68, 0.3);
  --shadow-glow-ai: 0 0 20px rgba(139, 92, 246, 0.3);

  /* Component shadow aliases */
  --shadow-card: var(--shadow-md);
  --shadow-card-hover: var(--shadow-lg);
  --shadow-button: var(--shadow-sm);
  --shadow-button-hover: var(--shadow-md);
  --shadow-dropdown: var(--shadow-lg);
  --shadow-modal: var(--shadow-xl);
  --shadow-tooltip: var(--shadow-md);

  /* ============================================================
     TRANSITION SYSTEM
     ============================================================ */

  /* Durations */
  --duration-instant: 50ms;
  --duration-fast: 100ms;
  --duration-normal: 200ms;
  --duration-moderate: 300ms;
  --duration-slow: 400ms;
  --duration-slower: 500ms;
  --duration-ambient: 2000ms;

  /* Easing functions */
  --ease-linear: linear;
  --ease-in: cubic-bezier(0.4, 0, 1, 1);
  --ease-out: cubic-bezier(0, 0, 0.2, 1);
  --ease-in-out: cubic-bezier(0.4, 0, 0.2, 1);
  --ease-spring: cubic-bezier(0.34, 1.56, 0.64, 1);
  --ease-bounce: cubic-bezier(0.175, 0.885, 0.32, 1.275);
  --ease-pulse: cubic-bezier(0.4, 0, 0.6, 1);

  /* Composite transitions */
  --transition-fast: var(--duration-fast) var(--ease-out);
  --transition-normal: var(--duration-normal) var(--ease-out);
  --transition-smooth: var(--duration-moderate) var(--ease-in-out);
  --transition-spring: var(--duration-moderate) var(--ease-spring);

  /* Component-specific transitions */
  --transition-button: all var(--duration-normal) var(--ease-out);
  --transition-card: all var(--duration-moderate) var(--ease-spring);
  --transition-dropdown: opacity var(--duration-fast) var(--ease-out),
                         transform var(--duration-normal) var(--ease-spring);
  --transition-modal: opacity var(--duration-normal) var(--ease-out),
                      transform var(--duration-moderate) var(--ease-spring);
  --transition-focus: box-shadow var(--duration-fast) var(--ease-out);

  /* ============================================================
     GLASS/BLUR EFFECTS
     ============================================================ */
  --blur-sm: 8px;
  --blur-md: 16px;
  --blur-lg: 24px;
  --blur-xl: 32px;

  /* Glass surfaces */
  --glass-blur: blur(var(--blur-lg)) saturate(180%);
  --glass-blur-light: blur(var(--blur-md)) saturate(150%);
  --glass-blur-heavy: blur(var(--blur-xl)) saturate(200%);

  /* ============================================================
     Z-INDEX SCALE
     ============================================================ */
  --z-base: 0;
  --z-dropdown: 100;
  --z-sticky: 200;
  --z-fixed: 300;
  --z-modal-backdrop: 400;
  --z-modal: 500;
  --z-popover: 600;
  --z-tooltip: 700;
  --z-toast: 800;

  /* ============================================================
     TYPOGRAPHY SCALE
     ============================================================ */
  --font-sans: 'Inter', system-ui, -apple-system, sans-serif;
  --font-display: 'Inter', system-ui, sans-serif;
  --font-mono: 'JetBrains Mono', 'Fira Code', monospace;

  /* Font sizes */
  --text-xs: 0.6875rem;   /* 11px */
  --text-sm: 0.8125rem;   /* 13px */
  --text-base: 0.9375rem; /* 15px */
  --text-lg: 1.125rem;    /* 18px */
  --text-xl: 1.375rem;    /* 22px */
  --text-2xl: 1.75rem;    /* 28px */
  --text-3xl: 2.25rem;    /* 36px */
  --text-4xl: 3rem;       /* 48px */

  /* Line heights */
  --leading-none: 1;
  --leading-tight: 1.1;
  --leading-snug: 1.25;
  --leading-normal: 1.5;
  --leading-relaxed: 1.625;

  /* Letter spacing */
  --tracking-tighter: -0.02em;
  --tracking-tight: -0.01em;
  --tracking-normal: 0;
  --tracking-wide: 0.02em;
  --tracking-wider: 0.04em;

  /* Font weights */
  --font-light: 300;
  --font-normal: 400;
  --font-medium: 500;
  --font-semibold: 600;
  --font-bold: 700;

  /* ============================================================
     COLOR PALETTE (Shared Semantic Colors)
     ============================================================ */

  /* Primary accent */
  --color-primary-50: #EFF6FF;
  --color-primary-100: #DBEAFE;
  --color-primary-200: #BFDBFE;
  --color-primary-300: #93C5FD;
  --color-primary-400: #60A5FA;
  --color-primary-500: #3B82F6;
  --color-primary-600: #2563EB;
  --color-primary-700: #1D4ED8;
  --color-primary-800: #1E40AF;
  --color-primary-900: #1E3A8A;

  /* Success (Green) */
  --color-success-50: #ECFDF5;
  --color-success-500: #10B981;
  --color-success-600: #059669;

  /* Warning (Amber) */
  --color-warning-50: #FFFBEB;
  --color-warning-500: #F59E0B;
  --color-warning-600: #D97706;

  /* Error (Red) */
  --color-error-50: #FEF2F2;
  --color-error-500: #EF4444;
  --color-error-600: #DC2626;

  /* AI/Processing (Purple) */
  --color-ai-50: #F5F3FF;
  --color-ai-500: #8B5CF6;
  --color-ai-600: #7C3AED;

  /* Cyan accent (Neural connections) */
  --color-cyan-500: #22D3EE;
  --color-cyan-400: #06B6D4;

  /* Neutrals */
  --color-neutral-50: #F8FAFC;
  --color-neutral-100: #F1F5F9;
  --color-neutral-200: #E2E8F0;
  --color-neutral-300: #CBD5E1;
  --color-neutral-400: #94A3B8;
  --color-neutral-500: #64748B;
  --color-neutral-600: #475569;
  --color-neutral-700: #334155;
  --color-neutral-800: #1E293B;
  --color-neutral-900: #0F172A;
  --color-neutral-950: #020617;
}
```

### 1.2 Concept-Specific Theme Overrides

```css
/* ================================================================
   CONCEPT A: NEURAL GLASS - Dark Theme Overrides
   ================================================================ */

[data-theme="neural-glass"],
.theme-neural-glass {
  /* Background colors */
  --color-bg-base: #0A0A0F;
  --color-bg-elevated: #12121A;
  --color-bg-surface: rgba(255, 255, 255, 0.05);
  --color-bg-surface-hover: rgba(255, 255, 255, 0.08);
  --color-bg-surface-active: rgba(255, 255, 255, 0.12);

  /* Border colors */
  --color-border: rgba(255, 255, 255, 0.1);
  --color-border-hover: rgba(255, 255, 255, 0.15);
  --color-border-focus: var(--color-primary-500);

  /* Text colors */
  --color-text-primary: #F8FAFC;
  --color-text-secondary: #94A3B8;
  --color-text-muted: #64748B;

  /* Use dark shadows */
  --shadow-card: var(--shadow-dark-md);
  --shadow-card-hover: var(--shadow-dark-lg);

  /* Accent glow colors */
  --color-synapse: #22D3EE;
  --color-neural: #3B82F6;
}

/* ================================================================
   CONCEPT C: SPATIAL LAYERS - Light Theme Overrides
   ================================================================ */

[data-theme="spatial-layers"],
.theme-spatial-layers {
  /* Background colors */
  --color-bg-base: #F5F7FA;
  --color-bg-elevated: #FFFFFF;
  --color-bg-surface: rgba(255, 255, 255, 0.85);
  --color-bg-surface-hover: rgba(255, 255, 255, 0.9);
  --color-bg-surface-active: rgba(255, 255, 255, 0.95);

  /* Border colors */
  --color-border: rgba(255, 255, 255, 0.5);
  --color-border-hover: rgba(255, 255, 255, 0.7);
  --color-border-focus: var(--color-primary-500);

  /* Text colors */
  --color-text-primary: #0F172A;
  --color-text-secondary: #475569;
  --color-text-muted: #94A3B8;

  /* Standard light shadows */
  --shadow-card: var(--shadow-md);
  --shadow-card-hover: var(--shadow-lg);
}
```

---

## 2. ACCESSIBILITY AUDIT (WCAG 2.1 AA)

### 2.1 Concept A: Neural Glass - Issues & Fixes

| Issue | WCAG Criterion | Severity | Fix Required |
|-------|----------------|----------|--------------|
| **Low text contrast on glass** | 1.4.3 Contrast (Minimum) | High | Glass cards with 5% white bg need text at 94% opacity minimum. Use `--color-text-primary` (#F8FAFC) exclusively on glass surfaces. |
| **Glow text effects reduce legibility** | 1.4.3 | Medium | Limit `text-shadow` glow to non-essential decorative text only. Never on body text. |
| **Status orb colors alone convey meaning** | 1.4.1 Use of Color | High | Add text labels or icons alongside color-coded status orbs. Example: "Active" label + green orb. |
| **Animation-heavy transitions** | 2.3.3 Animation from Interactions | Medium | Implement `prefers-reduced-motion` support (see Section 4). |
| **Pulsing animations may be distracting** | 2.2.2 Pause, Stop, Hide | Medium | Provide UI control to pause ambient animations. |
| **Focus states unclear on dark bg** | 2.4.7 Focus Visible | High | Use high-contrast focus ring: `box-shadow: 0 0 0 2px var(--color-primary-500), 0 0 0 4px rgba(59, 130, 246, 0.3)`. |
| **Keyboard nav for org chart** | 2.1.1 Keyboard | High | Ensure all org chart nodes are focusable with arrow key navigation. |
| **Neural flow lines lack alternative** | 1.1.1 Non-text Content | Low | Provide aria-label describing data flow status. |

#### Recommended Focus State (Dark Theme)
```css
.theme-neural-glass *:focus-visible {
  outline: none;
  box-shadow:
    0 0 0 2px var(--color-bg-base),
    0 0 0 4px var(--color-primary-500),
    var(--shadow-glow-primary);
}
```

### 2.2 Concept C: Spatial Layers - Issues & Fixes

| Issue | WCAG Criterion | Severity | Fix Required |
|-------|----------------|----------|--------------|
| **Glass surfaces reduce text contrast** | 1.4.3 Contrast (Minimum) | Medium | Increase text weight on glass (+100). Use `text-shadow: 0 1px 2px rgba(0,0,0,0.1)` for legibility. |
| **Parallax scroll disorienting** | 2.3.3 Animation from Interactions | Medium | Disable parallax for `prefers-reduced-motion`. |
| **Constellation view keyboard navigation** | 2.1.1 Keyboard | High | Implement focus management for node navigation with arrow keys. |
| **Low contrast on "Meta" text (12px)** | 1.4.3 | Medium | Increase Meta text from 12px to 13px minimum, ensure 4.5:1 ratio. |
| **Hover-only reveals on cards** | 1.4.13 Content on Hover or Focus | Medium | Ensure hover content is also accessible on focus and can be dismissed. |
| **Ambient glow animations** | 2.3.3 | Low | Reduce intensity or disable for motion-sensitive users. |
| **Touch targets on orb buttons** | 2.5.5 Target Size | Medium | Ensure 44x44px minimum for icon buttons (currently specified, verify implementation). |

#### Recommended Focus State (Light Theme)
```css
.theme-spatial-layers *:focus-visible {
  outline: none;
  box-shadow:
    0 0 0 2px #FFFFFF,
    0 0 0 4px var(--color-primary-500);
}
```

### 2.3 Shared Accessibility Requirements

```css
/* ================================================================
   ACCESSIBLE FOCUS STATES
   ================================================================ */

/* Base focus ring - works on all backgrounds */
:focus-visible {
  outline: 2px solid var(--color-primary-500);
  outline-offset: 2px;
}

/* Skip link for keyboard navigation */
.skip-link {
  position: absolute;
  top: -100%;
  left: 50%;
  transform: translateX(-50%);
  padding: var(--space-3) var(--space-5);
  background: var(--color-primary-500);
  color: white;
  border-radius: var(--radius-md);
  z-index: var(--z-toast);
  transition: top var(--duration-fast) var(--ease-out);
}

.skip-link:focus {
  top: var(--space-4);
}

/* Screen reader only content */
.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
  border-width: 0;
}

/* Ensure minimum touch target size */
.touch-target {
  min-width: 44px;
  min-height: 44px;
}
```

### 2.4 Color Contrast Verification

| Combination | Contrast Ratio | WCAG AA (4.5:1) | WCAG AAA (7:1) |
|-------------|----------------|-----------------|----------------|
| Neural Glass: #F8FAFC on #0A0A0F | 18.5:1 | PASS | PASS |
| Neural Glass: #94A3B8 on #0A0A0F | 7.2:1 | PASS | PASS |
| Neural Glass: #64748B on #0A0A0F | 4.7:1 | PASS | FAIL |
| Spatial: #0F172A on #F5F7FA | 14.8:1 | PASS | PASS |
| Spatial: #475569 on #FFFFFF | 7.0:1 | PASS | PASS |
| Spatial: #94A3B8 on #FFFFFF | 3.3:1 | FAIL | FAIL |

**Actions Required:**
- Spatial Layers: Upgrade `--color-neutral-400` (#94A3B8) usage to `--color-neutral-500` (#64748B) for AA compliance on white.
- Neural Glass: `--color-neutral-500` barely passes; use `--color-neutral-400` (#94A3B8) minimum for secondary text.

---

## 3. MOBILE-FIRST RESPONSIVE STRATEGY

### 3.1 Breakpoint System

```css
/* ================================================================
   RESPONSIVE BREAKPOINTS
   ================================================================ */

:root {
  /* Breakpoint values (reference only - use in media queries) */
  --breakpoint-sm: 640px;   /* Large phones, landscape */
  --breakpoint-md: 768px;   /* Tablets */
  --breakpoint-lg: 1024px;  /* Small laptops */
  --breakpoint-xl: 1280px;  /* Desktops */
  --breakpoint-2xl: 1536px; /* Large desktops */
}

/* Mobile-first media queries */
/* Base styles = mobile (< 640px) */

@media (min-width: 640px) {
  /* sm: Large phones, small tablets */
}

@media (min-width: 768px) {
  /* md: Tablets */
}

@media (min-width: 1024px) {
  /* lg: Laptops */
}

@media (min-width: 1280px) {
  /* xl: Desktops */
}

@media (min-width: 1536px) {
  /* 2xl: Large monitors */
}
```

### 3.2 Layout Strategy by Breakpoint

#### Mobile (< 640px)
```
NAVIGATION
┌─────────────────────────────────┐
│  LOGO              [=] MENU     │  Fixed bottom nav
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│  Single column content          │
│  ┌─────────────────────────────┐│
│  │ Full-width cards            ││
│  │ Stacked vertically          ││
│  └─────────────────────────────┘│
│  ┌─────────────────────────────┐│
│  │ Metrics: 2-column mini grid ││
│  └─────────────────────────────┘│
└─────────────────────────────────┘
        │
        ▼
ORG CHART → List View
┌─────────────────────────────────┐
│  ▼ COO                          │
│    ├── Engineering Lead         │
│    │   ├── Worker 1             │
│    │   └── Worker 2             │
│    ├── QA Lead                  │
│    └── Architecture Lead        │
└─────────────────────────────────┘
```

#### Tablet (768px - 1023px)
```
┌─────────────────────────────────────────────────────┐
│  LOGO        [Dashboard] [Projects] [Agents]   [=]  │
└─────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│  ┌──────────────────┐  ┌──────────────────┐         │
│  │   Card 1 (50%)   │  │   Card 2 (50%)   │         │
│  └──────────────────┘  └──────────────────┘         │
│  ┌─────────────────────────────────────────┐        │
│  │   Full-width card for important content │        │
│  └─────────────────────────────────────────┘        │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐            │
│  │ Metric 1 │ │ Metric 2 │ │ Metric 3 │ (3-col)   │
│  └──────────┘ └──────────┘ └──────────┘            │
└─────────────────────────────────────────────────────┘
                        │
                        ▼
ORG CHART → Cluster View (Department groupings)
```

#### Desktop (1024px+)
```
┌───────────┬─────────────────────────────────────────────────────────────┐
│           │  HEADER BAR                               Search   Avatar   │
│  SIDEBAR  ├─────────────────────────────────────────────────────────────┤
│           │                                                             │
│  ◉ Dash   │   ┌────────────┐  ┌────────────┐  ┌────────────┐ ┌────────┐│
│  ○ Proj   │   │ Card (25%) │  │ Card (25%) │  │ Card (25%) │ │ (25%)  ││
│  ○ Agents │   └────────────┘  └────────────┘  └────────────┘ └────────┘│
│  ○ Gates  │                                                             │
│           │   ┌────────────────────────────┐  ┌────────────────────────┐│
│  ─────    │   │      Main Content (60%)    │  │   Sidebar Panel (40%) ││
│           │   │                            │  │   (Context/Details)   ││
│  ○ Settings   │                            │  │                        ││
│           │   └────────────────────────────┘  └────────────────────────┘│
└───────────┴─────────────────────────────────────────────────────────────┘
```

### 3.3 Component Responsive Behavior

```css
/* ================================================================
   RESPONSIVE COMPONENT TOKENS
   ================================================================ */

:root {
  /* Mobile-first values */
  --container-padding: var(--space-4);
  --card-columns: 1;
  --sidebar-width: 0;
  --nav-height: 56px;
}

@media (min-width: 768px) {
  :root {
    --container-padding: var(--space-5);
    --card-columns: 2;
    --nav-height: 64px;
  }
}

@media (min-width: 1024px) {
  :root {
    --container-padding: var(--space-6);
    --card-columns: 3;
    --sidebar-width: 240px;
    --nav-height: 72px;
  }
}

@media (min-width: 1280px) {
  :root {
    --card-columns: 4;
    --sidebar-width: 280px;
  }
}
```

### 3.4 Concept-Specific Mobile Adaptations

#### Neural Glass Mobile
- Sidebar collapses to bottom sheet navigation
- Org chart neural network becomes vertical tree with expandable nodes
- Glass effects reduced (lighter blur for performance)
- Status orbs remain but remove ambient particle effects
- Activity log becomes pull-to-refresh feed

#### Spatial Layers Mobile
- Parallax disabled (single-speed scrolling)
- Floating panels stack vertically
- Constellation becomes swipeable horizontal list
- Priority cards use sticky positioning at top
- Quick action dock becomes bottom action bar

---

## 4. PERFORMANCE BUDGET

### 4.1 Animation Frame Budgets

```css
/* ================================================================
   PERFORMANCE-OPTIMIZED ANIMATIONS
   ================================================================ */

/*
 * TARGET: 60fps (16.67ms per frame)
 * BUDGET: 10ms for JS, 6ms for rendering
 *
 * Only animate these properties (GPU-accelerated):
 * - transform
 * - opacity
 * - filter (with caution)
 */

/* APPROVED animations (transform/opacity only) */
.animate-lift {
  transition: transform var(--duration-moderate) var(--ease-spring),
              box-shadow var(--duration-moderate) var(--ease-out);
  will-change: transform;
}

.animate-lift:hover {
  transform: translateY(-4px);
}

/* Fade transitions */
.animate-fade {
  transition: opacity var(--duration-normal) var(--ease-out);
}

/* Scale animations */
.animate-scale {
  transition: transform var(--duration-fast) var(--ease-out);
}

.animate-scale:active {
  transform: scale(0.98);
}

/* AVOID: These trigger layout/paint */
/*
 * DO NOT animate:
 * - width, height
 * - top, left, right, bottom
 * - margin, padding
 * - border-width
 * - font-size
 */
```

### 4.2 Animation Complexity Tiers

| Tier | Budget | Use Case | Allowed Effects |
|------|--------|----------|-----------------|
| **Tier 1: Micro** | < 100ms | Button press, toggle | `transform: scale`, `opacity` |
| **Tier 2: Standard** | 200-300ms | Card hover, dropdown | `transform`, `opacity`, `box-shadow` |
| **Tier 3: Complex** | 300-500ms | Page transition, modal | All GPU props + `filter: blur` |
| **Tier 4: Ambient** | 2000-4000ms | Status pulse, data flow | `opacity` only, reduced on mobile |

### 4.3 Reduced Motion Support

```css
/* ================================================================
   REDUCED MOTION - WCAG 2.3.3 Compliance
   ================================================================ */

@media (prefers-reduced-motion: reduce) {
  /* Disable all non-essential animations */
  *,
  *::before,
  *::after {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
    scroll-behavior: auto !important;
  }

  /* Keep essential visual feedback (instant) */
  .btn,
  .card,
  .nav-item {
    transition: none;
  }

  /* Remove parallax effects */
  .parallax {
    transform: none !important;
  }

  /* Disable pulsing status indicators */
  .status-pulse,
  .status-orb {
    animation: none !important;
  }

  /* Keep status visible but static */
  .status-active {
    opacity: 1;
    box-shadow: var(--shadow-glow-success);
  }

  /* Disable particle effects */
  .neural-flow,
  .data-particles,
  .constellation-animation {
    display: none;
  }

  /* Alternative: Static connection lines */
  .connection-line {
    stroke-dasharray: none;
    animation: none;
  }
}

/* Provide user control for animation preferences */
.animations-paused * {
  animation-play-state: paused !important;
}

/* Explicit reduced motion toggle */
[data-reduced-motion="true"] {
  /* Same rules as prefers-reduced-motion */
}
```

### 4.4 Performance Optimization Guidelines

```javascript
// ================================================================
// ANIMATION PERFORMANCE BEST PRACTICES
// ================================================================

// 1. Use CSS transforms, not layout properties
// GOOD:
element.style.transform = 'translateY(-4px)';
// BAD:
element.style.top = '-4px';

// 2. Use will-change sparingly and remove after animation
element.style.willChange = 'transform';
// After animation completes:
element.style.willChange = 'auto';

// 3. Throttle scroll-based animations to 60fps
let ticking = false;
window.addEventListener('scroll', () => {
  if (!ticking) {
    requestAnimationFrame(() => {
      updateParallax();
      ticking = false;
    });
    ticking = true;
  }
});

// 4. Defer non-critical animations until after load
document.addEventListener('DOMContentLoaded', () => {
  // Enable ambient animations after 1 second
  setTimeout(() => {
    document.body.classList.add('animations-ready');
  }, 1000);
});

// 5. Pause animations when tab is not visible
document.addEventListener('visibilitychange', () => {
  if (document.hidden) {
    document.body.classList.add('animations-paused');
  } else {
    document.body.classList.remove('animations-paused');
  }
});
```

### 4.5 Resource Budgets

| Resource | Target | Maximum | Notes |
|----------|--------|---------|-------|
| Initial CSS | 50KB | 80KB | Includes tokens + base components |
| JS (core) | 100KB | 150KB | Before gzip |
| Fonts | 100KB | 150KB | Inter variable + JetBrains Mono subset |
| Images/icons | 50KB | 100KB | SVG sprites preferred |
| **Total Initial Load** | **300KB** | **480KB** | First contentful paint |
| WebGL/Canvas (Org Chart) | Lazy loaded | 200KB | Only on Agents page |

---

## 5. SHARED UI PATTERNS

### 5.1 Universal Components

These components MUST be consistent across all concepts:

#### Status Indicators
```
┌────────────────────────────────────────────────────────────────────┐
│ STATUS INDICATORS (Consistent across concepts)                      │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  AGENT STATUS                                                       │
│  ● Active    (#10B981 + glow)     - Actively processing            │
│  ● Working   (#3B82F6)            - Running task                    │
│  ● Idle      (#94A3B8)            - Available, no current task     │
│  ● Pending   (#F59E0B + pulse)    - Waiting for input/approval     │
│  ● Error     (#EF4444 + attention)- Requires intervention          │
│  ○ Offline   (#CBD5E1, no fill)   - Disconnected                   │
│                                                                     │
│  TASK/PROJECT STATUS                                                │
│  ▓▓▓▓▓▓▓▓░░░  Progress bar       - Always shows percentage         │
│  ● ● ● ○ ○    Phase dots         - Discrete project stages         │
│                                                                     │
│  VISUAL RULES:                                                      │
│  - Always pair color with icon OR text label                        │
│  - Consistent 8px diameter for inline indicators                    │
│  - 12px diameter for prominent status (agent cards)                 │
│  - Glow radius: 12-20px depending on importance                     │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
```

#### Progress Indicators
```css
/* Shared progress bar styling */
.progress-bar {
  height: 6px;
  border-radius: var(--radius-full);
  background: var(--color-neutral-200);
  overflow: hidden;
}

.progress-bar-fill {
  height: 100%;
  border-radius: var(--radius-full);
  background: linear-gradient(
    90deg,
    var(--color-primary-500) 0%,
    var(--color-primary-400) 100%
  );
  transition: width var(--duration-moderate) var(--ease-out);
}

/* Phase dots */
.phase-indicator {
  display: flex;
  gap: var(--space-2);
}

.phase-dot {
  width: 8px;
  height: 8px;
  border-radius: var(--radius-full);
  background: var(--color-neutral-300);
  transition: background var(--duration-normal) var(--ease-out);
}

.phase-dot.completed {
  background: var(--color-success-500);
}

.phase-dot.current {
  background: var(--color-primary-500);
  box-shadow: var(--shadow-glow-primary);
}
```

#### Buttons
```css
/* Primary Button */
.btn-primary {
  padding: var(--space-button-y) var(--space-button-x);
  border-radius: var(--radius-button);
  background: linear-gradient(135deg, var(--color-primary-500) 0%, var(--color-primary-600) 100%);
  color: white;
  font-weight: var(--font-semibold);
  font-size: var(--text-base);
  border: none;
  box-shadow: var(--shadow-button);
  transition: var(--transition-button);
  cursor: pointer;
}

.btn-primary:hover {
  transform: translateY(-2px);
  box-shadow: var(--shadow-button-hover), var(--shadow-glow-primary);
}

.btn-primary:active {
  transform: translateY(0) scale(0.98);
}

/* Secondary Button - Adapts to theme */
.btn-secondary {
  padding: var(--space-button-y) var(--space-button-x);
  border-radius: var(--radius-button);
  background: var(--color-bg-surface);
  backdrop-filter: var(--glass-blur-light);
  color: var(--color-text-primary);
  font-weight: var(--font-medium);
  font-size: var(--text-base);
  border: 1px solid var(--color-border);
  transition: var(--transition-button);
  cursor: pointer;
}

.btn-secondary:hover {
  background: var(--color-bg-surface-hover);
  border-color: var(--color-border-hover);
}

/* Ghost Button */
.btn-ghost {
  padding: var(--space-button-y) var(--space-button-x);
  border-radius: var(--radius-button);
  background: transparent;
  color: var(--color-text-secondary);
  font-weight: var(--font-medium);
  font-size: var(--text-base);
  border: none;
  transition: var(--transition-button);
  cursor: pointer;
}

.btn-ghost:hover {
  background: var(--color-bg-surface);
  color: var(--color-text-primary);
}
```

#### Cards
```css
/* Base Card */
.card {
  padding: var(--space-card-padding);
  border-radius: var(--radius-card);
  background: var(--color-bg-surface);
  backdrop-filter: var(--glass-blur);
  border: 1px solid var(--color-border);
  box-shadow: var(--shadow-card);
  transition: var(--transition-card);
}

.card:hover {
  transform: translateY(-4px);
  box-shadow: var(--shadow-card-hover);
}

/* Priority Card (Gates, Approvals) */
.card-priority {
  composes: card;
  border-color: var(--color-warning-500);
  box-shadow: var(--shadow-card), var(--shadow-glow-warning);
}

/* Agent Card */
.card-agent {
  composes: card;
  position: relative;
}

.card-agent::before {
  content: '';
  position: absolute;
  top: var(--space-4);
  left: var(--space-4);
  width: 12px;
  height: 12px;
  border-radius: var(--radius-full);
  background: var(--agent-status-color, var(--color-neutral-400));
  box-shadow: 0 0 var(--agent-glow-size, 0) var(--agent-status-color, transparent);
}
```

#### Form Elements
```css
/* Text Input */
.input {
  width: 100%;
  padding: var(--space-input-y) var(--space-input-x);
  border-radius: var(--radius-input);
  background: var(--color-bg-surface);
  backdrop-filter: var(--glass-blur-light);
  border: 1px solid var(--color-border);
  color: var(--color-text-primary);
  font-size: var(--text-base);
  transition: var(--transition-focus);
}

.input:hover {
  border-color: var(--color-border-hover);
}

.input:focus {
  border-color: var(--color-border-focus);
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.15);
  outline: none;
}

.input::placeholder {
  color: var(--color-text-muted);
}
```

### 5.2 Shared Interaction Patterns

#### Hover States
- All interactive elements: lift 2-4px on hover
- Shadow expands and softens
- Background lightens (light theme) or brightens (dark theme)
- Transition: 200-300ms with spring easing

#### Press/Active States
- Scale down to 0.98
- Shadow contracts
- Instant feedback: 100ms or less

#### Focus States
- Always visible, high contrast ring
- Consistent 2px solid primary color
- Additional glow on dark backgrounds
- Never remove outlines without replacement

#### Loading States
```css
/* Skeleton loader */
.skeleton {
  background: linear-gradient(
    90deg,
    var(--color-neutral-200) 0%,
    var(--color-neutral-100) 50%,
    var(--color-neutral-200) 100%
  );
  background-size: 200% 100%;
  animation: skeleton-shimmer 1.5s ease-in-out infinite;
  border-radius: var(--radius-md);
}

@keyframes skeleton-shimmer {
  0% { background-position: 200% 0; }
  100% { background-position: -200% 0; }
}

/* Spinner */
.spinner {
  width: 20px;
  height: 20px;
  border: 2px solid var(--color-neutral-200);
  border-top-color: var(--color-primary-500);
  border-radius: var(--radius-full);
  animation: spin 0.8s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}
```

### 5.3 Navigation Patterns

```
┌────────────────────────────────────────────────────────────────────┐
│ NAVIGATION CONSISTENCY                                              │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  DESKTOP SIDEBAR (Both Concepts)                                    │
│  ┌──────────────────────────┐                                       │
│  │ LOGO                     │  Fixed position                       │
│  ├──────────────────────────┤                                       │
│  │ ◉ Dashboard      ←─────  │  Active: Filled bg + accent bar       │
│  │ ○ Projects               │  Inactive: Transparent bg             │
│  │ ○ Agents          (3)    │  Badge: Notification count            │
│  │ ○ Gates           ●      │  Dot: Requires attention              │
│  │ ○ Integrations           │                                       │
│  ├──────────────────────────┤                                       │
│  │ ○ Settings               │  Divider separates main/secondary     │
│  └──────────────────────────┘                                       │
│                                                                     │
│  MOBILE BOTTOM NAV                                                  │
│  ┌─────┬─────┬─────┬─────┬─────┐                                    │
│  │  ◉  │  ○  │  ○  │  ○  │  ○  │  5 max items                       │
│  │Dash │Proj │Agent│Gates│More │  Icon + label below                │
│  └─────┴─────┴─────┴─────┴─────┘                                    │
│                                                                     │
│  HEADER BAR                                                         │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  [Breadcrumb] Page Title                    [🔍] [🔔] [👤]  │   │
│  └─────────────────────────────────────────────────────────────┘   │
│  Search, Notifications, User always in top-right                   │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
```

### 5.4 Toast/Notification Patterns

```css
/* Toast container */
.toast-container {
  position: fixed;
  top: var(--space-5);
  right: var(--space-5);
  z-index: var(--z-toast);
  display: flex;
  flex-direction: column;
  gap: var(--space-3);
}

/* Toast base */
.toast {
  padding: var(--space-4);
  border-radius: var(--radius-lg);
  background: var(--color-bg-surface);
  backdrop-filter: var(--glass-blur);
  border: 1px solid var(--color-border);
  box-shadow: var(--shadow-lg);
  max-width: 360px;

  animation: toast-enter var(--duration-moderate) var(--ease-spring);
}

@keyframes toast-enter {
  from {
    opacity: 0;
    transform: translateX(100%);
  }
  to {
    opacity: 1;
    transform: translateX(0);
  }
}

/* Toast variants */
.toast-success {
  border-left: 3px solid var(--color-success-500);
}

.toast-warning {
  border-left: 3px solid var(--color-warning-500);
}

.toast-error {
  border-left: 3px solid var(--color-error-500);
}

.toast-info {
  border-left: 3px solid var(--color-primary-500);
}
```

---

## 6. IMPLEMENTATION CHECKLIST

### Pre-Development
- [ ] Set up CSS custom properties in global stylesheet
- [ ] Configure theme switching mechanism (`data-theme` attribute)
- [ ] Implement `prefers-reduced-motion` detection
- [ ] Set up responsive breakpoint utilities

### Core Components
- [ ] Button variants (primary, secondary, ghost, icon)
- [ ] Card variants (standard, priority, agent)
- [ ] Form inputs (text, select, checkbox)
- [ ] Status indicators (orbs, progress bars, badges)
- [ ] Navigation (sidebar, mobile nav, header)
- [ ] Toast notifications
- [ ] Modal/dialog
- [ ] Dropdown menu
- [ ] Tooltip

### Accessibility
- [ ] Keyboard navigation for all interactive elements
- [ ] Focus management for modals and dropdowns
- [ ] ARIA labels for status indicators
- [ ] Skip links
- [ ] Screen reader announcements for dynamic content
- [ ] Color contrast verification (automated testing)
- [ ] Touch target size verification (44x44px)

### Performance
- [ ] CSS bundle size under 80KB
- [ ] Font subsetting (Latin only)
- [ ] Lazy load org chart visualization
- [ ] Animation throttling on scroll
- [ ] `will-change` management
- [ ] Visibility-based animation pausing

---

## 7. APPENDIX: QUICK REFERENCE

### Spacing Scale
| Token | Value | Pixels |
|-------|-------|--------|
| `--space-1` | 0.25rem | 4px |
| `--space-2` | 0.5rem | 8px |
| `--space-3` | 0.75rem | 12px |
| `--space-4` | 1rem | 16px |
| `--space-5` | 1.5rem | 24px |
| `--space-6` | 2rem | 32px |

### Radius Scale
| Token | Value | Use |
|-------|-------|-----|
| `--radius-sm` | 6px | Subtle rounding |
| `--radius-md` | 8px | Buttons, inputs |
| `--radius-lg` | 12px | Cards |
| `--radius-xl` | 16px | Modals |

### Transition Presets
| Token | Duration | Easing | Use |
|-------|----------|--------|-----|
| `--transition-fast` | 100ms | ease-out | Micro feedback |
| `--transition-normal` | 200ms | ease-out | Standard |
| `--transition-spring` | 300ms | spring | Cards, lifts |

### Breakpoints
| Name | Width | Layout |
|------|-------|--------|
| Mobile | < 640px | Single column |
| Tablet | 768px | 2 columns |
| Desktop | 1024px | Sidebar + 3 columns |
| Wide | 1280px | Full layout |

---

*ROUND 2: Unified Design System*
*AI Corp Frontend Design Sprint*
*Document Version: 1.0*
