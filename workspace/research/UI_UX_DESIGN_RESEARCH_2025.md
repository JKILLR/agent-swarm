# UI/UX Design Best Practices 2025-2026
## Comprehensive Research for AI Design Agent Prompts

**Date**: 2026-01-05
**Purpose**: Expert-level knowledge base for design swarm agents

---

## Table of Contents
1. [Core UI/UX Principles](#1-core-uiux-principles)
2. [Modern Design Systems](#2-modern-design-systems)
3. [Design Trends 2025-2026](#3-design-trends-2025-2026)
4. [Accessibility Standards](#4-accessibility-standards)
5. [Tools & Frameworks](#5-tools--frameworks)
6. [Design-to-Code Workflow](#6-design-to-code-workflow)
7. [Mobile-First & Responsive Design](#7-mobile-first--responsive-design)
8. [Color Systems](#8-color-systems)
9. [Typography](#9-typography)
10. [Spacing Systems](#10-spacing-systems)

---

## 1. Core UI/UX Principles

### Fundamental Concepts
- **UI (User Interface)**: How a product looks - colors, buttons, typography, layout
- **UX (User Experience)**: How users interact with the product - flow, usability, satisfaction

### Essential Design Principles

#### 1.1 Simplicity & Clarity
- Keep designs clean and free of clutter
- Every element should be easy to understand
- If users ever have to stop and think, clarity is missing
- Reduce cognitive load by removing unnecessary elements

#### 1.2 Consistency
- Creates predictability and builds user confidence
- Reduces learning curves
- Inconsistent interfaces are a major source of user frustration and abandonment (Baymard Institute)
- Apply consistent patterns for: navigation, buttons, forms, spacing, colors

#### 1.3 Visual Hierarchy
- **60-30-10 Rule**: 60% dominant color (backgrounds), 30% secondary (key elements), 10% accent (CTAs)
- Use size, color, contrast, and position to guide attention
- Most important elements should be most prominent

#### 1.4 Proximity & Grouping
- Related elements should be visually close
- Reduces scan time and improves comprehension
- Place form labels next to their fields
- Use card layouts to cluster similar content

#### 1.5 Whitespace (Negative Space)
- Whitespace creates relationships between elements
- Do not cram elements together
- Provides visual breathing room
- Improves readability and focus

### Key UX Laws

#### Jakob's Law
Users expect your product to work the same way similar products work. Leverage familiar patterns.

#### Fitts's Law
The time to acquire a target is a function of distance and size. Make clickable areas large and easy to reach.

#### Hick's Law
Decision time increases with the number of choices. Minimize options to reduce cognitive load.

#### Tesler's Law
All systems have inherent complexity that cannot be removed. Do not sacrifice functional elements for absolute simplicity.

---

## 2. Modern Design Systems

### What is a Design System?
A design system is a complete set of standards, documentation, and reusable components that guide the creation of digital products.

### Design System Architecture (2025)

#### 2.1 Design Tokens (Foundation Layer)

**W3C Design Tokens Specification 2025.10** (Published October 28, 2025)
- First stable, production-ready, vendor-neutral format
- Organizations involved: Adobe, Amazon, Google, Microsoft, Meta, Figma, Salesforce, Shopify, Disney, and many others

**Token Types:**
1. **Primitive Tokens**: Base values (e.g., `color-blue-500: #3b82f6`)
2. **Semantic Tokens**: Meaning-based (e.g., `color-primary: {color-blue-500}`)
3. **Component Tokens**: Component-specific (e.g., `button-background: {color-primary}`)

**Token Categories:**
- Colors (now supporting OKLCH, Display P3, CSS Color Module 4)
- Typography (font family, size, weight, line height)
- Spacing (margins, padding, gaps)
- Shadows
- Border radius
- Animation/Motion
- Breakpoints

**Modern Token Features:**
- Theming support (light/dark modes, brand variants)
- Inheritance and aliases
- Cross-platform generation (iOS, Android, web, Flutter)

#### 2.2 Component Architecture

**Atomic Design Methodology:**
1. **Atoms**: Basic building blocks (buttons, inputs, labels)
2. **Molecules**: Simple combinations (search form = input + button)
3. **Organisms**: Complex components (navigation, hero sections)
4. **Templates**: Page-level layouts
5. **Pages**: Final implementations with real content

#### 2.3 Pattern Libraries
- Documented solutions to common UI problems
- Include usage guidelines and code examples
- Cover interaction patterns, not just visual patterns

### Notable Design Systems to Reference

1. **Adobe Spectrum**: Comprehensive with motion, copywriting, inclusive design guidance
2. **Gov.uk Design System**: User-tested, research-validated, accessibility-first
3. **Carbon (IBM)**: Strong token system, comprehensive documentation
4. **Material Design (Google)**: Well-documented, widely recognized patterns
5. **Lightning (Salesforce)**: Enterprise-focused, accessibility compliant

### Modern Best Practices
- Synchronization of Figma, design tokens, and frontend in single pipeline
- Automated accessibility checks in CI
- Documentation generation without manual editing
- Single source of truth across design and development

---

## 3. Design Trends 2025-2026

### Visual Styles

#### 3.1 Glass & Transparency Effects
- **Liquid Glass**: Apple-inspired, translucent surfaces with depth and motion
- **Glassmorphism**: Blurred backgrounds, subtle layering, frosted glass effect
- Creates depth and hierarchy without clutter
- Figma has added native glass effect support

#### 3.2 Typography Trends
- **Big Typography**: Oversized, bold text as primary visual element
- **Dynamic Typography**: Text that adapts and responds to interaction
- **Modular Type Systems**: Letters made from shapes that can rearrange
- Variable fonts for weight and style flexibility

#### 3.3 Motion Design
- Quieter, smarter animations focused on rhythm and flow
- Micro-interactions for feedback and delight
- Scrollytelling for data visualization
- Performance-conscious animations

#### 3.4 Depth & Dimension
- **Neumorphism 2.0**: Soft shadows and highlights with improved accessibility
- Flat design with subtle 3D details
- Spatial design for XR devices (Apple Vision Pro, Meta Quest)

#### 3.5 Layout Patterns
- **Bento Grids**: Clean, organized, responsive card layouts
- Asymmetrical layouts for visual interest
- Full-screen sections with clear visual breaks

### Color Trends
- Softer holographic effects (not rainbow overload)
- Bold contrasts (dark blue + neon orange)
- Dark mode as standard expectation
- OKLCH for perceptually uniform color systems

### Aesthetic Directions
- Nostalgic web elements (pixel icons, retro gradients) with modern execution
- Softened brutalism
- Emotional, imperfect designs
- Human expression over perfection

### AI Integration
- AI generating usable UI screens (Galileo, Uizard, Figma Make)
- Designers focusing on refining and strategizing
- Real-time interface personalization

---

## 4. Accessibility Standards

### WCAG 2.2 Overview (Current Standard)

**Published**: October 5, 2023 (Updated December 12, 2024)
**ISO Standard**: ISO/IEC 40500:2025

#### Four Principles (POUR)
1. **Perceivable**: Information must be presentable in ways users can perceive
2. **Operable**: UI must be operable by all users
3. **Understandable**: Information and operation must be understandable
4. **Robust**: Content must be robust enough for assistive technologies

#### Conformance Levels
- **Level A**: Minimum accessibility (must meet)
- **Level AA**: Standard target (most regulations require)
- **Level AAA**: Enhanced accessibility (optimal)

### Key WCAG 2.2 Success Criteria

#### Visual & Contrast
- **1.4.3 Contrast (Minimum)**: 4.5:1 for normal text, 3:1 for large text (AA)
- **1.4.6 Contrast (Enhanced)**: 7:1 for normal text, 4.5:1 for large text (AAA)
- **1.4.11 Non-text Contrast**: 3:1 for UI components and graphics

#### Keyboard & Navigation
- **2.1.1 Keyboard**: All functionality available via keyboard
- **2.4.7 Focus Visible**: Keyboard focus indicator must be visible
- **2.4.11 Focus Not Obscured (Minimum)**: Focus must not be hidden (New in 2.2)

#### Target Size (New in WCAG 2.2)
- **2.5.8 Target Size (Minimum)**: Interactive targets at least 24x24 CSS pixels
- Exception for inline links within text

#### Forms & Input
- **3.3.7 Redundant Entry**: Don't require re-entering previously provided info (New in 2.2)
- **3.3.8 Accessible Authentication (Minimum)**: No cognitive tests for login (New in 2.2)

### Regulatory Landscape 2025-2026
- **US DOJ**: WCAG 2.1 Level AA required by April 24, 2026
- **European Accessibility Act (EAA)**: Expected to adopt WCAG 2.2 in 2025
- **Section 508**: Updated to reference WCAG 2.0 Level AA

### Implementation Best Practices
- Semantic HTML first (proper heading hierarchy, landmarks, form labels)
- ARIA only when HTML is insufficient
- Color never as sole indicator of meaning
- Clear focus states
- Skip navigation links
- Descriptive alt text for images
- Accessible forms with proper labels and error messages
- Sufficient touch target sizes (minimum 44x44px recommended)

### Testing Tools
- axe DevTools
- WAVE
- Lighthouse
- Color contrast checkers
- Screen reader testing (VoiceOver, NVDA, JAWS)

---

## 5. Tools & Frameworks

### CSS Frameworks

#### Tailwind CSS v4 (2025)
**Key Features:**
- OKLCH color system (converted from HSL)
- `@theme` directive for design tokens
- `tw-animate-css` replacing `tailwindcss-animate`
- Improved performance and smaller bundles
- Container queries support

**Best Practices:**
```css
/* Use CSS variables for theming */
@theme {
  --color-primary: oklch(0.7 0.15 250);
  --spacing-sm: 0.5rem;
}

/* Responsive with mobile-first */
<div class="p-4 md:p-6 lg:p-8">

/* Container queries */
@container (min-width: 400px) { ... }
```

### Component Libraries

#### shadcn/ui
- Not a traditional library - copy-paste components into your project
- Built on Radix UI primitives + Tailwind CSS
- Full code ownership (no dependency lock-in)
- WCAG-compliant by default
- CLI-based installation

**When to Use:**
- Custom design systems
- Long-term maintainability priority
- Full control over component code

#### Radix UI
- Headless (unstyled) component primitives
- Battle-tested accessibility
- Composable and customizable
- **Note**: Creators announced reduced active maintenance (2025)

#### Other Popular Libraries
- **MUI (Material UI)**: Quick development, professional defaults
- **Chakra UI**: Good DX, accessible, themeable
- **Mantine**: Feature-rich, TypeScript-first
- **Ant Design**: Enterprise-focused, comprehensive

### Extended shadcn Ecosystem
- **Origin UI**: 400+ components, 25+ categories (free)
- **Cult-UI**: AI-powered blocks, Next.js templates
- **Magic UI**: Animation-focused components
- **Aceternity UI**: Creative, animated components

### Key Principles for Tool Selection
1. Radix primitives for behavior and accessibility
2. Tailwind for declarative, token-driven styling
3. Plain React - no custom renderers or wrappers
4. Keep `components/ui/overrides` folder for customizations
5. Declare dark variants early in development

---

## 6. Design-to-Code Workflow

### Figma Dev Mode (2025)

**Core Capabilities:**
- Inspect component properties (colors, tokens, spacing)
- View designs marked as "Ready for dev"
- Focus view for isolating specific components
- Generate code snippets (CSS, React, etc.)
- VS Code extension for in-editor inspection

**Key Features:**
- Real-time collaboration
- Design-to-code inspection
- Asset export
- Component property access
- Token value display

### Design Handoff Best Practices

#### For Designers
1. Use consistent naming conventions for styles and components
2. Mark designs as "Ready for dev" when complete
3. Use components and variants (not one-off elements)
4. Apply auto-layout for responsive behavior
5. Document interaction states (hover, focus, active, disabled)
6. Include spacing and padding specifications

#### For Collaboration
1. Define shared naming conventions early
2. Start collaboration with developers at design stage
3. Use a unified design language across the team
4. Document design decisions and rationale
5. Regular sync meetings to address questions

### Reality Check
- Generated code serves as foundation, not final product
- Developers must refine, optimize, and add interactivity
- Performance optimization still required
- Responsiveness needs developer attention

### Integration Tools
- **Figma + Jira/Confluence**: Project management
- **Figma + Notion/Slack**: Communication
- **Zeplin**: Alternative handoff tool
- **Storybook**: Component documentation and testing

### Modern Workflow Pipeline
```
Figma Design
    |
    v
Design Tokens (JSON/CSS Variables)
    |
    v
Component Library (shadcn/ui, custom)
    |
    v
Storybook (documentation, testing)
    |
    v
Production Application
```

---

## 7. Mobile-First & Responsive Design

### Mobile-First Principle
- **70%+ of web traffic** comes from mobile devices
- Design smallest viewport first
- Progressively enhance for larger screens
- Results in better Core Web Vitals

### Recommended Breakpoints (2025)

```css
/* Mobile First Approach */
/* Base styles = mobile (< 480px) */

/* Small devices */
@media (min-width: 480px) { }

/* Tablets */
@media (min-width: 768px) { }

/* Small desktop */
@media (min-width: 1024px) { }

/* Large desktop */
@media (min-width: 1280px) { }

/* Extra large */
@media (min-width: 1440px) { }
```

### Device Statistics (North America 2025)
- iPhone (375x812): 16.79%
- Android (390x844): 13.72%
- 480px breakpoint covers ~50% of mobile devices

### Best Practices

#### 1. Content-Driven Breakpoints
- Let content determine breakpoints
- Change layout when it starts to look bad
- Avoid rigid device-specific widths

#### 2. Layout vs. Component Breakpoints
- **Global breakpoints**: Page structure changes
- **Component breakpoints**: Local responsive behavior

#### 3. Modern CSS Features
```css
/* Container Queries */
@container (min-width: 400px) {
  .card { flex-direction: row; }
}

/* Fluid Typography */
font-size: clamp(1rem, 2vw + 0.5rem, 1.5rem);

/* Fluid Spacing */
padding: clamp(1rem, 5vw, 3rem);
```

#### 4. Flexible Units
- Use `rem` for consistent scaling
- Use `em` for component-relative sizing
- Use `%` and `vw/vh` for fluid layouts
- Pixels for breakpoints (consistent behavior)

#### 5. Touch-Friendly Design
- Minimum touch target: 44x44px (recommended)
- WCAG 2.2 minimum: 24x24px
- Adequate spacing between interactive elements
- Consider thumb zones on mobile

### Testing
- Test at various widths, not just breakpoints
- Test portrait and landscape orientations
- Use Chrome DevTools for live tuning
- Validate with real device testing

---

## 8. Color Systems

### OKLCH: The New Standard

**Why OKLCH over RGB/HSL:**
1. **Perceptual uniformity**: Equal changes in values = equal perceived changes
2. **Consistent contrast**: Same lightness values maintain contrast across hues
3. **Efficient token generation**: Calculate scales with simple math
4. **Wide gamut support**: P3, Rec.2020 color spaces

**OKLCH Structure:**
- **L** (Lightness): 0-1 (0 = black, 1 = white)
- **C** (Chroma): 0-0.4 (saturation intensity)
- **H** (Hue): 0-360 (color wheel position)

**Browser Support**: 93.1% as of September 2025

### CSS Implementation

```css
/* OKLCH with fallback */
:root {
  --color-primary: oklch(0.6 0.15 250);
  --color-primary-rgb: #3b82f6; /* Fallback */
}

/* Dynamic color manipulation */
.hover-state {
  background: oklch(from var(--color-primary) calc(l - 0.1) c h);
}

/* Tailwind v4 uses OKLCH by default */
```

### Color Scale Architecture

```css
:root {
  /* Primitive tokens */
  --blue-50: oklch(0.97 0.01 250);
  --blue-100: oklch(0.93 0.03 250);
  --blue-500: oklch(0.6 0.15 250);
  --blue-900: oklch(0.25 0.12 250);

  /* Semantic tokens */
  --color-primary: var(--blue-500);
  --color-primary-hover: var(--blue-600);
  --color-background: var(--gray-50);
  --color-text: var(--gray-900);

  /* Component tokens */
  --button-bg: var(--color-primary);
  --button-bg-hover: var(--color-primary-hover);
}
```

### Dark Mode Strategy

```css
:root {
  --color-background: oklch(0.98 0 0);
  --color-text: oklch(0.15 0 0);
}

[data-theme="dark"] {
  --color-background: oklch(0.15 0 0);
  --color-text: oklch(0.95 0 0);
}

/* Or with media query */
@media (prefers-color-scheme: dark) {
  :root { ... }
}
```

### Accessibility Considerations
- WCAG AA: 4.5:1 contrast for normal text, 3:1 for large text
- WCAG AAA: 7:1 for normal text, 4.5:1 for large text
- Use contrast checking tools
- Test in grayscale to verify hierarchy
- Never rely on color alone for meaning

---

## 9. Typography

### Typography Scale

**Base Size**: 16-18px (1rem)

**Common Scales:**
- **Minor Second (1.067)**: Subtle variation
- **Major Second (1.125)**: Moderate contrast
- **Minor Third (1.200)**: Good for body text
- **Major Third (1.250)**: Strong hierarchy
- **Perfect Fourth (1.333)**: Dramatic contrast
- **Golden Ratio (1.618)**: Classic proportions

**Example Scale (1.25 ratio):**
```css
:root {
  --text-xs: 0.64rem;   /* 10.24px */
  --text-sm: 0.8rem;    /* 12.8px */
  --text-base: 1rem;    /* 16px */
  --text-lg: 1.25rem;   /* 20px */
  --text-xl: 1.563rem;  /* 25px */
  --text-2xl: 1.953rem; /* 31.25px */
  --text-3xl: 2.441rem; /* 39px */
  --text-4xl: 3.052rem; /* 48.8px */
}
```

### Fluid Typography

```css
/* Using clamp() for responsive sizing */
h1 {
  font-size: clamp(2rem, 5vw + 1rem, 4rem);
}

body {
  font-size: clamp(1rem, 0.5vw + 0.875rem, 1.125rem);
}
```

### Line Height Guidelines
- **Body text**: 1.4 - 1.7 (1.5 is safe default)
- **Headings**: 1.1 - 1.3
- **Tight text**: 1.25
- **8pt grid alignment**: Line heights should be multiples of 8px

### Line Length
- **Optimal**: 45-85 characters
- **Ideal**: 65-75 characters
- Use `max-width: 65ch` for paragraph containers

### Font Weight Guidelines
- **Body text**: 400-500 (regular to medium)
- **Headings**: 600-700 (semi-bold to bold)
- **Avoid**: Ultra-light (300 and below) for small text

### Font Pairing Strategies

**Contrast Methods:**
1. **Style contrast**: Serif headline + sans-serif body
2. **Weight contrast**: Bold header + regular body
3. **Size contrast**: Dramatic size differences

**Classic Pairings:**
- Playfair Display (serif) + Lato (sans-serif)
- Montserrat (sans-serif) + Merriweather (serif)
- Inter (sans-serif) + Source Serif Pro (serif)
- Roboto (sans-serif) + Roboto Slab (slab-serif)

**General Rule**: Two fonts maximum, three only for specific roles

### Variable Fonts

**Benefits:**
- Single file for multiple weights/styles
- Smooth weight transitions
- Reduced HTTP requests
- Optical size adjustments

```css
@font-face {
  font-family: 'Inter';
  src: url('Inter-VariableFont.woff2') format('woff2');
  font-weight: 100 900;
}

.heading {
  font-variation-settings: 'wght' 700;
}
```

### Performance
- Use `font-display: swap` for most cases
- Preload critical fonts
- Self-host for better performance
- Use WOFF2 format

---

## 10. Spacing Systems

### The 8pt Grid System

**Core Principle**: Base all spacing on multiples of 8 pixels

**Scale:**
```css
:root {
  --space-1: 0.25rem;  /* 4px - half step */
  --space-2: 0.5rem;   /* 8px */
  --space-3: 0.75rem;  /* 12px */
  --space-4: 1rem;     /* 16px */
  --space-5: 1.25rem;  /* 20px */
  --space-6: 1.5rem;   /* 24px */
  --space-8: 2rem;     /* 32px */
  --space-10: 2.5rem;  /* 40px */
  --space-12: 3rem;    /* 48px */
  --space-16: 4rem;    /* 64px */
}
```

### Why 8pt?
1. Divides evenly into common screen sizes
2. Works across different pixel densities
3. Provides enough granularity without chaos
4. Recommended by Apple and Google

### 4pt Half-Step
- Use for fine adjustments (icons, small text blocks)
- Secondary information spacing
- Icon padding
- Small gaps where 8px is too much

### Spacing Token Naming

**Option 1: Numeric**
```css
--spacing-4, --spacing-8, --spacing-16
```

**Option 2: T-Shirt Sizes**
```css
--spacing-xxs, --spacing-xs, --spacing-sm, --spacing-md, --spacing-lg, --spacing-xl
```

**Option 3: Semantic**
```css
--spacing-component-padding
--spacing-section-gap
--spacing-card-internal
```

### Application Guidelines

**Component Internal Spacing:**
- Padding inside buttons, cards, inputs
- Use smaller values (8-16px)

**Component External Spacing:**
- Gaps between components
- Margins between sections
- Use larger values (24-64px)

**Rule**: Internal spacing <= External spacing

### CSS Implementation

```css
/* With root font-size: 16px */
/* 0.5rem = 8px, 1rem = 16px, etc. */

.card {
  padding: var(--space-4);  /* 16px */
  gap: var(--space-2);      /* 8px */
  margin-bottom: var(--space-6); /* 24px */
}
```

### Typography on 8pt Grid
- Font sizes can vary (e.g., 15px)
- Line heights MUST be multiples of 8 (e.g., 24px)
- Ensures baseline alignment

---

## Quick Reference for AI Agents

### Design Agent Checklist

#### Before Starting
- [ ] Understand user requirements and context
- [ ] Identify primary user personas
- [ ] Define accessibility requirements

#### Visual Design
- [ ] Apply 60-30-10 color rule
- [ ] Use 8pt grid for all spacing
- [ ] Establish clear visual hierarchy
- [ ] Ensure WCAG AA contrast ratios
- [ ] Design for mobile-first

#### Components
- [ ] Use semantic HTML structure
- [ ] Follow atomic design principles
- [ ] Include all interaction states
- [ ] Ensure minimum touch targets (44x44px)
- [ ] Document component API

#### Code Output
- [ ] Use Tailwind CSS utility classes
- [ ] Apply fluid typography with clamp()
- [ ] Use CSS custom properties for tokens
- [ ] Include responsive breakpoints
- [ ] Add ARIA labels where needed

### Code Patterns

```jsx
// Modern React component with Tailwind + shadcn approach
export function Card({ title, children, variant = "default" }) {
  return (
    <div className={cn(
      "rounded-lg border bg-card p-6",
      "shadow-sm transition-shadow hover:shadow-md",
      variants[variant]
    )}>
      <h3 className="text-lg font-semibold mb-4">{title}</h3>
      <div className="text-muted-foreground">{children}</div>
    </div>
  );
}
```

```css
/* Design token setup */
@theme {
  --color-background: oklch(0.98 0 0);
  --color-foreground: oklch(0.15 0 0);
  --color-primary: oklch(0.6 0.15 250);
  --color-muted: oklch(0.55 0 0);

  --radius-sm: 0.25rem;
  --radius-md: 0.5rem;
  --radius-lg: 1rem;

  --space-1: 0.25rem;
  --space-2: 0.5rem;
  --space-4: 1rem;
  --space-6: 1.5rem;
  --space-8: 2rem;
}
```

---

## Sources

### Core UI/UX Principles
- [FullStack UX/UI Design Trends 2025](https://www.fullstack.com/labs/resources/blog/top-5-ux-ui-design-trends-in-2025-the-future-of-user-experiences)
- [UI/UX Design in 2025: Principles, Process & Trends - DEV Community](https://dev.to/vrajparikh/uiux-design-in-2025-principles-process-trends-5dl5)
- [10 Fundamentals UX/UI Laws for Designers 2025](https://uxplaybook.org/articles/10-ui-ux-fundamental-laws-2025)
- [UX Planet: 7 Concepts Every UI/UX Designer Should Know in 2025](https://uxplanet.org/7-concepts-every-ui-ux-designer-should-know-in-2025-accea5d71b06)

### Design Systems
- [W3C Design Tokens Specification 2025.10](https://www.w3.org/community/design-tokens/2025/10/28/design-tokens-specification-reaches-first-stable-version/)
- [Contentful: Design Tokens Explained](https://www.contentful.com/blog/design-token-system/)
- [Modern Design Systems for React in 2025](https://inwald.com/2025/11/modern-design-systems-for-react-in-2025-a-pragmatic-comparison/)
- [Best Design System Examples](https://www.adhamdannaway.com/blog/design-systems/design-system-examples)

### Design Trends
- [Muzli: UI Design Trends 2025](https://muz.li/blog/ui-design-trends-in-2025/)
- [UX Studio: UI Trends 2026](https://www.uxstudioteam.com/ux-blog/ui-trends-2019)
- [UX/UI Design Trends for 2026](https://blog.prototypr.io/ux-ui-design-trends-for-2026-from-ai-to-xr-to-vibe-creation-7c5f8e35dc1d)
- [12 UI/UX Design Trends 2026 (Data-Backed)](https://www.index.dev/blog/ui-ux-design-trends)

### Accessibility
- [W3C WCAG 2 Overview](https://www.w3.org/WAI/standards-guidelines/wcag/)
- [WebAIM WCAG 2 Checklist](https://webaim.org/standards/wcag/checklist)
- [WCAG 2.2 AA Summary and Checklist](https://www.levelaccess.com/blog/wcag-2-2-aa-summary-and-checklist-for-website-owners/)
- [DigitalA11Y WCAG Checklist](https://www.digitala11y.com/wcag-checklist/)

### Tools & Frameworks
- [shadcn/ui Tailwind v4](https://ui.shadcn.com/docs/tailwind-v4)
- [ShadCN UI vs Radix UI vs Tailwind UI Comparison](https://javascript.plainenglish.io/shadcn-ui-vs-radix-ui-vs-tailwind-ui-which-should-you-choose-in-2025-b8b4cadeaa25)
- [Vercel Academy: React UI with shadcn/ui](https://vercel.com/academy/shadcn-ui)
- [React UI Libraries 2025 Comparison](https://makersden.io/blog/react-ui-libs-2025-comparing-shadcn-radix-mantine-mui-chakra)

### Design-to-Code Workflow
- [Figma Developer Handoff Guide](https://www.figma.com/best-practices/guide-to-developer-handoff/)
- [Figma Dev Mode](https://www.figma.com/dev-mode/)
- [Figma Developer Handoff in 2025](https://edesignify.com/blogs/figma-developer-handoff-in-2025-can-it-truly-deliver-productionready-code)

### Responsive Design
- [BrowserStack: Responsive Design Breakpoints 2025](https://www.browserstack.com/guide/responsive-design-breakpoints)
- [Responsive Design Best Practices 2025](https://nextnative.dev/blog/responsive-design-best-practices)
- [Responsive Design Breakpoints Playbook](https://dev.to/gerryleonugroho/responsive-design-breakpoints-2025-playbook-53ih)

### Color Systems
- [Evil Martians: OKLCH in CSS](https://evilmartians.com/chronicles/oklch-in-css-why-quit-rgb-hsl)
- [UX Collective: OKLCH Explained for Designers](https://uxdesign.cc/oklch-explained-for-designers-dc6af4433611)
- [LogRocket: OKLCH for Consistent Color Palettes](https://blog.logrocket.com/oklch-css-consistent-accessible-color-palettes)

### Typography
- [design.dev Typography Guide](https://design.dev/guides/typography-web-design/)
- [Guide to Responsive Typography Sizing and Scales](https://designshack.net/articles/typography/guide-to-responsive-typography-sizing-and-scales/)
- [Modern Web Typography Techniques 2025](https://www.frontendtools.tech/blog/modern-web-typography-techniques-2025-readability-guide)

### Spacing Systems
- [8pt Grid System Guide](https://www.rejuvenate.digital/news/designing-rhythm-power-8pt-grid-ui-design)
- [Spacing Best Practices](https://cieden.com/book/sub-atomic/spacing/spacing-best-practices)
- [Designer's Ultimate Spacing Guide](https://hakan-ertan.com/designers-ultimate-spacing-guide-from-design-tokens-to-final-design/)
- [Carbon Design System Spacing](https://carbondesignsystem.com/elements/spacing/overview/)
