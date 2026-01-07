# Design System Reference

**Last Updated**: 2026-01-05
**Version**: 1.0.0

This is the source of truth for the design swarm's design system tokens, patterns, and standards.

---

## 1. Color System (OKLCH)

### Why OKLCH?
- Perceptually uniform (equal changes = equal perceived changes)
- Consistent contrast across hues
- Wide gamut support (P3, Rec.2020)
- Browser support: 93.1%+

### Base Palette

```css
@theme {
  /* Neutrals */
  --color-gray-50: oklch(0.98 0 0);
  --color-gray-100: oklch(0.96 0 0);
  --color-gray-200: oklch(0.90 0 0);
  --color-gray-300: oklch(0.83 0 0);
  --color-gray-400: oklch(0.70 0 0);
  --color-gray-500: oklch(0.55 0 0);
  --color-gray-600: oklch(0.45 0 0);
  --color-gray-700: oklch(0.37 0 0);
  --color-gray-800: oklch(0.27 0 0);
  --color-gray-900: oklch(0.18 0 0);
  --color-gray-950: oklch(0.10 0 0);

  /* Primary (Blue) */
  --color-primary-50: oklch(0.97 0.01 250);
  --color-primary-100: oklch(0.93 0.03 250);
  --color-primary-200: oklch(0.86 0.06 250);
  --color-primary-300: oklch(0.76 0.10 250);
  --color-primary-400: oklch(0.66 0.13 250);
  --color-primary-500: oklch(0.60 0.15 250);
  --color-primary-600: oklch(0.52 0.14 250);
  --color-primary-700: oklch(0.44 0.12 250);
  --color-primary-800: oklch(0.36 0.10 250);
  --color-primary-900: oklch(0.30 0.08 250);

  /* Success (Green) */
  --color-success-500: oklch(0.65 0.15 145);

  /* Warning (Amber) */
  --color-warning-500: oklch(0.75 0.15 85);

  /* Error (Red) */
  --color-error-500: oklch(0.55 0.20 25);
}
```

### Semantic Tokens

```css
@theme {
  /* Light Mode (Default) */
  --color-background: var(--color-gray-50);
  --color-foreground: var(--color-gray-900);
  --color-muted: var(--color-gray-100);
  --color-muted-foreground: var(--color-gray-500);
  --color-card: var(--color-gray-50);
  --color-card-foreground: var(--color-gray-900);
  --color-border: var(--color-gray-200);
  --color-input: var(--color-gray-200);
  --color-ring: var(--color-primary-500);
  --color-primary: var(--color-primary-500);
  --color-primary-foreground: oklch(0.98 0 0);
}

[data-theme="dark"] {
  --color-background: var(--color-gray-950);
  --color-foreground: var(--color-gray-50);
  --color-muted: var(--color-gray-800);
  --color-muted-foreground: var(--color-gray-400);
  --color-card: var(--color-gray-900);
  --color-card-foreground: var(--color-gray-50);
  --color-border: var(--color-gray-800);
  --color-input: var(--color-gray-800);
}
```

### Contrast Requirements (WCAG AA)

| Element | Minimum Ratio |
|---------|---------------|
| Normal text (< 18px) | 4.5:1 |
| Large text (>= 18px bold, >= 24px) | 3:1 |
| UI components | 3:1 |
| Focus indicators | 3:1 |

---

## 2. Typography

### Font Stack

```css
@theme {
  --font-sans: 'Inter', ui-sans-serif, system-ui, sans-serif;
  --font-mono: 'JetBrains Mono', ui-monospace, monospace;
}
```

### Type Scale (1.25 ratio - Major Third)

```css
@theme {
  --text-xs: 0.64rem;     /* 10.24px */
  --text-sm: 0.8rem;      /* 12.8px */
  --text-base: 1rem;      /* 16px */
  --text-lg: 1.25rem;     /* 20px */
  --text-xl: 1.563rem;    /* 25px */
  --text-2xl: 1.953rem;   /* 31.25px */
  --text-3xl: 2.441rem;   /* 39px */
  --text-4xl: 3.052rem;   /* 48.8px */
  --text-5xl: 3.815rem;   /* 61px */
}
```

### Fluid Typography

```css
/* Responsive headings using clamp() */
h1 { font-size: clamp(2rem, 5vw + 1rem, 4rem); }
h2 { font-size: clamp(1.5rem, 3vw + 0.75rem, 2.5rem); }
h3 { font-size: clamp(1.25rem, 2vw + 0.5rem, 1.75rem); }
```

### Line Heights

| Use Case | Line Height |
|----------|-------------|
| Headings | 1.2 |
| Body text | 1.5 |
| Tight text | 1.25 |
| Relaxed | 1.75 |

### Line Length
- Optimal: 65-75 characters
- Use `max-width: 65ch` for paragraphs

---

## 3. Spacing (8pt Grid)

### Base Scale

```css
@theme {
  --space-0: 0;
  --space-px: 1px;
  --space-0.5: 0.125rem;  /* 2px */
  --space-1: 0.25rem;     /* 4px - half step */
  --space-2: 0.5rem;      /* 8px */
  --space-3: 0.75rem;     /* 12px */
  --space-4: 1rem;        /* 16px */
  --space-5: 1.25rem;     /* 20px */
  --space-6: 1.5rem;      /* 24px */
  --space-8: 2rem;        /* 32px */
  --space-10: 2.5rem;     /* 40px */
  --space-12: 3rem;       /* 48px */
  --space-16: 4rem;       /* 64px */
  --space-20: 5rem;       /* 80px */
  --space-24: 6rem;       /* 96px */
}
```

### Usage Guidelines

| Context | Recommended Spacing |
|---------|---------------------|
| Component padding | 16-24px (space-4 to space-6) |
| Section padding | 48-96px (space-12 to space-24) |
| Element gap (tight) | 8px (space-2) |
| Element gap (normal) | 16px (space-4) |
| Element gap (loose) | 24px (space-6) |

---

## 4. Border Radius

```css
@theme {
  --radius-none: 0;
  --radius-sm: 0.25rem;   /* 4px */
  --radius-md: 0.5rem;    /* 8px */
  --radius-lg: 0.75rem;   /* 12px */
  --radius-xl: 1rem;      /* 16px */
  --radius-2xl: 1.5rem;   /* 24px */
  --radius-full: 9999px;
}
```

### Usage

| Component | Radius |
|-----------|--------|
| Buttons | md (8px) |
| Cards | lg (12px) |
| Inputs | md (8px) |
| Badges | full |
| Modals | xl (16px) |

---

## 5. Shadows

```css
@theme {
  --shadow-sm: 0 1px 2px 0 oklch(0 0 0 / 0.05);
  --shadow: 0 1px 3px 0 oklch(0 0 0 / 0.1), 0 1px 2px -1px oklch(0 0 0 / 0.1);
  --shadow-md: 0 4px 6px -1px oklch(0 0 0 / 0.1), 0 2px 4px -2px oklch(0 0 0 / 0.1);
  --shadow-lg: 0 10px 15px -3px oklch(0 0 0 / 0.1), 0 4px 6px -4px oklch(0 0 0 / 0.1);
  --shadow-xl: 0 20px 25px -5px oklch(0 0 0 / 0.1), 0 8px 10px -6px oklch(0 0 0 / 0.1);
}
```

---

## 6. Breakpoints

```css
/* Mobile First */
/* Base: < 480px (mobile) */
--breakpoint-sm: 480px;   /* Small mobile */
--breakpoint-md: 768px;   /* Tablet */
--breakpoint-lg: 1024px;  /* Small desktop */
--breakpoint-xl: 1280px;  /* Large desktop */
--breakpoint-2xl: 1440px; /* Extra large */
```

### Tailwind Usage
```html
<div class="
  grid grid-cols-1      /* Mobile: 1 column */
  sm:grid-cols-2        /* 480px+: 2 columns */
  md:grid-cols-3        /* 768px+: 3 columns */
  lg:grid-cols-4        /* 1024px+: 4 columns */
">
```

---

## 7. Animation

```css
@theme {
  /* Durations */
  --duration-fast: 150ms;
  --duration-normal: 200ms;
  --duration-slow: 300ms;

  /* Easings */
  --ease-default: cubic-bezier(0.4, 0, 0.2, 1);
  --ease-in: cubic-bezier(0.4, 0, 1, 1);
  --ease-out: cubic-bezier(0, 0, 0.2, 1);
  --ease-in-out: cubic-bezier(0.4, 0, 0.2, 1);
  --ease-bounce: cubic-bezier(0.68, -0.55, 0.265, 1.55);
}
```

### Motion Guidelines
- Use subtle animations (don't distract)
- Respect `prefers-reduced-motion`
- Duration: 150-300ms for most interactions
- Provide feedback, don't just decorate

```css
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: 0.01ms !important;
    transition-duration: 0.01ms !important;
  }
}
```

---

## 8. Component Standards

### Buttons

```tsx
// Sizes
const sizes = {
  sm: "h-9 px-3 text-sm",      // 36px height
  md: "h-10 px-4 text-sm",     // 40px height
  lg: "h-11 px-8 text-base",   // 44px height (touch target)
}

// Variants
const variants = {
  primary: "bg-primary text-primary-foreground hover:bg-primary/90",
  secondary: "bg-secondary text-secondary-foreground hover:bg-secondary/80",
  outline: "border border-input bg-background hover:bg-accent",
  ghost: "hover:bg-accent hover:text-accent-foreground",
  destructive: "bg-destructive text-destructive-foreground hover:bg-destructive/90",
}
```

### Inputs

- Height: 40px minimum (44px for touch)
- Padding: 12px horizontal
- Border: 1px solid border color
- Focus: 2px ring with ring color
- Error: Red border + error message below

### Cards

- Padding: 24px (space-6)
- Border radius: lg (12px)
- Shadow: shadow-sm, hover:shadow-md
- Border: 1px solid border color

---

## 9. Accessibility Checklist

### Required for All Components

- [ ] 4.5:1 contrast for text
- [ ] 3:1 contrast for UI elements
- [ ] Visible focus states (focus-visible:ring-2)
- [ ] Keyboard accessible
- [ ] Touch targets >= 44x44px
- [ ] ARIA labels where needed
- [ ] Works with reduced motion

### Focus States

```css
.interactive {
  @apply focus:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2;
}
```

---

## 10. File Structure

```
swarms/design/
├── agents/
│   ├── design_lead.md
│   ├── design_researcher.md
│   ├── ui_implementer.md
│   └── ux_reviewer.md
├── workspace/
│   ├── STATE.md
│   └── design_system.md (this file)
└── components/
    └── (implemented components go here)
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-01-05 | Initial design system |
