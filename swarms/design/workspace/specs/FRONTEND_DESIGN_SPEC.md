# AI Corp - Frontend Design Specification

A design framework for building a web UI around the AI Corp autonomous agent system.

---

## Executive Summary

**What is AI Corp?**
An autonomous AI corporation where multiple Claude instances work as a unified organization - with hierarchy, departments, work queues, and quality gates - just like a real company.

**User Role:** The human user is the **CEO** - they provide high-level direction, approve major decisions, and monitor progress. The AI agents handle execution.

**Core Metaphor:** A corporate org chart that actually runs itself.

---

## Information Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         AI CORP UI                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │  DASHBOARD  │  │  PROJECTS   │  │   AGENTS    │             │
│  │  (Home)     │  │  (Molecules)│  │  (Org Chart)│             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │  DISCOVERY  │  │   GATES     │  │  SETTINGS   │             │
│  │  (New Work) │  │  (Approvals)│  │  (Config)   │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Screens

### 1. Dashboard (Home) - Command Center
### 2. Discovery / New Project - Conversational Requirements
### 3. Project Detail (Molecule View)
### 4. Agents / Org Chart - Interactive Graph Visualization
### 5. Gates / Approvals
### 6. Settings / Configuration
### 7. Global UI Patterns & Component Library
### 8. Integrations Hub - Connected Intelligence

---

## Design Principles

1. **CEO Perspective**: User is executive, not operator. Show outcomes, not implementation details.
2. **Progressive Disclosure**: Dashboard → Project → Step → Detail. Don't overwhelm.
3. **Status at a Glance**: Health indicators visible without clicking. Problems surface automatically.
4. **Trust but Verify**: AI handles execution, but human approves gates and can intervene.
5. **Activity Over Configuration**: Most time spent monitoring, not configuring. Optimize for that.

---

## Design Requirements for This Sprint

### Goal
Create **3 distinct, original UI design concepts** that:
- Follow 2025/2026 design trends
- Are creative and innovative
- Maintain usability and accessibility
- Feel premium and professional
- Work for the CEO persona (executive oversight)

### Key Screens to Design
1. **Dashboard** - The command center view
2. **Org Chart / Agents** - Interactive visualization
3. **Project Detail** - Workflow and progress view

### Deliverables Per Concept
- Visual direction / mood board
- Color palette and typography
- Key component styles
- Dashboard layout wireframe
- Interaction patterns

---

*Full spec continues in attached document*
*See: FRONTEND_DESIGN_SPEC_FULL.md for complete wireframes*
