# AI Corp Frontend UI Design Review

**Author**: Design Lead Agent
**Date**: 2026-01-05
**Status**: Comprehensive Architecture Analysis

---

## Executive Summary

This document provides a thorough UI/UX design analysis of the AI Corp architecture, a fully autonomous AI corporation system featuring a 5-level hierarchy, 5 departments, and sophisticated workflow management through Molecules, Hooks, Beads, Channels, and Gates. The design recommendations focus on creating an intuitive, scalable interface that surfaces complex multi-agent operations while maintaining clarity for the human CEO.

---

## Part 1: Organizational Hierarchy Analysis

### 1.1 Hierarchy Structure

```
Level 0: CEO (Human Owner) - Strategic oversight
Level 1: COO (AI) - Operational command
Level 2: VPs (5 total) - Department leadership
Level 3: Directors (Multiple per VP) - Tactical management
Level 4: Workers (Pooled) - Task execution
```

### 1.2 UI Components for Hierarchy

#### **Org Chart Visualization**

| Component | Purpose | Recommended Implementation |
|-----------|---------|---------------------------|
| `<OrgTree />` | Interactive hierarchy tree | D3.js force-directed graph with collapsible nodes |
| `<RoleCard />` | Individual agent info card | Shows status, current task, department color |
| `<HierarchyBreadcrumb />` | Navigation context | CEO > COO > VP Engineering > Frontend Director |
| `<ReportingLine />` | Visual connection | Animated SVG paths showing active communication |
| `<PoolIndicator />` | Worker pool status | Circular gauge showing pool utilization |

#### **Visual Hierarchy Design Principles**

1. **Size Gradient**: CEO node largest, workers smallest
2. **Color Coding**: Each department gets a distinct color family
3. **Status Indicators**: Real-time pulse animations for active agents
4. **Depth Perception**: Subtle shadows/elevation increase at higher levels
5. **Interaction Zones**: Click to drill-down, hover for quick stats

### 1.3 Org Chart Component Specifications

```
┌────────────────────────────────────────────────────────────────────┐
│                         ORG CHART VIEW                              │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│                        ┌─────────────┐                              │
│                        │   CEO (You) │  ← Crown icon, gold accent   │
│                        │  ● Online   │                              │
│                        └──────┬──────┘                              │
│                               │                                     │
│                        ┌──────┴──────┐                              │
│                        │     COO     │  ← AI badge, silver accent   │
│                        │  ● Working  │                              │
│                        └──────┬──────┘                              │
│                               │                                     │
│    ┌──────────┬───────────┬───┴───┬───────────┬──────────┐         │
│    │          │           │       │           │          │         │
│ ┌──┴──┐   ┌───┴──┐   ┌───┴──┐ ┌──┴───┐  ┌───┴──┐   [+2 more]     │
│ │VP Eng│  │VP Res │  │VP Prod│ │VP Qual│ │VP Ops│                  │
│ │ 12   │  │  4   │   │  6   │  │  5   │  │  3   │  ← Worker count │
│ └──────┘  └──────┘   └──────┘  └──────┘  └──────┘                  │
│                                                                     │
│  [Expand All]  [Collapse]  [Filter by Dept]  [Search Agent]        │
└────────────────────────────────────────────────────────────────────┘
```

---

## Part 2: Department Analysis & UI Components

### 2.1 Department Overview

| Department | VP Role | Primary Color | Icon Suggestion |
|------------|---------|---------------|-----------------|
| Engineering | VP Engineering | Blue (#3B82F6) | Code brackets `</>` |
| Research | VP Research | Purple (#8B5CF6) | Magnifying glass |
| Product | VP Product | Green (#10B981) | Cube/Package |
| Quality | VP Quality | Orange (#F59E0B) | Shield/Checkmark |
| Operations | VP Operations | Gray (#6B7280) | Gear/Cog |

### 2.2 Department Dashboard Components

#### **Engineering Department**

```tsx
// Components needed:
<EngineeringDashboard>
  <DirectorPanel directors={['Architecture', 'Frontend', 'Backend', 'DevOps']} />
  <WorkerPoolGrid pools={['frontend_workers', 'backend_workers', 'devops_workers']} />
  <ActiveMolecules filter="engineering" />
  <CodeMetrics commits={} linesChanged={} prOpen={} />
  <SkillsInventory skills={['frontend-design', 'aws-skills', 'terraform-skills']} />
</EngineeringDashboard>
```

**Key Metrics to Display**:
- Active workers / Max workers per pool
- Current sprint velocity
- Code review queue depth
- Build/Deploy status

#### **Research Department**

```tsx
<ResearchDashboard>
  <DirectorPanel directors={['Market Research', 'Technical Research']} />
  <ResearchQueue items={activeResearchTasks} />
  <KnowledgeBaseStatus entries={} lastUpdated={} />
  <ResearchTimeline completed={} inProgress={} />
</ResearchDashboard>
```

**Key Metrics to Display**:
- Research tickets open/closed
- Knowledge base growth
- Time to insight (avg)

#### **Product Department**

```tsx
<ProductDashboard>
  <DirectorPanel directors={['Product', 'Design']} />
  <RoadmapView milestones={} features={} />
  <DesignSystemStatus components={} coverage={} />
  <UXWorkflow wireframes={} prototypes={} approved={} />
</ProductDashboard>
```

**Key Metrics to Display**:
- Features in pipeline by stage
- Design approval rate
- Spec completion percentage

#### **Quality Department**

```tsx
<QualityDashboard>
  <DirectorPanel directors={['QA', 'Security']} />
  <TestSuiteStatus passed={} failed={} skipped={} />
  <SecurityAuditLog findings={} resolved={} />
  <GateStatusPanel gates={allGates} />
  <BugTracker critical={} high={} medium={} low={} />
</QualityDashboard>
```

**Key Metrics to Display**:
- Test coverage percentage
- Gates passed vs blocked
- Security vulnerabilities (CVSS score distribution)
- Bug burn-down rate

#### **Operations Department**

```tsx
<OperationsDashboard>
  <DirectorPanel directors={['Project', 'Documentation']} />
  <ResourceAllocation agents={} utilization={} />
  <ProjectTimeline projects={} milestones={} />
  <DocumentationCoverage docs={} stale={} />
  <SystemHealthMonitor uptime={} performance={} />
</OperationsDashboard>
```

**Key Metrics to Display**:
- Resource utilization
- Documentation freshness
- Project status summary
- System health indicators

### 2.3 Cross-Department Role Matrix Component

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         DEPARTMENT ROLE MATRIX                            │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  Department     │ Directors          │ Workers        │ Skills           │
│  ───────────────┼────────────────────┼────────────────┼─────────────────│
│  Engineering    │ Arch, FE, BE, DevOps│ FE(5), BE(5), │ frontend-design │
│                 │                     │ DevOps(3)     │ aws-skills      │
│  ───────────────┼────────────────────┼────────────────┼─────────────────│
│  Research       │ Market, Technical  │ Researchers   │ -               │
│  ───────────────┼────────────────────┼────────────────┼─────────────────│
│  Product        │ Product, Design    │ PMs, UX       │ frontend-design │
│  ───────────────┼────────────────────┼────────────────┼─────────────────│
│  Quality        │ QA, Security       │ QA Eng,       │ webapp-testing  │
│                 │                     │ Reviewers     │ security-bluebook│
│  ───────────────┼────────────────────┼────────────────┼─────────────────│
│  Operations     │ Project, Docs      │ PMs, Writers  │ docx, pdf       │
│                                                                           │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Part 3: Core Systems UI Design

### 3.1 Molecules (Persistent Workflows)

#### **Molecule Visualization Requirements**

Molecules are the heart of AI Corp's workflow persistence. The UI must communicate:

1. **Workflow Identity**: Unique ID, name, creation context
2. **Step Progression**: Visual timeline of completed/active/pending steps
3. **Dependencies**: Which steps block others
4. **Checkpoints**: Recovery points for crash resilience
5. **RACI Assignment**: Who's responsible, accountable, consulted, informed

#### **Molecule Card Component**

```
┌──────────────────────────────────────────────────────────────────┐
│  MOL-123: Build User Dashboard                        [ACTIVE]   │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Progress: ████████░░░░░░░░░░░░  40%                             │
│                                                                   │
│  Steps:                                                           │
│  ✓ Design Review (design_director) ─────────────────┐            │
│  ● Component Implementation (frontend_worker_pool)  │ ← Current  │
│    └─ Checkpoint: "Completed Header component"      │            │
│  ○ QA Review (qa_engineer_pool)  ←──────────────────┘ Blocked    │
│  ○ Security Review [GATE] (security_director)                    │
│                                                                   │
│  RACI: A:frontend_director  R:frontend_workers  C:design  I:vp   │
│                                                                   │
│  [View Details]  [View Log]  [Escalate]                          │
└──────────────────────────────────────────────────────────────────┘
```

#### **Molecule List View**

```tsx
<MoleculeList>
  <MoleculeFilters
    status={['active', 'completed', 'blocked']}
    department={departments}
    stage={pipelineStages}
  />
  <MoleculeGrid>
    {molecules.map(mol => <MoleculeCard key={mol.id} molecule={mol} />)}
  </MoleculeGrid>
  <MoleculeTimeline molecules={molecules} /> {/* Gantt-style view */}
</MoleculeList>
```

### 3.2 Hooks (Work Queues)

#### **Hook Dashboard Design**

Hooks represent agent work queues. The UI should show:

1. **Agent Identity**: Who owns this hook
2. **Queue Depth**: How many tasks waiting
3. **Current Task**: What's being executed now
4. **Priority Sorting**: Visual distinction for priority levels

```
┌──────────────────────────────────────────────────────────────────┐
│                        HOOK MONITOR                               │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Agent: frontend_worker_01                    Status: ● Working  │
│  Department: Engineering                      Role: Frontend      │
│                                                                   │
│  CURRENT TASK                                                     │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ TASK-001: Implement Dashboard Header                        │  │
│  │ Priority: HIGH  │  Molecule: MOL-123  │  Started: 2m ago   │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                   │
│  QUEUE (3 items)                                                  │
│  ┌──────┬──────────────────────────────┬──────────┬───────────┐  │
│  │ PRI  │ TASK                         │ MOLECULE │ QUEUED    │  │
│  ├──────┼──────────────────────────────┼──────────┼───────────┤  │
│  │ 🔴   │ Fix Navigation Bug           │ MOL-125  │ 5m        │  │
│  │ 🟡   │ Add Footer Component         │ MOL-123  │ 10m       │  │
│  │ 🟢   │ Update Styles                │ MOL-128  │ 15m       │  │
│  └──────┴──────────────────────────────┴──────────┴───────────┘  │
│                                                                   │
│  [Reprioritize]  [Reassign]  [Clear Queue]                       │
└──────────────────────────────────────────────────────────────────┘
```

#### **Hook Grid View (All Agents)**

```tsx
<HookGrid>
  <HookFilters department={} role={} status={} />
  <AgentHookCards>
    {agents.map(agent => (
      <HookCard
        agent={agent}
        queueDepth={agent.queue.length}
        currentTask={agent.current_task}
        status={agent.status}
      />
    ))}
  </AgentHookCards>
  <QueueMetrics avgWait={} maxDepth={} throughput={} />
</HookGrid>
```

### 3.3 Beads (Git-Backed Ledger)

#### **Ledger Visualization**

Beads provide the audit trail. The UI should enable:

1. **Timeline View**: Chronological event log
2. **Filter by Type**: Tasks, decisions, handoffs
3. **Git Integration**: Link to actual commits
4. **Search**: Full-text search across ledger

```
┌──────────────────────────────────────────────────────────────────┐
│                        BEAD LEDGER                                │
├──────────────────────────────────────────────────────────────────┤
│  Filter: [All ▼]  [Tasks]  [Decisions]  [Handoffs]   🔍 Search   │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  2026-01-05T14:32:00Z                                            │
│  ├─ 📋 TASK: frontend_worker_01 completed Header component       │
│  │   Molecule: MOL-123 │ Commit: abc123f │ Duration: 45m         │
│  │                                                                │
│  2026-01-05T13:45:00Z                                            │
│  ├─ 🔀 HANDOFF: design_director → frontend_director              │
│  │   Molecule: MOL-123 │ Stage: DESIGN → BUILD                   │
│  │   Notes: "Design approved, ready for implementation"          │
│  │                                                                │
│  2026-01-05T12:30:00Z                                            │
│  ├─ ⚖️ DECISION: vp_engineering approved tech stack              │
│  │   Context: MOL-123 │ Options considered: 3                    │
│  │   Selected: React + TypeScript │ Rationale: [View]            │
│  │                                                                │
│  [Load More]                                                      │
└──────────────────────────────────────────────────────────────────┘
```

### 3.4 Channels (Communication)

#### **Channel Types & UI Treatment**

| Channel Type | Direction | UI Visualization |
|--------------|-----------|------------------|
| DOWN-CHAIN | CEO→Worker | Red/Orange arrow, top-to-bottom |
| UP-CHAIN | Worker→CEO | Blue/Green arrow, bottom-to-top |
| PEER-TO-PEER | Same level | Horizontal bidirectional |
| BROADCAST | One→Many | Radial burst animation |

#### **Communication Hub Component**

```
┌──────────────────────────────────────────────────────────────────┐
│                    COMMUNICATION HUB                              │
├────────────────────┬─────────────────────────────────────────────┤
│  CHANNELS          │  MESSAGE STREAM                              │
│  ─────────────     │  ───────────────                             │
│                    │                                              │
│  ▼ Down-Chain (3)  │  ┌─────────────────────────────────────────┐│
│    → VP Eng        │  │ ↓ COO → VP Engineering        2m ago    ││
│    → VP Prod       │  │ "Prioritize MOL-123, CEO request"       ││
│    → VP Quality    │  └─────────────────────────────────────────┘│
│                    │                                              │
│  ▲ Up-Chain (5)    │  ┌─────────────────────────────────────────┐│
│    ← frontend_dir  │  │ ↑ Frontend Director → VP Eng   5m ago   ││
│    ← qa_director   │  │ "MOL-123 blocked: need design assets"   ││
│    ← ...           │  └─────────────────────────────────────────┘│
│                    │                                              │
│  ↔ Peer (2)        │  ┌─────────────────────────────────────────┐│
│    ↔ VP Prod       │  │ ↔ VP Eng ↔ VP Prod             10m ago  ││
│                    │  │ "Syncing on dashboard requirements"     ││
│  📢 Broadcast (1)  │  └─────────────────────────────────────────┘│
│    All Eng         │                                              │
│                    │  [Compose Message]  [View All]              │
└────────────────────┴─────────────────────────────────────────────┘
```

### 3.5 Gates (Quality Checkpoints)

#### **Gate Status Dashboard**

Gates are critical control points. The UI must clearly communicate:

1. **Gate Identity**: Which quality checkpoint
2. **Current Status**: Open/Closed/Blocked
3. **Blocking Items**: What's waiting for approval
4. **Approval Authority**: Who can open the gate
5. **History**: Past approvals/rejections

```
┌──────────────────────────────────────────────────────────────────┐
│                      QUALITY GATES                                │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐            │
│  │ GATE 1       │  │ GATE 2       │  │ GATE 3       │            │
│  │ Research     │→→│ Design       │→→│ Code Review  │            │
│  │ ✓ OPEN       │  │ ● REVIEWING  │  │ ○ WAITING    │            │
│  │ 0 blocked    │  │ 2 blocked    │  │ 0 blocked    │            │
│  └──────────────┘  └──────────────┘  └──────────────┘            │
│                                                                   │
│  ┌──────────────┐  ┌──────────────┐                              │
│  │ GATE 4       │  │ GATE 5       │                              │
│  │ QA Passed    │→→│ Security     │                              │
│  │ ○ WAITING    │  │ ○ WAITING    │                              │
│  │ 0 blocked    │  │ 0 blocked    │                              │
│  └──────────────┘  └──────────────┘                              │
│                                                                   │
│  GATE 2 QUEUE (Design Approved)                                   │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ MOL-123: Build User Dashboard    [Approve] [Request Changes]│  │
│  │ MOL-125: Navigation Redesign     [Approve] [Request Changes]│  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

---

## Part 4: Pipeline Stages UI Design

### 4.1 Pipeline Overview

```
INBOX → RESEARCH → DESIGN → BUILD → QA → SECURITY → DEPLOY → MONITOR → COMPLETE
  │        │          │        │      │       │         │        │         │
  │      GATE 1    GATE 2   GATE 3  GATE 4  GATE 5     │        │         │
  │    (Research) (Design)  (Code)   (QA) (Security)   │        │         │
  │                                                     │        │         │
  └─────────────────────── ARCHIVE ────────────────────┴────────┴─────────┘
```

### 4.2 Kanban Pipeline View

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                              PIPELINE VIEW                                        │
├────────┬─────────┬─────────┬─────────┬─────────┬──────────┬────────┬────────────┤
│ INBOX  │RESEARCH │ DESIGN  │  BUILD  │   QA    │ SECURITY │ DEPLOY │  COMPLETE  │
│  (3)   │   (2)   │   (1)   │   (4)   │   (2)   │   (0)    │  (1)   │    (12)    │
├────────┼─────────┼─────────┼─────────┼─────────┼──────────┼────────┼────────────┤
│        │         │         │         │         │          │        │            │
│┌──────┐│┌───────┐│┌───────┐│┌───────┐│┌───────┐│          │┌──────┐│  ✓ MOL-100 │
││MOL-  ││|MOL-   ││|MOL-   ││|MOL-   ││|MOL-   ││          ││MOL-  ││  ✓ MOL-101 │
││130   │││122    │││123    │││115    │││120    ││          ││118   ││  ✓ MOL-102 │
│└──────┘│└───────┘│└───────┘│└───────┘│└───────┘│          │└──────┘│  ...       │
│        │         │         │         │         │          │        │            │
│┌──────┐│┌───────┐│         │┌───────┐│┌───────┐│          │        │            │
││MOL-  ││|MOL-   ││         ││|MOL-   ││|MOL-   ││          │        │            │
││131   │││124    ││         │││116    │││121    ││          │        │            │
│└──────┘│└───────┘│         │└───────┘│└───────┘│          │        │            │
│        │         │         │         │         │          │        │            │
│┌──────┐│   🚧    │         │┌───────┐│         │          │        │            │
││MOL-  ││ GATE 1  ││   🚧    ││|MOL-   ││   🚧    │   🚧     │        │            │
││132   ││        ││ GATE 2  │││117    ││ GATE 4  │ GATE 5   │        │            │
│└──────┘│        ││         │└───────┘│         │          │        │            │
│        │        ││         │         │         │          │        │            │
│        │        ││         │┌───────┐│         │          │        │            │
│        │        ││   🚧    ││|MOL-   ││         │          │        │            │
│        │        ││ GATE 3  │││119    ││         │          │        │            │
│        │        ││         │└───────┘│         │          │        │            │
│        │         │         │         │         │          │        │            │
├────────┴─────────┴─────────┴─────────┴─────────┴──────────┴────────┴────────────┤
│  [+ New Molecule]  │  Filter: [All ▼]  │  View: [Kanban] [Timeline] [List]      │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### 4.3 Stage-Specific Views

Each pipeline stage should have a detailed view:

```tsx
<PipelineStageView stage="BUILD">
  <StageHeader
    name="BUILD"
    moleculeCount={4}
    avgTimeInStage="2.5 days"
    blockedCount={1}
  />
  <GateStatus
    entryGate="GATE 2 (Design Approved)"
    exitGate="GATE 3 (Code Review)"
  />
  <MoleculeList molecules={buildStageMolecules} />
  <ResourceAllocation>
    <WorkerPool pool="frontend_workers" assigned={3} available={5} />
    <WorkerPool pool="backend_workers" assigned={2} available={5} />
  </ResourceAllocation>
  <StageMetrics>
    <Metric label="Throughput" value="5/week" trend="up" />
    <Metric label="Avg Cycle Time" value="2.3 days" trend="down" />
    <Metric label="WIP Limit" value="4/6" status="ok" />
  </StageMetrics>
</PipelineStageView>
```

### 4.4 Pipeline Analytics Dashboard

```
┌──────────────────────────────────────────────────────────────────┐
│                    PIPELINE ANALYTICS                             │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  FLOW METRICS                                                     │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │  Throughput: ███████████░░░░░  24/30 molecules this week    │ │
│  │  Cycle Time: Average 4.2 days (Target: 5 days) ✓            │ │
│  │  WIP: 12 active (Limit: 15) ✓                               │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                   │
│  BOTTLENECK ANALYSIS                                              │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │  Stage        │ Avg Time │ Queue Depth │ Status              │ │
│  │  ─────────────┼──────────┼─────────────┼──────────           │ │
│  │  RESEARCH     │ 1.2d     │ 2           │ ✓ Normal            │ │
│  │  DESIGN       │ 0.8d     │ 1           │ ✓ Normal            │ │
│  │  BUILD        │ 2.1d     │ 4           │ ⚠️ High Load        │ │
│  │  QA           │ 0.5d     │ 2           │ ✓ Normal            │ │
│  │  SECURITY     │ 0.3d     │ 0           │ ✓ Normal            │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                   │
│  CUMULATIVE FLOW DIAGRAM                                          │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │     ▲                                                        │ │
│  │  30 │  ████████████████████████████ Complete                │ │
│  │     │  ░░░░████████████████████████ Deploy                  │ │
│  │  20 │  ░░░░░░░░████████████████████ Security                │ │
│  │     │  ░░░░░░░░░░░░████████████████ QA                      │ │
│  │  10 │  ░░░░░░░░░░░░░░░░████████████ Build                   │ │
│  │     │  ░░░░░░░░░░░░░░░░░░░░████████ Design                  │ │
│  │   0 └────────────────────────────────▶ Time                 │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

---

## Part 5: Information Architecture

### 5.1 Global Navigation Structure

```
┌──────────────────────────────────────────────────────────────────┐
│  AI CORP                                              [CEO] [⚙️]  │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  PRIMARY NAV (Top Bar)                                           │
│  ┌───────────────────────────────────────────────────────────┐   │
│  │ [Dashboard] [Pipeline] [Org Chart] [Molecules] [Ledger]   │   │
│  └───────────────────────────────────────────────────────────┘   │
│                                                                   │
│  SECONDARY NAV (Sidebar - Context Dependent)                     │
│  ┌─────────────────┐                                             │
│  │ DEPARTMENTS     │  When viewing Org Chart / Dashboard         │
│  │ ├─ Engineering  │                                             │
│  │ ├─ Research     │                                             │
│  │ ├─ Product      │                                             │
│  │ ├─ Quality      │                                             │
│  │ └─ Operations   │                                             │
│  │                 │                                             │
│  │ QUICK ACTIONS   │                                             │
│  │ ├─ New Molecule │                                             │
│  │ ├─ View Alerts  │                                             │
│  │ └─ Agent Status │                                             │
│  └─────────────────┘                                             │
│                                                                   │
│  UTILITY NAV (Bottom)                                            │
│  ┌───────────────────────────────────────────────────────────┐   │
│  │ [Settings] [Skills] [Templates] [Help] [System Status]    │   │
│  └───────────────────────────────────────────────────────────┘   │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### 5.2 Page Hierarchy

```
/                           → CEO Dashboard (Overview)
├── /dashboard              → Main dashboard with key metrics
│   ├── /dashboard/ceo      → CEO-specific view
│   └── /dashboard/ops      → Operations overview
│
├── /pipeline               → Kanban pipeline view
│   ├── /pipeline/inbox     → Inbox stage detail
│   ├── /pipeline/research  → Research stage detail
│   ├── /pipeline/design    → Design stage detail
│   ├── /pipeline/build     → Build stage detail
│   ├── /pipeline/qa        → QA stage detail
│   ├── /pipeline/security  → Security stage detail
│   ├── /pipeline/deploy    → Deploy stage detail
│   └── /pipeline/analytics → Pipeline analytics
│
├── /org                    → Organization views
│   ├── /org/chart          → Full org chart
│   ├── /org/departments    → Department list
│   │   └── /org/departments/:id  → Department detail
│   ├── /org/roles          → Role definitions
│   └── /org/agents         → All agents list
│       └── /org/agents/:id → Agent detail/hook view
│
├── /molecules              → Workflow management
│   ├── /molecules/active   → Active molecules
│   ├── /molecules/completed → Completed molecules
│   ├── /molecules/templates → Molecule templates
│   └── /molecules/:id      → Molecule detail view
│
├── /channels               → Communication hub
│   ├── /channels/inbox     → CEO message inbox
│   ├── /channels/sent      → Sent messages
│   └── /channels/broadcast → Broadcast messages
│
├── /gates                  → Quality gates
│   ├── /gates/status       → All gates overview
│   └── /gates/:id          → Gate detail/approval queue
│
├── /ledger                 → Bead ledger (audit trail)
│   ├── /ledger/tasks       → Task history
│   ├── /ledger/decisions   → Decision log
│   └── /ledger/handoffs    → Handoff history
│
├── /skills                 → Skill management
│   └── /skills/:id         → Skill detail
│
└── /settings               → System settings
    ├── /settings/pools     → Worker pool config
    ├── /settings/gates     → Gate rules
    └── /settings/integrations → External integrations
```

### 5.3 Information Density Guidelines

| View Type | Density | Use Case |
|-----------|---------|----------|
| Dashboard | Low | Overview, status at a glance |
| Pipeline Kanban | Medium | Work tracking, flow visibility |
| Molecule Detail | High | Deep dive, troubleshooting |
| Ledger | High | Audit, historical analysis |
| Org Chart | Low | Navigation, structure understanding |
| Settings | Medium | Configuration |

---

## Part 6: Visual Hierarchy Recommendations

### 6.1 Typography Scale

```
Heading 1 (Page Titles):      32px / Bold / Primary color
Heading 2 (Section Headers):  24px / Semibold / Primary color
Heading 3 (Card Headers):     18px / Semibold / Secondary color
Body (Content):               14px / Regular / Text color
Caption (Metadata):           12px / Regular / Muted color
Code/Mono (IDs, Status):      13px / Monospace / Accent color
```

### 6.2 Color System

```
SEMANTIC COLORS
───────────────
Primary:     #2563EB (Blue)     - Actions, links, active states
Success:     #10B981 (Green)    - Completed, approved, healthy
Warning:     #F59E0B (Amber)    - Attention, pending, review
Error:       #EF4444 (Red)      - Failed, blocked, critical
Info:        #6366F1 (Indigo)   - Informational, tips

DEPARTMENT COLORS
─────────────────
Engineering: #3B82F6 (Blue)
Research:    #8B5CF6 (Purple)
Product:     #10B981 (Green)
Quality:     #F59E0B (Orange)
Operations:  #6B7280 (Gray)

HIERARCHY LEVELS
────────────────
CEO:         #FFD700 (Gold)     - Crown accent
COO:         #C0C0C0 (Silver)   - AI leadership
VP:          Department color   - Saturated
Director:    Department color   - Medium saturation
Worker:      Department color   - Low saturation/pastel

STATUS INDICATORS
─────────────────
● Online/Active:   #10B981 (Green pulse)
● Working:         #3B82F6 (Blue solid)
● Blocked:         #EF4444 (Red solid)
● Idle:            #9CA3AF (Gray)
○ Offline:         #E5E7EB (Light gray outline)
```

### 6.3 Spacing System

```
Base unit: 4px

xs:  4px   (tight padding, icon gaps)
sm:  8px   (list item padding, small gaps)
md:  16px  (card padding, section gaps)
lg:  24px  (major section separation)
xl:  32px  (page margins, hero spacing)
2xl: 48px  (full-width container margins)
```

### 6.4 Component Elevation

```
Level 0: Page background       - No shadow
Level 1: Cards, panels         - shadow-sm (0 1px 2px rgba)
Level 2: Dropdowns, popovers   - shadow-md (0 4px 6px rgba)
Level 3: Modals, dialogs       - shadow-lg (0 10px 15px rgba)
Level 4: Tooltips              - shadow-xl (0 20px 25px rgba)
```

---

## Part 7: Dashboard Layout Concepts

### 7.1 CEO Dashboard (Main View)

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│  AI CORP CEO DASHBOARD                              Mon Jan 5, 2026 │ [⚙️] [👤]  │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│  ┌─────────────────────────────────────────┬──────────────────────────────────┐  │
│  │         CORPORATION HEALTH              │       URGENT ATTENTION           │  │
│  │  ┌─────────┬─────────┬─────────┐       │  ┌───────────────────────────┐   │  │
│  │  │ Agents  │ Molecules│ Gates  │        │  │ ⚠️ 2 molecules blocked     │   │  │
│  │  │  24/30  │   12    │  2/5   │        │  │ 🔴 Security review needed  │   │  │
│  │  │ Active  │ Active  │ Open   │        │  │ 📢 3 up-chain messages     │   │  │
│  │  └─────────┴─────────┴─────────┘       │  └───────────────────────────┘   │  │
│  └─────────────────────────────────────────┴──────────────────────────────────┘  │
│                                                                                   │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                        PIPELINE SUMMARY                                    │   │
│  │  INBOX(3)→RESEARCH(2)→DESIGN(1)→BUILD(4)→QA(2)→SECURITY(0)→DEPLOY(1)→✓12│   │
│  │  ══════════════════════════════════════════════════════════════════════   │   │
│  │  [View Pipeline →]                                                         │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                   │
│  ┌────────────────────────────────────┬─────────────────────────────────────┐   │
│  │       DEPARTMENT STATUS             │        RECENT ACTIVITY              │   │
│  │  ┌──────────────────────────────┐  │  ┌────────────────────────────────┐ │   │
│  │  │ ● Engineering    12 agents   │  │  │ ✓ MOL-118 deployed (2m ago)    │ │   │
│  │  │ ● Research        4 agents   │  │  │ → MOL-123 moved to BUILD       │ │   │
│  │  │ ● Product         6 agents   │  │  │ ✗ MOL-125 blocked at GATE 2    │ │   │
│  │  │ ● Quality         5 agents   │  │  │ ↑ VP Eng: "Need clarification" │ │   │
│  │  │ ● Operations      3 agents   │  │  │ + MOL-132 created (INBOX)      │ │   │
│  │  └──────────────────────────────┘  │  └────────────────────────────────┘ │   │
│  │  [View Org Chart →]                │  [View All Activity →]              │   │
│  └────────────────────────────────────┴─────────────────────────────────────┘   │
│                                                                                   │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                     KEY METRICS THIS WEEK                                  │   │
│  │  ┌────────────┬────────────┬────────────┬────────────┬────────────┐      │   │
│  │  │ Molecules  │ Avg Cycle  │  Quality   │  Uptime    │  Cost Est  │      │   │
│  │  │ Completed  │   Time     │   Score    │            │            │      │   │
│  │  │            │            │            │            │            │      │   │
│  │  │    24      │   4.2d     │   98.5%    │   99.9%    │   $124     │      │   │
│  │  │   ↑ 12%    │   ↓ 0.5d   │   → same   │   → same   │   ↓ 8%     │      │   │
│  │  └────────────┴────────────┴────────────┴────────────┴────────────┘      │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                   │
│  QUICK ACTIONS: [+ New Molecule] [Send Directive] [View Reports] [System Check] │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### 7.2 Department Dashboard Layout

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│  ENGINEERING DEPARTMENT                                          [VP: Online ●]  │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│  ┌────────────────────────────────────────┬─────────────────────────────────────┐│
│  │         DIRECTORS                       │        WORKER POOLS                 ││
│  │  ┌──────────────────────────────────┐  │  ┌─────────────────────────────┐   ││
│  │  │ 👤 Architecture Dir    ● Active  │  │  │ Frontend Pool               │   ││
│  │  │ 👤 Frontend Dir        ● Active  │  │  │ ████████░░ 3/5 active       │   ││
│  │  │ 👤 Backend Dir         ○ Idle    │  │  │                              │   ││
│  │  │ 👤 DevOps Dir          ● Working │  │  │ Backend Pool                │   ││
│  │  └──────────────────────────────────┘  │  │ ████░░░░░░ 2/5 active       │   ││
│  │                                         │  │                              │   ││
│  │                                         │  │ DevOps Pool                 │   ││
│  │                                         │  │ ██░░░░░░░░ 1/3 active       │   ││
│  │                                         │  └─────────────────────────────┘   ││
│  └────────────────────────────────────────┴─────────────────────────────────────┘│
│                                                                                   │
│  ┌──────────────────────────────────────────────────────────────────────────────┐│
│  │                    ACTIVE MOLECULES IN ENGINEERING                            ││
│  │  ┌────────────────────────────────────────────────────────────────────────┐  ││
│  │  │ MOL-123  Build User Dashboard       BUILD    ████████░░  80%  [View]   │  ││
│  │  │ MOL-119  API Refactoring            BUILD    █████░░░░░  50%  [View]   │  ││
│  │  │ MOL-117  Mobile Responsive          BUILD    ███░░░░░░░  30%  [View]   │  ││
│  │  │ MOL-115  Performance Optimization   BUILD    ██░░░░░░░░  20%  [View]   │  ││
│  │  └────────────────────────────────────────────────────────────────────────┘  ││
│  └──────────────────────────────────────────────────────────────────────────────┘│
│                                                                                   │
│  ┌───────────────────────────────────┬──────────────────────────────────────────┐│
│  │       INSTALLED SKILLS            │       RECENT COMMUNICATIONS              ││
│  │  ┌─────────────────────────────┐  │  ┌────────────────────────────────────┐ ││
│  │  │ ✓ frontend-design           │  │  │ ↓ COO: "Prioritize MOL-123"        │ ││
│  │  │ ✓ aws-skills                │  │  │ ↔ VP Prod: "Design sync complete"  │ ││
│  │  │ ✓ terraform-skills          │  │  │ ↑ FE Dir: "Need 2 more workers"    │ ││
│  │  └─────────────────────────────┘  │  └────────────────────────────────────┘ ││
│  └───────────────────────────────────┴──────────────────────────────────────────┘│
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### 7.3 Molecule Detail Layout

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│  MOL-123: Build User Dashboard                              [ACTIVE]  [BUILD]    │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│  PROGRESS: ████████████████░░░░░░░░░░░░░░  40%                                   │
│                                                                                   │
│  ┌──────────────────────────────────────────────────────────────────────────────┐│
│  │                           WORKFLOW TIMELINE                                   ││
│  │                                                                               ││
│  │  ✓ Design Review ──────● Component Impl ──────○ QA Review ──────○ Security   ││
│  │    └ design_dir         └ FE workers           └ QA pool        └ [GATE]     ││
│  │    2h                     In Progress           Blocked          Pending      ││
│  │                           "Header done"                                       ││
│  │                                                                               ││
│  └──────────────────────────────────────────────────────────────────────────────┘│
│                                                                                   │
│  ┌────────────────────────────────────┬─────────────────────────────────────────┐│
│  │            RACI MATRIX              │         CURRENT CHECKPOINT              ││
│  │  ┌──────────────────────────────┐  │  ┌───────────────────────────────────┐ ││
│  │  │ Accountable: frontend_dir    │  │  │ Step: Component Implementation    │ ││
│  │  │ Responsible: FE worker pool  │  │  │ Status: "Completed Header comp"   │ ││
│  │  │ Consulted:   design_dir      │  │  │ Worker: frontend_worker_01        │ ││
│  │  │ Informed:    vp_engineering  │  │  │ Started: 45 minutes ago           │ ││
│  │  └──────────────────────────────┘  │  └───────────────────────────────────┘ ││
│  └────────────────────────────────────┴─────────────────────────────────────────┘│
│                                                                                   │
│  ┌──────────────────────────────────────────────────────────────────────────────┐│
│  │                           ACTIVITY LOG                                        ││
│  │  ┌────────────────────────────────────────────────────────────────────────┐  ││
│  │  │ 14:32  frontend_worker_01  Completed Header component                  │  ││
│  │  │ 14:15  frontend_worker_01  Started Component Implementation            │  ││
│  │  │ 13:45  design_director     Approved design, passed GATE 2              │  ││
│  │  │ 13:30  design_director     Completed Design Review                     │  ││
│  │  │ 12:00  vp_engineering      Created molecule, assigned to frontend_dir  │  ││
│  │  │ 11:45  coo                 Delegated "Build User Dashboard" request    │  ││
│  │  └────────────────────────────────────────────────────────────────────────┘  ││
│  │  [View Full Log]  [View Git History]  [Export]                               ││
│  └──────────────────────────────────────────────────────────────────────────────┘│
│                                                                                   │
│  ACTIONS: [Escalate] [Reassign] [Add Step] [Cancel Molecule]                     │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Part 8: Design System Requirements

### 8.1 Core Component Library

#### **Layout Components**

| Component | Purpose | Props |
|-----------|---------|-------|
| `<AppShell />` | Main app container | sidebar, header, footer |
| `<PageHeader />` | Page title + actions | title, breadcrumb, actions |
| `<SplitView />` | Two-panel layout | left, right, ratio |
| `<CardGrid />` | Responsive card grid | columns, gap |
| `<Sidebar />` | Navigation sidebar | items, collapsed |

#### **Data Display Components**

| Component | Purpose | Props |
|-----------|---------|-------|
| `<DataTable />` | Sortable/filterable table | columns, data, pagination |
| `<StatCard />` | Single metric display | label, value, trend, icon |
| `<ProgressBar />` | Progress indicator | value, max, color, label |
| `<Timeline />` | Event timeline | events, orientation |
| `<KanbanBoard />` | Kanban columns | columns, cards, onDrag |
| `<OrgChart />` | Hierarchical tree | nodes, onNodeClick |
| `<FlowDiagram />` | Pipeline visualization | stages, connections |

#### **Agent-Specific Components**

| Component | Purpose | Props |
|-----------|---------|-------|
| `<AgentCard />` | Agent info card | agent, status, task |
| `<AgentAvatar />` | Agent icon + status | agent, size, showStatus |
| `<MoleculeCard />` | Workflow summary | molecule, compact |
| `<MoleculeTimeline />` | Workflow steps visual | steps, currentStep |
| `<HookQueue />` | Agent work queue | queue, onReorder |
| `<GateStatus />` | Gate indicator | gate, blocked, queue |
| `<ChannelMessage />` | Communication item | message, direction |
| `<RACIMatrix />` | RACI display | responsible, accountable, consulted, informed |

#### **Interactive Components**

| Component | Purpose | Props |
|-----------|---------|-------|
| `<CommandPalette />` | Quick actions (⌘K) | commands, onSelect |
| `<SearchBar />` | Global search | placeholder, onSearch |
| `<FilterBar />` | Multi-filter controls | filters, onChange |
| `<ActionMenu />` | Context actions | items, trigger |
| `<NotificationCenter />` | Alerts/notifications | notifications, onDismiss |

### 8.2 Animation & Motion Guidelines

```
TRANSITIONS
───────────
Default duration:   150ms
Complex transitions: 300ms
Page transitions:   200ms
Easing:            cubic-bezier(0.4, 0, 0.2, 1)

MICRO-INTERACTIONS
──────────────────
Button hover:       Scale 1.02, brightness +5%
Card hover:         Elevation increase, border highlight
Status change:      Color fade 200ms
Progress update:    Width transition 300ms
New item:           Fade in + slide down 200ms

LOADING STATES
──────────────
Skeleton:           Shimmer animation 1.5s infinite
Spinner:            Rotate 1s linear infinite
Progress:           Indeterminate bar animation
Pulse:              Agent status indicator 2s ease-in-out infinite
```

### 8.3 Responsive Breakpoints

```
Mobile:     < 640px    Single column, stacked layout
Tablet:     640-1024px  Two columns, collapsible sidebar
Desktop:    1024-1440px Full layout, all panels visible
Wide:       > 1440px    Extra space for expanded views
```

### 8.4 Accessibility Requirements

| Requirement | Implementation |
|-------------|----------------|
| Color contrast | WCAG AA minimum (4.5:1 for text) |
| Keyboard navigation | Full keyboard support, visible focus states |
| Screen readers | ARIA labels on all interactive elements |
| Reduced motion | Respect `prefers-reduced-motion` |
| Color blind support | Don't rely solely on color; use icons/patterns |
| Focus management | Logical focus order, focus trapping in modals |

### 8.5 Dark Mode Support

```
LIGHT MODE                    DARK MODE
──────────                    ─────────
Background: #FFFFFF           Background: #1F2937
Surface:    #F9FAFB           Surface:    #374151
Border:     #E5E7EB           Border:     #4B5563
Text:       #111827           Text:       #F9FAFB
Muted:      #6B7280           Muted:      #9CA3AF
```

---

## Part 9: Technical Implementation Recommendations

### 9.1 Frontend Stack Recommendation

```
Framework:        Next.js 14+ (App Router)
Language:         TypeScript
Styling:          Tailwind CSS + CSS Variables
Components:       shadcn/ui (headless primitives)
State:            Zustand (global) + TanStack Query (server)
Charts:           Recharts or Visx
Diagrams:         React Flow (org charts, pipelines)
Animation:        Framer Motion
WebSocket:        Socket.io-client
Forms:            React Hook Form + Zod
```

### 9.2 Key Data Flows

```
REAL-TIME UPDATES
─────────────────
WebSocket channels:
  - agent:status     → Agent status changes
  - molecule:update  → Molecule progress updates
  - gate:status      → Gate open/close events
  - channel:message  → Communication events
  - ledger:entry     → Bead ledger additions

POLLING (Backup)
────────────────
  - /api/pipeline/status    (5s)
  - /api/agents/health      (10s)
  - /api/metrics/summary    (30s)
```

### 9.3 Performance Considerations

| Concern | Mitigation |
|---------|------------|
| Large org charts | Virtual rendering, lazy load levels |
| Real-time updates | Debounce UI updates, batch state changes |
| Ledger history | Pagination, virtual scrolling |
| Dashboard metrics | Server-side aggregation, caching |
| Search | Debounced input, server-side search |

---

## Part 10: Priority Implementation Order

### Phase 1: Foundation
1. `<AppShell />` with navigation
2. CEO Dashboard (overview)
3. Basic Org Chart (static)
4. Molecule List (table view)

### Phase 2: Core Workflows
5. Pipeline Kanban view
6. Molecule Detail view
7. Gate Status dashboard
8. Agent Hook view

### Phase 3: Communication
9. Communication Hub (Channels)
10. Bead Ledger viewer
11. Notification Center

### Phase 4: Analytics
12. Pipeline Analytics
13. Department Dashboards
14. Historical reports

### Phase 5: Polish
15. Dark mode
16. Mobile responsive
17. Keyboard shortcuts
18. Performance optimization

---

## Appendix A: Component Mockup Reference

### Status Badge Variants

```
┌─────────────────────────────────────────────────────────────┐
│  STATUS BADGES                                               │
│                                                              │
│  ● Online    ● Working    ● Blocked    ○ Idle    ○ Offline  │
│  (green)     (blue)       (red)        (gray)    (outline)  │
│                                                              │
│  PRIORITY BADGES                                             │
│                                                              │
│  🔴 Critical  🟠 High  🟡 Medium  🟢 Low                     │
│                                                              │
│  STAGE BADGES                                                │
│                                                              │
│  [INBOX] [RESEARCH] [DESIGN] [BUILD] [QA] [SECURITY] [DEPLOY]│
│  (gray)  (purple)   (green)  (blue)  (orange)(red)  (teal)  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Icon Recommendations

```
HIERARCHY ICONS
───────────────
CEO:        👑 Crown
COO:        🤖 Robot
VP:         🎯 Target
Director:   📋 Clipboard
Worker:     ⚙️ Gear

DEPARTMENT ICONS
────────────────
Engineering: </> Code brackets
Research:    🔍 Magnifying glass
Product:     📦 Package
Quality:     🛡️ Shield
Operations:  ⚙️ Settings

ACTION ICONS
────────────
Create:      + Plus
Edit:        ✏️ Pencil
Delete:      🗑️ Trash
View:        👁️ Eye
Approve:     ✓ Checkmark
Reject:      ✗ X
Escalate:    ⬆️ Arrow up
```

---

## Conclusion

The AI Corp architecture presents a sophisticated multi-agent system that requires a thoughtfully designed UI to surface complexity while maintaining usability. Key design priorities:

1. **Visual Clarity**: The 5-level hierarchy and 5 departments need clear visual distinction through consistent color coding and layout patterns.

2. **Real-Time Awareness**: WebSocket-driven updates for agent status, molecule progress, and gate changes are essential for the CEO to maintain situational awareness.

3. **Progressive Disclosure**: Summary views (dashboard, pipeline kanban) with drill-down to detail views (molecule detail, agent hooks) to manage information density.

4. **Clear Navigation**: The page hierarchy should mirror the organizational structure, making navigation intuitive.

5. **Actionability**: Every view should offer clear actions the CEO can take, from creating molecules to approving gates to sending directives.

The recommended component library and design system provide a foundation for consistent, accessible, and performant UI development. Phase-based implementation allows for iterative delivery while building toward the complete vision.

---

*Document generated by Design Lead Agent*
*AI Corp Frontend UI Design Review v1.0*
