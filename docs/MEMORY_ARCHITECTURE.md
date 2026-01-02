# Agent Swarm Memory Architecture

## Overview

The memory system provides persistent context across sessions, enabling agents to:
- Understand organizational goals and vision
- Track progress on ongoing work
- Remember key decisions and their rationale
- Maintain awareness of cross-swarm dependencies

## Memory Hierarchy

```
memory/
├── core/
│   ├── vision.md          # Organization mission, values, long-term goals
│   ├── priorities.md      # Current strategic priorities (updated regularly)
│   └── decisions.md       # Major organizational decisions log
│
├── swarms/
│   ├── {swarm_name}/
│   │   ├── context.md     # Swarm mission, current state, key info
│   │   ├── progress.md    # Active work, blockers, recent completions
│   │   ├── history.md     # Session summaries (rolling, last N sessions)
│   │   └── knowledge.md   # Swarm-specific knowledge, patterns, lessons
│   │
│   └── cross_swarm.md     # Dependencies and coordination between swarms
│
└── sessions/
    └── {session_id}.md    # Raw session logs (for reference/debugging)
```

## Context Loading Rules

### COO (Supreme Orchestrator)
Loads on every chat:
1. `core/vision.md` - Full content
2. `core/priorities.md` - Full content
3. `core/decisions.md` - Last 10 entries
4. `swarms/cross_swarm.md` - Full content
5. Each swarm's `context.md` - Summary section only
6. Current chat session history

### VP of Operations
Loads on spawn:
1. `core/vision.md` - Summary
2. `core/priorities.md` - Full content
3. `swarms/cross_swarm.md` - Full content
4. All swarms' `progress.md` - Recent entries
5. All swarms' `context.md` - Full content

### Swarm Orchestrators
Loads on spawn:
1. `core/vision.md` - Summary
2. `core/priorities.md` - Their swarm's section
3. Own swarm's `context.md` - Full content
4. Own swarm's `progress.md` - Full content
5. Own swarm's `history.md` - Last 3 sessions
6. `swarms/cross_swarm.md` - Relevant dependencies

### Individual Agents (researcher, implementer, etc.)
Loads on spawn:
1. Own swarm's `context.md` - Mission and current focus
2. Own swarm's `progress.md` - Active work section
3. Relevant files from swarm workspace

## Memory Update Triggers

### Automatic Updates
1. **After each chat session**:
   - Summarize conversation into `progress.md`
   - Log any decisions to appropriate `decisions.md`

2. **After agent task completion**:
   - Update `progress.md` with what was done
   - Update `knowledge.md` if patterns/lessons learned

3. **Daily rollup** (if system runs continuously):
   - Consolidate session logs into `history.md`
   - Prune old entries, keep summaries

### Manual Updates
- CEO can edit any memory file directly
- Agents can propose updates (logged, CEO can review)

## Memory File Formats

### vision.md
```markdown
# Organization Vision

## Mission
[One paragraph mission statement]

## Core Values
- Value 1: Description
- Value 2: Description

## Long-term Goals
1. Goal with timeline
2. Goal with timeline

## Summary
[2-3 sentence summary for quick loading]
```

### context.md (per swarm)
```markdown
# {Swarm Name} Context

## Mission
[What this swarm exists to do]

## Current Focus
[What we're actively working on right now]

## Key Files
- `path/to/important/file.py` - What it does
- `path/to/config.yaml` - Configuration for X

## Team
- orchestrator: Coordinates work
- researcher: Investigates solutions
- implementer: Builds features

## Dependencies
- Depends on: [other swarms/external]
- Depended on by: [other swarms]

## Summary
[2-3 sentence summary for quick loading]
```

### progress.md (per swarm)
```markdown
# {Swarm Name} Progress

## Active Work
- [ ] Task 1 - assigned to X - status
- [x] Task 2 - completed by Y

## Blockers
- Blocker description - waiting on Z

## Recently Completed
### {Date}
- Completed item with brief description

## Next Up
- Planned work items
```

### decisions.md
```markdown
# Decision Log

## {Date} - {Decision Title}
**Context**: Why this decision was needed
**Decision**: What was decided
**Rationale**: Why this choice
**Impact**: What changes as a result
**Owner**: Who made/approved this
```

## Implementation Notes

### Context Size Management
- Use summaries for broad context
- Full content only for immediately relevant files
- Implement token counting to stay within limits
- Prioritize recent over old

### Consistency
- All writes go through MemoryManager
- Atomic updates (read-modify-write with locks)
- Backup before major changes

### Search/Retrieval (Future)
- Could add vector embeddings for semantic search
- For now, structured files + grep is sufficient
