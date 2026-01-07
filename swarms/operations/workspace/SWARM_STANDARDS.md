# Swarm Workspace Standards

> **This document defines organizational standards for all swarm workspaces.**
> QA Agent enforces these standards through regular audits.

**Version:** 1.0
**Effective Date:** 2026-01-02
**Owner:** QA Agent
**Approved By:** Operations Coordinator

---

## 1. Required Folder Structure

Every active swarm workspace MUST maintain this structure:

```
swarms/{swarm_name}/
  workspace/
    STATE.md              # REQUIRED - Swarm shared memory
    sessions/             # REQUIRED - Historical work sessions
    research/             # OPTIONAL - Research outputs (research swarms only)
    decisions/            # RECOMMENDED - Architecture Decision Records
    archive/              # RECOMMENDED - Completed/obsolete materials
```

### Folder Purposes

#### `STATE.md` (Required)
The single source of truth for swarm status. Every agent reads this first.

**Must contain:**
- Last Updated date
- Current Objectives
- Progress Log (most recent at top)
- Key Files table
- Architecture Decisions (or link to decisions/)
- Known Issues / Blockers
- Next Steps

#### `sessions/` (Required)
Historical records of work sessions, organized by date.

**Structure:**
```
sessions/
  2026-01-02/           # One folder per active day
    session_notes.md    # Summary of what happened
    [outputs...]        # Files created during session
  2026-01-01/
    ...
```

**When to create a session folder:**
- Multi-file research output
- Complex implementation spanning multiple commits
- Brainstorming or exploration work
- Any work generating 3+ related files

**Session folder naming:** `YYYY-MM-DD` or `YYYY-MM-DD_topic` for multiple sessions per day.

#### `research/` (Optional)
For research-focused swarms (asa_research, etc.). Contains research outputs that are not tied to specific sessions.

**Good candidates for research/:**
- Literature reviews
- Competitive analysis
- Technology evaluations
- Long-term reference documents

#### `decisions/` (Recommended)
Architecture Decision Records (ADRs) that are too detailed for STATE.md.

**ADR Format:**
```markdown
# ADR-XXX: Decision Title

**Date:** YYYY-MM-DD
**Status:** Proposed | Accepted | Deprecated | Superseded
**Deciders:** [who made the decision]

## Context
[Why was this decision needed?]

## Decision
[What was decided?]

## Consequences
[What are the implications?]

## Alternatives Considered
[What else was evaluated?]
```

#### `archive/` (Recommended)
For files that are no longer actively used but should be preserved.

**When to archive:**
- Work is superseded by newer version
- Project/feature was cancelled
- Reference material no longer needed for active work

**Archive structure:**
```
archive/
  2026-Q1/              # Organize by quarter
    old_research/
    deprecated_plans/
  2025-Q4/
    ...
```

---

## 2. STATE.md Requirements

### Mandatory Sections

Every STATE.md MUST include these sections:

```markdown
# Swarm State

> **This file is the shared memory for all agents working on this swarm.**
> Always read this file first. Update it after completing work.

## Last Updated
[DATE] - [AGENT_TYPE] ([Brief description])

## Current Objectives
1. [Numbered list of active objectives]

## Progress Log
<!-- Most recent entries at top -->

### [DATE] [AGENT_TYPE] - [Title]
- [What was done]
- Files changed: `file1.py`, `file2.ts`
- Outcome: success/partial/blocked

## Key Files
| File | Purpose | Last Modified By |
|------|---------|------------------|
| ... | ... | ... |

## Architecture Decisions
[Brief summaries or links to decisions/ folder]

## Known Issues / Blockers
[Tracked problems with status]

## Next Steps
[What should happen next]
```

### Currency Requirements

| Swarm Status | Max STATE.md Age | Action if Exceeded |
|--------------|------------------|-------------------|
| Active | 7 days | Flag in weekly report |
| Active (high priority) | 3 days | Alert Ops Coordinator |
| Paused | 30 days | Acceptable |
| Inactive | 90 days | Consider archiving |

### Progress Log Requirements

- **Date format:** `YYYY-MM-DD`
- **Agent identification:** Include agent type (e.g., RESEARCHER, IMPLEMENTER)
- **Brief title:** One-line description of work
- **Files changed:** List modified/created files
- **Outcome:** Always state success/partial/blocked

**Good example:**
```markdown
### 2026-01-02 IMPLEMENTER - Fixed Authentication Bug
- Updated token validation logic to handle edge case
- Files changed: `auth/validator.py`, `tests/test_auth.py`
- Outcome: success
```

**Bad example:**
```markdown
### Fixed stuff
- Made some changes
```

---

## 3. File Naming Conventions

### General Rules

| Element | Convention | Example |
|---------|------------|---------|
| Dates in filenames | `YYYY-MM-DD` prefix | `2026-01-02_research_notes.md` |
| Multi-word names | `UPPER_SNAKE_CASE` for docs | `IMPLEMENTATION_PLAN.md` |
| Code files | `snake_case` | `trading_bot.py` |
| Session folders | `YYYY-MM-DD` or `YYYY-MM-DD_topic` | `2026-01-02_brainstorm/` |
| Archive folders | Quarter-based | `2026-Q1/` |

### Markdown File Naming

**Research/Analysis documents:**
- Use `UPPER_SNAKE_CASE`
- Prefix with category if helpful
- Examples: `RESEARCH_COERCION_MECHANISMS.md`, `ANALYSIS_COMPETITOR_REVIEW.md`

**Session-specific documents:**
- Prefix with date
- Examples: `2026-01-02_brainstorm_notes.md`, `2026-01-02_meeting_summary.md`

**Decisions:**
- Use ADR format
- Examples: `ADR-001_authentication_approach.md`

### Avoid

- Spaces in filenames (use underscores)
- Special characters except `-` and `_`
- Overly long filenames (max 50 characters)
- Generic names like `notes.md`, `temp.md`, `test.md`
- Version numbers in filenames (use git for versioning)

---

## 4. Documentation Standards

### Required Documentation

| Document | Required For | Location |
|----------|--------------|----------|
| STATE.md | All swarms | `workspace/STATE.md` |
| swarm.yaml | All swarms | `swarm.yaml` (root of swarm folder) |
| Agent prompts | All agents | `agents/{agent_name}.md` |
| README.md | Code projects | Project root |

### Markdown Formatting

**Headers:**
- Use ATX-style headers (`#`, `##`, etc.)
- One blank line before and after headers
- No skipping levels (don't go from `#` to `###`)

**Lists:**
- Use `-` for unordered lists
- Use `1.` for ordered lists
- Indent with 2 spaces for nested items

**Tables:**
- Align columns with pipes
- Use header separators (`|---|---|`)

**Code:**
- Use fenced code blocks with language identifier
- Use inline code for filenames, commands, and identifiers

**Links:**
- Use relative links for internal documents
- Use absolute links for external resources

### Timestamp Metadata (Optional but Recommended)

For research swarms, include YAML front matter:

```yaml
---
created: YYYY-MM-DD HH:MM
updated: YYYY-MM-DD
author: [agent or person]
---
```

---

## 5. Session Archival Rules

### When to Create Session Records

Create a session folder when:
- Work session generates 3+ related files
- Complex exploration or research occurs
- Brainstorming produces multiple artifacts
- Multi-day work on single topic

### Session Folder Contents

**Required:**
- `session_notes.md` or `README.md` summarizing the session

**Optional:**
- Research outputs
- Code experiments
- Decision documents
- Meeting notes

### Session Notes Format

```markdown
# Session: [Brief Description]
**Date:** YYYY-MM-DD
**Agent(s):** [Who worked on this]

## Summary
[1-3 sentence overview]

## Objectives
- [What was the goal?]

## Work Completed
- [Bullet points of what was done]

## Files Created
| File | Purpose |
|------|---------|
| ... | ... |

## Outcomes
[Key results, decisions made, blockers identified]

## Next Steps
[What should happen next?]
```

### When to Archive

Move to `archive/` when:
- Work is superseded by newer version
- Project direction changed
- Material is 30+ days old and no longer referenced
- Feature was cancelled

### Archive Procedure

1. Create quarter folder if needed (`archive/2026-Q1/`)
2. Move files to appropriate subfolder
3. Update STATE.md to remove references or note archival
4. Add brief note in Progress Log about archival

---

## 6. Swarm-Specific Requirements

### Research Swarms (asa_research)

Additional requirements:
- `research/` folder for major research outputs
- Literature citations where applicable
- Clear distinction between hypotheses and validated findings
- Version tracking for evolving theories

### Development Swarms (swarm_dev)

Additional requirements:
- Code must follow language-specific conventions (PEP 8, ESLint, etc.)
- Tests for new functionality
- Docstrings/comments for complex logic
- PR descriptions for merged changes

### Trading/Finance Swarms (trading_bots)

Additional requirements:
- Risk parameters documented
- Capital limits clearly stated
- Paper trading results tracked
- No production credentials in files

### Application Swarms (mynd_app)

Additional requirements:
- API documentation for endpoints
- Database schema documentation
- Deployment runbook
- Environment configuration documented

---

## 7. Compliance Enforcement

### Audit Frequency

| Swarm Status | Audit Frequency |
|--------------|-----------------|
| Active (high priority) | Weekly |
| Active | Bi-weekly |
| Paused | Monthly |
| Inactive | Quarterly |

### Compliance Scoring

Each audit evaluates:

| Category | Weight | Criteria |
|----------|--------|----------|
| STATE.md Quality | 30% | All sections present, current, accurate |
| Folder Structure | 25% | Required folders exist, properly used |
| File Naming | 15% | Conventions followed |
| Documentation | 15% | Required docs present, current |
| Archive Hygiene | 15% | Old files archived, no clutter |

### Grades

| Grade | Score | Meaning |
|-------|-------|---------|
| A | 90-100% | Excellent compliance |
| B | 80-89% | Good, minor issues |
| C | 70-79% | Fair, needs improvement |
| D | 60-69% | Poor, significant issues |
| F | <60% | Critical, immediate remediation required |

### Remediation Process

1. **Score D or F:** QA Agent creates remediation issue in STATE.md
2. **Issue communicated:** Ops Coordinator notifies COO and swarm lead
3. **Remediation deadline:** 1 week for D, 3 days for F
4. **Re-audit:** QA Agent verifies remediation complete
5. **Escalation:** If not resolved, escalate to COO

---

## 8. Exemptions

### Requesting an Exemption

Swarms may request exemptions from specific standards if:
- Standard is inappropriate for swarm type
- Technical limitation prevents compliance
- Temporary exception during transition

**Exemption Process:**
1. Swarm lead documents reason in STATE.md
2. Ops Coordinator reviews and approves/denies
3. Exemption recorded in this document (Section 8)
4. Review exemption quarterly

### Current Exemptions

| Swarm | Standard | Exemption | Reason | Expires |
|-------|----------|-----------|--------|---------|
| _none_ | | | | |

---

## Document Control

**Version History:**
- v1.0 (2026-01-02): Initial standards document

**Next Review Date:** 2026-02-02

**Document Owner:** QA Agent

**Approval Authority:** Operations Coordinator
