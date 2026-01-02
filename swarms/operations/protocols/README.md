# Operations Protocols

This directory contains the formal operational protocols for the agent-swarm system.

## Documents

### 1. Coordination Model (`coordination_model.md`)
**Purpose:** Complete protocol documentation for the Hybrid Coordination Model

**Audience:** All stakeholders (COO, Operations Coordinator, QA Agent, Swarms)

**Contents:**
- Full model overview and rationale
- Tier 1 (Direct Execution) procedures
- Tier 2 (Operations-Coordinated) procedures
- Escalation triggers and decision criteria
- QA engagement rules (pre-review and post-audit)
- Commit message standards with examples
- Success metrics and monitoring
- Operational workflows and timelines
- Status reporting formats
- Continuous improvement process

**When to read:**
- Initial onboarding
- Monthly protocol reviews
- When escalation patterns change
- For detailed reference during complex decisions

### 2. COO Quick Reference (`coo_quick_reference.md`)
**Purpose:** Fast decision-making guide for the COO

**Audience:** COO (Supreme Orchestrator)

**Contents:**
- 30-second decision tree
- 5-question escalation checklist
- Quick task type guide
- Commit message verification checklist
- Communication templates
- Common mistake patterns
- Real-world decision examples

**When to use:**
- Every time a new task arrives from CEO
- When uncertain about Tier 1 vs Tier 2
- For quick communication templates
- As a daily reference tool

## Quick Start

### For COO:
1. Read `coo_quick_reference.md` first (10 minutes)
2. Bookmark it for daily use
3. Refer to `coordination_model.md` for detailed scenarios

### For Operations Coordinator:
1. Read full `coordination_model.md` (30 minutes)
2. Understand all escalation triggers
3. Memorize QA engagement rules
4. Review monthly with team

### For QA Agent:
1. Read QA sections in `coordination_model.md`
2. Focus on pre-review vs post-audit criteria
3. Understand escalation authority

### For Swarms:
1. Read commit message standards in `coordination_model.md`
2. Understand when to escalate to Operations
3. Follow [SWARM:name] [PRIORITY:1-5] [STATUS:status] format

## Implementation Status

**Effective Date:** 2026-01-01
**Approved By:** CEO
**Status:** Active

**Current Version:** 1.0
- Hybrid model implemented
- Two-tier coordination system active
- Commit message standards enforced
- QA engagement rules defined

## Key Contacts

- **Protocol Owner:** Operations Coordinator
- **Approval Authority:** COO (with CEO oversight)
- **Questions:** Contact Operations Coordinator

## Protocol Updates

Protocol updates require:
1. Operations Coordinator proposal
2. COO approval
3. CEO notification for major changes
4. Version increment and change log update

**Next Review:** 2026-02-01 (monthly)

## Related Files

- `/swarms/operations/swarm.yaml` - Operations swarm configuration
- `/swarms/operations/agents/ops_coordinator.md` - Operations Coordinator agent prompt
- `/swarms/operations/agents/qa_agent.md` - QA Agent prompt

## Metrics Dashboard

Track hybrid model effectiveness:
- Tier 1 completion time: Target <24h
- Tier 2 coordination overhead: Target <4h planning
- Escalation rate: Target <20%
- Commit compliance: Target >95%
- QA pass rate: Target >90%

Review these monthly with COO and adjust protocols as needed.

