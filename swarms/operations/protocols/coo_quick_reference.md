# COO Quick Reference Guide: Hybrid Coordination Model

**Version:** 1.0 | **Last Updated:** 2026-01-01

---

## When You Receive a Task from CEO

### The 30-Second Decision

Ask yourself these **5 questions** in order:

```
1. Is this Priority 1-2? ────────────────► YES → Engage Operations (Tier 2)
                              │
                              NO
                              ▼
2. Does this span multiple swarms? ─────► YES → Engage Operations (Tier 2)
                              │
                              NO
                              ▼
3. Are there cross-swarm dependencies? ─► YES → Engage Operations (Tier 2)
                              │
                              NO
                              ▼
4. Is this infrastructure/architecture? ► YES → Engage Operations (Tier 2)
                              │
                              NO
                              ▼
5. Could this conflict with other work? ► YES → Engage Operations (Tier 2)
                              │
                              NO
                              ▼
                     DIRECT SWARM EXECUTION (Tier 1)
```

**If ANY answer is YES → Tier 2 (Operations)**
**If ALL answers are NO → Tier 1 (Direct)**

---

## Tier 1: Direct Swarm Execution (DEFAULT)

### Use When:
- Single swarm can handle it
- Priority 3-5
- No dependencies
- Standard work (features, fixes, docs)

### Your Actions:
1. Select appropriate swarm
2. Create brief task description
3. Delegate to swarm
4. Monitor commit messages
5. Verify completion

### Timeline: Hours to 1-2 days

---

## Tier 2: Operations Coordination (ESCALATE)

### Use When:
- **Priority 1-2** (critical/high)
- **Multiple swarms** needed
- **Dependencies** across swarms
- **Infrastructure** changes
- **Resource conflicts** possible

### Your Actions:
1. Brief Operations Coordinator
2. Provide full context
3. Set expectations/deadlines
4. Receive periodic updates
5. Get completion report

### Timeline: 2-5 days

---

## Quick Task Type Guide

| Task Type | Coordination Tier |
|-----------|------------------|
| Single feature in one swarm | Tier 1 |
| Bug fix in existing code | Tier 1 |
| Documentation update | Tier 1 |
| Code refactoring (single swarm) | Tier 1 |
| Configuration tweaks | Tier 1 |
| **Backend + Frontend integration** | **Tier 2** |
| **Database migration** | **Tier 2** |
| **API contract changes** | **Tier 2** |
| **Multi-swarm feature** | **Tier 2** |
| **Production incident** | **Tier 2** |
| **Security vulnerability** | **Tier 2** |
| **Core infrastructure** | **Tier 2** |

---

## Commit Message Verification

All commits should follow this format:

```
[SWARM:name] [PRIORITY:1-5] [STATUS:status] Brief description
```

### Check for:
- All three tags present
- Priority matches your assignment
- Status is accurate
- Description is clear

### If Non-Compliant:
- Follow up with swarm
- Remind of standard
- Operations will track compliance metrics

---

## Red Flags (Auto-Escalate)

If you see any of these, **immediately engage Operations**:

- Swarm reports "blocked" status
- Multiple failed commits on same task
- Task involves >1 swarm
- Swarm asks about dependencies
- Resource conflicts mentioned
- Security/infrastructure keywords
- Priority 1-2 assigned

---

## Communication Templates

### Tier 1 Delegation
```
To: [Swarm Name]
Priority: [1-5]
Task: [Brief description]

Details:
[What needs to be done]

Expected completion: [timeframe]
Questions? Let me know.
```

### Tier 2 Escalation
```
To: Operations Coordinator
Priority: [1-5]
Complexity: Multi-swarm / Infrastructure / High-priority

Task Overview:
[What needs to be done]

Swarms Potentially Involved:
[List swarms]

Dependencies/Concerns:
[Any known issues]

Deadline: [date/time]
```

---

## Status Checking

### Tier 1 Monitoring
- Check recent commits for tags
- Review swarm status files daily
- Look for Operations alerts

### Tier 2 Updates
- Operations sends periodic updates
- Request on-demand status anytime
- Final report on completion

---

## Escalation Patterns to Watch

After a few weeks, you'll notice patterns:

**Common Tier 2 Scenarios:**
- Anything touching authentication
- Database schema changes
- API endpoint additions
- Shared utility modifications
- Deployment pipeline changes
- Cross-swarm integrations

**Learn from experience:**
- Track which tasks became complex
- Note false escalations (Tier 2 that could have been Tier 1)
- Adjust decision-making over time

---

## Monthly Metrics Review

Operations will provide monthly report showing:

- **Tier 1 vs Tier 2 distribution** (target: 80/20)
- **Average completion times** (Tier 1: <24h, Tier 2: 2-5d)
- **Escalation accuracy** (false escalation rate <5%)
- **Quality metrics** (commit compliance, QA pass rate)

Use these to refine your decision-making.

---

## Common Mistakes to Avoid

1. **Tier 1 when should be Tier 2:**
   - Symptom: Swarm gets blocked or asks for help mid-task
   - Fix: Use decision tree more carefully

2. **Tier 2 when should be Tier 1:**
   - Symptom: Simple task takes too long
   - Fix: Trust swarms more for standard work

3. **Incomplete briefings:**
   - Symptom: Swarm asks many clarifying questions
   - Fix: Provide more context upfront

4. **Missing commit tag checks:**
   - Symptom: Operations complains about non-compliance
   - Fix: Spot-check commits regularly

---

## Emergency Protocol

### Production Incident (Priority 1)
1. **Immediately** engage Operations Coordinator
2. Provide incident details and impact
3. Operations will coordinate all swarms
4. Stay available for executive decisions
5. Get post-incident report

### Security Vulnerability (Priority 1)
1. **Immediately** engage Operations Coordinator
2. Coordinate with QA for security review
3. Fast-track through Tier 2 process
4. Verify fix thoroughly before deployment

---

## Key Contacts

- **Operations Coordinator:** Primary contact for Tier 2
- **QA Agent:** Quality concerns, standards questions
- **CEO:** Strategic direction, major decisions

---

## Tips for Success

**Speed vs. Quality Balance:**
- Tier 1 = Speed (direct execution)
- Tier 2 = Quality (coordinated, reviewed)
- Use the right tier for the right task

**Trust Your Swarms:**
- They're capable of Tier 1 work independently
- Don't micromanage
- Monitor outcomes, not process

**Leverage Operations:**
- They're there to help with complexity
- Don't hesitate to escalate when uncertain
- Better to escalate early than fix issues later

**Learn and Adapt:**
- Track your decisions and outcomes
- Adjust patterns based on results
- Share learnings with CEO and Operations

---

## Decision Examples

### Example 1: Simple Feature
**Task:** "Add a new button to the MYND app dashboard"

**Decision Process:**
- Priority? 3 (Medium) → NO
- Multiple swarms? No, just mynd_app → NO
- Dependencies? None → NO
- Infrastructure? No → NO
- Conflicts? Unlikely → NO

**Result:** Tier 1 - Direct to mynd_app swarm

---

### Example 2: API Integration
**Task:** "Add user analytics endpoint that requires frontend dashboard"

**Decision Process:**
- Priority? 2 (High) → YES → STOP

**Result:** Tier 2 - Engage Operations
**Reason:** High priority AND multi-swarm (backend + frontend)

---

### Example 3: Database Change
**Task:** "Add a new column to users table"

**Decision Process:**
- Priority? 3 (Medium) → NO
- Multiple swarms? Potentially affects all swarms using users table → NO (continue)
- Dependencies? Yes, any swarm querying users table → YES → STOP

**Result:** Tier 2 - Engage Operations
**Reason:** Cross-swarm dependencies (schema change)

---

### Example 4: Documentation Update
**Task:** "Update README with new installation steps"

**Decision Process:**
- Priority? 4 (Low) → NO
- Multiple swarms? No → NO
- Dependencies? No → NO
- Infrastructure? No → NO
- Conflicts? No → NO

**Result:** Tier 1 - Direct to swarm_dev

---

### Example 5: Bug Fix
**Task:** "Fix crash in ASA research module"

**Decision Process:**
- Priority? 3 (Medium) → NO
- Multiple swarms? No, just asa_research → NO
- Dependencies? No → NO
- Infrastructure? No → NO
- Conflicts? No → NO

**Result:** Tier 1 - Direct to asa_research swarm

---

## Remember

**Default to Tier 1 for speed.**
**Escalate to Tier 2 when complexity demands it.**
**Trust the decision tree.**
**Monitor outcomes and improve.**

---

**Questions?**
Contact Operations Coordinator or refer to full protocol:
`swarms/operations/protocols/coordination_model.md`

