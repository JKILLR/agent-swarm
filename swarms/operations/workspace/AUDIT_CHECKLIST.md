# Swarm Audit Checklist

> **This checklist is used by QA Agent for weekly swarm audits.**
> Complete all sections and record results in the audit report.

**Version:** 1.0
**Effective Date:** 2026-01-02
**Owner:** QA Agent

---

## Audit Schedule

| Week | Swarm to Audit | Notes |
|------|----------------|-------|
| Week 1 | asa_research | Research swarm, high file count |
| Week 2 | trading_bots | Finance swarm, risk-sensitive |
| Week 3 | swarm_dev | Core development, infrastructure |
| Week 4 | mynd_app | Application swarm (if active) |

Rotate monthly. Adjust if swarm status changes.

---

## Pre-Audit Preparation

Before starting the audit:

1. [ ] Read swarm's `swarm.yaml` for context
2. [ ] Check swarm status (active/paused/inactive)
3. [ ] Review previous audit results if any
4. [ ] Note any exemptions granted (see SWARM_STANDARDS.md Section 8)

---

## Weekly Audit Checklist

### Section 1: STATE.md Health (30 points)

**Location:** `swarms/{name}/workspace/STATE.md`

| Item | Points | Pass/Fail | Notes |
|------|--------|-----------|-------|
| STATE.md exists | 5 | | |
| "Last Updated" is within 7 days | 5 | | |
| Current Objectives section populated | 3 | | |
| Progress Log has entries from past 14 days | 4 | | |
| Key Files table is populated and accurate | 4 | | |
| Architecture Decisions documented | 3 | | |
| Known Issues tracked with status | 3 | | |
| Next Steps defined | 3 | | |

**Section 1 Score:** ___ / 30

---

### Section 2: Folder Structure (25 points)

**Location:** `swarms/{name}/workspace/`

| Item | Points | Pass/Fail | Notes |
|------|--------|-----------|-------|
| `sessions/` folder exists | 5 | | |
| Recent sessions properly organized by date | 5 | | |
| No more than 15 files in workspace root | 5 | | |
| `archive/` folder exists (or no old files) | 5 | | |
| `decisions/` or ADRs in STATE.md | 5 | | |

**For research swarms only:**
| Item | Points | Pass/Fail | Notes |
|------|--------|-----------|-------|
| `research/` folder exists | +5 | | |

**Section 2 Score:** ___ / 25 (or ___ / 30 for research swarms)

---

### Section 3: File Naming (15 points)

| Item | Points | Pass/Fail | Notes |
|------|--------|-----------|-------|
| Markdown files use UPPER_SNAKE_CASE or date prefix | 5 | | |
| No spaces in filenames | 3 | | |
| No generic names (temp.md, test.md, notes.md) | 3 | | |
| Code files follow language conventions | 2 | | |
| Filenames under 50 characters | 2 | | |

**Section 3 Score:** ___ / 15

---

### Section 4: Documentation Quality (15 points)

| Item | Points | Pass/Fail | Notes |
|------|--------|-----------|-------|
| README.md exists for code projects | 3 | | |
| Agent prompt files are complete | 3 | | |
| swarm.yaml is accurate and current | 3 | | |
| Markdown formatting is consistent | 3 | | |
| Links are functional (not broken) | 3 | | |

**Section 4 Score:** ___ / 15

---

### Section 5: Archive Hygiene (15 points)

| Item | Points | Pass/Fail | Notes |
|------|--------|-----------|-------|
| Files older than 30 days are archived or still relevant | 5 | | |
| Archive organized by quarter | 3 | | |
| No duplicate files | 4 | | |
| Obsolete/superseded files removed or archived | 3 | | |

**Section 5 Score:** ___ / 15

---

## Score Calculation

**Total Possible:** 100 points

| Section | Score | Weight |
|---------|-------|--------|
| STATE.md Health | ___ / 30 | 30% |
| Folder Structure | ___ / 25 | 25% |
| File Naming | ___ / 15 | 15% |
| Documentation | ___ / 15 | 15% |
| Archive Hygiene | ___ / 15 | 15% |
| **TOTAL** | ___ / 100 | |

**Grade:**
- A: 90-100
- B: 80-89
- C: 70-79
- D: 60-69
- F: <60

---

## Red Flags to Watch For

### Critical (Immediate Escalation Required)

| Red Flag | Why It Matters | Action |
|----------|----------------|--------|
| STATE.md missing | Swarm has no shared memory | Create immediately |
| STATE.md >30 days stale | Swarm may be abandoned | Escalate to Ops Coordinator |
| Credentials in files | Security vulnerability | Remove immediately, rotate credentials |
| No progress in 14+ days (active swarm) | Work may be blocked | Investigate with swarm lead |
| Critical blocker unaddressed for 7+ days | Business impact | Escalate to COO |

### Warning (Flag in Report)

| Red Flag | Why It Matters | Action |
|----------|----------------|--------|
| 20+ files in workspace root | Navigation difficulty | Schedule cleanup |
| No session folders | History not preserved | Recommend folder creation |
| Conflicting or duplicate files | Confusion risk | Identify and deduplicate |
| Stale Next Steps (>14 days unchanged) | Planning may be outdated | Request update |
| Missing agent prompt files | Agent behavior undefined | Flag to swarm lead |

### Informational (Note for Improvement)

| Flag | Suggestion |
|------|------------|
| No timestamp metadata on docs | Recommend adding YAML headers |
| Inconsistent markdown formatting | Minor cleanup opportunity |
| Archive folder empty | May be fine, or cleanup needed |
| Long filenames | Suggest shortening |

---

## How to Assess Swarm Health

### Health Categories

**GOOD (Grade A-B):**
- STATE.md current and complete
- Folder structure in place
- Regular progress visible
- No open blockers
- Files well-organized

**FAIR (Grade C):**
- STATE.md exists but partially stale
- Some structure missing
- Progress visible but irregular
- Minor blockers present
- Some disorganization

**NEEDS_ATTENTION (Grade D):**
- STATE.md stale or incomplete
- Poor folder structure
- Infrequent progress
- Significant blockers
- Disorganized files

**CRITICAL (Grade F):**
- STATE.md missing or very stale
- No structure
- No recent progress
- Major blockers unaddressed
- Files scattered/confusing

---

## Escalation Criteria

### Escalate to Ops Coordinator When:

1. **Grade F audit** - Immediate remediation required
2. **Critical blocker >7 days** - Business impact likely
3. **Credentials found in files** - Security issue
4. **No progress >14 days** (active swarm) - May indicate larger problem
5. **Cross-swarm inconsistency** - Standards drift

### Escalate to COO When:

1. **Remediation not completed after 1 week**
2. **Same issues found in consecutive audits**
3. **Swarm lead unresponsive**
4. **Security or data integrity concern**
5. **Pattern of declining health across swarms**

---

## Audit Report Template

```markdown
# Swarm Audit Report: {SWARM_NAME}

**Date:** YYYY-MM-DD
**Auditor:** QA Agent
**Swarm Status:** Active/Paused/Inactive

## Summary
[1-2 sentence overall assessment]

## Scores

| Section | Score | Grade |
|---------|-------|-------|
| STATE.md Health | _/30 | |
| Folder Structure | _/25 | |
| File Naming | _/15 | |
| Documentation | _/15 | |
| Archive Hygiene | _/15 | |
| **TOTAL** | _/100 | _ |

## Health Assessment: [GOOD/FAIR/NEEDS_ATTENTION/CRITICAL]

## Findings

### Critical Issues (Must Fix)
- [Issue] - [Location] - [Recommendation]

### Warnings (Should Fix)
- [Issue] - [Location] - [Recommendation]

### Suggestions (Nice to Have)
- [Issue] - [Location] - [Recommendation]

## Positive Notes
- [Things done well]

## Remediation Deadline
[If Grade D or F: specify deadline]

## Next Audit
[Date of next scheduled audit]
```

---

## Post-Audit Actions

### After Every Audit:

1. [ ] Record audit results in Operations STATE.md Progress Log
2. [ ] Update swarm health status in Operations STATE.md managed swarms table
3. [ ] If Grade D/F: Create issue in swarm's STATE.md
4. [ ] If critical red flags: Escalate immediately
5. [ ] Save audit report to `swarms/operations/audits/{swarm}_{date}.md`

### After Remediation:

1. [ ] Conduct follow-up audit within 1 week
2. [ ] Verify issues resolved
3. [ ] Update health status
4. [ ] Close remediation issue

---

## Quick Reference: Audit in 10 Minutes

If time-constrained, focus on:

1. **STATE.md exists and is current?** (30 seconds)
2. **Progress in past 7 days?** (1 minute)
3. **Any blockers?** (1 minute)
4. **Files reasonably organized?** (2 minutes)
5. **Any red flags?** (1 minute)

Record findings even if partial. A quick audit is better than no audit.

---

## Document Control

**Version History:**
- v1.0 (2026-01-02): Initial checklist

**Next Review Date:** 2026-02-02

**Document Owner:** QA Agent
