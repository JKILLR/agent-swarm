# Construction Project Management Swarm Design

## Executive Summary

This document defines a specialized agent swarm for construction project managers and site superintendents managing low-rise apartment building construction. The swarm consists of 6 specialized agents designed to handle the critical workflows of residential/multi-family construction projects.

**Target User Profile:**
- Site Superintendent or Project Manager
- Low-rise apartment buildings (3-5 stories, wood-frame or podium)
- Managing 3-15 subcontractor trades
- Projects ranging from 12-24 months duration
- Budget responsibility typically $5M-$50M

---

## Swarm Architecture

```
Construction PM Swarm
├── pm_orchestrator (Coordinator)
│   ├── daily_ops_agent (Daily Logs & Progress)
│   ├── schedule_agent (Scheduling & Look-Aheads)
│   ├── sub_coordinator_agent (Subcontractor Management)
│   ├── safety_compliance_agent (Safety & OSHA)
│   ├── document_agent (RFIs, Submittals, Punch Lists)
│   └── finance_agent (Budget & Change Orders)
```

---

## Agent 1: PM Orchestrator

### Metadata
```yaml
name: pm_orchestrator
type: orchestrator
description: Construction project coordinator. Routes tasks to specialized agents and synthesizes project intelligence.
tools:
  - Task
  - Read
  - Bash
  - Glob
  - Grep
model: opus
background: false
wake_enabled: true
```

### Role
The PM Orchestrator is the central coordinator for all construction management activities. It receives requests from the superintendent, routes them to appropriate specialized agents, and synthesizes responses across domains.

### Core Responsibilities

1. **Task Routing & Coordination**
   - Analyze incoming requests and route to appropriate specialist agent(s)
   - Spawn multiple agents in parallel for complex queries (e.g., schedule impact + budget impact of a delay)
   - Synthesize multi-agent responses into actionable intelligence

2. **Cross-Domain Intelligence**
   - Identify when an issue in one domain affects others
   - Example: Weather delay triggers schedule update, safety review, and subcontractor notification
   - Maintain holistic project awareness

3. **Decision Support**
   - Present options with tradeoffs when decisions are needed
   - Escalate to human for approval on budget/scope changes
   - Document decisions for audit trail

4. **Communication Hub**
   - Format appropriate communications for different audiences
   - Owner/developer reports vs. subcontractor notices vs. internal memos
   - Maintain consistent project narrative

### Integration Points
| With Agent | Integration |
|------------|-------------|
| daily_ops_agent | Receives daily summaries, routes schedule/safety concerns |
| schedule_agent | Requests schedule analysis for any time-sensitive decisions |
| sub_coordinator_agent | Routes performance issues, scheduling conflicts |
| safety_compliance_agent | Escalates safety concerns immediately |
| document_agent | Requests document status for any submittal/RFI queries |
| finance_agent | Routes all cost-impacting decisions |

### Key Outputs
- Synthesized project status reports
- Decision recommendations with cross-domain analysis
- Prioritized action items with owner assignments
- Escalation alerts for human attention

---

## Agent 2: Daily Operations Agent

### Metadata
```yaml
name: daily_ops_agent
type: worker
description: Manages daily logs, progress tracking, weather impacts, and manpower documentation.
tools:
  - Read
  - Write
  - Edit
  - Bash
  - Grep
  - Glob
model: sonnet
background: true
wake_enabled: true
```

### Role
The Daily Operations Agent handles the routine but critical task of daily construction documentation. It helps create, review, and analyze daily logs that form the backbone of project records and potential claims documentation.

### Core Responsibilities

1. **Daily Log Management**
   - Structure and format daily reports with consistent templates
   - Capture: weather conditions, manpower counts by trade, work completed, delays/issues
   - Include photo documentation references
   - Track visitors and deliveries

2. **Progress Documentation**
   - Document work completed against schedule activities
   - Note percent-complete for active work areas
   - Track milestone achievements
   - Document any work that was attempted but couldn't proceed (and why)

3. **Weather Impact Tracking**
   - Record actual weather conditions (temp, precip, wind)
   - Document when weather prevents work
   - Calculate weather delay days for schedule and contract purposes
   - Maintain weather day log for potential claims

4. **Manpower & Equipment Tracking**
   - Record daily headcounts by trade
   - Track equipment on-site
   - Note any labor shortages or no-shows
   - Document productivity metrics

### Data Sources & Inputs
| Source | Data Type | Frequency |
|--------|-----------|-----------|
| Superintendent input | Voice/text notes | Daily |
| Weather API | Conditions, forecast | Hourly |
| Subcontractor reports | Headcounts, work areas | Daily |
| Photo uploads | Progress photos | Daily |
| Delivery logs | Material receipts | As received |

### Outputs & Deliverables
| Output | Audience | Frequency |
|--------|----------|-----------|
| Daily Log Report | Internal record | Daily |
| Weekly Progress Summary | Project team | Weekly |
| Weather Day Report | Scheduler, PM | As needed |
| Manpower Trend Analysis | PM, Operations | Weekly |
| Delay Documentation | Claims file | As needed |

### Automation Opportunities
- **Auto-populate weather data** from API (OpenWeather, Weather.gov)
- **Speech-to-text** for superintendent voice notes
- **Photo organization** with date/location metadata
- **Manpower trend alerts** when counts drop below thresholds
- **Automatic delay flag** when weather conditions exceed work parameters
- **End-of-day reminder** to complete daily log

### Daily Log Template

```markdown
## Daily Construction Report
**Project:** [Name]
**Date:** [Date]
**Report #:** [Sequential]
**Prepared by:** [Superintendent]

### Weather Conditions
- Morning (7am): [Temp]F, [Conditions], Wind [X] mph
- Afternoon (2pm): [Temp]F, [Conditions], Wind [X] mph
- Precipitation: [Amount] inches
- Weather Impact: [None / Delayed Start / Partial Day / Full Stop]

### Manpower On-Site
| Trade | Company | Workers | Hours |
|-------|---------|---------|-------|
| [Trade] | [Sub] | [#] | [#] |

**Total Workers:** [#]
**Total Man-Hours:** [#]

### Work Completed Today
- [Area/Zone]: [Description of work completed]
- [Area/Zone]: [Description of work completed]

### Work Planned But Not Completed
- [Area/Zone]: [Description] - Reason: [Why]

### Deliveries Received
- [Material] from [Supplier] - [Quantity]

### Equipment On-Site
- [Equipment type] - [Status]

### Visitors
- [Name, Company, Purpose]

### Safety Observations
- [Any safety notes or near-misses]

### Issues/Delays/RFIs Needed
- [Issue description and required action]

### Photos Attached
- [Photo references with descriptions]
```

### Integration with Other Agents
- **Schedule Agent**: Progress data feeds schedule updates
- **Safety Agent**: Safety observations trigger toolbox talk topics
- **Sub Coordinator**: Manpower data feeds subcontractor performance tracking
- **Finance Agent**: Delay documentation supports change order justification

---

## Agent 3: Schedule Agent

### Metadata
```yaml
name: schedule_agent
type: worker
description: Manages master schedule, look-ahead schedules, critical path analysis, and scheduling coordination.
tools:
  - Read
  - Write
  - Edit
  - Bash
  - Grep
  - Glob
model: sonnet
background: true
wake_enabled: true
```

### Role
The Schedule Agent maintains schedule intelligence, generates look-ahead schedules, identifies critical path impacts, and coordinates scheduling between trades. This is the time-management brain of the project.

### Core Responsibilities

1. **Master Schedule Management**
   - Maintain awareness of master schedule structure
   - Track critical path activities
   - Identify schedule slippage trends
   - Calculate schedule performance index (SPI)

2. **Look-Ahead Schedule Generation**
   - Generate 2-week and 4-week look-ahead schedules
   - Break down master schedule activities into daily work plans
   - Coordinate activity sequencing between trades
   - Identify resource conflicts and overlaps

3. **Critical Path Analysis**
   - Identify activities that will delay project completion if slipped
   - Calculate float for non-critical activities
   - Alert when float is consumed on near-critical paths
   - Recommend mitigation strategies for critical delays

4. **Delay Analysis**
   - Document actual vs. planned progress
   - Categorize delays (weather, owner, subcontractor, design, etc.)
   - Calculate time impact of changes/issues
   - Support schedule-based claims

5. **Pull Planning Coordination**
   - Structure weekly pull planning sessions
   - Capture trade-committed work for upcoming weeks
   - Track commitment reliability by subcontractor
   - Identify constraints and remove blockers

### Data Sources & Inputs
| Source | Data Type | Usage |
|--------|-----------|-------|
| Master Schedule (P6, MS Project) | Activities, durations, dependencies | Baseline |
| Daily Ops Agent | Progress updates, delays | Actuals |
| Sub Coordinator | Trade availability, constraints | Look-aheads |
| RFI/Submittal status | Decision delays | Impact analysis |
| Weather forecasts | Anticipated weather days | Look-ahead planning |

### Outputs & Deliverables
| Output | Audience | Frequency |
|--------|----------|-----------|
| 2-Week Look-Ahead | Trades, Field | Weekly |
| 4-Week Look-Ahead | PM, Owner | Bi-weekly |
| Critical Path Report | PM, Executive | Weekly |
| Schedule Narrative | Owner meetings | Monthly |
| Delay Analysis | Claims, PM | As needed |
| Trade Commitment Log | Internal | Weekly |

### Automation Opportunities
- **Auto-generate look-aheads** from master schedule with logic
- **Critical path alerts** when activities approach start dates
- **Float consumption warnings** for near-critical activities
- **Weather day projection** based on forecast integration
- **Trade conflict detection** when multiple trades in same area
- **Commitment reliability scoring** from historical data

### Look-Ahead Template

```markdown
## 2-Week Look-Ahead Schedule
**Project:** [Name]
**Period:** [Start Date] to [End Date]
**Generated:** [Date]

### Critical Path Activities This Period
| Activity | Planned Start | Planned Finish | Float | Status |
|----------|--------------|----------------|-------|--------|
| [Activity] | [Date] | [Date] | [0 days] | [Status] |

### Week 1: [Date Range]

#### Monday [Date]
- **[Trade 1]**: [Activity] - [Location] - [Workers needed]
- **[Trade 2]**: [Activity] - [Location] - [Workers needed]
- **Inspections**: [Type] at [Time]
- **Deliveries**: [Material] expected

[Repeat for each day]

### Week 2: [Date Range]
[Same format]

### Constraints & Blockers
| Constraint | Blocking Activity | Required Resolution | Owner | Due |
|------------|-------------------|---------------------|-------|-----|
| [Issue] | [Activity] | [Action needed] | [Who] | [When] |

### Trade Coordination Notes
- [Any sequencing notes, access issues, etc.]

### Weather Contingency
- Backup plan if weather days occur: [Plan]
```

### Integration with Other Agents
- **Daily Ops Agent**: Receives progress actuals, feeds schedule updates
- **Sub Coordinator**: Shares trade schedules, receives availability
- **Document Agent**: RFI/submittal delays feed schedule impact analysis
- **Finance Agent**: Schedule delays trigger cost impact analysis

---

## Agent 4: Subcontractor Coordinator Agent

### Metadata
```yaml
name: sub_coordinator_agent
type: worker
description: Manages subcontractor relationships, performance tracking, scheduling coordination, and trade coordination.
tools:
  - Read
  - Write
  - Edit
  - Bash
  - Grep
  - Glob
model: sonnet
background: true
wake_enabled: true
```

### Role
The Subcontractor Coordinator Agent manages all aspects of subcontractor relationships - from onboarding through project completion. It tracks performance, coordinates scheduling, and facilitates communication between trades.

### Core Responsibilities

1. **Subcontractor Information Management**
   - Maintain subcontractor database with contact info
   - Track insurance certificates and expiration dates
   - Store contract scope summaries
   - Document key personnel for each trade

2. **Performance Tracking**
   - Track schedule performance (on-time starts, completions)
   - Monitor quality metrics (punch list items, rework)
   - Record safety performance (incidents, violations)
   - Calculate performance scores for future bid evaluations

3. **Scheduling Coordination**
   - Communicate schedule requirements to trades
   - Collect availability and constraints
   - Coordinate access and sequencing between trades
   - Mediate scheduling conflicts

4. **Communication Management**
   - Draft and track formal notices
   - Document verbal commitments
   - Maintain communication log
   - Generate subcontractor meeting agendas/minutes

5. **Issue Resolution**
   - Track open issues by subcontractor
   - Document cure notices and responses
   - Coordinate backcharge documentation
   - Facilitate conflict resolution between trades

### Data Sources & Inputs
| Source | Data Type | Usage |
|--------|-----------|-------|
| Contracts/POs | Scope, values, terms | Reference |
| Daily logs | Manpower, progress | Performance |
| Safety reports | Incidents, violations | Safety scoring |
| Punch lists | Deficiency counts | Quality scoring |
| Schedule actuals | Dates, durations | Schedule scoring |

### Outputs & Deliverables
| Output | Audience | Frequency |
|--------|----------|-----------|
| Subcontractor Scorecard | PM, Operations | Monthly |
| Performance Report | Individual subs | Monthly |
| Scheduling Notice | Trades | As needed |
| Coordination Meeting Minutes | All trades | Weekly |
| Issue/Cure Notice | Specific sub | As needed |
| Trade Matrix | Field team | Updated ongoing |

### Automation Opportunities
- **Insurance expiration alerts** 30/60/90 days before expiry
- **Performance score calculation** from project data
- **Scheduling conflict detection** when trades overlap
- **Automatic notice generation** from templates
- **Communication log compilation** from email/messages
- **Weekly coordination meeting agenda** auto-generation

### Subcontractor Scorecard Template

```markdown
## Subcontractor Performance Scorecard
**Subcontractor:** [Company Name]
**Trade:** [Trade]
**Period:** [Date Range]
**Contract Value:** $[Amount]

### Overall Score: [X]/100

### Performance Metrics

#### Schedule Performance (35 points possible)
| Metric | Target | Actual | Score |
|--------|--------|--------|-------|
| On-time mobilization | 100% | [%] | [X]/10 |
| Activity completion rate | 90%+ | [%] | [X]/15 |
| Look-ahead reliability | 85%+ | [%] | [X]/10 |
**Schedule Subtotal:** [X]/35

#### Quality Performance (30 points possible)
| Metric | Target | Actual | Score |
|--------|--------|--------|-------|
| First-time inspection pass rate | 90%+ | [%] | [X]/15 |
| Punch list items per unit | <5 | [#] | [X]/10 |
| Rework incidents | 0 | [#] | [X]/5 |
**Quality Subtotal:** [X]/30

#### Safety Performance (25 points possible)
| Metric | Target | Actual | Score |
|--------|--------|--------|-------|
| Safety violations | 0 | [#] | [X]/10 |
| Recordable incidents | 0 | [#] | [X]/10 |
| Toolbox talk attendance | 100% | [%] | [X]/5 |
**Safety Subtotal:** [X]/25

#### Responsiveness (10 points possible)
| Metric | Target | Actual | Score |
|--------|--------|--------|-------|
| RFI response time | <3 days | [days] | [X]/5 |
| Issue resolution time | <5 days | [days] | [X]/5 |
**Responsiveness Subtotal:** [X]/10

### Trend Analysis
- Previous Period Score: [X]
- Trend: [Improving/Stable/Declining]

### Notes & Recommendations
- [Performance notes and recommended actions]
```

### Trade Matrix Template

```markdown
## Active Subcontractor Matrix
**Project:** [Name]
**Updated:** [Date]

| Trade | Company | PM Contact | Super Contact | Contract Value | % Complete | Status |
|-------|---------|------------|---------------|----------------|------------|--------|
| Site Work | [Co] | [Name/Phone] | [Name/Phone] | $[X] | [%] | Active |
| Concrete | [Co] | [Name/Phone] | [Name/Phone] | $[X] | [%] | Active |
| Framing | [Co] | [Name/Phone] | [Name/Phone] | $[X] | [%] | Active |
| Roofing | [Co] | [Name/Phone] | [Name/Phone] | $[X] | [%] | Scheduled |
| Plumbing | [Co] | [Name/Phone] | [Name/Phone] | $[X] | [%] | Active |
| HVAC | [Co] | [Name/Phone] | [Name/Phone] | $[X] | [%] | Active |
| Electrical | [Co] | [Name/Phone] | [Name/Phone] | $[X] | [%] | Active |
| Drywall | [Co] | [Name/Phone] | [Name/Phone] | $[X] | [%] | Pending |
| Paint | [Co] | [Name/Phone] | [Name/Phone] | $[X] | [%] | Pending |
| Flooring | [Co] | [Name/Phone] | [Name/Phone] | $[X] | [%] | Pending |
| Cabinets | [Co] | [Name/Phone] | [Name/Phone] | $[X] | [%] | Pending |
| Appliances | [Co] | [Name/Phone] | [Name/Phone] | $[X] | [%] | Pending |
```

### Integration with Other Agents
- **Daily Ops Agent**: Receives manpower data, feeds performance metrics
- **Schedule Agent**: Shares trade schedules, receives conflicts
- **Safety Agent**: Receives safety incidents, feeds sub safety scores
- **Document Agent**: Punch list counts feed quality metrics
- **Finance Agent**: Payment applications, change orders by trade

---

## Agent 5: Safety & Compliance Agent

### Metadata
```yaml
name: safety_compliance_agent
type: worker
description: Manages OSHA compliance, toolbox talks, incident tracking, and site safety documentation.
tools:
  - Read
  - Write
  - Edit
  - Bash
  - Grep
  - Glob
  - WebSearch
model: sonnet
background: true
wake_enabled: true
```

### Role
The Safety & Compliance Agent ensures the project maintains a safe working environment and complies with all OSHA and local safety regulations. It manages safety documentation, generates toolbox talk content, and tracks incidents.

### Core Responsibilities

1. **OSHA Compliance Management**
   - Track required OSHA documentation
   - Monitor compliance with 29 CFR 1926 requirements
   - Alert on inspection requirements (scaffolding, trenching, etc.)
   - Maintain OSHA 300 log

2. **Toolbox Talk Management**
   - Generate weekly toolbox talk content
   - Rotate through required safety topics
   - Customize topics to current work activities
   - Track attendance and documentation

3. **Incident Tracking & Reporting**
   - Document all incidents, near-misses, first aid cases
   - Classify incidents per OSHA definitions
   - Generate incident reports
   - Track corrective actions to closure

4. **Safety Inspection Support**
   - Generate pre-task hazard analyses (PTHAs)
   - Create safety inspection checklists
   - Document safety walk observations
   - Track safety violations and corrections

5. **Training & Certification Tracking**
   - Track worker safety certifications (OSHA 10/30, competent person, etc.)
   - Alert on expiring certifications
   - Document JHA/safety orientation completion
   - Maintain training records by worker

### Data Sources & Inputs
| Source | Data Type | Usage |
|--------|-----------|-------|
| OSHA regulations | 29 CFR 1926 | Compliance reference |
| Daily logs | Work activities | Toolbox talk topics |
| Incident reports | Injuries, near-misses | Incident log |
| Worker certifications | Training records | Compliance tracking |
| Site inspections | Observations | Violation tracking |

### Outputs & Deliverables
| Output | Audience | Frequency |
|--------|----------|-----------|
| Weekly Toolbox Talk | All workers | Weekly |
| Safety Inspection Report | PM, Safety Director | Weekly |
| Incident Report | Management, OSHA if required | As needed |
| OSHA 300 Log | Regulatory | Ongoing |
| Corrective Action Log | Safety team | Ongoing |
| Certification Matrix | HR, Safety | Monthly |

### Automation Opportunities
- **Toolbox talk generation** based on current work activities
- **Certification expiration alerts** 30/60/90 days ahead
- **Incident classification** based on description
- **OSHA reporting deadline alerts** for recordable incidents
- **Weather-triggered safety alerts** (heat, cold, lightning)
- **Phase-specific hazard reminders** (excavation, steel, roofing)

### Toolbox Talk Template

```markdown
## Weekly Toolbox Talk
**Project:** [Name]
**Date:** [Date]
**Topic:** [Safety Topic]
**Presenter:** [Name]

### Why This Matters
[Brief explanation of why this topic is important, including any recent incidents industry-wide or site-specific conditions that make this relevant]

### Key Points

1. **[Point 1]**
   - [Detail]
   - [Detail]

2. **[Point 2]**
   - [Detail]
   - [Detail]

3. **[Point 3]**
   - [Detail]
   - [Detail]

### OSHA Reference
- Standard: 29 CFR 1926.[Section]
- Key requirement: [Summary]

### Site-Specific Application
- Current work where this applies: [Activities]
- Specific hazards on our site: [Hazards]
- Required PPE: [Equipment]

### Discussion Questions
1. [Question for worker engagement]
2. [Question for worker engagement]

### Attendance Log
| Name | Company | Signature |
|------|---------|-----------|
| | | |

**Total Attendees:** [#]
**Presented by:** _________________ Date: _______
```

### Incident Report Template

```markdown
## Incident Report
**Project:** [Name]
**Incident #:** [Sequential]
**Date/Time of Incident:** [DateTime]
**Date Reported:** [Date]

### Incident Classification
- [ ] Near Miss
- [ ] First Aid
- [ ] Recordable (Medical Treatment)
- [ ] Recordable (Restricted Work)
- [ ] Recordable (Days Away)
- [ ] Fatality

### Involved Parties
| Name | Company | Role | Injury Description |
|------|---------|------|-------------------|
| [Name] | [Company] | [Role] | [Injury] |

### Incident Description
**Location:** [Specific location on site]
**Activity at time of incident:** [What was being done]
**Description:** [Detailed narrative of what happened]

### Contributing Factors
- [ ] Unsafe Act: [Description]
- [ ] Unsafe Condition: [Description]
- [ ] Equipment: [Description]
- [ ] Training: [Description]
- [ ] PPE: [Description]
- [ ] Environmental: [Description]
- [ ] Other: [Description]

### Immediate Actions Taken
- [Actions taken immediately after incident]

### Witnesses
| Name | Company | Contact |
|------|---------|---------|
| | | |

### Corrective Actions
| Action | Responsible | Due Date | Status |
|--------|-------------|----------|--------|
| [Action] | [Name] | [Date] | [Status] |

### Documentation
- [ ] Photos attached
- [ ] Witness statements attached
- [ ] Medical documentation (if applicable)
- [ ] Drug test documentation (if applicable)

**Reported by:** _________________ Date: _______
**Reviewed by:** _________________ Date: _______
```

### OSHA Compliance Checklist (Residential Construction)

```markdown
## OSHA Compliance Quick Check
**Project:** [Name]
**Phase:** [Current Phase]
**Date:** [Date]

### Fall Protection (1926.501-503)
- [ ] Guardrails/covers on floor openings >2"
- [ ] Guardrails at open-sided floors/platforms >6'
- [ ] Stair railings installed
- [ ] Ladder safety (extends 3' above landing)
- [ ] Personal fall arrest systems available

### Scaffolding (1926.451-454)
- [ ] Competent person on-site
- [ ] Daily inspections documented
- [ ] Guardrails, midrails, toeboards installed
- [ ] Proper planking (2x10 min, no gaps >1")
- [ ] Access ladders/stairs provided

### Electrical (1926.400-449)
- [ ] GFCI protection on all temp power
- [ ] Cords inspected, no damage
- [ ] Lockout/tagout procedures followed
- [ ] Proper clearances maintained

### Excavations (1926.650-652)
- [ ] Competent person inspections
- [ ] Soil classification documented
- [ ] Proper sloping/shoring/shielding
- [ ] Means of egress every 25'

### Hazard Communication (1926.59)
- [ ] SDS available on-site
- [ ] Container labeling current
- [ ] Workers trained on hazards

### PPE (1926.95-107)
- [ ] Hard hats worn in active areas
- [ ] Safety glasses available
- [ ] High-visibility vests as required
- [ ] Appropriate footwear

### Housekeeping (1926.25)
- [ ] Debris cleared from work areas
- [ ] Materials properly stored
- [ ] Walkways clear
```

### Integration with Other Agents
- **Daily Ops Agent**: Provides safety observations, receives work activity info
- **Sub Coordinator**: Provides safety scores, receives sub safety performance
- **Schedule Agent**: Safety training scheduled in look-aheads
- **PM Orchestrator**: Escalates incidents immediately

---

## Agent 6: Document Management Agent

### Metadata
```yaml
name: document_agent
type: worker
description: Manages RFIs, submittals, punch lists, and inspection documentation.
tools:
  - Read
  - Write
  - Edit
  - Bash
  - Grep
  - Glob
model: sonnet
background: true
wake_enabled: true
```

### Role
The Document Management Agent handles all construction documentation workflows including RFIs (Requests for Information), submittals, punch lists, and inspection scheduling/tracking. It ensures timely document processing and maintains organized records.

### Core Responsibilities

1. **RFI Management**
   - Draft RFIs from superintendent questions
   - Track RFI status and response deadlines
   - Log responses and distribute to affected parties
   - Analyze RFI patterns and trends
   - Connect RFIs to schedule impacts and change orders

2. **Submittal Management**
   - Track submittal schedule and deadlines
   - Monitor review status (pending, approved, revise-resubmit)
   - Alert on long lead-time items
   - Document approval stamps and conditions
   - Coordinate resubmittal requirements

3. **Punch List Management**
   - Create and organize punch lists by area/unit
   - Track punch list item status
   - Coordinate with trades for completion
   - Document completion and re-inspection
   - Generate punch list reports for closeout

4. **Inspection Coordination**
   - Schedule municipal inspections (building, fire, etc.)
   - Track inspection results (pass, fail, corrections)
   - Coordinate re-inspections
   - Maintain inspection log
   - Alert on upcoming required inspections

5. **Document Organization**
   - Maintain project document registers
   - Organize photos by date/location
   - Archive completed documentation
   - Support project closeout documentation

### Data Sources & Inputs
| Source | Data Type | Usage |
|--------|-----------|-------|
| Superintendent questions | RFI content | RFI generation |
| Subcontractor submittals | Product data, shop drawings | Submittal log |
| Walkthroughs | Deficiency observations | Punch lists |
| Inspector results | Pass/fail, corrections | Inspection log |
| Project schedule | Required inspection dates | Scheduling |

### Outputs & Deliverables
| Output | Audience | Frequency |
|--------|----------|-----------|
| RFI Log/Status Report | PM, Design team | Weekly |
| Submittal Log/Status | PM, Design team | Weekly |
| Punch List Report | PM, Subs | As needed |
| Inspection Schedule | Field, Subs | Weekly |
| Document Register | Project team | Ongoing |
| Closeout Package | Owner | At completion |

### Automation Opportunities
- **RFI response deadline alerts** at 5, 10, 15 day marks
- **Submittal schedule tracking** with lead time warnings
- **Punch list photo linking** with automatic organization
- **Inspection scheduling** based on work completion
- **Approval status notifications** to affected trades
- **Document log auto-updates** from activity

### RFI Template

```markdown
## Request for Information (RFI)
**Project:** [Name]
**RFI #:** [Sequential]
**Date Submitted:** [Date]
**Response Required By:** [Date - typically 10-15 business days]

### From
- **Company:** [GC Name]
- **Contact:** [Superintendent/PM Name]
- **Phone:** [Number]
- **Email:** [Email]

### To
- **Company:** [Architect/Engineer]
- **Contact:** [Name]

### Subject
[Brief description - one line]

### Specification/Drawing Reference
- Spec Section: [Section Number and Title]
- Drawing Sheet: [Sheet Number]
- Detail/Grid: [Detail or Grid Reference]

### Question
[Clear, specific question. One question per RFI.]

### Background/Context
[Why this question arose - what condition exists in field, what clarification is needed]

### Suggested Solution
[Contractor's recommended resolution - shows expertise and expedites response]

### Attachments
- [ ] Photos
- [ ] Marked-up drawings
- [ ] Product data
- [ ] Other: [Description]

### Impact If Not Resolved
- **Schedule Impact:** [Activity affected, delay potential]
- **Cost Impact:** [Potential cost if delayed or if change required]

---
### FOR DESIGN TEAM USE ONLY

**Response Date:** _______
**Responded By:** _______

**Response:**
[Design team response]

**Attachments:** [List]

**Distribution:**
- [ ] Owner
- [ ] GC
- [ ] Affected Subcontractor: ________
```

### Submittal Log Template

```markdown
## Submittal Log
**Project:** [Name]
**Updated:** [Date]

| Sub # | Spec Section | Description | Subcontractor | Received | Sent to Design | Status | Returned | Lead Time Notes |
|-------|--------------|-------------|---------------|----------|----------------|--------|----------|-----------------|
| 01-001 | 07 2100 | Thermal Insulation | [Sub] | [Date] | [Date] | Approved | [Date] | |
| 02-001 | 09 2900 | Gypsum Board | [Sub] | [Date] | [Date] | Revise/Resubmit | [Date] | Rev 2 needed |
| 03-001 | 08 7100 | Door Hardware | [Sub] | [Date] | [Date] | Under Review | Pending | 12-week lead time |

### Status Legend
- **Not Submitted** - Awaiting from subcontractor
- **Under Review** - With design team
- **Approved** - Approved as submitted
- **Approved as Noted** - Approved with comments
- **Revise/Resubmit** - Rejected, resubmittal required
- **For Record Only** - Filed, no action required

### Long Lead Items Alert
| Item | Required On-Site | Order By | Status |
|------|------------------|----------|--------|
| [Item] | [Date] | [Date] | [Status] |
```

### Punch List Template

```markdown
## Punch List
**Project:** [Name]
**Building/Area:** [Location]
**Walkthrough Date:** [Date]
**Participants:** [Names]

### Summary
- **Total Items:** [#]
- **Completed:** [#]
- **Remaining:** [#]

### Items

| # | Location | Description | Trade | Priority | Status | Completed Date |
|---|----------|-------------|-------|----------|--------|----------------|
| 1 | Unit 101 - Kitchen | Cabinet door alignment | Cabinets | 2 | Open | |
| 2 | Unit 101 - Bath | Caulk at tub surround | Plumber | 3 | Open | |
| 3 | Unit 101 - LR | Touch-up paint at corner | Painter | 3 | Complete | [Date] |

### Priority Legend
- **1** - Safety/Code issue - immediate
- **2** - Functional issue - before occupancy
- **3** - Cosmetic - before final

### By Trade Summary
| Trade | Total | Completed | Remaining |
|-------|-------|-----------|-----------|
| [Trade] | [#] | [#] | [#] |

### Photo Documentation
- [Photo references by item number]
```

### Inspection Log Template

```markdown
## Inspection Log
**Project:** [Name]
**Permit #:** [Number]

| Date | Inspection Type | Inspector | Building/Area | Result | Re-Inspect Date | Notes |
|------|-----------------|-----------|---------------|--------|-----------------|-------|
| [Date] | Foundation | [Name] | Bldg A | Pass | N/A | |
| [Date] | Rough Plumbing | [Name] | Bldg A, Units 101-110 | Corrections | [Date] | See correction list |
| [Date] | Rough Electrical | [Name] | Bldg A, Units 101-110 | Pass | N/A | |
| [Date] | Framing | [Name] | Bldg A | Pass | N/A | |
| [Date] | Insulation | [Name] | Bldg A | Pass | N/A | |

### Upcoming Inspections Needed
| Inspection Type | Building/Area | Ready Date | Scheduled | Prerequisites |
|-----------------|---------------|------------|-----------|---------------|
| [Type] | [Location] | [Date] | [Date/TBD] | [What must be complete first] |

### Correction Items (Open)
| Inspection | Correction Required | Trade | Status |
|------------|---------------------|-------|--------|
| [Insp Date/Type] | [Correction] | [Trade] | [Status] |
```

### Integration with Other Agents
- **Schedule Agent**: RFI/submittal delays feed schedule impacts
- **Sub Coordinator**: Punch items assigned to trades
- **Finance Agent**: RFIs/submittals may trigger change orders
- **PM Orchestrator**: Document status included in owner reports

---

## Agent 7: Finance & Budget Agent

### Metadata
```yaml
name: finance_agent
type: worker
description: Manages budget tracking, change orders, cost forecasting, and payment application processing.
tools:
  - Read
  - Write
  - Edit
  - Bash
  - Grep
  - Glob
model: sonnet
background: true
wake_enabled: true
```

### Role
The Finance & Budget Agent maintains financial intelligence for the project. It tracks budgets, processes change orders, forecasts costs, and supports payment application review. This is the financial brain of the project.

### Core Responsibilities

1. **Budget Tracking**
   - Maintain budget vs. actual tracking by cost code
   - Track committed costs (subcontracts, POs)
   - Monitor cost performance index (CPI)
   - Alert on budget variances

2. **Change Order Management**
   - Draft change order requests (CORs) with supporting documentation
   - Track change order status (pending, approved, rejected)
   - Maintain change order log
   - Calculate cumulative contract adjustments
   - Connect change orders to RFIs, delays, scope changes

3. **Cost Forecasting**
   - Project final cost based on trends
   - Identify cost-to-complete by category
   - Flag potential budget overruns early
   - Support monthly cost reports

4. **Payment Application Review**
   - Review subcontractor pay applications
   - Verify progress against billing
   - Track retention
   - Support owner billing preparation

5. **Potential Change Item (PCI) Tracking**
   - Log issues that may become change orders
   - Track cost exposure
   - Document time and material work
   - Maintain contemporaneous records

### Data Sources & Inputs
| Source | Data Type | Usage |
|--------|-----------|-------|
| Budget/estimate | Original budget | Baseline |
| Subcontractor invoices | Payment requests | Cost tracking |
| Change orders | Approved changes | Budget adjustment |
| Daily logs | T&M documentation | Change support |
| RFIs/submittals | Scope changes | Change triggers |
| Schedule | Delay claims | Time-related costs |

### Outputs & Deliverables
| Output | Audience | Frequency |
|--------|----------|-----------|
| Budget Status Report | PM, Ownership | Monthly |
| Change Order Log | PM, Owner | Ongoing |
| Cost Forecast | PM, Accounting | Monthly |
| Payment Application Summary | PM, Accounting | Monthly |
| Potential Change Log | PM | Ongoing |
| Variance Analysis | PM, Ownership | Monthly |

### Automation Opportunities
- **Variance alerts** when cost codes exceed thresholds
- **Forecast calculations** based on burn rate
- **Change order log updates** from approved items
- **Payment application math verification**
- **Retention tracking** and release scheduling
- **Cost report generation** from project data

### Budget Tracking Template

```markdown
## Project Budget Status
**Project:** [Name]
**Original Contract:** $[Amount]
**Approved Changes:** $[Amount]
**Current Contract:** $[Amount]
**Report Date:** [Date]

### Executive Summary
- **Budget Status:** [On Budget / Over Budget / Under Budget]
- **Cost Performance Index (CPI):** [X.XX] ([>1.0 = under budget])
- **Forecasted Final Cost:** $[Amount]
- **Forecasted Variance:** $[Amount] [Over/Under]

### Cost Summary by Category

| Category | Original Budget | Approved Changes | Current Budget | Committed | Actual to Date | Forecast Final | Variance |
|----------|-----------------|------------------|----------------|-----------|----------------|----------------|----------|
| General Conditions | $X | $X | $X | $X | $X | $X | $X |
| Site Work | $X | $X | $X | $X | $X | $X | $X |
| Concrete | $X | $X | $X | $X | $X | $X | $X |
| Masonry | $X | $X | $X | $X | $X | $X | $X |
| Structural Steel | $X | $X | $X | $X | $X | $X | $X |
| Wood Framing | $X | $X | $X | $X | $X | $X | $X |
| Roofing | $X | $X | $X | $X | $X | $X | $X |
| Windows/Doors | $X | $X | $X | $X | $X | $X | $X |
| Drywall/Finishes | $X | $X | $X | $X | $X | $X | $X |
| Plumbing | $X | $X | $X | $X | $X | $X | $X |
| HVAC | $X | $X | $X | $X | $X | $X | $X |
| Electrical | $X | $X | $X | $X | $X | $X | $X |
| Contingency | $X | $X | $X | $X | $X | $X | $X |
| **TOTAL** | **$X** | **$X** | **$X** | **$X** | **$X** | **$X** | **$X** |

### Variance Analysis (Items >5% or >$10,000)
| Category | Variance | Reason | Mitigation |
|----------|----------|--------|------------|
| [Category] | $[X] | [Explanation] | [Action plan] |

### Cash Flow
| Month | Projected Spend | Actual Spend | Cumulative Projected | Cumulative Actual |
|-------|-----------------|--------------|---------------------|------------------|
| [Month] | $X | $X | $X | $X |
```

### Change Order Log Template

```markdown
## Change Order Log
**Project:** [Name]
**Original Contract:** $[Amount]
**Updated:** [Date]

### Summary
- **Total CORs Submitted:** [#]
- **Approved:** [#] = $[Amount]
- **Pending:** [#] = $[Amount] (exposure)
- **Rejected:** [#]
- **Current Contract Value:** $[Original + Approved]

### Change Order Detail

| COR # | Date | Description | RFI Ref | Schedule Impact | Cost | Status | Approval Date |
|-------|------|-------------|---------|-----------------|------|--------|---------------|
| COR-001 | [Date] | [Description] | RFI-### | [Days] | $[X] | Approved | [Date] |
| COR-002 | [Date] | [Description] | N/A | [Days] | $[X] | Pending | - |
| COR-003 | [Date] | [Description] | RFI-### | [Days] | $[X] | Rejected | [Date] |

### By Category
| Category | Count | Approved | Pending |
|----------|-------|----------|---------|
| Owner-Directed | [#] | $[X] | $[X] |
| Design Error | [#] | $[X] | $[X] |
| Unforeseen Condition | [#] | $[X] | $[X] |
| Regulatory | [#] | $[X] | $[X] |
| Other | [#] | $[X] | $[X] |

### Pending COR Detail
[Detail on each pending item with supporting documentation references]
```

### Change Order Request Template

```markdown
## Change Order Request (COR)
**Project:** [Name]
**COR #:** [Sequential]
**Date:** [Date]
**Response Required By:** [Date]

### Change Description
[Clear description of the changed work]

### Reason for Change
- [ ] Owner-Directed Change
- [ ] Design Error or Omission
- [ ] Unforeseen Site Condition
- [ ] Regulatory/Code Requirement
- [ ] Value Engineering (credit)
- [ ] Other: [Explain]

### Reference Documents
- RFI #: [If applicable]
- Drawing/Spec Reference: [Reference]
- Owner Direction: [Date/Method of direction]
- Other: [Documentation]

### Cost Breakdown
| Item | Description | Quantity | Unit | Unit Price | Extended |
|------|-------------|----------|------|------------|----------|
| Labor | [Description] | [#] | [hrs/days] | $[X] | $[X] |
| Material | [Description] | [#] | [unit] | $[X] | $[X] |
| Equipment | [Description] | [#] | [days] | $[X] | $[X] |
| Subcontractor | [Name] | [LS] | [LS] | $[X] | $[X] |
| **Subtotal** | | | | | **$[X]** |
| Overhead & Profit ([X]%) | | | | | $[X] |
| **TOTAL** | | | | | **$[X]** |

### Schedule Impact
- **Days Added/Reduced:** [#] days
- **Critical Path Impact:** [Yes/No]
- **Affected Activities:** [List]
- **New Completion Date:** [If applicable]

### Supporting Documentation Attached
- [ ] Subcontractor quotes
- [ ] Material pricing
- [ ] Labor records
- [ ] Photos
- [ ] RFI response
- [ ] Other: [Description]

### Approval

**Contractor Authorization:**
Signature: _________________ Date: _______
Name/Title: _______________________________

**Owner Approval:**
Signature: _________________ Date: _______
Name/Title: _______________________________
```

### Integration with Other Agents
- **Schedule Agent**: Time impacts for change orders
- **Document Agent**: RFIs trigger change orders
- **Daily Ops Agent**: T&M records support change documentation
- **Sub Coordinator**: Subcontractor costs feed budget
- **PM Orchestrator**: Budget status for owner reporting

---

## Swarm Configuration

```yaml
# Construction PM Swarm Configuration
# swarms/construction_pm/swarm.yaml

name: "Construction PM"
description: "Construction project management swarm for site superintendents managing low-rise apartment construction"
version: "1.0.0"
status: "active"

agents:
  - name: "pm_orchestrator"
    type: "orchestrator"
    prompt_file: "agents/pm_orchestrator.md"
    max_turns: 30
    background: false
  - name: "daily_ops_agent"
    type: "worker"
    prompt_file: "agents/daily_ops_agent.md"
    max_turns: 25
    background: true
  - name: "schedule_agent"
    type: "worker"
    prompt_file: "agents/schedule_agent.md"
    max_turns: 25
    background: true
  - name: "sub_coordinator_agent"
    type: "worker"
    prompt_file: "agents/sub_coordinator_agent.md"
    max_turns: 25
    background: true
  - name: "safety_compliance_agent"
    type: "worker"
    prompt_file: "agents/safety_compliance_agent.md"
    max_turns: 25
    background: true
  - name: "document_agent"
    type: "worker"
    prompt_file: "agents/document_agent.md"
    max_turns: 25
    background: true
  - name: "finance_agent"
    type: "worker"
    prompt_file: "agents/finance_agent.md"
    max_turns: 25
    background: true

workspace: "./workspace"

settings:
  max_concurrent_tasks: 6
  require_consensus: false
  timeout: 300
  parallel_execution: true
  wake_messaging: true

priorities:
  - "Daily log completion and progress tracking"
  - "Schedule maintenance and look-ahead generation"
  - "RFI and submittal tracking"
  - "Safety compliance and documentation"
  - "Budget tracking and change order management"
  - "Subcontractor coordination and performance"
```

---

## Integration Architecture

### Data Flow Diagram

```
                            ┌─────────────────────┐
                            │   Superintendent    │
                            │   (Human Input)     │
                            └──────────┬──────────┘
                                       │
                                       ▼
                            ┌─────────────────────┐
                            │   PM Orchestrator   │
                            │   (Task Router)     │
                            └──────────┬──────────┘
                                       │
           ┌───────────┬───────────┬───┴───┬───────────┬───────────┐
           │           │           │       │           │           │
           ▼           ▼           ▼       ▼           ▼           ▼
    ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
    │  Daily   │ │ Schedule │ │   Sub    │ │  Safety  │ │ Document │ │ Finance  │
    │   Ops    │ │  Agent   │ │  Coord   │ │  Agent   │ │  Agent   │ │  Agent   │
    └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘
         │            │            │            │            │            │
         └────────────┴────────────┴─────┬──────┴────────────┴────────────┘
                                         │
                                         ▼
                              ┌─────────────────────┐
                              │   Shared Project    │
                              │   Data Store        │
                              │   (workspace/)      │
                              └─────────────────────┘
```

### Cross-Agent Data Dependencies

| Agent | Produces | Consumed By |
|-------|----------|-------------|
| Daily Ops | Progress updates, manpower, weather | Schedule, Sub Coord, Safety |
| Schedule | Look-aheads, critical path, delays | All agents |
| Sub Coordinator | Performance scores, trade matrix | Safety, Finance |
| Safety | Incidents, compliance status | Sub Coord, Daily Ops |
| Document | RFI/submittal status, punch lists | Schedule, Finance |
| Finance | Budget status, change orders | PM Orchestrator |

### External Integrations

| External System | Integration Type | Data Exchanged |
|-----------------|------------------|----------------|
| Weather API | REST API | Current conditions, forecast |
| Scheduling Software (P6, MS Project) | File import/export | Schedule activities |
| Accounting System | API or file | Cost data, invoices |
| Document Management | File system | PDFs, photos |
| Email | IMAP/SMTP | Notifications, RFI distribution |

---

## Use Case Examples

### Use Case 1: Morning Site Briefing

**Superintendent Request:** "Give me this morning's briefing"

**PM Orchestrator Actions:**
1. Spawns Daily Ops Agent for yesterday's summary
2. Spawns Schedule Agent for today's look-ahead
3. Spawns Safety Agent for today's safety focus
4. Spawns Document Agent for pending items needing attention

**Output:**
```
## Morning Briefing - [Date]

### Yesterday's Progress
- Framing: Units 201-205 walls complete
- Plumbing: Rough-in units 101-105 complete
- Electrical: Rough-in units 101-103 complete
- Weather: Clear, high 75F, no impact

### Today's Work Plan
**Critical Path Items:**
- Framing roof trusses Building A (8am delivery)

**Trade Schedule:**
- Framing: Truss set (8 workers)
- Roofing: Mobilizing, staging materials
- Plumbing: Continue rough-in 106-110
- Electrical: Continue rough-in 104-110

**Inspections:**
- 2pm: Rough plumbing inspection, Units 101-105

### Safety Focus
- Truss setting operations - crane safety zone
- Toolbox talk: Fall protection during roof work

### Action Items Requiring Attention
- RFI-023: Response overdue 5 days (window specs)
- Submittal-031: Cabinet shop drawings pending 12 days
- Electrical sub short 2 workers yesterday
```

### Use Case 2: Delay Impact Analysis

**Superintendent Request:** "The drywall sub says they can't start until next Thursday. What's the impact?"

**PM Orchestrator Actions:**
1. Spawns Schedule Agent for critical path analysis
2. Spawns Sub Coordinator for drywall sub status
3. Spawns Finance Agent for cost impact

**Output:**
```
## Delay Impact Analysis: Drywall Start Delay

### Schedule Impact
- **Original Start:** Monday 1/15
- **Proposed Start:** Thursday 1/18
- **Delay:** 3 working days
- **Critical Path Impact:** YES - drywall is on critical path
- **Project Completion Impact:** 3 days (from 4/15 to 4/18)
- **Float Consumed:** All remaining float in finish sequence

### Downstream Activities Affected
| Activity | Original Start | New Start | Trade |
|----------|----------------|-----------|-------|
| Drywall | 1/15 | 1/18 | ABC Drywall |
| Paint Prime | 1/29 | 2/1 | XYZ Paint |
| Flooring | 2/5 | 2/8 | Floors Inc |
| Cabinets | 2/8 | 2/11 | Cabinet Co |

### Subcontractor Status
- ABC Drywall current performance score: 72/100
- This is their 3rd schedule commitment miss this project
- They cite material delivery delay (drywall shortage)

### Cost Impact
- GC extended general conditions: 3 days x $2,500 = $7,500
- Potential liquidated damages: 3 days x $1,000 = $3,000
- Total exposure: $10,500

### Mitigation Options
1. **Compress drywall duration** - Add crew, work Saturday
   - Cost: ~$5,000 premium
   - Recovers: 2 days
2. **Accept delay** - Push completion to 4/18
   - Cost: $10,500 exposure
3. **Partial release** - Start paint in units as drywall completes
   - Risk: Quality issues with multiple mobilizations

### Recommendation
Request ABC Drywall provide written recovery plan with Saturday work to recover 2 of 3 days. Document delay cause for potential backcharge or liquidated damages offset.
```

### Use Case 3: Weekly Owner Report Generation

**Superintendent Request:** "Generate this week's owner report"

**PM Orchestrator Actions:**
1. Spawns Daily Ops Agent for weekly progress summary
2. Spawns Schedule Agent for schedule status
3. Spawns Document Agent for RFI/submittal status
4. Spawns Finance Agent for budget status
5. Spawns Safety Agent for safety metrics

**Output:**
```
## Weekly Owner Progress Report
**Project:** Parkview Apartments
**Week Ending:** [Date]
**Report #:** 15

### Executive Summary
Project remains on schedule with substantial completion forecasted for April 15. Budget is tracking 2.3% under forecast. No safety incidents this week.

### Schedule Status
- **Overall:** On Schedule
- **Percent Complete:** 42%
- **Forecasted Completion:** April 15 (no change)

**This Week's Accomplishments:**
- Building A framing 100% complete
- Roofing Building A 60% complete
- MEP rough-in Units 101-110 complete
- Site utilities 100% complete

**Next Week's Plan:**
- Complete roofing Building A
- Begin framing Building B
- Continue MEP rough-in Building A
- Drywall mobilization

### Budget Status
- **Original Contract:** $12,500,000
- **Approved Changes:** +$145,000
- **Current Contract:** $12,645,000
- **Cost to Date:** $5,280,000
- **Forecasted Final:** $12,350,000
- **Projected Savings:** $295,000 (2.3%)

**Pending Change Orders:** 2 items totaling $45,000

### Open Items Requiring Owner Attention
1. **COR-007:** Upgraded lobby flooring per owner request - $28,000 - AWAITING APPROVAL
2. **RFI-025:** Confirm exterior paint colors - 5 days overdue

### Safety
- **Recordable Incidents:** 0
- **Near Misses:** 1 (documented, corrective action complete)
- **Man-Hours This Week:** 2,400
- **Project TRIR:** 0.0

### Photos
[Attached: 10 progress photos with captions]

### Issues/Risks
| Issue | Impact | Mitigation | Status |
|-------|--------|------------|--------|
| Drywall material shortage | Potential 3-day delay | Early order placed | Monitoring |

### Next Report: [Date]
```

---

## Life OS Integration

### How This Fits Into Life OS Swarm Architecture

The Construction PM Swarm integrates with the broader Life OS ecosystem as a **domain-specific swarm** that can be activated when the user is working on construction projects.

```
Life OS (Supreme Orchestrator)
├── Operations Swarm
├── Personal Productivity Swarm
├── Finance Swarm (personal)
├── Construction PM Swarm  <-- This swarm
│   └── [Project-specific instances]
└── [Other domain swarms]
```

### Context Switching

When the user context switches to construction work:
1. Life OS activates the Construction PM Swarm
2. Project-specific data loads from workspace
3. PM Orchestrator becomes the primary interface
4. Other Life OS swarms remain available for cross-domain queries

### Cross-Swarm Interactions

| Life OS Function | Construction PM Interaction |
|------------------|----------------------------|
| Calendar | Inspection schedules, meetings sync to personal calendar |
| Tasks | Project action items can surface in personal task list |
| Finance (Personal) | Project pay periods align with personal cash flow |
| Communication | Email notifications routed appropriately |

### Data Persistence

Each project maintains its own workspace:
```
swarms/construction_pm/
├── workspace/
│   ├── STATE.md
│   └── projects/
│       ├── parkview_apartments/
│       │   ├── daily_logs/
│       │   ├── schedules/
│       │   ├── submittals/
│       │   ├── rfis/
│       │   ├── safety/
│       │   ├── financials/
│       │   └── photos/
│       └── [other_projects]/
```

---

## Implementation Roadmap

### Phase 1: Core Foundation (Weeks 1-2)
1. Create swarm directory structure
2. Implement PM Orchestrator
3. Implement Daily Ops Agent with basic templates
4. Create workspace structure

### Phase 2: Scheduling & Documents (Weeks 3-4)
1. Implement Schedule Agent
2. Implement Document Agent (RFIs, Submittals)
3. Create look-ahead templates
4. Build document tracking system

### Phase 3: Subcontractors & Safety (Weeks 5-6)
1. Implement Sub Coordinator Agent
2. Implement Safety Agent
3. Create performance tracking system
4. Build safety documentation templates

### Phase 4: Finance & Integration (Weeks 7-8)
1. Implement Finance Agent
2. Build cross-agent data flows
3. Create owner reporting system
4. Integrate with external systems (weather API)

### Phase 5: Refinement & Testing (Weeks 9-10)
1. User testing with real project data
2. Template refinement
3. Automation tuning
4. Documentation completion

---

## Sources

### Construction Software Research
- [Procore Daily Log](https://www.procore.com/quality-safety/daily-log)
- [Raken Daily Reports](https://www.rakenapp.com/features/daily-reports)
- [Best Construction PM Software 2026](https://www.permitflow.com/blog/construction-project-management-software)
- [Buildertrend Daily Logs](https://buildertrend.com/project-management/daily-logs/)
- [Procore Construction Daily Reports Guide](https://www.procore.com/library/construction-daily-reports)

### OSHA Safety Compliance
- [OSHA Construction Compliance Quick Start](https://www.osha.gov/complianceassistance/quickstarts/construction)
- [OSHA Construction Compliance](https://www.osha.gov/construction/compliance)
- [OSHA 2025 Safety Rules](https://www.forconstructionpros.com/business/construction-safety/article/22954763/heavy-construction-systems-specialists-inc-hcss-oshas-2025-safety-rules-redefine-compliance-and-recordkeeping-in-construction)
- [29 CFR Part 1926](https://www.ecfr.gov/current/title-29/subtitle-B/chapter-XVII/part-1926)
- [Site Safety & Health Officer Guide](https://www.360training.com/blog/become-an-ssho)

### RFI & Document Management
- [RFI Construction Guide](https://www.mastt.com/guide/rfi-construction)
- [Ultimate Guide to Construction RFIs](https://www.linarc.com/buildspace/the-ultimate-guide-to-construction-rfis-everything-you-need-to-know)
- [Procore RFI Guide](https://www.procore.com/library/rfi-construction)
- [Autodesk Construction RFI Tracking](https://construction.autodesk.com/tools/construction-rfi-tracking/)
- [ESUB RFI Best Practices](https://esub.com/blog/tips-managing-rfi-document-process-field)

### Scheduling
- [Buildertrend Scheduling](https://buildertrend.com/project-management/schedule/)
- [Best Construction Scheduling Software 2026](https://www.projectmanager.com/blog/best-construction-scheduling-software)
- [Outbuild Scheduling](https://www.outbuild.com/construction-scheduling-software)
- [InEight Schedule](https://ineight.com/products/ineight-schedule/)
- [Procore Schedule](https://www.procore.com/project-management/schedule)

### Subcontractor Management
- [Subcontractor Management Guide 2025](https://www.fieldservicely.com/subcontractor-management)
- [Deltek Subcontractor Management](https://www.deltek.com/en/construction/subcontractor-management)
- [Workyard Subcontractor Best Practices](https://www.workyard.com/construction-management/subcontractor-management)
- [Procore Subcontractor Management Tips](https://www.procore.com/library/subcontractor-management)
- [Outbuild Managing Subcontractors](https://www.outbuild.com/blog/how-to-manage-subcontractors)

### Budget & Change Orders
- [Autodesk Cost Management](https://construction.autodesk.com/workflows/construction-cost-management/)
- [4castplus Budgeting](https://4castplus.com/construction-budgeting-software/)
- [Best Cost Management Software 2026](https://www.smartsheet.com/content/best-construction-cost-management-software)
- [Bauwise Cost Management](https://www.bauwise.com/)
- [Buildertrend Change Orders](https://buildertrend.com/project-management/construction-change-order-software/)

### Owner Communication & Meetings
- [Autodesk Construction Meetings Guide](https://www.autodesk.com/blogs/construction/construction-meetings/)
- [Sitemate Meeting Minutes](https://sitemate.com/resources/articles/commercial/construction-meeting-minutes/)
- [OAC Meeting Guide](https://www.cogram.com/blog/oac-meeting)
- [Deltek Meeting Minutes Best Practices](https://www.deltek.com/en/blog/construction-meeting-minutes)
- [Procore Meeting Minutes Template](https://www.procore.com/library/construction-meeting-minutes-template)

### Punch Lists & Inspections
- [Construction Punch List Guide 2026](https://monday.com/blog/project-management/construction-punch-list/)
- [Building Inspection Guide](https://bluerithm.com/guide-to-building-inspections/)
- [Building Inspection Checklist](https://www.projectmanager.com/blog/building-inspection)
- [Construction Site Inspection Checklist](https://buildern.com/resources/blog/construction-site-inspection-checklist/)

---

*Document generated by Research Specialist Agent*
*Date: 2026-01-06*
