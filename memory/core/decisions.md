# Organizational Decision Log

## 2025-01-02 - Memory System Architecture
**Context**: Agents lacked persistent context across sessions, limiting their effectiveness
**Decision**: Implement hierarchical file-based memory system
**Rationale**:
- Markdown files are human-readable and editable
- Hierarchical structure mirrors organization structure
- Can be version controlled with git
- Simpler than database, sufficient for current needs
**Impact**: All agents now load relevant context on spawn
**Owner**: CEO + Claude Code

---

## 2025-01-02 - Git Workflow for Agents
**Context**: Agents needed ability to make permanent changes
**Decision**: Agents commit to `swarm/*` feature branches, CEO merges to main
**Rationale**:
- Maintains human oversight of all changes
- Follows standard git workflow
- Easy to review and rollback
- Protects main branch
**Impact**: Agents can now make autonomous changes with safety guardrails
**Owner**: CEO

---

## 2025-01-02 - Communication Style Guidelines
**Context**: COO responses were too verbose and repetitive
**Decision**: Add explicit anti-duplication rules and decision highlighting format
**Rationale**:
- CEO time is limited; responses should be scannable
- Decisions need to stand out visually
**Impact**: `⚡ **CEO DECISION REQUIRED**` format for decisions
**Owner**: CEO

---

## 2025-01-01 - Priority Order Established
**Context**: Multiple swarms exist; need clear prioritization
**Decision**: Swarm Dev > ASA Research > MYND App
**Rationale**:
- System must be self-developing before it can effectively work on projects
- ASA research is higher priority than MYND (foundational)
- MYND paused until ASA progress
**Impact**: All resources focus on Swarm Dev until operational
**Owner**: CEO
