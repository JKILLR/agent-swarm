# Skills Integration with Agent-Swarm System

**Date**: 2026-01-05
**Research from**: 3 parallel agents (architect, implementer, critic)

---

## Executive Summary

Claude Code Skills can significantly enhance our agent-swarm orchestration system by:
1. **Providing quick skill-based triggers** for swarm operations via Claude Code
2. **Giving each agent type specialized knowledge** via custom skills
3. **Creating a bridge** between Claude Code's native interface and our REST API agents

---

## Key Integration Opportunities

### 1. Skills as Swarm Triggers

Create Claude Code skills that call our REST API endpoints:

```yaml
---
name: spawn-swarm-agent
description: Spawn a specialized agent from the agent-swarm system. Use when user asks to delegate work, spawn workers, or run parallel agents.
---

# Spawn Swarm Agent

When the user wants to delegate work to a specialized agent:

1. Identify the appropriate swarm and agent type
2. Call the REST API:

```bash
curl -X POST http://localhost:8000/api/agents/execute \
  -H "Content-Type: application/json" \
  -d '{"swarm": "SWARM_NAME", "agent": "AGENT_NAME", "prompt": "TASK"}'
```

Available swarms and agents:
- **swarm_dev**: architect, implementer, critic, reviewer, refactorer
- **operations**: ops_coordinator, qa_agent
- **mynd_app**: orchestrator, worker, critic
```

### 2. Agent-Specific Skills

Each agent type could have a corresponding skill loaded into its context:

| Agent | Skill Purpose |
|-------|--------------|
| **implementer** | Code patterns, file conventions, test requirements |
| **architect** | System design templates, architecture decision records |
| **critic** | Code review checklist, security patterns |
| **reviewer** | PR review guidelines, approval criteria |
| **researcher** | Search strategies, source evaluation |

Example skill for implementer:
```yaml
---
name: swarm-implementer
description: Code implementation patterns for agent-swarm. Use when implementing features in the agent-swarm codebase.
---

# Implementer Guidelines

## File Conventions
- Backend: `backend/` - FastAPI services
- Frontend: `frontend/` - Next.js app
- Agents: `swarms/{swarm_name}/agents/{agent}.md`

## Patterns
- Always update STATE.md after changes
- Use TypeScript strict mode
- Follow existing code style
```

### 3. Skill-to-REST Bridge Pattern

Skills can't make network calls directly, but they CAN instruct Claude to use Bash:

```yaml
---
name: delegate-to-swarm
description: Delegate complex tasks to the agent-swarm REST API
allowed-tools: Bash, Read
---

# Delegation Pattern

For complex tasks, delegate to specialized agents:

## Single Agent
```bash
curl -X POST http://localhost:8000/api/agents/execute \
  -H "Content-Type: application/json" \
  -d '{"swarm": "swarm_dev", "agent": "implementer", "prompt": "..."}'
```

## Parallel Agents
```bash
curl -X POST http://localhost:8000/api/agents/execute-batch \
  -H "Content-Type: application/json" \
  -d '{"agents": [...]}'
```

## Async (Background)
```bash
curl -X POST http://localhost:8000/api/agents/execute-async ...
# Returns execution_id immediately
curl http://localhost:8000/api/agents/execute-async/EXECUTION_ID
```
```

---

## Implementation Architecture

### Option A: Skills Directory per Swarm

```
swarms/
├── swarm_dev/
│   ├── agents/
│   │   ├── implementer.md
│   │   └── architect.md
│   └── skills/                    # NEW
│       ├── implementing/
│       │   └── SKILL.md
│       └── architecting/
│           └── SKILL.md
└── operations/
    └── skills/
        └── coordinating/
            └── SKILL.md
```

### Option B: Centralized Skills with Agent Tags

```
.claude/skills/
├── spawn-agent/
│   └── SKILL.md           # Spawns any swarm agent
├── swarm-status/
│   └── SKILL.md           # Checks swarm/pool status
├── delegate-task/
│   └── SKILL.md           # Intelligent delegation
└── review-work/
    └── SKILL.md           # Triggers critic/reviewer
```

### Option C: Dynamic Skill Generation (Advanced)

Generate skills from agent markdown files:

```python
# backend/services/skill_generator.py
def generate_skill_from_agent(agent_path: str) -> str:
    """Convert agent.md to SKILL.md format"""
    agent_config = parse_agent_md(agent_path)
    return f"""---
name: swarm-{agent_config.name}
description: {agent_config.description}. Use when {agent_config.triggers}.
---

{agent_config.instructions}

## API Access
Call via: curl -X POST http://localhost:8000/api/agents/execute ...
"""
```

---

## Recommended Skills for Agent-Swarm

### Must-Have Skills

| Skill | Purpose |
|-------|---------|
| **spawn-swarm-agent** | Trigger single agent via REST API |
| **batch-agents** | Run multiple agents in parallel |
| **swarm-status** | Check pool status, running agents |
| **delegate-intelligent** | Smart routing based on task type |

### Nice-to-Have Skills

| Skill | Purpose |
|-------|---------|
| **state-management** | Read/update workspace STATE.md |
| **memory-api** | Store/retrieve facts and preferences |
| **coordination-protocols** | When to escalate to operations tier |

---

## Technical Considerations

### 1. Skills vs MCP for Agent-Swarm

| Aspect | Skills Approach | MCP Approach |
|--------|----------------|--------------|
| **Setup** | Simple - just markdown files | Requires MCP server |
| **Token efficiency** | Better - progressive loading | All tools loaded upfront |
| **Flexibility** | Instructions only | Can add new tools |
| **Portability** | Now an open standard | Already standardized |

**Recommendation**: Use Skills for orchestration instructions, keep REST API for actual execution.

### 2. Limitations to Address

1. **Subagents can't access skills** - Our REST API agents won't have skill access
   - Solution: Embed skill knowledge directly in agent prompts

2. **No state persistence** - Skills can't remember between sessions
   - Solution: Use our memory API (`/api/memory/facts`)

3. **15k character limit** - Many skills can exceed budget
   - Solution: Keep skill descriptions minimal, put details in referenced files

4. **Auto-invocation unreliability** - 20-50% failure rate
   - Solution: Use WHEN/WHEN NOT patterns in descriptions

### 3. Integration with Existing Skills System

We already have skills defined in `.claude/commands/`:
- `/spawn-swarm` - Spawns agents
- `/delegate` - Delegates tasks
- `/swarm-status` - Checks status
- `/morning-standup` - Daily standup

These are **slash commands** (user-invoked), not **skills** (auto-invoked).

**Migration Path**:
1. Keep slash commands for explicit invocation
2. Add corresponding skills for auto-invocation
3. Skills can reference slash commands: "For explicit control, user can run /spawn-swarm"

---

## Proposed Implementation Plan

### Phase 1: Core Skills
1. Create `.claude/skills/` directory
2. Add `spawn-agent/SKILL.md` - basic agent spawning
3. Add `swarm-status/SKILL.md` - pool/status checks
4. Test auto-invocation reliability

### Phase 2: Smart Delegation
1. Add `delegate-intelligent/SKILL.md` - routes to appropriate agent
2. Add `batch-work/SKILL.md` - parallel execution patterns
3. Add tier escalation logic (swarm_dev → operations)

### Phase 3: Agent Enhancement
1. Generate skills from existing agent.md files
2. Add domain-specific skills per swarm
3. Create skill bundles for common workflows

---

## Example: Complete Spawn-Agent Skill

```yaml
---
name: spawn-swarm-agent
description: |
  Spawn specialized agents from the agent-swarm orchestration system.
  USE WHEN: user asks to delegate work, run parallel tasks, spawn workers,
  or needs specialized agent capabilities (architect, implementer, critic).
  DO NOT USE WHEN: simple tasks Claude can do directly, or when user
  explicitly wants to do something themselves.
---

# Agent-Swarm Spawning

## Available Agents

### swarm_dev (Development Work)
- **implementer** - Write code, create/modify files
- **architect** - Design solutions, create plans
- **critic** - Review code for bugs/issues
- **reviewer** - Code review and quality checks
- **refactorer** - Clean up and improve code

### operations (Cross-Swarm)
- **ops_coordinator** - Multi-swarm coordination
- **qa_agent** - Quality audits

## Spawning Patterns

### Single Agent
```bash
curl -X POST http://localhost:8000/api/agents/execute \
  -H "Content-Type: application/json" \
  -d '{"swarm": "swarm_dev", "agent": "implementer", "prompt": "Your task here"}'
```

### Parallel Batch
```bash
curl -X POST http://localhost:8000/api/agents/execute-batch \
  -H "Content-Type: application/json" \
  -d '{"agents": [
    {"swarm": "swarm_dev", "agent": "implementer", "prompt": "Task 1"},
    {"swarm": "swarm_dev", "agent": "critic", "prompt": "Task 2"}
  ]}'
```

### Async (Background)
```bash
curl -X POST http://localhost:8000/api/agents/execute-async \
  -H "Content-Type: application/json" \
  -d '{"swarm": "swarm_dev", "agent": "implementer", "prompt": "Long task..."}'
# Returns execution_id - check status with:
curl http://localhost:8000/api/agents/execute-async/EXECUTION_ID
```

## Best Practices

1. Always include clear, detailed prompts
2. Tell agents to read workspace/STATE.md for context
3. Tell agents to update STATE.md when done
4. Use batch execution for independent parallel work
5. Use async for long-running background tasks
```

---

## Sources

- Research files in `workspace/research/`:
  - `claude_skills_overview.md` - How skills work
  - `skills_technical_analysis.md` - Limitations and gotchas
  - `claude-code-skills-research.md` - Comprehensive overview
- Official docs: https://code.claude.com/docs/en/skills
- GitHub: https://github.com/anthropics/skills
