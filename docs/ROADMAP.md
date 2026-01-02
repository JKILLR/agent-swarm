# Agent Swarm Development Roadmap - Phase 1

## Vision
The Supreme Orchestrator is the COO of my life. Direct agents under it are executive assistants and VPs. Each swarm is a department. Software swarms (ASA, MYND) are first priority. Future swarms will include construction management (my real job), personal finances (mortgages, taxes, budgeting), and more.

## Current Priority Order

1. **Swarm Dev** - PRIMARY FOCUS until system is self-developing
2. **ASA Research** - Secondary, activate when Swarm Dev is operational
3. **MYND App** - Paused, activate after ASA progress
4. **Operations** - Active, manages all swarms

## Immediate Goal: Self-Developing System

Before focusing on ASA or other projects, the agent-swarm system must be able to:
- Execute code changes autonomously
- Push/pull from GitHub
- Run its own tests
- Modify its own codebase

**The system should be as capable as Claude Code in the terminal.**

## Phase 0: Execution Layer (CURRENT PRIORITY)

### 0.1 Claude Agent SDK Integration
Wire up actual agent execution so agents can:
- Read/write files in their workspace
- Execute bash commands
- Use all tools listed in their prompts
- Stream responses back to the UI

### 0.2 Git Integration for Agents
Enable agents to push/pull from GitHub:
- Configure git credentials for agent subprocesses
- Add git tools to Swarm Dev agents
- Test: Agent can commit and push a change

### 0.3 Self-Modification Capability
Swarm Dev must be able to modify agent-swarm itself:
- Implementer can edit files in the codebase
- Reviewer can run tests
- Critic can check for security issues
- Orchestrator can coordinate multi-file changes

## Phase 1: Core Functionality

### 1.1 Fix and Validate
- Fix query() keyword args bug (if still present)
- Test: python main.py chat
- Verify parallel agent spawning works
- Verify wake messaging works

### 1.2 Enhance Supreme Orchestrator
The supreme orchestrator needs to act like a COO with direct reports:

Create supreme/agents/ with these direct agents:
- chief_of_staff.md - Manages priorities across all swarms, daily briefings
- project_manager.md - Tracks status of all projects, dependencies, blockers
- context_keeper.md - Maintains cross-swarm knowledge, spots connections

Update supreme/agents/supreme.md to delegate to these before routing to swarms.

### 1.3 Swarm Dev (PRIMARY FOCUS)
Swarm Dev is the team that builds the agent-swarm system itself.

swarms/swarm_dev/swarm.yaml priorities:
- Claude Agent SDK execution layer
- Git integration for autonomous commits
- Self-modification capability
- Web UI enhancements

swarms/swarm_dev/agents/implementer.md should:
- Have full file system access to agent-swarm/
- Be able to run git commands
- Execute python tests
- Modify any file in the codebase

swarms/swarm_dev/agents/reviewer.md should:
- Run linters and type checkers
- Execute test suites
- Verify changes don't break existing functionality

### 1.4 ASA Swarm (SECONDARY - after Swarm Dev works)
Update swarms/asa_research/ with ASA-specific context:

swarms/asa_research/swarm.yaml priorities:
- Implement true sparse attention O(n×k)
- Long-context benchmarks (4096+ tokens)
- Scale testing at 100M+ parameters
- Wall-clock measurements

Context:
- H6 validated: 73.9% attention overlap with linguistic structure
- 21% faster convergence than baseline
- Current code: asa_v2_2.py, train_asa.py, h6_correlation.py
- Bottleneck: Still O(n²) compute with masking, need true sparse kernels
- Target: xformers or triton for sparse attention kernels

### 1.5 MYND Swarm (PAUSED)
Keep paused for now. Will activate after ASA sparse attention works.

swarms/mynd_app/swarm.yaml:
- status: paused
- description: Personal AI companion app - cognitive operating system

## Phase 2: Operational Excellence

### 2.1 Consensus Protocol
Implement actual consensus before major decisions:
- shared/consensus.py should track rounds, votes, outcomes
- Log to logs/consensus/ for audit trail
- Critic must participate in every consensus round

### 2.2 Memory and Context
- Swarm workspaces should persist state between sessions
- Supreme should summarize each session to logs/daily/
- Context keeper should read these for continuity

### 2.3 Background Monitors
Each swarm gets a monitor agent that:
- Runs in background during work sessions
- Watches for test failures, build errors
- Wakes main thread only on problems

## Phase 3: Expand Swarms (Later)

### 3.1 Construction Management Swarm
- Project tracking
- Subcontractor coordination
- Budget monitoring
- Timeline management

### 3.2 Personal Finance Swarm
- Mortgage tracking
- Tax preparation
- Budget analysis
- Investment monitoring

### 3.3 Future Swarms
- Health/fitness tracking
- Learning/education goals
- Social/networking management

## File Structure After Phase 1

```
agent-swarm/
├── main.py
├── config.yaml
├── shared/
│   ├── agent_definitions.py
│   ├── swarm_interface.py
│   ├── consensus.py
│   ├── agent_executor.py      # NEW: Claude SDK execution
│   └── agent_base.py
├── supreme/
│   ├── orchestrator.py
│   └── agents/
│       ├── supreme.md           # COO - routes and delegates
│       ├── chief_of_staff.md    # Priorities and briefings
│       ├── project_manager.md   # Status tracking
│       └── context_keeper.md    # Cross-swarm knowledge
├── swarms/
│   ├── _template/
│   ├── swarm_dev/              # ACTIVE - PRIMARY FOCUS
│   │   ├── swarm.yaml
│   │   ├── agents/
│   │   │   ├── orchestrator.md
│   │   │   ├── architect.md
│   │   │   ├── implementer.md   # Can modify codebase
│   │   │   ├── reviewer.md      # Can run tests
│   │   │   ├── critic.md
│   │   │   └── refactorer.md
│   │   └── workspace -> ../..   # Points to project root
│   ├── asa_research/            # SECONDARY - after Swarm Dev works
│   │   ├── swarm.yaml
│   │   ├── agents/
│   │   └── workspace/
│   ├── mynd_app/                # PAUSED
│   │   ├── swarm.yaml
│   │   └── agents/
│   └── operations/              # Cross-swarm management
│       ├── swarm.yaml
│       └── agents/
└── logs/
    ├── conversations/
    ├── consensus/
    └── daily/
```

## Next Actions (In Order)

1. **Wire up Claude Agent SDK execution layer**
   - Create shared/agent_executor.py
   - Enable agents to actually run tools
   - Stream tool results back to UI

2. **Add git credentials for agents**
   - Configure CLAUDE_CODE_OAUTH_TOKEN for agent subprocesses
   - Test agent can commit and push

3. **Test Swarm Dev self-modification**
   - Ask Swarm Dev to make a small change to itself
   - Verify it can edit, test, and commit

4. Fix query() keyword args bug (if still blocking)
5. Test parallel agent spawning
6. Once Swarm Dev works, switch focus to ASA

## Success Criteria

### Phase 0 Complete When:
- [ ] Swarm Dev agents can read/write files
- [ ] Swarm Dev agents can run git commands
- [ ] Swarm Dev agents can execute tests
- [ ] A Swarm Dev agent successfully modifies and commits code

### Phase 1 Complete When:
- [ ] python main.py chat works without errors
- [ ] Parallel agents spawn and wake properly
- [ ] Swarm Dev is self-maintaining
- [ ] ASA swarm can research and propose implementations
- [ ] Consensus rounds are logged
- [ ] Supreme acts like a COO, not just a router
