# Agent Swarm Development Roadmap - Phase 1

## Vision
The Supreme Orchestrator is the COO of my life. Direct agents under it are executive assistants and VPs. Each swarm is a department. Software swarms (ASA, MYND) are first priority. Future swarms will include construction management (my real job), personal finances (mortgages, taxes, budgeting), and more.

## Immediate Fix
There's a bug in the query() calls - they need keyword arguments, not positional.

Find all instances of:
```python
query(prompt, options)
```

Replace with:
```python
query(prompt=prompt, options=options)
```

Check: supreme/orchestrator.py, shared/swarm_interface.py, and any other files using query().

## Current State
- Basic structure built
- 2 swarms discovered (ASA, MYND)
- 7 agents total
- CLI working (list, chat commands)
- Bug: query() positional args error blocking chat

## Phase 1: Core Functionality (This Week)

### 1.1 Fix and Validate
- Fix query() keyword args bug
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

### 1.3 ASA Swarm (Primary Focus)
Update swarms/asa_research/ with ASA-specific context:

swarms/asa_research/swarm.yaml priorities:
- Implement true sparse attention O(n×k)
- Long-context benchmarks (4096+ tokens)
- Scale testing at 100M+ parameters
- Wall-clock measurements

swarms/asa_research/agents/researcher.md should know:
- H6 validated: 73.9% attention overlap with linguistic structure
- 21% faster convergence than baseline
- Current code: asa_v2_2.py, train_asa.py, h6_correlation.py
- Bottleneck: Still O(n²) compute with masking, need true sparse kernels

swarms/asa_research/agents/implementer.md should know:
- Target: xformers or triton for sparse attention kernels
- Must preserve bonding mask functionality
- Test on WikiText-2 first, then scale

swarms/asa_research/agents/critic.md should challenge:
- Does sparse implementation actually save FLOPs?
- Are we measuring wall-clock or just theoretical complexity?
- What's the memory overhead of the bonding mask?

### 1.4 MYND Swarm (Secondary)
Keep paused for now. Will activate after ASA sparse attention works.

swarms/mynd_app/swarm.yaml:
- status: paused
- description: Personal AI companion app - cognitive operating system

## Phase 2: Operational Excellence (Next Week)

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
│   ├── asa_research/            # ACTIVE - primary focus
│   │   ├── swarm.yaml
│   │   ├── agents/
│   │   │   ├── orchestrator.md
│   │   │   ├── researcher.md    # Knows ASA papers, prior art
│   │   │   ├── implementer.md   # Knows ASA codebase
│   │   │   ├── critic.md        # Challenges all proposals
│   │   │   └── monitor.md       # Watches tests
│   │   └── workspace/
│   │       └── (ASA project files, notes, experiments)
│   ├── mynd_app/                # PAUSED - activate later
│   │   ├── swarm.yaml
│   │   └── agents/
│   └── operations/              # Cross-swarm management
│       ├── swarm.yaml
│       └── agents/
│           ├── vp_operations.md
│           ├── project_coordinator.md
│           └── qa_agent.md
└── logs/
    ├── conversations/
    ├── consensus/
    └── daily/
```

## Next Actions (In Order)
1. Fix query() keyword args bug
2. Test: python main.py chat with simple query
3. Test: parallel agent spawning on ASA research task
4. Add ASA-specific context to swarms/asa_research/agents/*.md
5. Create supreme direct agents (chief_of_staff, project_manager, context_keeper)
6. Run first real ASA task: "Research sparse attention kernel implementations"

## Success Criteria
- python main.py chat works without errors
- Parallel agents spawn and wake properly
- ASA swarm can research and propose implementation approaches
- Consensus rounds are logged
- Supreme acts like a COO, not just a router
