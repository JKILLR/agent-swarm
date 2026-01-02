# Cross-Swarm Dependencies

## Dependency Graph
```
Swarm Dev (foundational)
    │
    ├──→ Operations (uses system capabilities)
    │
    ├──→ ASA Research (needs self-modification working)
    │         │
    │         └──→ MYND App (will use ASA for efficient memory)
    │
    └──→ [Future Swarms]
```

## Active Dependencies

### Swarm Dev → All Swarms
- All swarms depend on Swarm Dev for:
  - Tool execution infrastructure
  - Git workflow capabilities
  - Memory/context persistence
  - Error recovery mechanisms
- **Current Status**: Most capabilities working, memory in progress

### ASA Research → MYND App
- MYND will use ASA's sparse attention for:
  - Efficient long-context memory
  - Scalable personal knowledge base
- **Current Status**: Dependency paused (both on hold)

## Coordination Protocols

### Cross-Swarm Task Handoff
1. Source swarm completes their portion
2. Source orchestrator notifies Operations
3. Operations validates deliverable
4. Operations notifies target swarm
5. Target swarm acknowledges and begins

### Shared Resource Conflicts
- Git branches: Each swarm uses `swarm/{swarm_name}/*` namespace
- Workspace files: Each swarm has isolated workspace
- System files: Swarm Dev coordinates, others request changes

### Breaking Changes
When Swarm Dev modifies core infrastructure:
1. Announce change scope to Operations
2. Operations assesses impact on other swarms
3. Create migration plan if needed
4. Execute with monitoring
5. Verify all swarms still functional

## Current Active Coordination Needs
- None active (most swarms paused or focused internally)

## Historical Coordination
- 2025-01-02: Memory system design - affects all swarms, led by Swarm Dev
