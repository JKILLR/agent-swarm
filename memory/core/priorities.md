# Strategic Priorities

*Last updated: 2025-01-02*

## Priority 1: Self-Developing System (ACTIVE)
**Owner**: Swarm Dev
**Goal**: Agent swarm can autonomously modify, test, and deploy its own code

### Current State
- ✅ Tool execution working (Read, Write, Edit, Bash, Git)
- ✅ Git workflow implemented (commit to feature branches)
- ✅ Live activity feed for visibility
- ✅ Chat persistence across sessions
- 🔄 Memory system being implemented
- ⬜ Testing infrastructure needs completion
- ⬜ Error recovery for failed tasks

### Key Metrics
- Can agents make a code change? **Yes**
- Can agents run tests? **Partially** (Bash works, pytest setup incomplete)
- Can agents commit and push? **Yes** (swarm/* branches)
- Can agents recover from errors? **No** (needs implementation)

### Blockers
- Testing infrastructure created but not committed
- No automatic error recovery/retry

---

## Priority 2: ASA Research (STANDBY)
**Owner**: ASA Research Swarm
**Goal**: Implement true sparse attention with O(n×k) complexity

### Current State
- H6 hypothesis validated (73.9% attention overlap with linguistic structure)
- 21% faster convergence demonstrated
- Bottleneck: Still using O(n²) compute with masking

### Next Steps (when activated)
1. Research xformers/triton sparse attention kernels
2. Implement true sparse forward/backward pass
3. Benchmark at 4096+ token context
4. Scale testing at 100M+ parameters

### Waiting On
- Swarm Dev fully operational
- Self-testing capability confirmed

---

## Priority 3: MYND App (PAUSED)
**Owner**: MYND Swarm
**Goal**: Personal AI companion application

### Status
Paused pending ASA sparse attention implementation. Will leverage ASA for efficient long-context memory.

---

## Cross-Swarm Dependencies
- All swarms depend on Swarm Dev for system capabilities
- ASA depends on Swarm Dev for self-modification to work
- MYND depends on ASA for efficient attention mechanisms
- Operations coordinates all swarms
