# Swarm Dev Context

## Mission
Build and maintain the agent-swarm system itself. Make the system self-developing so it can autonomously improve without human coding intervention.

## Current Focus
**Making the system production-ready for self-development:**
- Complete memory/context persistence system
- Implement error recovery for failed tasks
- Validate testing infrastructure
- Document agent capabilities and limitations

## Key Files
- `backend/main.py` - FastAPI backend, WebSocket chat, Claude CLI integration
- `backend/tools.py` - Tool definitions and execution (Read, Write, Edit, Bash, Git, etc.)
- `frontend/` - Next.js dashboard and chat interface
- `supreme/agents/supreme.md` - COO prompt and behavior
- `shared/agent_definitions.py` - Agent type definitions and tool assignments
- `memory/` - Persistent context system (NEW)

## Architecture
```
User ─→ Frontend ─→ WebSocket ─→ Backend ─→ Claude CLI
                                    │
                                    └─→ Tool Executor ─→ Subagent Spawn
                                                             │
                                                             └─→ Claude API
```

## Team
- **orchestrator** - Coordinates development work, routes to specialists
- **architect** - Designs system changes, reviews architecture decisions
- **implementer** - Writes code, creates files, runs commands
- **reviewer** - Tests changes, validates functionality
- **critic** - Reviews for security, edge cases, improvements
- **refactorer** - Cleans up code, improves maintainability

## Recent Accomplishments
- ✅ Tool execution working for all defined tools
- ✅ Git workflow (GitCommit, GitSync, GitStatus)
- ✅ Live activity feed in UI
- ✅ Chat history persistence
- ✅ Edit tool for targeted file modifications
- ✅ Fixed response duplication bug

## Dependencies
- **Depends on**: None (foundational)
- **Depended on by**: All other swarms

## Summary
Swarm Dev is the foundational team that builds the agent infrastructure. Current focus is completing memory persistence and error recovery to enable true autonomous operation.
