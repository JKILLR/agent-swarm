# Boris's Claude Code Usage: Analysis for Agent-Swarm Optimization

**Date**: 2026-01-03
**Analyst**: Researcher Agent
**Source**: Twitter thread from Boris (Claude Code creator)

---

## Executive Summary

Boris's 13-point thread reveals production-hardened patterns from someone who built Claude Code. Our agent-swarm system has implemented approximately 40% of these patterns, with several high-impact opportunities remaining. The most actionable insights are:

1. **Slash Commands** (missing) - Could dramatically speed up common workflows
2. **PostToolUse Hooks for Formatting** (missing) - Low-effort quality improvement
3. **Verification Loops** (partially missing) - Key to "2-3x quality"
4. **Background Agent for Long Tasks** (missing) - Would solve our concurrency model

---

## Point-by-Point Analysis

### Point 1: Parallel Execution (5 Claudes in Terminal)

**Boris's Practice**: Runs 5 Claude instances in parallel in terminal tabs with system notifications for input.

**Our Current State**: PARTIALLY IMPLEMENTED
- We have `AgentExecutorPool` in `/Users/jellingson/agent-swarm/shared/agent_executor_pool.py` with semaphore-based concurrency limits (default 5)
- We have parallel Task tool execution capability documented in `/.claude/workflow.md`
- We have `ParallelTasks` tool documented for multi-agent work

**Gap Analysis**:
- No system notifications when agents need input (agents run without human attention)
- Our WebSocket chat bypasses the AgentExecutorPool (known critical issue)
- Task tool does NOT spawn real agents (known critical issue from STATE.md)

**Impact if Fixed**: HIGH - Enables true parallel development velocity

**Recommended Actions**:
1. Fix Task tool to spawn real agents via AgentExecutorPool (already in Known Issues #12)
2. Add notification hooks when agents are blocked waiting for input
3. Wire WebSocket chat through AgentExecutorPool (already in Next Steps #16)

---

### Point 2: Multi-Platform Parallelism (claude.ai/code + local)

**Boris's Practice**: Runs 5-10 additional Claudes on claude.ai/code in parallel. Uses `&` for session handoff, `--teleport` to switch between.

**Our Current State**: NOT IMPLEMENTED
- Single-instance model per chat session
- No handoff mechanism between agents
- No teleport capability

**Gap Analysis**:
- We use Agent Mailbox (`/Users/jellingson/agent-swarm/shared/agent_mailbox.py`) but it's not wired to Task flow
- No cross-instance session continuation

**Impact if Fixed**: MEDIUM - Useful for human operators, less relevant for automated swarm

**Recommended Actions**:
1. Wire Agent Mailbox to Task completion results (already in Next Steps #20)
2. Consider session export/import for debugging handoff scenarios

---

### Point 3: Model Choice (Opus 4.5 for Everything)

**Boris's Practice**: Uses Opus 4.5 with thinking for everything. "Even though it's bigger & slower, you have to steer it less and it's better at tool use."

**Our Current State**: IMPLEMENTED
- `/.claude/workflow.md` lines 246-252 explicitly state: "CRITICAL: All agents MUST use `claude-opus-4-5-20251101`"
- Already enforcing Opus 4.5 for all agents

**Gap Analysis**: None - we're aligned with Boris's recommendation.

**Impact**: Already realized.

---

### Point 4: Shared CLAUDE.md Checked into Git

**Boris's Practice**: Team shares a single CLAUDE.md, checked into git. "Anytime Claude does something incorrectly, add it to CLAUDE.md."

**Our Current State**: PARTIALLY IMPLEMENTED
- We have `/.claude/workflow.md` (411 lines) which serves similar purpose
- It IS checked into git
- Contains "What NOT to Do" section with common errors

**Gap Analysis**:
- We don't have a CLAUDE.md at root (we have workflow.md instead)
- Not clear if agents update workflow.md when they make mistakes

**Impact if Improved**: MEDIUM - Could create a learning loop for error prevention

**Recommended Actions**:
1. Add guidance for COO/Critic to append to workflow.md when errors occur
2. Consider renaming workflow.md to CLAUDE.md for standard compatibility
3. Add a "Lessons Learned" section that agents can append to

---

### Point 5: Code Review Integration (@.claude on PRs)

**Boris's Practice**: Tags @.claude on PRs to add things to CLAUDE.md. Uses Claude Code GitHub action. Calls it "Compounding Engineering."

**Our Current State**: NOT IMPLEMENTED
- No GitHub action for Claude
- No automated PR review integration
- Manual review process only

**Gap Analysis**: Significant automation opportunity

**Impact if Fixed**: HIGH - Enables continuous learning from PR feedback

**Recommended Actions**:
1. Set up Claude Code GitHub Action for PR reviews
2. Configure action to suggest CLAUDE.md updates when catching errors
3. Create `/api/add-to-claudemd` endpoint for automated additions

---

### Point 6: Plan Mode First (Shift+Tab Twice)

**Boris's Practice**: Most sessions start in Plan mode. Goes back and forth until plan is good, THEN switches to auto-accept edits. "A good plan is really important!"

**Our Current State**: PARTIALLY IMPLEMENTED
- We have Architect agent that designs before implementation
- We have Delegation Pipeline: researcher -> architect -> implementer -> critic -> tester
- STATE.md captures planning decisions in Architecture Decisions section

**Gap Analysis**:
- No explicit Plan Mode for COO interactions
- No formal design approval gate before implementation

**Impact if Improved**: MEDIUM - Would improve design quality

**Recommended Actions**:
1. Add explicit "design review" step where Critic reviews Architect's plan
2. Consider adding `--plan` flag to agent execution for planning-only mode
3. Add ADR requirement before major implementations

---

### Point 7: Slash Commands for Inner Loops

**Boris's Practice**: Uses slash commands for every workflow done many times a day. Commands checked into `.claude/commands/`. Example: `/commit-push-pr` uses inline bash to pre-compute git status.

**Our Current State**: NOT IMPLEMENTED
- No `.claude/commands/` directory
- No custom slash commands
- Repetitive workflows done manually

**Gap Analysis**: Major opportunity - we have many repetitive workflows:
- Starting a task with STATE.md reading
- Committing with proper format
- Running validation
- Creating PRs

**Impact if Fixed**: HIGH - Immediate productivity gain

**Recommended Actions**:
1. Create `.claude/commands/` directory
2. Create common commands:
   - `/read-state` - Read STATE.md for current context
   - `/commit-push` - Stage, commit with convention, push
   - `/validate` - Run syntax checks and tests
   - `/new-task` - Initialize new task with STATE.md update
   - `/review-pr` - Run critic agent on PR changes
3. Add commands for swarm-specific workflows

---

### Point 8: Subagents for Workflows

**Boris's Practice**: Uses subagents like code-simplifier, verify-app. "Think of subagents as automating the most common workflows."

**Our Current State**: FULLY IMPLEMENTED
- We have 5 subagents defined in `/.claude/agents/`:
  - `researcher.md` - Deep research
  - `architect.md` - System design
  - `implementer.md` - Code implementation
  - `critic.md` - Code review
  - `tester.md` - Test execution
- Additionally, each swarm has its own agents in `swarms/{name}/agents/`

**Gap Analysis**:
- We may be missing specialized workflow agents (code-simplifier, verify-app equivalent)
- Our subagents are role-based rather than workflow-based

**Impact if Improved**: MEDIUM - Could add specialized workflow agents

**Recommended Actions**:
1. Consider adding specialized agents:
   - `simplifier.md` - Simplify/refactor code after implementation
   - `verifier.md` - Verify implementations work correctly
   - `reviewer.md` - PR-style code review
2. Review if existing agents cover all needed workflows

---

### Point 9: PostToolUse Hook for Formatting

**Boris's Practice**: Formats Claude's code automatically. "Claude usually generates well-formatted code, and the hook handles the last 10%."

**Our Current State**: NOT IMPLEMENTED
- We have `PreToolUse` hooks but no `PostToolUse` hooks
- Our hooks are for coordination, not quality
- No auto-formatting on file writes

**Gap Analysis**: Easy win for code quality

**Impact if Fixed**: MEDIUM-HIGH - Consistent code formatting with minimal effort

**Recommended Actions**:
1. Add `PostToolUse` hook for Write/Edit tools
2. Run formatters based on file type:
   - `.py` -> `black` or `ruff format`
   - `.ts`/`.tsx` -> `prettier`
   - `.json` -> `jq .`
3. Create `/Users/jellingson/agent-swarm/scripts/hooks/post_write.py`

---

### Point 10: Permissions Management (/permissions)

**Boris's Practice**: Doesn't use `--dangerously-skip-permissions`. Uses `/permissions` to pre-allow safe bash commands. Checked into `.claude/settings.json`.

**Our Current State**: IMPLEMENTED
- We have `/.claude/settings.json` with permissions:
  ```json
  {
    "permissions": {
      "allow": [
        "WebSearch", "WebFetch",
        "Bash(curl:*)", "Bash(wget:*)",
        "Bash(git clone:*)", "Bash(git pull:*)",
        "Bash(git push:*)", "Bash(git fetch:*)"
      ]
    }
  }
  ```

**Gap Analysis**:
- Could expand allowed commands (npm, python, pytest, etc.)
- No documentation about using `/permissions` to add more

**Recommended Actions**:
1. Expand permissions to include common dev commands:
   - `Bash(npm:*)`
   - `Bash(python:*)`
   - `Bash(pytest:*)`
   - `Bash(pip:*)`
2. Document permission management process

---

### Point 11: Tool Integration (Slack MCP, BigQuery, Sentry)

**Boris's Practice**: Claude uses all his tools - Slack MCP server, BigQuery CLI, Sentry for error logs. Config checked into `.mcp.json`.

**Our Current State**: NOT IMPLEMENTED
- No `.mcp.json` file found
- No MCP server integrations
- Tools are built-in Claude tools only

**Gap Analysis**: We could integrate with external tools for better coordination

**Impact if Fixed**: MEDIUM - Depends on which integrations would be useful

**Recommended Actions**:
1. Evaluate MCP servers that would benefit agent-swarm:
   - GitHub MCP for issue/PR management
   - Slack MCP for notifications
   - Memory MCP for persistent context
2. Create `.mcp.json` with relevant integrations
3. Consider Sentry for error tracking in production

---

### Point 12: Long-Running Task Strategies

**Boris's Practice**: Three strategies:
1. Prompt Claude to verify with background agent when done
2. Use agent Stop hook for deterministic verification
3. Use ralph-wiggum plugin
4. Use `--permission-mode=dontAsk` or `--dangerously-skip-permissions` in sandbox

**Our Current State**: PARTIALLY IMPLEMENTED
- We have `SubagentStop` hook in settings.json that calls `agent_complete.py`
- Hook updates task status in coordination database
- No background verification agent
- No ralph-wiggum equivalent

**Gap Analysis**:
- Missing verification loop (most important pattern per Boris)
- No background agent for quality checks

**Impact if Fixed**: HIGH - Boris says verification is "most important thing"

**Recommended Actions**:
1. Add verification prompt to agent completion hook
2. Create background verifier agent that runs after implementer completes
3. Integrate verification into Task completion flow
4. Add deterministic checks (syntax, tests) to SubagentStop hook

---

### Point 13: Verification is Key (2-3x Quality)

**Boris's Practice**: "Probably the most important thing to get great results - give Claude a way to verify its work. If Claude has that feedback loop, it will 2-3x the quality." Uses Claude Chrome extension to test UI changes.

**Our Current State**: PARTIALLY IMPLEMENTED
- We have Critic agent for code review
- We have Tester agent for test execution
- Delegation pipeline includes review step

**Gap Analysis**:
- Verification is not automatic - requires COO to trigger
- No self-verification loop within agent tasks
- No UI testing capability for frontend changes

**Impact if Fixed**: HIGH - "2-3x quality" is Boris's claim

**Recommended Actions**:
1. Add automatic verification to SubagentStop hook
2. Create verification slash command
3. Add Playwright/browser testing capability for UI changes
4. Make verifier agent run automatically after implementer
5. Add instruction to all agent prompts: "Verify your work before reporting complete"

---

## Priority Ranking of Improvements

Based on impact and effort analysis:

### Tier 1: High Impact, Low-Medium Effort

| Priority | Improvement | Impact | Effort | Quick Win? |
|----------|------------|--------|--------|------------|
| 1 | **Slash Commands** (.claude/commands/) | HIGH | LOW | Yes |
| 2 | **PostToolUse Formatting Hook** | MEDIUM-HIGH | LOW | Yes |
| 3 | **Auto-Verification Loop** | HIGH | MEDIUM | No |
| 4 | **Expand Permissions** | MEDIUM | LOW | Yes |

### Tier 2: High Impact, High Effort

| Priority | Improvement | Impact | Effort | Depends On |
|----------|------------|--------|--------|------------|
| 5 | **Fix Task Tool Spawning** | HIGH | HIGH | Architecture |
| 6 | **GitHub Action Integration** | HIGH | MEDIUM | CI/CD |
| 7 | **WebSocket through Pool** | HIGH | HIGH | Architecture |

### Tier 3: Medium Impact, Variable Effort

| Priority | Improvement | Impact | Effort | Notes |
|----------|------------|--------|--------|-------|
| 8 | **MCP Server Integration** | MEDIUM | MEDIUM | Nice to have |
| 9 | **Workflow-Based Subagents** | MEDIUM | MEDIUM | Specialization |
| 10 | **CLAUDE.md Learning Loop** | MEDIUM | LOW | Cultural change |

---

## Recommended Quick Wins (Can implement today)

### 1. Create Slash Commands Directory

```bash
mkdir -p /Users/jellingson/agent-swarm/.claude/commands
```

Example command - `/read-state.md`:
```markdown
Read the current STATE.md to understand context:

$ARGUMENTS

Context:
$(cat workspace/STATE.md 2>/dev/null || echo "No STATE.md in workspace")
```

### 2. Add PostToolUse Formatting Hook

Add to `/.claude/settings.json`:
```json
{
  "PostToolUse": [
    {
      "matcher": "Write|Edit",
      "hooks": [
        {"type": "command", "command": "./scripts/hooks/post_write.py"}
      ]
    }
  ]
}
```

### 3. Expand Permissions

Update `/.claude/settings.json`:
```json
{
  "permissions": {
    "allow": [
      "WebSearch", "WebFetch",
      "Bash(curl:*)", "Bash(wget:*)",
      "Bash(git:*)",
      "Bash(npm:*)", "Bash(npx:*)",
      "Bash(python:*)", "Bash(python3:*)",
      "Bash(pytest:*)", "Bash(pip:*)"
    ]
  }
}
```

### 4. Add Verification to Agent Prompts

Add to all agent .md files:
```markdown
## Verification Requirement

Before reporting completion:
1. Verify syntax: `python3 -m py_compile` for .py files
2. Run relevant tests if modified test files
3. Check for import errors
4. Confirm changes match the request
```

---

## Connections to Known Issues

These improvements relate to existing issues in STATE.md:

| Boris Insight | Related Known Issue | Connection |
|---------------|---------------------|------------|
| Parallel Execution | #12 Task tool doesn't spawn real agents | Same root cause |
| Verification Loop | None documented | NEW priority |
| Slash Commands | None documented | NEW opportunity |
| Agent Mailbox | #8 Work Ledger/Mailbox not integrated | Same gap |
| WebSocket Pool | #7 Main WebSocket bypasses pool | Same issue |

---

## Conclusion

Boris's thread validates our architectural direction while exposing specific gaps:

**We're doing well on**:
- Model choice (Opus 4.5 only)
- Subagent structure (role-based agents)
- Permission management (settings.json)
- Workflow documentation (workflow.md)

**We're missing**:
- Slash commands (easy win)
- PostToolUse hooks (easy win)
- Verification loops (Boris's top recommendation)
- True parallel agent spawning (known critical issue)
- GitHub Action integration (compounding engineering)

**Most impactful single change**: Add automatic verification loop to SubagentStop hook. Boris claims this alone provides "2-3x quality improvement."

---

## Files Referenced

| File | Purpose |
|------|---------|
| `/Users/jellingson/agent-swarm/.claude/settings.json` | Permissions and hooks configuration |
| `/Users/jellingson/agent-swarm/.claude/workflow.md` | Development workflow documentation |
| `/Users/jellingson/agent-swarm/.claude/agents/*.md` | Subagent definitions |
| `/Users/jellingson/agent-swarm/scripts/hooks/pre_task.py` | Task coordination hook |
| `/Users/jellingson/agent-swarm/scripts/hooks/agent_complete.py` | Agent completion hook |
| `/Users/jellingson/agent-swarm/shared/agent_executor_pool.py` | Parallel execution pool |
| `/Users/jellingson/agent-swarm/shared/agent_mailbox.py` | Agent communication system |
| `/Users/jellingson/agent-swarm/swarms/swarm_dev/workspace/STATE.md` | Current system state |
