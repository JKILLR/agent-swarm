# Claude Code Skills - Technical Analysis

**Date**: 2026-01-05
**Focus**: Technical limitations, gotchas, what skills CANNOT do, comparison to MCP

---

## Executive Summary

Claude Code Skills are markdown-based instruction packages that teach Claude domain-specific workflows. While powerful, they have significant technical limitations including unreliable auto-invocation, token budget constraints, platform dependencies, and gaps in subagent integration. This analysis documents real-world issues from GitHub bug reports, community findings, and technical comparisons with MCP.

---

## 1. Technical Limitations

### 1.1 Token/Character Budget Limits

**Critical Issue**: Skills have a hard character limit that can silently prevent activation.

- As of Claude Code 2.0.70, the limit for skill and command descriptions defaults to **15,000 characters** (~4,000 tokens)
- **No warning when exceeded** - skills simply won't appear in Claude's context
- System prompt tells Claude never to use skills that aren't listed
- With multiple plugins installed, skills can consume **50k+ tokens** before any conversation starts

**GitHub Issue #14882**: Skills consume full token count at startup instead of progressive disclosure (frontmatter only). Only frontmatter (~100 tokens per skill) should be loaded at startup, but the full content is being loaded.

**Source**: [Claude Code skills not triggering? It might not see them](https://blog.fsck.com/2025/12/17/claude-code-skills-not-triggering/)

### 1.2 Unreliable Auto-Invocation

**The Problem**: Skills are supposed to activate autonomously based on descriptions, but this fails frequently.

| Approach | Success Rate |
|----------|-------------|
| Simple instructions | ~20% |
| Basic hook approach | ~50% |
| Optimized WHEN/WHEN NOT pattern | 80-84% |

**Root Cause**: There is **no algorithmic routing or intent classification**. The system:
- Formats all available skills into text in the Skill tool's prompt
- Relies entirely on Claude's language model to make selection decisions
- Description quality directly determines auto-invocation accuracy

**Solution Patterns**:
```yaml
# BAD - vague description
description: Helps with documents

# GOOD - structured WHEN/WHEN NOT pattern
description: |
  Generate PDF reports from data.
  USE WHEN: user asks for PDF output, report generation, document export.
  DO NOT USE WHEN: viewing existing PDFs, simple text output.
```

**Source**: [How to Make Claude Code Skills Activate Reliably](https://scottspence.com/posts/how-to-make-claude-code-skills-activate-reliably)

### 1.3 Subagent Limitations

**Critical Restriction**: Built-in agents (Explore, Plan, Verify) and the Task tool do **NOT** have access to your Skills.

Only custom subagents defined in `.claude/agents/` with an explicit `skills` field can use Skills:

```yaml
# .claude/agents/my-agent.yaml
name: my-custom-agent
skills:
  - my-skill-name
```

**Implication**: You cannot leverage skills in the standard exploration, planning, or verification workflows without custom agent definitions.

**Source**: [Agent Skills - Claude Code Docs](https://code.claude.com/docs/en/skills)

### 1.4 Environment and Platform Restrictions

| Limitation | Details |
|------------|---------|
| **Plan Required** | Custom skills require Pro, Max, Team, or Enterprise plan |
| **Code Execution Required** | Skills require code execution to be enabled |
| **No Runtime Package Installation** | All dependencies must be pre-installed; cannot install packages at runtime with API Skills |
| **Sandboxed Execution** | Skills run in restricted sandbox with no persistence between sessions |
| **Windows Issues** | Skills fail to load on Windows due to path normalization and glob pattern incompatibility (Issue #11620) |

### 1.5 Skills Cannot Explicitly Reference Each Other

Skills cannot directly call or reference other skills. While Claude can use multiple skills together automatically, there's no mechanism for skill composition or chaining at the skill definition level.

---

## 2. Known Bugs and Issues (from GitHub)

### 2.1 Discovery and Loading Failures

| Issue | Description | Status |
|-------|-------------|--------|
| **#14577** | `/skills` command shows "No skills found" despite skills being loaded | Open |
| **#14733** | User skills in `~/.claude/skills/` not appearing in `/skills` output | Open |
| **#9716** | Claude not aware of skills in `.claude/skills/` directory | Open |
| **#11266** | User skills not auto-discovered despite correct structure | Open |
| **#9954** | Built-in skills not available in Claude Code | Open |
| **#11004** | Skills not loaded after `/clear` command without restart | Open |

### 2.2 Platform-Specific Bugs

| Issue | Description |
|-------|-------------|
| **#11620** | Windows: Skills fail with "Error: unknown skill" due to path backslash issues |
| **#10001** | Skills system references non-existent `file_read` tool |
| **#9928** | Incorrect metadata labels "(project, gitignored)" for user-level skills |

### 2.3 Integration Failures

| Issue | Description |
|-------|-------------|
| **#10568** | Marketplace skills not exposed through Skill tool |
| **#10766** | Skills not triggered in Plan Mode |
| **#12178** | Plugin-dev skills fail due to missing `build.sh` script |
| **#14549** | Skill inflation - 3218 skills reported when only 169 exist |
| **SDK #36** | Skills not discovered in Claude Agent SDK despite correct configuration |

**Source**: [GitHub anthropics/claude-code issues](https://github.com/anthropics/claude-code/issues)

---

## 3. What Skills CANNOT Do

### 3.1 Fundamental Limitations

| Cannot Do | Explanation |
|-----------|-------------|
| **Connect to external systems** | Skills cannot make API calls, database connections, or network requests - that's MCP's job |
| **Persist state between sessions** | Sandbox isolation prevents any state persistence |
| **Run outside Claude ecosystem** | Skills require Claude's agent framework and file system |
| **Install packages at runtime** | All dependencies must be pre-installed |
| **Access restricted system resources** | Sandbox blocks SSH keys, system configs, etc. |
| **Override tool permissions** | Skills can restrict tools but cannot grant additional permissions |
| **Recover from context compaction** | After compaction, skills may need to be re-read |

### 3.2 Tool Restriction Gotchas

**The `allowed-tools` field has limitations**:

1. **Inconsistent with agent permissions** - Agents use `tools` field, skills use `allowed-tools`
2. **No path-based matching** - Cannot restrict Edit/Write to specific paths within skill
3. **Denial bypass possible** - Denying `Read(**/.env)` only blocks Read tool; Bash with `cat` can still access the file

```yaml
# This restriction can be bypassed
allowed-tools: Read, Grep, Glob
# Bash is blocked, but if user has Bash permissions elsewhere,
# they could still cat sensitive files
```

### 3.3 Context Window Behavior

| Issue | Impact |
|-------|--------|
| **Context compaction** | Claude is "dumber" after compaction; forgets skills it was using |
| **No selective unloading** | Cannot unload a skill once it's in context |
| **Token competition** | Skills compete with conversation history and MCP tool definitions |
| **Performance degradation** | Avoid using final 20% of context window for complex tasks |

---

## 4. Skills vs MCP: Technical Comparison

### 4.1 Architecture Differences

| Aspect | Skills | MCP |
|--------|--------|-----|
| **Purpose** | Teach Claude HOW to do things | Give Claude access to WHAT it needs |
| **Format** | Markdown + YAML frontmatter | JSON-RPC 2.0 with typed schemas |
| **Transport** | File system read | stdio, HTTP+SSE, custom transports |
| **Execution** | Runs in Claude's context | Runs in separate server process |
| **Token Loading** | Progressive disclosure (metadata → body → resources) | All tool definitions loaded upfront |
| **Portability** | Claude ecosystem only | Vendor-neutral, works across AI systems |

### 4.2 Token Efficiency

**Skills win on token efficiency**:

```
Skills:
├── Startup: ~30-50 tokens per skill (name + description only)
├── On-demand: Full content loaded when relevant
└── Progressive: Resources loaded only when needed

MCP:
├── Startup: ALL tool definitions loaded upfront
├── GitHub MCP alone: ~10,000+ tokens
└── Multiple MCPs: Can consume majority of context before work begins
```

**Example**: Sentry MCP consumes ~8k tokens just by being loaded.

### 4.3 When to Use Each

| Use Case | Skills | MCP |
|----------|--------|-----|
| **Workflow procedures** | Yes | No |
| **Style guides & conventions** | Yes | No |
| **API access** | No | Yes |
| **Database connections** | No | Yes |
| **External data sources** | No | Yes |
| **Code formatting rules** | Yes | No |
| **Tool orchestration logic** | Yes | No |
| **Real-time data fetching** | No | Yes |

### 4.4 Why Both Are Needed

**The pattern**: MCP provides connectivity, Skills provide procedural knowledge.

```
Without Skills: Claude can access tools but must figure out correct sequence each time
Without MCP: Claude knows procedures but cannot connect to external systems

Together: MCP gives access, Skills give the logic
```

**Rule of thumb**:
- MCP instructions cover HOW TO USE the server and its tools correctly
- Skill instructions cover HOW TO USE them for a given process or multi-server workflow

### 4.5 Why NOT to Expose Skills via MCP

Exposing Skills through MCP would:
1. **Negate progressive disclosure** - Adds protocol overhead for simple filesystem read
2. **Create redundant abstraction** - Skills already require local code execution
3. **Defeat both purposes** - Skills for context efficiency, MCP for standardized integration

---

## 5. Edge Cases and Gotchas

### 5.1 YAML Frontmatter Errors

Common configuration mistakes that silently break skills:

```yaml
# WRONG: Blank line before frontmatter

---
name: my-skill
---

# WRONG: Tabs instead of spaces
---
name: my-skill
	description: Uses tab	# Will fail

# WRONG: Missing closing ---
---
name: my-skill
description: No closing delimiter

# CORRECT:
---
name: my-skill
description: Brief description here
---
```

### 5.2 Path and Permission Issues

```bash
# Scripts need execute permissions
chmod +x scripts/*.py

# Use forward slashes (Unix style) even concept
scripts/helper.py    # Correct
scripts\helper.py    # Wrong (Windows-style breaks cross-platform)
```

### 5.3 Similar Skills Confusion

If Claude uses the wrong skill or seems confused between similar skills, their descriptions are too similar. Make descriptions distinct with specific trigger conditions.

### 5.4 Context Compaction Recovery

After context compaction:
- Claude doesn't know what files it was looking at
- Will make mistakes you corrected earlier in session
- May need to re-read skill content

**Workaround**: Use `/compact` manually when you notice Claude getting lost, or `/clear` and re-prompt for severe cases.

### 5.5 Task Sizing

> "The smaller and more isolated the problem, the better."

Breaking large tasks into small, isolated pieces is critical. One developer spent 2 days and ~$100 in tokens arguing with Claude Code on a combined task that exhibited many failure modes.

---

## 6. Security Considerations

### 6.1 Sandbox Protection

Skills run in OS-level sandbox with:
- **Filesystem isolation**: Only specific directories accessible
- **Network isolation**: Internet access only through controlled proxy
- **No persistence**: State doesn't survive between sessions

This protects against:
- Prompt injection leading to code execution
- Exfiltration of SSH keys or credentials
- Backdoor installation in system resources

### 6.2 Allowed-Tools Security

```yaml
# DON'T list every tool - defeats security model
allowed-tools: Read, Write, Edit, Bash, Grep, Glob, WebFetch...

# DO use minimum necessary
allowed-tools: Read, Grep, Glob
```

### 6.3 Risky Permissions

Avoid allowing writes to:
- Directories containing executables in $PATH
- System configuration directories
- User shell configuration files (.bashrc, .zshrc)

---

## 7. Best Practices Summary

### 7.1 Description Writing

```yaml
# Include WHEN and WHEN NOT patterns
description: |
  Generate test cases using PICT for pairwise coverage.
  USE WHEN: user needs test case generation, combinatorial testing,
  parameter coverage analysis.
  DO NOT USE WHEN: running existing tests, unit test debugging.
```

### 7.2 Skill Structure

```
my-skill/
├── SKILL.md          # Keep under 500 lines
├── scripts/          # Executable code (chmod +x)
├── references/       # Docs loaded on-demand
└── assets/           # Templates, binary files
```

### 7.3 Testing

- Test with incomplete/unusual inputs
- Document what skill cannot do
- Handle edge cases gracefully
- Test incrementally after each change

### 7.4 Token Management

- Keep individual skill descriptions concise
- Use progressive disclosure for large content
- Monitor total skill count vs. 15k character budget
- Check `/skills` output to verify activation

---

## 8. Community Resources

### 8.1 Official Repositories

- [anthropics/skills](https://github.com/anthropics/skills) - Official examples (docx, pdf, pptx, xlsx, etc.)
- [Agent Skills Docs](https://code.claude.com/docs/en/skills) - Official documentation

### 8.2 Community Collections

- [travisvn/awesome-claude-skills](https://github.com/travisvn/awesome-claude-skills) - Curated list
- [VoltAgent/awesome-claude-skills](https://github.com/VoltAgent/awesome-claude-skills) - Resources collection
- [alirezarezvani/claude-skills](https://github.com/alirezarezvani/claude-skills) - Real-world usage examples
- [ComposioHQ/awesome-claude-skills](https://github.com/ComposioHQ/awesome-claude-skills) - Integration-focused

### 8.3 Technical Deep Dives

- [Claude Agent Skills: A First Principles Deep Dive](https://leehanchung.github.io/blogs/2025/10/26/claude-skills-deep-dive/)
- [Inside Claude Code Skills: Structure, prompts, invocation](https://mikhail.io/2025/10/claude-code-skills/)
- [Skills vs Dynamic MCP Loadouts](https://lucumr.pocoo.org/2025/12/13/skills-vs-mcp/)
- [Claude Code Gotchas](https://www.dolthub.com/blog/2025-06-30-claude-code-gotchas/)

---

## 9. Key Takeaways

1. **Skills are NOT magic** - 20-50% failure rate without optimized descriptions
2. **Token budget is real** - 15k character limit can silently disable skills
3. **Subagents can't use skills** - Only custom agents with explicit `skills` field
4. **MCP vs Skills is NOT either/or** - Use MCP for connectivity, Skills for procedures
5. **Windows support is broken** - Path handling issues prevent skill loading
6. **Context compaction hurts** - Skills may need re-reading after compaction
7. **Test incrementally** - Complex skills fail in surprising ways
8. **Security sandbox is strong** - But `allowed-tools` can be bypassed via Bash

---

## Sources

- [Agent Skills - Claude Code Docs](https://code.claude.com/docs/en/skills)
- [Claude Skills vs. MCP: A Technical Comparison](https://intuitionlabs.ai/articles/claude-skills-vs-mcp)
- [Skills explained: How Skills compares to prompts, Projects, MCP, and subagents](https://claude.com/blog/skills-explained)
- [Claude Code skills not triggering? It might not see them](https://blog.fsck.com/2025/12/17/claude-code-skills-not-triggering/)
- [How to Make Claude Code Skills Activate Reliably](https://scottspence.com/posts/how-to-make-claude-code-skills-activate-reliably)
- [GitHub anthropics/claude-code issues](https://github.com/anthropics/claude-code/issues)
- [Claude Code Gotchas - DoltHub](https://www.dolthub.com/blog/2025-06-30-claude-code-gotchas/)
- [Making Claude Code more secure and autonomous](https://www.anthropic.com/engineering/claude-code-sandboxing)
- [Equipping agents for the real world with Agent Skills](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills)
- [Claude Agent Skills: A First Principles Deep Dive](https://leehanchung.github.io/blogs/2025/10/26/claude-skills-deep-dive/)
- [Skills vs Dynamic MCP Loadouts](https://lucumr.pocoo.org/2025/12/13/skills-vs-mcp/)
