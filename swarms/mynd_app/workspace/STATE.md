# MYND App Comprehensive Code Review

## Review Type: READ-ONLY (NO CHANGES)
**CRITICAL: Do NOT modify any code. This is a review and analysis task only.**

## Repository Location
`/Users/jellingson/agent-swarm/swarms/mynd_app/workspace/mynd-v3`

## Project Overview

MYND is a mind-mapping application with an AI assistant called "Axel". The architecture consists of:

1. **Frontend** (`self-dev.html`) - Served locally via `python3 -m http.server 8080` on Mac
2. **Brain Server** (`mynd-brain/server.py`) - FastAPI Python server running on Runpod GPU, port 8420
3. **Claude CLI** - Chat functionality uses Claude Code CLI (Max subscription) instead of Anthropic API

## Current Status
- Review in progress
- Multiple agents analyzing different aspects

## Known Issues to Investigate

### PRIMARY ISSUE: LocalBrain Connection Not Working
The frontend keeps trying to reach an OLD Cloudflare tunnel URL (`maple-leisure-conference-ericsson.trycloudflare.com`) instead of the current one, even when the correct URL is passed via `?brain=` parameter.

**Key Questions:**
1. Why is the old Cloudflare URL cached/persisted?
2. Is `CONFIG.BRAIN_SERVER_URL` being evaluated correctly?
3. Is there localStorage or another cache storing the old URL?
4. Script load order issue - does `local-brain-client.js` load before `config.js`?
5. Is there somewhere else `serverUrl` is being set with hardcoded URLs?

### Files to Review (Priority Order)
1. `js/local-brain-client.js` - LocalBrain connection logic
2. `js/config.js` - Configuration including BRAIN_SERVER_URL
3. `js/app-module.js` - Main app, calls LocalBrain
4. `mynd-brain/server.py` - Python server, CORS config
5. `self-dev.html` - Check script load order

## Code Concerns
- **Code is bloated and complicated** - Any suggested fixes need to be careful and minimal
- Large files: `self-dev.html` (376KB), `index.html` (1.9MB)
- Mixed patterns and potential technical debt

## Review Areas

### 1. Architecture Review (architect agent)
- Overall code structure and organization
- Separation of concerns
- Module dependencies
- Data flow patterns
- Configuration management approach

### 2. Code Quality Review (critic agent)
- Bug patterns and potential issues
- Error handling
- Race conditions
- Memory leaks
- Security concerns
- Code duplication

### 3. Connection Issue Analysis (reviewer agent)
- Deep dive into LocalBrain connection flow
- URL handling and caching behavior
- Script initialization order
- localStorage usage
- CORS configuration

## Review Progress

| Agent | Status | Findings |
|-------|--------|----------|
| architect | pending | - |
| critic | pending | - |
| reviewer | pending | - |

## Findings Summary
(To be populated by review agents)

---

Last Updated: Starting review
