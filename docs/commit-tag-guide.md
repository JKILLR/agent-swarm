# Automated Tracking: Commit Tag Guide

## Overview

Our agent swarm system uses **structured commit message tags** to automatically update task status across swarms. This eliminates manual synchronization and keeps Operations Swarm informed in real-time.

## Tag Format

Add tags to your commit messages using this format:

```
<conventional-commit-type>: <description> [swarm:NAME] [task:ID] [status:STATE]
```

### Tag Components

| Tag | Description | Example Values |
|-----|-------------|----------------|
| `[swarm:NAME]` | Target swarm identifier | `swarm_dev`, `asa_research`, `mynd_app`, `operations` |
| `[task:ID]` | Task/priority identifier | `P1`, `P2`, `testing-infrastructure` |
| `[status:STATE]` | New task status | `in_progress`, `completed`, `blocked`, `not_started` |

## Examples

### Example 1: Mark Testing Infrastructure Complete
```bash
git commit -m "feat: Add comprehensive testing infrastructure [swarm:swarm_dev] [task:P1] [status:completed]"
```

### Example 2: Update Research Task
```bash
git commit -m "docs: Document ASA attention mechanism [swarm:asa_research] [task:P2] [status:in_progress]"
```

### Example 3: Mark Task as Blocked
```bash
git commit -m "fix: Attempt API integration [swarm:mynd_app] [task:api-integration] [status:blocked]"
```

### Example 4: Multiple Changes (Use Multiple Commits)
```bash
git commit -m "feat: Implement feature A [swarm:swarm_dev] [task:P3] [status:completed]"
git commit -m "test: Add tests for feature A [swarm:swarm_dev] [task:P4] [status:completed]"
```

## How It Works

### Automatic Updates

1. **You commit** with structured tags
2. **Git hook** (`commit-msg`) parses the tags
3. **Swarm YAML** file is automatically updated
4. **Operations Swarm** sees changes in next sync

### What Gets Updated

The hook automatically updates these fields in `swarms/<swarm_name>/swarm.yaml`:
- `status`: Task status changed to your specified value
- `last_updated`: Timestamp of the update
- `notes`: Optionally adds commit reference

### Status Values

| Status | Meaning | When to Use |
|--------|---------|-------------|
| `not_started` | Task hasn't begun | Initial state |
| `in_progress` | Actively working | Work has started |
| `completed` | Task finished | Work is done |
| `blocked` | Can't proceed | Waiting on dependency |
| `on_hold` | Paused temporarily | Deprioritized |
| `cancelled` | Won't complete | No longer needed |

## Best Practices

### ✅ DO:
- Use tags for **significant milestones** (feature complete, major progress)
- Match task IDs **exactly** as they appear in swarm.yaml
- Use conventional commit format (`feat:`, `fix:`, `docs:`, etc.)
- Add descriptive commit messages beyond just tags

### ❌ DON'T:
- Tag every minor commit (reserve for meaningful updates)
- Guess task IDs (check swarm.yaml first)
- Use tags for commits outside swarm work
- Forget to push commits (tracking only works on pushed commits)

## Task ID Reference

### Swarm Dev (swarm_dev)
- `P1`: Testing infrastructure - COMPLETED
- `P2`: Feature parity with Claude Desktop
- `P3`: Self-modification capability with safety guardrails

### ASA Research (asa_research)
- Check `swarms/asa_research/swarm.yaml` for current task IDs

### MYND App (mynd_app)
- Check `swarms/mynd_app/swarm.yaml` for current task IDs

### Operations (operations)
- Check `swarms/operations/swarm.yaml` for current task IDs

## Troubleshooting

### Tag Not Working?

1. **Check hook is executable:**
   ```bash
   ls -la .git/hooks/commit-msg
   # Should show: -rwxr-xr-x
   ```

2. **Verify tag format:**
   - No spaces inside brackets: `[swarm:name]` not `[swarm: name]`
   - Correct swarm name matches directory name
   - Task ID matches swarm.yaml exactly

3. **Check hook output:**
   After commit, you should see:
   ```
   ✅ Auto-updated: swarm_dev/P1 → completed
   ```

### Manual Update

If automatic tracking fails, you can manually update:
```bash
# Edit the swarm.yaml file directly
vim swarms/<swarm_name>/swarm.yaml

# Then commit without tags
git commit -m "docs: Manual status update"
```

## Integration with Operations Swarm

The Operations Swarm's **ops_coordinator** periodically syncs with swarm.yaml files to:
- Track cross-swarm progress
- Identify blocked tasks
- Report status to COO
- Maintain organizational visibility

By using commit tags, you ensure Operations stays informed **automatically** without manual reporting overhead.

## Questions?

Contact the Operations Swarm or COO for:
- New swarm setup with tracking
- Custom task ID conventions
- Integration issues
- Feature requests for tracking system
