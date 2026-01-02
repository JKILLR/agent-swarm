# Automated Tracking System - Deployment Instructions

## What Has Been Created

The automated task tracking system (Recommendation 3) has been fully implemented with the following components:

### 📁 Files Created:

1. **docs/commit-tag-guide.md** - Complete user guide for commit tag format
2. **docs/git-hooks-setup.md** - Git hook installation instructions
3. **tools/sync_tracking.py** - Manual tracking sync utility
4. **.github/workflows/update-tracking.yml** - CI/CD automation
5. **swarms/operations/briefings/** - Operations Swarm sync documentation

### 🔧 What It Does:

- **Auto-updates task status** when you commit with tags like `[swarm:name] [task:id] [status:state]`
- **Syncs swarm.yaml files** automatically based on git commits
- **Notifies Operations Swarm** of all changes
- **CI/CD integration** for automated tracking on push

## 🚀 To Complete Deployment:

### Step 1: Install Git Hook (Local)

```bash
# Create and install the commit-msg hook
cat > .git/hooks/commit-msg << 'EOF'
#!/usr/bin/env python3
import sys, re, yaml
from pathlib import Path
from datetime import datetime

commit_msg_file = sys.argv[1]
with open(commit_msg_file, 'r') as f:
    commit_msg = f.read()

swarm_match = re.search(r'\[swarm:(\w+)\]', commit_msg)
task_match = re.search(r'\[task:([\w-]+)\]', commit_msg)
status_match = re.search(r'\[status:(\w+)\]', commit_msg)

if swarm_match and task_match and status_match:
    swarm_name, task_id, new_status = swarm_match.group(1), task_match.group(1), status_match.group(1)
    swarm_yaml_path = Path(__file__).parent.parent.parent / 'swarms' / swarm_name / 'swarm.yaml'
    if swarm_yaml_path.exists():
        try:
            with open(swarm_yaml_path, 'r') as f:
                swarm_config = yaml.safe_load(f)
            if 'priorities' in swarm_config:
                for priority in swarm_config['priorities']:
                    if priority.get('id') == task_id or priority.get('title', '').lower().replace(' ', '-') == task_id.lower():
                        priority['status'] = new_status
                        priority['last_updated'] = datetime.now().strftime('%Y-%m-%d')
                        with open(swarm_yaml_path, 'w') as f:
                            yaml.dump(swarm_config, f, default_flow_style=False, sort_keys=False)
                        print(f"✅ Auto-updated: {swarm_name}/{task_id} → {new_status}")
                        break
        except Exception as e:
            print(f"⚠️  Warning: {e}", file=sys.stderr)
sys.exit(0)
EOF

chmod +x .git/hooks/commit-msg
```

### Step 2: Commit and Push

```bash
# Stage all new files
git add docs/ tools/ .github/workflows/update-tracking.yml swarms/operations/

# Commit with tracking tag (this will test the hook!)
git commit -m "feat: Implement automated task tracking system [swarm:operations] [task:automated-tracking] [status:completed]

Comprehensive automated tracking infrastructure:
- Git commit-msg hook for auto-updates
- Manual sync tool (tools/sync_tracking.py)
- GitHub Actions workflow integration
- Full documentation and setup guides

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"

# Push to repository
git push origin main
```

### Step 3: Verify Installation

```bash
# Test the manual sync tool
python tools/sync_tracking.py swarm_dev P1 completed

# Verify git hook is installed
ls -la .git/hooks/commit-msg
# Should show: -rwxr-xr-x
```

## 📖 Usage Examples

### Example 1: Mark Task Complete
```bash
git commit -m "feat: Add new feature [swarm:swarm_dev] [task:P2] [status:completed]"
# Output: ✅ Auto-updated: swarm_dev/P2 → completed
```

### Example 2: Update Progress
```bash
git commit -m "wip: Working on feature [swarm:mynd_app] [task:user-auth] [status:in_progress]"
```

### Example 3: Mark as Blocked
```bash
git commit -m "fix: Attempted fix [swarm:asa_research] [task:benchmarking] [status:blocked]"
```

## 🎯 Benefits

- **Zero manual tracking overhead** - Status updates happen automatically
- **Operations Swarm stays informed** - Real-time visibility across all swarms
- **Audit trail** - Git history shows all status changes
- **CI/CD integration** - Automated workflows keep everything in sync

## 📋 Next Steps After Deployment

1. **Test the system** with a small commit
2. **Update Operations Swarm** config to monitor automated updates
3. **Train team members** on commit tag format
4. **Monitor GitHub Actions** to ensure CI/CD tracking works

## ❓ Questions?

See full documentation:
- `docs/commit-tag-guide.md` - Complete usage guide
- `docs/git-hooks-setup.md` - Installation details
- Contact Operations Swarm for support
